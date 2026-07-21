import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import numpy as np
import pymupdf
import pytest
from moto import mock_aws

from graderbot import Box
from storage import (
    WorksheetRecord,
    _default_s3_client,
    compute_sty_hash,
    deserialize_boxes,
    generate_answer_key_pdf,
    generate_presigned_url,
    get_worksheet_by_public_id,
    image_to_pdf,
    images_to_pdf,
    init_db,
    insert_worksheet,
    list_worksheets,
    parse_s3_url,
    record_sty_version,
    serialize_boxes,
    slugify_title,
    store_worksheet,
    upload_to_s3,
)
from worksheet_synth import WORKSHEET_STY_PATH
from worksheetbot import Question


def _sample_record(**overrides) -> WorksheetRecord:
    fields = dict(
        prompt="10 question algebra worksheet",
        tex_source=r"\documentclass{article}\begin{document}hi\end{document}",
        questions_json='[{"id": "1", "text": "1+1=?", "answer": "2"}]',
        model="claude-sonnet-4-6",
        num_questions=1,
        student_pdf_s3url="https://bucket.s3.amazonaws.com/student.pdf",
        cv_pdf_s3url="https://bucket.s3.amazonaws.com/cv.pdf",
        answers_pdf_s3url="https://bucket.s3.amazonaws.com/answers.pdf",
        sty_hash="deadbeef",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    fields.update(overrides)
    return WorksheetRecord(**fields)


# --------------------------------------------------------------------------
# init_db / insert_worksheet
# --------------------------------------------------------------------------

def test_init_db_creates_worksheet_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}

    assert columns == {
        "id",
        "prompt",
        "tex_source",
        "questions_json",
        "model",
        "num_questions",
        "title",
        "public_id",
        "boxes_json",
        "student_pdf_s3url",
        "cv_pdf_s3url",
        "answers_pdf_s3url",
        "sty_hash",
        "created_at",
    }


def test_init_db_sets_wal_journal_mode(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "worksheets.sqlite3"
    init_db(db_path).close()

    conn = init_db(db_path)  # should not raise on pre-existing table

    assert conn.execute("SELECT COUNT(*) FROM WORKSHEET").fetchone()[0] == 0


def test_init_db_drops_legacy_git_sha_column(tmp_path):
    db_path = tmp_path / "worksheets.sqlite3"
    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.execute(
        """
        CREATE TABLE WORKSHEET (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            tex_source TEXT,
            questions_json TEXT,
            git_sha TEXT,
            model TEXT,
            num_questions INTEGER,
            student_pdf_s3url TEXT,
            cv_pdf_s3url TEXT,
            answers_pdf_s3url TEXT,
            created_at TEXT
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    conn = init_db(db_path)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    assert "git_sha" not in columns


def test_init_db_adds_sty_hash_column_to_pre_existing_table(tmp_path):
    db_path = tmp_path / "worksheets.sqlite3"
    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.execute(
        """
        CREATE TABLE WORKSHEET (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            tex_source TEXT,
            questions_json TEXT,
            model TEXT,
            num_questions INTEGER,
            student_pdf_s3url TEXT,
            cv_pdf_s3url TEXT,
            answers_pdf_s3url TEXT,
            created_at TEXT
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    conn = init_db(db_path)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    assert "sty_hash" in columns


def test_init_db_adds_title_column_to_pre_existing_table(tmp_path):
    db_path = tmp_path / "worksheets.sqlite3"
    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.execute(
        """
        CREATE TABLE WORKSHEET (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            tex_source TEXT,
            questions_json TEXT,
            model TEXT,
            num_questions INTEGER,
            student_pdf_s3url TEXT,
            cv_pdf_s3url TEXT,
            answers_pdf_s3url TEXT,
            sty_hash TEXT,
            created_at TEXT
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    conn = init_db(db_path)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    assert "title" in columns


def test_init_db_creates_sty_version_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(STY_VERSION)")}

    assert columns == {"hash", "content", "created_at"}


def test_insert_worksheet_returns_id_and_persists_row(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record()

    new_id = insert_worksheet(conn, record)

    row = conn.execute(
        "SELECT prompt, tex_source, questions_json, model, num_questions, "
        "student_pdf_s3url, cv_pdf_s3url, answers_pdf_s3url, sty_hash, created_at "
        "FROM WORKSHEET WHERE id = ?",
        (new_id,),
    ).fetchone()

    assert row == (
        record.prompt,
        record.tex_source,
        record.questions_json,
        record.model,
        record.num_questions,
        record.student_pdf_s3url,
        record.cv_pdf_s3url,
        record.answers_pdf_s3url,
        record.sty_hash,
        record.created_at,
    )


def test_insert_worksheet_persists_title(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record(title="Linear Equations Practice")

    new_id = insert_worksheet(conn, record)

    row = conn.execute("SELECT title FROM WORKSHEET WHERE id = ?", (new_id,)).fetchone()
    assert row == ("Linear Equations Practice",)


def test_insert_worksheet_allows_null_title(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record(title=None)

    new_id = insert_worksheet(conn, record)

    row = conn.execute("SELECT title FROM WORKSHEET WHERE id = ?", (new_id,)).fetchone()
    assert row == (None,)


def test_insert_worksheet_allows_null_pdf_urls(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record(cv_pdf_s3url=None, answers_pdf_s3url=None)

    new_id = insert_worksheet(conn, record)

    row = conn.execute(
        "SELECT cv_pdf_s3url, answers_pdf_s3url FROM WORKSHEET WHERE id = ?",
        (new_id,),
    ).fetchone()
    assert row == (None, None)


# --------------------------------------------------------------------------
# compute_sty_hash / record_sty_version
# --------------------------------------------------------------------------

def test_compute_sty_hash_is_deterministic_for_same_content(tmp_path):
    sty_path = tmp_path / "gbworksheet.sty"
    sty_path.write_text(r"\ProvidesPackage{gbworksheet}[2026/07/08 Worksheet Layout]")

    assert compute_sty_hash(sty_path) == compute_sty_hash(sty_path)


def test_compute_sty_hash_differs_for_different_content(tmp_path):
    sty_a = tmp_path / "a.sty"
    sty_b = tmp_path / "b.sty"
    sty_a.write_text("version one")
    sty_b.write_text("version two")

    assert compute_sty_hash(sty_a) != compute_sty_hash(sty_b)


def test_record_sty_version_inserts_row_and_returns_hash(tmp_path):
    sty_path = tmp_path / "gbworksheet.sty"
    sty_path.write_text("version one")
    conn = init_db(tmp_path / "worksheets.sqlite3")

    sty_hash = record_sty_version(conn, sty_path)

    assert sty_hash == compute_sty_hash(sty_path)
    row = conn.execute(
        "SELECT content FROM STY_VERSION WHERE hash = ?", (sty_hash,)
    ).fetchone()
    assert row == ("version one",)


def test_record_sty_version_is_idempotent_for_unchanged_content(tmp_path):
    sty_path = tmp_path / "gbworksheet.sty"
    sty_path.write_text("version one")
    conn = init_db(tmp_path / "worksheets.sqlite3")

    record_sty_version(conn, sty_path)
    record_sty_version(conn, sty_path)

    count = conn.execute("SELECT COUNT(*) FROM STY_VERSION").fetchone()[0]
    assert count == 1


def test_record_sty_version_adds_new_row_when_content_changes(tmp_path):
    sty_path = tmp_path / "gbworksheet.sty"
    sty_path.write_text("version one")
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record_sty_version(conn, sty_path)

    sty_path.write_text("version two")
    record_sty_version(conn, sty_path)

    count = conn.execute("SELECT COUNT(*) FROM STY_VERSION").fetchone()[0]
    assert count == 2


# --------------------------------------------------------------------------
# slugify_title
# --------------------------------------------------------------------------

def test_slugify_title_replaces_spaces_and_punctuation():
    assert slugify_title("Linear Equations, Grade 9!") == "Linear_Equations_Grade_9"


def test_slugify_title_strips_leading_and_trailing_separators():
    assert slugify_title("  Fractions!!  ") == "Fractions"


def test_slugify_title_falls_back_to_worksheet_for_empty_slug():
    assert slugify_title("!!!") == "worksheet"


# --------------------------------------------------------------------------
# _default_s3_client
# --------------------------------------------------------------------------

def test_default_s3_client_uses_aws_region_env_var(monkeypatch):
    """boto3 only auto-reads AWS_DEFAULT_REGION, not AWS_REGION (which this
    repo's .env/README use), so an unset region silently defaults to
    us-east-1 and breaks presigned URLs for buckets in other regions."""
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    client = _default_s3_client()

    assert client.meta.region_name == "us-east-2"


# --------------------------------------------------------------------------
# upload_to_s3
# --------------------------------------------------------------------------

@mock_aws
def test_upload_to_s3_returns_url_and_object_exists(tmp_path):
    bucket = "graderbot-test-bucket"
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)

    local_file = tmp_path / "worksheet.pdf"
    local_file.write_bytes(b"%PDF-1.4 fake pdf bytes")

    url = upload_to_s3(local_file, bucket, "worksheets/1/student.pdf", s3_client=s3_client)

    assert url == f"https://{bucket}.s3.amazonaws.com/worksheets/1/student.pdf"
    obj = s3_client.get_object(Bucket=bucket, Key="worksheets/1/student.pdf")
    assert obj["Body"].read() == b"%PDF-1.4 fake pdf bytes"


# --------------------------------------------------------------------------
# image_to_pdf
# --------------------------------------------------------------------------

def test_image_to_pdf_writes_single_page_readable_pdf(tmp_path):
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    out_path = tmp_path / "answers.pdf"

    result = image_to_pdf(image, out_path)

    assert result == out_path
    assert out_path.exists()
    doc = pymupdf.open(out_path)
    assert doc.page_count == 1


def test_images_to_pdf_writes_one_page_per_image(tmp_path):
    images = [np.full((100, 200, 3), 255, dtype=np.uint8) for _ in range(3)]
    out_path = tmp_path / "batch.pdf"

    result = images_to_pdf(images, out_path)

    assert result == out_path
    assert pymupdf.open(out_path).page_count == 3


def test_images_to_pdf_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError):
        images_to_pdf([], tmp_path / "empty.pdf")


# --------------------------------------------------------------------------
# generate_answer_key_pdf
# --------------------------------------------------------------------------

def test_generate_answer_key_pdf_fills_worksheet_and_converts_to_pdf(tmp_path):
    fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
    out_path = tmp_path / "answers.pdf"

    with patch("storage.fill_worksheet", return_value=fake_image) as mock_fill, \
         patch("storage.image_to_pdf", return_value=out_path) as mock_convert:
        result = generate_answer_key_pdf("demo.tex", {"1": "2"}, out_path)

    mock_fill.assert_called_once_with("demo.tex", {"1": "2"})
    mock_convert.assert_called_once_with(fake_image, out_path)
    assert result == out_path


# --------------------------------------------------------------------------
# store_worksheet orchestration
# --------------------------------------------------------------------------

def test_store_worksheet_orchestrates_compile_upload_and_insert(tmp_path):
    tex_path = tmp_path / "worksheet.tex"
    tex_path.write_text(r"\documentclass{article}\begin{document}hi\end{document}")
    db_path = tmp_path / "worksheets.sqlite3"
    questions = [Question(id="1", text="1+1=?", answer="2")]

    student_pdf = tmp_path / "build_blank" / "worksheet.pdf"
    cv_pdf = tmp_path / "build_cv" / "worksheet.pdf"
    answers_pdf = tmp_path / "answers.pdf"

    def fake_latexmk(tex_filename, cv_mode):
        return str(cv_pdf if cv_mode else student_pdf)

    sample_boxes = {"1": Box(0.1, 0.2, 0.3, 0.1), "name": Box(0.1, 0.9, 0.4, 0.05)}

    with patch("storage.latexmk_worksheet", side_effect=fake_latexmk) as mock_latexmk, \
         patch("storage.generate_answer_key_pdf", return_value=answers_pdf) as mock_answer_key, \
         patch("storage.extract_answer_boxes", return_value=sample_boxes) as mock_boxes, \
         patch("storage.upload_to_s3", side_effect=lambda path, bucket, key, s3_client=None: f"https://{bucket}.s3.amazonaws.com/{key}") as mock_upload:
        record = store_worksheet(
            tex_path=tex_path,
            questions=questions,
            prompt="algebra worksheet",
            model="claude-sonnet-4-6",
            bucket="graderbot-test-bucket",
            db_path=db_path,
        )

    mock_boxes.assert_called_once_with(str(cv_pdf))
    assert deserialize_boxes(record.boxes_json) == sample_boxes
    assert mock_latexmk.call_count == 2
    mock_latexmk.assert_any_call(str(tex_path), cv_mode=False)
    mock_latexmk.assert_any_call(str(tex_path), cv_mode=True)
    mock_answer_key.assert_called_once_with(str(tex_path), {"1": "2"}, tmp_path / "worksheet_answers.pdf")
    assert mock_upload.call_count == 3

    assert record.id is not None
    assert record.prompt == "algebra worksheet"
    assert record.model == "claude-sonnet-4-6"
    assert record.num_questions == 1
    assert record.tex_source == tex_path.read_text()
    assert record.questions_json == '[{"id": "1", "text": "1+1=?", "answer": "2"}]'
    assert record.student_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/worksheet_student.pdf"
    assert record.cv_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/worksheet_cv.pdf"
    assert record.answers_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/worksheet_answers.pdf"
    assert record.sty_hash == compute_sty_hash()

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT prompt, sty_hash FROM WORKSHEET WHERE id = ?", (record.id,)).fetchone()
    assert row == ("algebra worksheet", record.sty_hash)

    sty_row = conn.execute(
        "SELECT content FROM STY_VERSION WHERE hash = ?", (record.sty_hash,)
    ).fetchone()
    assert sty_row == (WORKSHEET_STY_PATH.read_text(),)


def test_store_worksheet_uses_slugified_title_as_filename_prefix(tmp_path):
    tex_path = tmp_path / "worksheet.tex"
    tex_path.write_text(r"\documentclass{article}\begin{document}hi\end{document}")
    db_path = tmp_path / "worksheets.sqlite3"
    questions = [Question(id="1", text="1+1=?", answer="2")]

    student_pdf = tmp_path / "build_blank" / "worksheet.pdf"
    cv_pdf = tmp_path / "build_cv" / "worksheet.pdf"
    answers_pdf = tmp_path / "answers.pdf"

    def fake_latexmk(tex_filename, cv_mode):
        return str(cv_pdf if cv_mode else student_pdf)

    with patch("storage.latexmk_worksheet", side_effect=fake_latexmk), \
         patch("storage.generate_answer_key_pdf", return_value=answers_pdf), \
         patch("storage.extract_answer_boxes", return_value={"1": Box(0.1, 0.2, 0.3, 0.1)}), \
         patch("storage.upload_to_s3", side_effect=lambda path, bucket, key, s3_client=None: f"https://{bucket}.s3.amazonaws.com/{key}"):
        record = store_worksheet(
            tex_path=tex_path,
            questions=questions,
            prompt="algebra worksheet",
            model="claude-sonnet-4-6",
            bucket="graderbot-test-bucket",
            db_path=db_path,
            title="Linear Equations!",
        )

    assert record.title == "Linear Equations!"
    assert record.student_pdf_s3url == (
        "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/Linear_Equations_student.pdf"
    )
    assert record.cv_pdf_s3url == (
        "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/Linear_Equations_cv.pdf"
    )
    assert record.answers_pdf_s3url == (
        "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/Linear_Equations_answers.pdf"
    )


# --------------------------------------------------------------------------
# list_worksheets
# --------------------------------------------------------------------------

def test_list_worksheets_returns_rows_ordered_by_created_at_desc(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    older_id = insert_worksheet(conn, _sample_record(prompt="older", created_at="2026-01-01T00:00:00+00:00"))
    newer_id = insert_worksheet(conn, _sample_record(prompt="newer", created_at="2026-06-01T00:00:00+00:00"))

    records = list_worksheets(conn)

    assert [r.id for r in records] == [newer_id, older_id]
    assert [r.prompt for r in records] == ["newer", "older"]
    assert records[0].student_pdf_s3url == "https://bucket.s3.amazonaws.com/student.pdf"


def test_list_worksheets_round_trips_title(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    insert_worksheet(conn, _sample_record(title="Fractions Warmup"))

    records = list_worksheets(conn)

    assert records[0].title == "Fractions Warmup"


def test_list_worksheets_empty_db_returns_empty_list(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    assert list_worksheets(conn) == []


# --------------------------------------------------------------------------
# boxes_json: serialize/deserialize + persistence
# --------------------------------------------------------------------------

def test_serialize_boxes_round_trips():
    boxes = {
        "add001": Box(0.1, 0.2, 0.3, 0.05),
        "name": Box(0.1, 0.9, 0.4, 0.05),
    }

    assert deserialize_boxes(serialize_boxes(boxes)) == boxes


def test_insert_worksheet_round_trips_boxes_json(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    boxes_json = serialize_boxes({"1": Box(0.1, 0.2, 0.3, 0.05)})
    insert_worksheet(conn, _sample_record(boxes_json=boxes_json))

    records = list_worksheets(conn)

    assert records[0].boxes_json == boxes_json


def test_init_db_adds_boxes_json_column_to_pre_existing_table(tmp_path):
    db_path = tmp_path / "worksheets.sqlite3"
    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.execute(
        """
        CREATE TABLE WORKSHEET (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            tex_source TEXT,
            questions_json TEXT,
            model TEXT,
            num_questions INTEGER,
            student_pdf_s3url TEXT,
            cv_pdf_s3url TEXT,
            answers_pdf_s3url TEXT,
            sty_hash TEXT,
            created_at TEXT
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    conn = init_db(db_path)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(WORKSHEET)")}
    assert "boxes_json" in columns


# --------------------------------------------------------------------------
# get_worksheet_by_public_id
# --------------------------------------------------------------------------

def test_get_worksheet_by_public_id_returns_matching_record(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    insert_worksheet(conn, _sample_record(public_id="ws_aaaa1111", prompt="target"))
    insert_worksheet(conn, _sample_record(public_id="ws_bbbb2222", prompt="other"))

    record = get_worksheet_by_public_id(conn, "ws_aaaa1111")

    assert record is not None
    assert record.public_id == "ws_aaaa1111"
    assert record.prompt == "target"


def test_get_worksheet_by_public_id_returns_none_when_missing(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    insert_worksheet(conn, _sample_record(public_id="ws_aaaa1111"))

    assert get_worksheet_by_public_id(conn, "ws_nope0000") is None


# --------------------------------------------------------------------------
# parse_s3_url
# --------------------------------------------------------------------------

def test_parse_s3_url_round_trips_upload_to_s3_output():
    url = "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/student.pdf"

    bucket, key = parse_s3_url(url)

    assert bucket == "graderbot-test-bucket"
    assert key == "worksheet/student.pdf"


# --------------------------------------------------------------------------
# generate_presigned_url
# --------------------------------------------------------------------------

@mock_aws
def test_generate_presigned_url_contains_bucket_and_key():
    bucket = "graderbot-test-bucket"
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)

    url = generate_presigned_url(bucket, "worksheet/student.pdf", s3_client=s3_client)

    assert bucket in url
    assert "worksheet/student.pdf" in url
