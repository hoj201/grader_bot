import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import numpy as np
import pymupdf
import pytest
from moto import mock_aws

from graderbot.models import Box
from graderbot.storage import (
    NameEmbeddingRecord,
    NameImageRecord,
    WorksheetRecord,
    _default_s3_client,
    compute_sty_hash,
    delete_from_s3,
    delete_student,
    delete_worksheet,
    deserialize_boxes,
    generate_answer_key_pdf,
    generate_presigned_url,
    get_or_create_classroom,
    get_or_create_student,
    get_worksheet_by_public_id,
    image_to_pdf,
    images_to_pdf,
    import_students_csv,
    init_db,
    insert_name_embedding,
    insert_name_image,
    insert_worksheet,
    list_classrooms,
    list_name_images,
    list_students,
    list_unembedded_name_images,
    list_worksheets,
    name_image_exists,
    parse_s3_url,
    record_sty_version,
    serialize_boxes,
    slugify_title,
    store_worksheet,
    upload_to_s3,
)
from graderbot.worksheet_synth import WORKSHEET_STY_PATH
from graderbot.worksheetbot import Question


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
        "header",
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


def test_init_db_adds_header_column_to_pre_existing_table(tmp_path):
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
            title TEXT,
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
    assert "header" in columns


def test_init_db_creates_sty_version_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(STY_VERSION)")}

    assert columns == {"hash", "content", "created_at"}


def test_init_db_creates_classroom_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(CLASSROOM)")}

    assert columns == {"id", "label", "created_at"}


def test_init_db_creates_student_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(STUDENT)")}

    assert columns == {"id", "classroom_id", "first_name", "last_name", "nickname", "created_at"}


def test_init_db_creates_name_images_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(NAME_IMAGES)")}

    assert columns == {
        "id",
        "student_id",
        "box_id",
        "image_s3url",
        "image_sha256",
        "created_at",
    }


def test_init_db_creates_name_embeddings_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(NAME_EMBEDDINGS)")}

    assert columns == {"id", "student_id", "name_image_id", "embedding_s3url", "created_at"}


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


def test_insert_worksheet_persists_header(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record(header="Show your work.")

    new_id = insert_worksheet(conn, record)

    row = conn.execute("SELECT header FROM WORKSHEET WHERE id = ?", (new_id,)).fetchone()
    assert row == ("Show your work.",)


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
    fake_images = [np.zeros((10, 10, 3), dtype=np.uint8)]
    out_path = tmp_path / "answers.pdf"

    with patch("graderbot.storage.fill_worksheet", return_value=fake_images) as mock_fill, \
         patch("graderbot.storage.images_to_pdf", return_value=out_path) as mock_convert:
        result = generate_answer_key_pdf("demo.tex", {"1": "2"}, out_path)

    mock_fill.assert_called_once_with("demo.tex", {"1": "2"})
    mock_convert.assert_called_once_with(fake_images, out_path)
    assert result == out_path


def test_generate_answer_key_pdf_passes_every_page_through(tmp_path):
    """A multi-page worksheet must produce a multi-page answer key: every image
    fill_worksheet returns is forwarded to images_to_pdf in order."""
    fake_images = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.full((10, 10, 3), 255, dtype=np.uint8),
    ]
    out_path = tmp_path / "answers.pdf"

    with patch("graderbot.storage.fill_worksheet", return_value=fake_images), \
         patch("graderbot.storage.images_to_pdf", return_value=out_path) as mock_convert:
        generate_answer_key_pdf("demo.tex", {"1": "2"}, out_path)

    (passed_images, _), _ = mock_convert.call_args
    assert passed_images is fake_images


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

    with patch("graderbot.storage.latexmk_worksheet", side_effect=fake_latexmk) as mock_latexmk, \
         patch("graderbot.storage.generate_answer_key_pdf", return_value=answers_pdf) as mock_answer_key, \
         patch("graderbot.storage.extract_answer_boxes", return_value=sample_boxes) as mock_boxes, \
         patch("graderbot.storage.upload_to_s3", side_effect=lambda path, bucket, key, s3_client=None: f"https://{bucket}.s3.amazonaws.com/{key}") as mock_upload:
        record = store_worksheet(
            tex_path=tex_path,
            questions=questions,
            prompt="algebra worksheet",
            model="claude-sonnet-4-6",
            bucket="graderbot-test-bucket",
            db_path=db_path,
            public_id="ws_abcd1234",
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
    assert record.student_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/ws_abcd1234/worksheet_student.pdf"
    assert record.cv_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/ws_abcd1234/worksheet_cv.pdf"
    assert record.answers_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/ws_abcd1234/worksheet_answers.pdf"
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

    with patch("graderbot.storage.latexmk_worksheet", side_effect=fake_latexmk), \
         patch("graderbot.storage.generate_answer_key_pdf", return_value=answers_pdf), \
         patch("graderbot.storage.extract_answer_boxes", return_value={"1": Box(0.1, 0.2, 0.3, 0.1)}), \
         patch("graderbot.storage.upload_to_s3", side_effect=lambda path, bucket, key, s3_client=None: f"https://{bucket}.s3.amazonaws.com/{key}"):
        record = store_worksheet(
            tex_path=tex_path,
            questions=questions,
            prompt="algebra worksheet",
            model="claude-sonnet-4-6",
            bucket="graderbot-test-bucket",
            db_path=db_path,
            title="Linear Equations!",
            header="Show all work.",
            public_id="ws_abcd1234",
        )

    assert record.title == "Linear Equations!"
    assert record.header == "Show all work."
    assert record.student_pdf_s3url == (
        "https://graderbot-test-bucket.s3.amazonaws.com/ws_abcd1234/Linear_Equations_student.pdf"
    )
    assert record.cv_pdf_s3url == (
        "https://graderbot-test-bucket.s3.amazonaws.com/ws_abcd1234/Linear_Equations_cv.pdf"
    )
    assert record.answers_pdf_s3url == (
        "https://graderbot-test-bucket.s3.amazonaws.com/ws_abcd1234/Linear_Equations_answers.pdf"
    )


def test_store_worksheet_same_title_distinct_public_ids_no_key_collision(tmp_path):
    """Two worksheets sharing a title must get distinct S3 keys (issue #33):
    the unique public_id namespaces the key so the second upload can't
    overwrite the first's PDFs."""
    tex_path = tmp_path / "worksheet.tex"
    tex_path.write_text(r"\documentclass{article}\begin{document}hi\end{document}")
    db_path = tmp_path / "worksheets.sqlite3"
    questions = [Question(id="1", text="1+1=?", answer="2")]

    student_pdf = tmp_path / "build_blank" / "worksheet.pdf"
    cv_pdf = tmp_path / "build_cv" / "worksheet.pdf"
    answers_pdf = tmp_path / "answers.pdf"

    def fake_latexmk(tex_filename, cv_mode):
        return str(cv_pdf if cv_mode else student_pdf)

    def store(public_id):
        with patch("graderbot.storage.latexmk_worksheet", side_effect=fake_latexmk), \
             patch("graderbot.storage.generate_answer_key_pdf", return_value=answers_pdf), \
             patch("graderbot.storage.extract_answer_boxes", return_value={"1": Box(0.1, 0.2, 0.3, 0.1)}), \
             patch("graderbot.storage.upload_to_s3", side_effect=lambda path, bucket, key, s3_client=None: f"https://{bucket}.s3.amazonaws.com/{key}"):
            return store_worksheet(
                tex_path=tex_path,
                questions=questions,
                prompt="algebra worksheet",
                model="claude-sonnet-4-6",
                bucket="graderbot-test-bucket",
                db_path=db_path,
                title="Fractions",
                public_id=public_id,
            )

    first = store("ws_11111111")
    second = store("ws_22222222")

    assert first.student_pdf_s3url != second.student_pdf_s3url
    assert first.cv_pdf_s3url != second.cv_pdf_s3url
    assert first.answers_pdf_s3url != second.answers_pdf_s3url


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


# --------------------------------------------------------------------------
# delete_from_s3
# --------------------------------------------------------------------------

@mock_aws
def test_delete_from_s3_removes_object(tmp_path):
    bucket = "graderbot-test-bucket"
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)
    s3_client.put_object(Bucket=bucket, Key="worksheet/student.pdf", Body=b"pdf")

    delete_from_s3(bucket, "worksheet/student.pdf", s3_client=s3_client)

    with pytest.raises(s3_client.exceptions.NoSuchKey):
        s3_client.get_object(Bucket=bucket, Key="worksheet/student.pdf")


# --------------------------------------------------------------------------
# delete_worksheet
# --------------------------------------------------------------------------

@mock_aws
def test_delete_worksheet_removes_blobs_and_row(tmp_path):
    bucket = "graderbot-test-bucket"
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)
    keys = ("ws/student.pdf", "ws/cv.pdf", "ws/answers.pdf")
    for key in keys:
        s3_client.put_object(Bucket=bucket, Key=key, Body=b"pdf")

    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record(
        student_pdf_s3url=f"https://{bucket}.s3.amazonaws.com/ws/student.pdf",
        cv_pdf_s3url=f"https://{bucket}.s3.amazonaws.com/ws/cv.pdf",
        answers_pdf_s3url=f"https://{bucket}.s3.amazonaws.com/ws/answers.pdf",
    )
    record.id = insert_worksheet(conn, record)

    delete_worksheet(conn, record, s3_client=s3_client)

    assert list_worksheets(conn) == []
    for key in keys:
        with pytest.raises(s3_client.exceptions.NoSuchKey):
            s3_client.get_object(Bucket=bucket, Key=key)


@mock_aws
def test_delete_worksheet_skips_null_urls(tmp_path):
    bucket = "graderbot-test-bucket"
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)
    s3_client.put_object(Bucket=bucket, Key="ws/student.pdf", Body=b"pdf")

    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record(
        student_pdf_s3url=f"https://{bucket}.s3.amazonaws.com/ws/student.pdf",
        cv_pdf_s3url=None,
        answers_pdf_s3url=None,
    )
    record.id = insert_worksheet(conn, record)

    delete_worksheet(conn, record, s3_client=s3_client)  # must not raise on None urls

    assert list_worksheets(conn) == []


def test_delete_worksheet_aborts_and_keeps_row_when_s3_delete_fails(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record()
    record.id = insert_worksheet(conn, record)

    with patch("graderbot.storage.delete_from_s3", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            delete_worksheet(conn, record, s3_client=object())

    # Strict: the row must survive when an S3 delete fails.
    assert [r.id for r in list_worksheets(conn)] == [record.id]


# --------------------------------------------------------------------------
# CLASSROOM / STUDENT / NAME_IMAGES / NAME_EMBEDDINGS (issue #43)
# --------------------------------------------------------------------------

def test_get_or_create_classroom_is_idempotent(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    first = get_or_create_classroom(conn, "Room 101")
    second = get_or_create_classroom(conn, "Room 101")

    assert first.id == second.id
    assert [c.label for c in list_classrooms(conn)] == ["Room 101"]


def test_get_or_create_student_is_idempotent(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")

    first = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    second = get_or_create_student(conn, classroom.id, "Anna", "Smith")

    assert first.id == second.id
    assert len(list_students(conn, classroom.id)) == 1


def test_list_students_scoped_to_classroom(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    room_a = get_or_create_classroom(conn, "Room A")
    room_b = get_or_create_classroom(conn, "Room B")
    get_or_create_student(conn, room_a.id, "Anna", "Smith")
    get_or_create_student(conn, room_b.id, "Zeke", "Jones")

    names = [(s.first_name, s.last_name) for s in list_students(conn, room_a.id)]
    assert names == [("Anna", "Smith")]


def test_import_students_csv_adds_students_with_nickname_column(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    csv_text = "first_name,last_name,nickname\nAnna,Smith,\nZeke,Jones,Z\n"

    result = import_students_csv(conn, classroom.id, csv_text)

    assert [(s.first_name, s.last_name, s.nickname) for s in result.added] == [
        ("Anna", "Smith", None),
        ("Zeke", "Jones", "Z"),
    ]
    assert result.skipped == []
    assert len(list_students(conn, classroom.id)) == 2


def test_import_students_csv_works_without_nickname_column(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    csv_text = "first_name,last_name\nAnna,Smith\n"

    result = import_students_csv(conn, classroom.id, csv_text)

    assert [(s.first_name, s.last_name) for s in result.added] == [("Anna", "Smith")]
    assert result.skipped == []


def test_import_students_csv_is_case_insensitive_to_column_order_and_case(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    csv_text = "Last_Name,First_Name\nSmith,Anna\n"

    result = import_students_csv(conn, classroom.id, csv_text)

    assert [(s.first_name, s.last_name) for s in result.added] == [("Anna", "Smith")]


def test_import_students_csv_skips_rows_missing_a_name(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    csv_text = "first_name,last_name\nAnna,Smith\n,Jones\nZeke,\n"

    result = import_students_csv(conn, classroom.id, csv_text)

    assert [(s.first_name, s.last_name) for s in result.added] == [("Anna", "Smith")]
    assert result.skipped == [
        "row 3: missing first or last name",
        "row 4: missing first or last name",
    ]


def test_import_students_csv_ignores_blank_lines(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    csv_text = "first_name,last_name\nAnna,Smith\n\nZeke,Jones\n"

    result = import_students_csv(conn, classroom.id, csv_text)

    assert len(result.added) == 2
    assert result.skipped == []


def test_import_students_csv_is_idempotent_with_existing_students(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    get_or_create_student(conn, classroom.id, "Anna", "Smith")
    csv_text = "first_name,last_name\nAnna,Smith\nZeke,Jones\n"

    import_students_csv(conn, classroom.id, csv_text)

    assert len(list_students(conn, classroom.id)) == 2


def test_import_students_csv_raises_on_missing_required_columns(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    csv_text = "name,nickname\nAnna Smith,\n"

    with pytest.raises(ValueError, match="missing required column"):
        import_students_csv(conn, classroom.id, csv_text)


def test_import_students_csv_raises_on_empty_csv(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")

    with pytest.raises(ValueError, match="empty"):
        import_students_csv(conn, classroom.id, "")


def test_insert_name_image_and_exists(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    student = get_or_create_student(conn, classroom.id, "Anna", "Smith")

    assert not name_image_exists(conn, "sha123")

    record = NameImageRecord(
        student_id=student.id,
        box_id="name1",
        image_s3url="https://bucket.s3.amazonaws.com/handwriting/1/1/sha123.png",
        image_sha256="sha123",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    insert_name_image(conn, record)

    assert name_image_exists(conn, "sha123")
    assert len(list_name_images(conn, student.id)) == 1


def test_list_unembedded_name_images_excludes_embedded(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    student = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    image_id = insert_name_image(
        conn,
        NameImageRecord(
            student_id=student.id,
            box_id="name1",
            image_s3url="https://bucket.s3.amazonaws.com/img.png",
            image_sha256="sha123",
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    assert len(list_unembedded_name_images(conn)) == 1

    insert_name_embedding(
        conn,
        NameEmbeddingRecord(
            student_id=student.id,
            name_image_id=image_id,
            embedding_s3url="https://bucket.s3.amazonaws.com/vec.npy",
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    assert list_unembedded_name_images(conn) == []


@mock_aws
def test_delete_student_removes_blobs_and_rows(tmp_path):
    bucket = "graderbot-test-bucket"
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)
    s3_client.put_object(Bucket=bucket, Key="img.png", Body=b"png")
    s3_client.put_object(Bucket=bucket, Key="vec.npy", Body=b"vec")

    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    student = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    image_id = insert_name_image(
        conn,
        NameImageRecord(
            student_id=student.id,
            box_id="name1",
            image_s3url=f"https://{bucket}.s3.amazonaws.com/img.png",
            image_sha256="sha123",
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    insert_name_embedding(
        conn,
        NameEmbeddingRecord(
            student_id=student.id,
            name_image_id=image_id,
            embedding_s3url=f"https://{bucket}.s3.amazonaws.com/vec.npy",
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    delete_student(conn, student.id, s3_client=s3_client)

    assert list_students(conn, classroom.id) == []
    assert list_name_images(conn) == []
    for key in ("img.png", "vec.npy"):
        with pytest.raises(s3_client.exceptions.NoSuchKey):
            s3_client.get_object(Bucket=bucket, Key=key)


def test_delete_student_aborts_and_keeps_rows_when_s3_delete_fails(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    student = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    insert_name_image(
        conn,
        NameImageRecord(
            student_id=student.id,
            box_id="name1",
            image_s3url="https://bucket.s3.amazonaws.com/img.png",
            image_sha256="sha123",
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    with patch("graderbot.storage.delete_from_s3", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            delete_student(conn, student.id, s3_client=object())

    # Strict: rows must survive when an S3 delete fails.
    assert len(list_students(conn, classroom.id)) == 1
    assert len(list_name_images(conn)) == 1
