import sqlite3
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import numpy as np
import pymupdf
import pytest
from moto import mock_aws

from storage import (
    WorksheetRecord,
    generate_answer_key_pdf,
    generate_presigned_url,
    get_git_sha,
    image_to_pdf,
    init_db,
    insert_worksheet,
    list_worksheets,
    parse_s3_url,
    store_worksheet,
    upload_to_s3,
)
from worksheetbot import Question


def _sample_record(**overrides) -> WorksheetRecord:
    fields = dict(
        prompt="10 question algebra worksheet",
        tex_source=r"\documentclass{article}\begin{document}hi\end{document}",
        questions_json='[{"id": "1", "text": "1+1=?", "answer": "2"}]',
        git_sha="deadbeef",
        model="claude-sonnet-4-6",
        num_questions=1,
        student_pdf_s3url="https://bucket.s3.amazonaws.com/student.pdf",
        cv_pdf_s3url="https://bucket.s3.amazonaws.com/cv.pdf",
        answers_pdf_s3url="https://bucket.s3.amazonaws.com/answers.pdf",
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
        "git_sha",
        "model",
        "num_questions",
        "student_pdf_s3url",
        "cv_pdf_s3url",
        "answers_pdf_s3url",
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


def test_insert_worksheet_returns_id_and_persists_row(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    record = _sample_record()

    new_id = insert_worksheet(conn, record)

    row = conn.execute(
        "SELECT prompt, tex_source, questions_json, git_sha, model, num_questions, "
        "student_pdf_s3url, cv_pdf_s3url, answers_pdf_s3url, created_at "
        "FROM WORKSHEET WHERE id = ?",
        (new_id,),
    ).fetchone()

    assert row == (
        record.prompt,
        record.tex_source,
        record.questions_json,
        record.git_sha,
        record.model,
        record.num_questions,
        record.student_pdf_s3url,
        record.cv_pdf_s3url,
        record.answers_pdf_s3url,
        record.created_at,
    )


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
# get_git_sha
# --------------------------------------------------------------------------

def test_get_git_sha_returns_stripped_output_of_git_rev_parse():
    with patch("storage.subprocess.run") as mock_run:
        mock_run.return_value.stdout = "abc123\n"
        sha = get_git_sha()

    assert sha == "abc123"
    args, kwargs = mock_run.call_args
    assert args[0] == ["git", "rev-parse", "HEAD"]


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

    with patch("storage.latexmk_worksheet", side_effect=fake_latexmk) as mock_latexmk, \
         patch("storage.generate_answer_key_pdf", return_value=answers_pdf) as mock_answer_key, \
         patch("storage.upload_to_s3", side_effect=lambda path, bucket, key, s3_client=None: f"https://{bucket}.s3.amazonaws.com/{key}") as mock_upload, \
         patch("storage.get_git_sha", return_value="deadbeef"):
        record = store_worksheet(
            tex_path=tex_path,
            questions=questions,
            prompt="algebra worksheet",
            model="claude-sonnet-4-6",
            bucket="graderbot-test-bucket",
            db_path=db_path,
        )

    assert mock_latexmk.call_count == 2
    mock_latexmk.assert_any_call(str(tex_path), cv_mode=False)
    mock_latexmk.assert_any_call(str(tex_path), cv_mode=True)
    mock_answer_key.assert_called_once_with(str(tex_path), {"1": "2"}, tmp_path / "worksheet_answers.pdf")
    assert mock_upload.call_count == 3

    assert record.id is not None
    assert record.prompt == "algebra worksheet"
    assert record.model == "claude-sonnet-4-6"
    assert record.num_questions == 1
    assert record.git_sha == "deadbeef"
    assert record.tex_source == tex_path.read_text()
    assert record.questions_json == '[{"id": "1", "text": "1+1=?", "answer": "2"}]'
    assert record.student_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/student.pdf"
    assert record.cv_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/cv.pdf"
    assert record.answers_pdf_s3url == "https://graderbot-test-bucket.s3.amazonaws.com/worksheet/answers.pdf"

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT prompt FROM WORKSHEET WHERE id = ?", (record.id,)).fetchone()
    assert row == ("algebra worksheet",)


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


def test_list_worksheets_empty_db_returns_empty_list(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    assert list_worksheets(conn) == []


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
