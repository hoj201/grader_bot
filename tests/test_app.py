from pathlib import Path

from streamlit.testing.v1 import AppTest

import storage

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _seed_worksheet(db_path):
    conn = storage.init_db(db_path)
    storage.insert_worksheet(
        conn,
        storage.WorksheetRecord(
            prompt="10 question algebra worksheet",
            tex_source=r"\documentclass{article}",
            questions_json="[]",
            git_sha="deadbeef",
            model="claude-sonnet-4-6",
            num_questions=10,
            student_pdf_s3url="https://bucket.s3.amazonaws.com/worksheet/student.pdf",
            cv_pdf_s3url="https://bucket.s3.amazonaws.com/worksheet/cv.pdf",
            answers_pdf_s3url="https://bucket.s3.amazonaws.com/worksheet/answers.pdf",
            created_at="2026-07-18T00:00:00+00:00",
        ),
    )
    conn.close()


def _set_env(monkeypatch, db_path):
    # app.py calls load_dotenv() on every run. python-dotenv's find_dotenv()
    # walks up from app.py's own location (not the cwd), so it will pick up
    # a real .env in the repo root and repopulate any var that monkeypatch
    # has deleted from os.environ. Neutralize it so these tests are hermetic
    # regardless of what's in a developer's local .env.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("WORKSHEETS_DB_PATH", str(db_path))
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Dummy credentials so boto3 can locally sign presigned URLs without
    # a real network call or real AWS account.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def test_gallery_tab_renders_seeded_worksheet_without_error(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_worksheet(db_path)
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    markdown_texts = " ".join(md.value for md in at.markdown)
    assert "10 question algebra worksheet" in markdown_texts


def test_gallery_tab_shows_empty_state_with_no_worksheets(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("No worksheets yet" in info.value for info in at.info)


def test_app_errors_when_bucket_not_configured(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)
    monkeypatch.delenv("S3_BUCKET", raising=False)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("S3_BUCKET" in err.value for err in at.error)
