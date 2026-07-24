import logging
from pathlib import Path

import boto3
from moto import mock_aws
from streamlit.testing.v1 import AppTest

from graderbot import scan_grader, storage

APP_PATH = str(Path(__file__).resolve().parent.parent / "graderbot" / "app.py")


def _seed_worksheet(db_path, title=None, questions_json="[]"):
    conn = storage.init_db(db_path)
    storage.insert_worksheet(
        conn,
        storage.WorksheetRecord(
            prompt="10 question algebra worksheet",
            tex_source=r"\documentclass{article}",
            questions_json=questions_json,
            model="claude-sonnet-4-6",
            num_questions=10,
            title=title,
            student_pdf_s3url="https://bucket.s3.amazonaws.com/worksheet/student.pdf",
            cv_pdf_s3url="https://bucket.s3.amazonaws.com/worksheet/cv.pdf",
            answers_pdf_s3url="https://bucket.s3.amazonaws.com/worksheet/answers.pdf",
            sty_hash="deadbeef",
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
    caption_texts = " ".join(c.value for c in at.caption)
    assert "sty=deadbeef" in caption_texts


def test_gallery_tab_shows_title_with_prompt_below_it(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_worksheet(db_path, title="Linear Equations Practice")
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    markdown_texts = " ".join(md.value for md in at.markdown)
    assert "Linear Equations Practice" in markdown_texts
    caption_texts = " ".join(c.value for c in at.caption)
    assert "10 question algebra worksheet" in caption_texts


def test_gallery_tab_shows_unknown_sty_version_when_hash_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.insert_worksheet(
        conn,
        storage.WorksheetRecord(
            prompt="legacy worksheet",
            tex_source=r"\documentclass{article}",
            questions_json="[]",
            model="claude-sonnet-4-6",
            num_questions=5,
            created_at="2026-07-18T00:00:00+00:00",
        ),
    )
    conn.close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    caption_texts = " ".join(c.value for c in at.caption)
    assert "sty=unknown" in caption_texts


def test_gallery_tab_shows_empty_state_with_no_worksheets(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("No worksheets yet" in info.value for info in at.info)


def test_grade_tab_renders_without_error(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    # The Grade tab's uploader and roster field are present in the app tree.
    assert any("Student work" in fu.label for fu in at.get("file_uploader"))


def test_grade_tab_uploader_accepts_pdf_jpeg_and_png(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    assert set(uploader.allowed_type) == {".pdf", ".jpg", ".jpeg", ".png"}


def test_grade_tab_writes_uploaded_png_with_png_suffix(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    seen_paths = []

    def fake_mark_scan(hws, roster, db_path, out_path, on_step=None):
        seen_paths.extend(str(p) for p in hws)
        return scan_grader.ScanBatchResult()

    monkeypatch.setattr("graderbot.scan_grader.mark_scan", fake_mark_scan)

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("photo.png", b"not-a-real-png", "image/png"))
    at.run()
    button = next(b for b in at.button if b.label == "Grade")
    button.click().run()

    assert not at.exception
    assert len(seen_paths) == 1
    assert seen_paths[0].endswith(".png")


def test_create_tab_has_model_selectbox_defaulting_to_haiku(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    model_selects = [sb for sb in at.selectbox if sb.label == "Claude model"]
    assert model_selects, "Create tab should expose a Claude model dropdown"
    assert model_selects[0].value == "claude-haiku-4-5"


def test_gallery_tab_exposes_questions_json_expander(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    questions_json = '[{"id": "1", "text": "$2+2=$", "answer": "4"}]'
    _seed_worksheet(db_path, questions_json=questions_json)
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    expander_labels = [ex.label for ex in at.expander]
    assert "View questions JSON" in expander_labels
    code_values = [c.value for c in at.code]
    assert questions_json in code_values


def test_create_tab_exposes_manual_json_entry(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    text_area_labels = [ta.label for ta in at.text_area]
    assert "Questions JSON" in text_area_labels
    button_labels = [b.label for b in at.button]
    assert "Create from JSON" in button_labels


def test_logging_is_configured_with_default_level(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert logging.getLogger("graderbot.app").getEffectiveLevel() == logging.INFO


def test_create_from_json_logs_created_worksheet(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    def fake_create(questions_json, template_path, out, title, header="", bucket=None,
                     db_path=None, on_step=None):
        record = storage.WorksheetRecord(
            id=1,
            prompt=title,
            tex_source=r"\documentclass{article}",
            questions_json=questions_json,
            model="manual",
            num_questions=1,
            title=title,
            created_at="2026-07-18T00:00:00+00:00",
        )
        return Path("unused.pdf"), [], record

    monkeypatch.setattr("graderbot.worksheetbot.create_worksheet_from_questions", fake_create)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_area(key="manual_questions_json").set_value(
        '[{"id": "1", "text": "$2+2=$", "answer": "4"}]'
    )
    at.text_input(key="manual_title").set_value("Test Worksheet")
    at.run()

    with caplog.at_level(logging.INFO, logger="graderbot.app"):
        button = next(b for b in at.button if b.label == "Create from JSON")
        button.click().run()

    assert not at.exception
    assert any("created worksheet id=1" in r.message for r in caplog.records)


@mock_aws
def test_delete_worksheet_logs_deletion(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_worksheet(db_path)
    _set_env(monkeypatch, db_path)
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bucket")

    at = AppTest.from_file(APP_PATH)
    at.run()
    ask_button = next(b for b in at.button if b.key and b.key.startswith("ask_delete_"))
    ask_button.click().run()

    with caplog.at_level(logging.INFO, logger="graderbot.app"):
        confirm_button = next(b for b in at.button if b.key and b.key.startswith("do_delete_"))
        confirm_button.click().run()

    assert not at.exception
    assert any("deleted worksheet id=" in r.message for r in caplog.records)


def test_roster_tab_vectorizes_samples_after_ingest(tmp_path, monkeypatch):
    from graderbot.name_dataset import IngestResult
    from graderbot.storage import NameImageRecord

    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.get_or_create_classroom(conn, "Room 101")
    conn.close()
    _set_env(monkeypatch, db_path)

    fake_result = IngestResult(
        records=[
            NameImageRecord(
                student_id=1,
                box_id="box-0",
                image_s3url="https://bucket.s3.amazonaws.com/name_images/1.png",
                image_sha256="deadbeef",
            )
        ],
        skipped=[],
    )
    monkeypatch.setattr(
        "graderbot.name_dataset.ingest_name_sheets", lambda *args, **kwargs: fake_result
    )
    vectorize_calls = []
    monkeypatch.setattr(
        "graderbot.embedding.vectorize_samples",
        lambda *args, **kwargs: vectorize_calls.append(kwargs) or 1,
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if "Scanned PDF" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    button = next(b for b in at.button if b.label == "Ingest")
    button.click().run()

    assert not at.exception
    assert len(vectorize_calls) == 1


def test_app_errors_when_bucket_not_configured(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)
    monkeypatch.delenv("S3_BUCKET", raising=False)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("S3_BUCKET" in err.value for err in at.error)
