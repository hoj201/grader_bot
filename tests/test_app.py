import logging
from pathlib import Path

import boto3
import numpy as np
from moto import mock_aws
from streamlit.testing.v1 import AppTest

from graderbot import app, name_classifier, scan_grader, storage
from graderbot.answer_reader import EasyOcrAnswerReader
from graderbot.worksheetbot import Question, WorksheetDocument

APP_PATH = str(Path(__file__).resolve().parent.parent / "graderbot" / "app.py")


def _seed_worksheet(db_path, title=None, questions_json="[]", public_id=None):
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
            public_id=public_id,
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

    def fake_mark_scan(hws, roster, db_path, out_path, on_step=None, name_reader=None, answer_reader=None):
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


def test_gallery_tab_shows_permanent_download_link(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_worksheet(db_path, public_id="ws_a1b2c3d4")
    _set_env(monkeypatch, db_path)
    monkeypatch.setenv("BASE_URL", "https://grader-bot.fly.dev")

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    code_values = " ".join(c.value for c in at.code)
    assert "https://grader-bot.fly.dev/?dl=ws_a1b2c3d4" in code_values


def test_gallery_tab_omits_permanent_link_without_public_id(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_worksheet(db_path)
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    caption_texts = " ".join(c.value for c in at.caption)
    assert "Permanent link" not in caption_texts


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


def _fake_document():
    return WorksheetDocument(
        title="Auto Title",
        header="Auto Header",
        questions=[Question(id="1", text="$2+2=$", answer="4")],
    )


def test_create_ai_shows_preview_before_compiling(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    monkeypatch.setattr(
        "graderbot.worksheetbot.generate_worksheet_document",
        lambda *args, **kwargs: _fake_document(),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_worksheet should not run before Accept")

    monkeypatch.setattr("graderbot.worksheetbot.build_worksheet", fail_if_called)

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(ta for ta in at.text_area if ta.label == "Worksheet prompt").set_value(
        "arithmetic worksheet"
    )
    at.run()
    button = next(b for b in at.button if b.label == "Generate questions")
    button.click().run()

    assert not at.exception
    button_labels = [b.label for b in at.button]
    assert "Accept and compile" in button_labels
    assert "Reject" in button_labels
    code_values = [c.value for c in at.code]
    assert any('"id": "1"' in v and '"text": "$2+2=$"' in v for v in code_values)


def test_create_ai_accept_compiles_and_stores(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    monkeypatch.setattr(
        "graderbot.worksheetbot.generate_worksheet_document",
        lambda *args, **kwargs: _fake_document(),
    )

    def fake_build(document, template_path, out, max_repairs, client=None, bucket=None,
                   db_path=None, prompt="", model=None, on_step=None):
        record = storage.WorksheetRecord(
            id=1,
            prompt=prompt,
            tex_source=r"\documentclass{article}",
            questions_json="[]",
            model=model,
            num_questions=len(document.questions),
            title=document.title,
            created_at="2026-07-18T00:00:00+00:00",
        )
        return Path("unused.pdf"), document.questions, record

    monkeypatch.setattr("graderbot.worksheetbot.build_worksheet", fake_build)

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(ta for ta in at.text_area if ta.label == "Worksheet prompt").set_value(
        "arithmetic worksheet"
    )
    at.run()
    next(b for b in at.button if b.label == "Generate questions").click().run()

    with caplog.at_level(logging.INFO, logger="graderbot.app"):
        next(b for b in at.button if b.label == "Accept and compile").click().run()

    assert not at.exception
    assert any("created worksheet id=1" in r.message for r in caplog.records)
    # issue #64: st.success() is followed by st.rerun(), which used to
    # discard the message before it ever reached the client.
    assert "Created worksheet id=1" in [s.value for s in at.success]
    # The preview is cleared once accepted.
    assert "Review generated questions" not in [s.value for s in at.subheader]


def test_create_ai_reject_prefills_manual_json_form(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    monkeypatch.setattr(
        "graderbot.worksheetbot.generate_worksheet_document",
        lambda *args, **kwargs: _fake_document(),
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(ta for ta in at.text_area if ta.label == "Worksheet prompt").set_value(
        "arithmetic worksheet"
    )
    at.run()
    next(b for b in at.button if b.label == "Generate questions").click().run()
    next(b for b in at.button if b.label == "Reject").click().run()

    assert not at.exception
    assert '"id": "1"' in at.text_area(key="manual_questions_json").value
    assert at.text_input(key="manual_title").value == "Auto Title"
    assert at.text_area(key="manual_header").value == "Auto Header"


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
    # issue #64: st.success() is followed by st.rerun(), which used to
    # discard the message before it ever reached the client.
    assert "Created worksheet id=1" in [s.value for s in at.success]


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


def test_roster_tab_shows_error_when_vectorization_fails(tmp_path, monkeypatch):
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

    def raise_vectorize_error(*args, **kwargs):
        raise EnvironmentError("VOYAGE_API_KEY is not set")

    monkeypatch.setattr("graderbot.embedding.vectorize_samples", raise_vectorize_error)

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if "Scanned PDF" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    button = next(b for b in at.button if b.label == "Ingest")
    button.click().run()

    # The failure must be surfaced legibly, not as an unhandled traceback,
    # and the earlier (already-persisted) ingest success should still show.
    assert not at.exception
    successes = " ".join(s.value for s in at.success)
    assert "Ingested 1 handwriting sample" in successes
    errors = " ".join(e.value for e in at.error)
    assert "embedding failed" in errors
    assert "VOYAGE_API_KEY is not set" in errors


def test_roster_tab_manual_add_student_creates_student(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.get_or_create_classroom(conn, "Room 101")
    conn.close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input(key="manual_student_first_name").set_value("Anna")
    at.text_input(key="manual_student_last_name").set_value("Smith")
    at.run()
    next(b for b in at.button if b.label == "Add student").click().run()

    assert not at.exception
    conn = storage.init_db(db_path)
    classroom = storage.list_classrooms(conn)[0]
    students = storage.list_students(conn, classroom.id)
    conn.close()
    assert [(s.first_name, s.last_name) for s in students] == [("Anna", "Smith")]
    successes = " ".join(s.value for s in at.success)
    assert "Added Anna Smith" in successes


def test_roster_tab_manual_add_student_requires_first_and_last_name(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.get_or_create_classroom(conn, "Room 101")
    conn.close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input(key="manual_student_first_name").set_value("Anna")
    at.run()
    next(b for b in at.button if b.label == "Add student").click().run()

    assert not at.exception
    assert any("First and last name are required" in e.value for e in at.error)
    conn = storage.init_db(db_path)
    classroom = storage.list_classrooms(conn)[0]
    assert storage.list_students(conn, classroom.id) == []
    conn.close()


def test_roster_tab_csv_import_adds_students_and_reports_skips(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.get_or_create_classroom(conn, "Room 101")
    conn.close()
    _set_env(monkeypatch, db_path)

    csv_bytes = b"first_name,last_name\nAnna,Smith\n,Jones\n"

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if fu.label == "Roster CSV")
    uploader.set_value(("roster.csv", csv_bytes, "text/csv"))
    at.run()
    next(b for b in at.button if b.label == "Import CSV").click().run()

    assert not at.exception
    conn = storage.init_db(db_path)
    classroom = storage.list_classrooms(conn)[0]
    students = storage.list_students(conn, classroom.id)
    conn.close()
    assert [(s.first_name, s.last_name) for s in students] == [("Anna", "Smith")]
    successes = " ".join(s.value for s in at.success)
    assert "Added 1 student(s) from CSV" in successes
    warnings = " ".join(w.value for w in at.warning)
    assert "row 3: missing first or last name" in warnings


def test_roster_tab_csv_import_shows_error_for_bad_header(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.get_or_create_classroom(conn, "Room 101")
    conn.close()
    _set_env(monkeypatch, db_path)

    csv_bytes = b"name\nAnna Smith\n"

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if fu.label == "Roster CSV")
    uploader.set_value(("roster.csv", csv_bytes, "text/csv"))
    at.run()
    next(b for b in at.button if b.label == "Import CSV").click().run()

    assert not at.exception
    assert any("missing required column" in e.value for e in at.error)


def test_roster_tab_transfer_student_moves_classroom(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    room_a = storage.get_or_create_classroom(conn, "Room A")
    room_b = storage.get_or_create_classroom(conn, "Room B")
    storage.get_or_create_student(conn, room_a.id, "Anna", "Smith")
    conn.close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(b for b in at.button if b.label == "Transfer").click().run()
    next(b for b in at.button if b.label == "Confirm").click().run()

    assert not at.exception
    conn = storage.init_db(db_path)
    assert storage.list_students(conn, room_a.id) == []
    students_b = storage.list_students(conn, room_b.id)
    conn.close()
    assert [(s.first_name, s.last_name) for s in students_b] == [("Anna", "Smith")]


def test_roster_tab_transfer_student_shows_error_on_name_collision(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    room_a = storage.get_or_create_classroom(conn, "Room A")
    room_b = storage.get_or_create_classroom(conn, "Room B")
    storage.get_or_create_student(conn, room_a.id, "Anna", "Smith")
    storage.get_or_create_student(conn, room_b.id, "Anna", "Smith")
    conn.close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(b for b in at.button if b.label == "Transfer").click().run()
    next(b for b in at.button if b.label == "Confirm").click().run()

    assert not at.exception
    assert any("already exists in the target classroom" in e.value for e in at.error)
    conn = storage.init_db(db_path)
    students_a = storage.list_students(conn, room_a.id)
    conn.close()
    assert [(s.first_name, s.last_name) for s in students_a] == [("Anna", "Smith")]


def test_visualize_tab_evaluate_classifier_shows_accuracy_and_confusion(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    classroom = storage.get_or_create_classroom(conn, "Room 101")
    anna = storage.get_or_create_student(conn, classroom.id, "Anna", "Smith")
    zeke = storage.get_or_create_student(conn, classroom.id, "Zeke", "Jones")
    lonely = storage.get_or_create_student(conn, classroom.id, "Lonely", "Student")
    conn.close()
    _set_env(monkeypatch, db_path)

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3, 4)).astype(np.float32)
    student_ids = np.array([anna.id, zeke.id, lonely.id])
    name_image_ids = np.array([1, 2, 3])
    monkeypatch.setattr(
        "graderbot.embedding.load_training_vectors",
        lambda *args, **kwargs: (vectors, student_ids, name_image_ids, 0),
    )
    monkeypatch.setattr(
        "graderbot.name_classifier.loo_cross_validate",
        lambda *args, **kwargs: (
            {anna.id: 1.0, zeke.id: 0.5},
            [lonely.id],
            {anna.id: {anna.id: 1}, zeke.id: {zeke.id: 1, anna.id: 1}},
        ),
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    button = next(b for b in at.button if b.label == "Evaluate classifier")
    button.click().run()

    assert not at.exception
    table_rows = at.table[0].value.to_dict("records")
    assert {"Student": "Anna Smith", "LOO accuracy": "100%"} in table_rows
    assert {"Student": "Zeke Jones", "LOO accuracy": "50%"} in table_rows

    matrix_rows = at.dataframe[0].value.to_dict("records")
    assert {"Actual": "Anna Smith", "Anna Smith": 1, "Zeke Jones": 0} in matrix_rows
    assert {"Actual": "Zeke Jones", "Anna Smith": 1, "Zeke Jones": 1} in matrix_rows

    captions = " ".join(c.value for c in at.caption)
    assert "Lonely Student" in captions


def _seed_classroom(db_path):
    conn = storage.init_db(db_path)
    classroom = storage.get_or_create_classroom(conn, "Room 101")
    anna = storage.get_or_create_student(conn, classroom.id, "Anna", "Smith")
    conn.close()
    return classroom, anna


def _patch_saved_classifier(monkeypatch, exists: bool):
    """Control what `app._classifier_exists` sees. It can't be patched directly:
    AppTest re-executes app.py as a fresh module, so only the dependencies it
    imports (here, storage's S3 client) can be stubbed."""

    class _FakeS3:
        def head_object(self, Bucket, Key):
            if not exists:
                raise RuntimeError("404 Not Found")
            return {}

    monkeypatch.setattr("graderbot.storage._default_s3_client", lambda: _FakeS3())


def _patch_embeddings(monkeypatch, student_id):
    """Three samples: `embedding_viz.project_3d` needs at least that many to
    project, and the Train button renders below the projection."""
    monkeypatch.setattr(
        "graderbot.embedding.load_training_vectors",
        lambda *args, **kwargs: (
            np.eye(3, 4, dtype=np.float32),
            np.array([student_id] * 3),
            np.array([1, 2, 3]),
            0,
        ),
    )


def test_visualize_tab_train_classifier_reports_the_fit(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    classroom, anna = _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_embeddings(monkeypatch, anna.id)

    train_calls = []
    monkeypatch.setattr(
        "graderbot.name_classifier.train_classroom_classifier",
        lambda *args, **kwargs: train_calls.append((args, kwargs))
        or name_classifier.TrainingReport(
            n_samples=9,
            n_students=2,
            s3_url="https://bucket.s3.amazonaws.com/name_classifier/1.joblib",
            embedding_dim=1024,
            discarded_wrong_dim=3,
            students_with_no_samples=["Nora None"],
            students_with_one_sample=["Solo One"],
        ),
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(b for b in at.button if b.label == "Train classifier").click().run()

    assert not at.exception
    assert len(train_calls) == 1
    # Trained for the selected classroom, not the whole database.
    assert train_calls[0][0][2] == classroom.id

    successes = " ".join(s.value for s in at.success)
    assert "9 sample(s)" in successes and "name_classifier/1.joblib" in successes
    warnings = " ".join(w.value for w in at.warning)
    assert "Nora None" in warnings
    assert "Solo One" in warnings
    assert "Skipped 3 embedding(s)" in warnings


def test_visualize_tab_train_classifier_surfaces_missing_data(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _, anna = _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_embeddings(monkeypatch, anna.id)

    def raise_no_data(*args, **kwargs):
        raise ValueError("no 1024-dimensional handwriting embeddings for classroom 1")

    monkeypatch.setattr(
        "graderbot.name_classifier.train_classroom_classifier", raise_no_data
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    next(b for b in at.button if b.label == "Train classifier").click().run()

    assert not at.exception
    assert any("no 1024-dimensional handwriting embeddings" in e.value for e in at.error)


def test_grade_tab_defaults_to_ocr_when_no_classifier_is_trained(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_saved_classifier(monkeypatch, exists=False)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    source = next(s for s in at.selectbox if "Read student names with" in s.label)
    assert source.value == "OCR (Tesseract)"
    assert any("No handwriting classifier has been trained" in c.value for c in at.caption)


def test_grade_tab_defaults_to_the_classifier_when_one_exists(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_saved_classifier(monkeypatch, exists=True)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    source = next(s for s in at.selectbox if "Read student names with" in s.label)
    assert source.value == "Handwriting classifier"


def test_grade_tab_passes_a_classifier_reader_when_selected(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_saved_classifier(monkeypatch, exists=True)

    sentinel = object()
    monkeypatch.setattr(
        "graderbot.name_reader.ClassifierNameReader.from_classroom",
        classmethod(lambda cls, *args, **kwargs: sentinel),
    )
    seen = {}

    def fake_mark_scan(hws, roster, db_path, out_path, on_step=None, name_reader=None, answer_reader=None):
        seen["name_reader"] = name_reader
        return scan_grader.ScanBatchResult()

    monkeypatch.setattr("graderbot.scan_grader.mark_scan", fake_mark_scan)

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    next(b for b in at.button if b.label == "Grade").click().run()

    assert not at.exception
    assert seen["name_reader"] is sentinel


def test_grade_tab_errors_rather_than_silently_using_ocr(tmp_path, monkeypatch):
    """Picking the classifier when none is saved must not quietly fall back --
    the whole point of the dropdown is knowing which one graded the pile."""
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_saved_classifier(monkeypatch, exists=True)
    monkeypatch.setattr(
        "graderbot.name_reader.ClassifierNameReader.from_classroom",
        classmethod(lambda cls, *args, **kwargs: None),
    )
    calls = []
    monkeypatch.setattr(
        "graderbot.scan_grader.mark_scan",
        lambda *args, **kwargs: calls.append(1) or scan_grader.ScanBatchResult(),
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    next(b for b in at.button if b.label == "Grade").click().run()

    assert not at.exception
    assert not calls
    assert any("No handwriting classifier is saved" in e.value for e in at.error)


def test_grade_tab_defaults_to_mathpix_for_answers(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    source = next(s for s in at.selectbox if "Read answers with" in s.label)
    assert source.value == "Mathpix"


def test_grade_tab_passes_an_easyocr_reader_when_selected(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)
    monkeypatch.setenv("EASYOCR_SERVICE_URL", "http://localhost:8080")

    seen = {}

    def fake_mark_scan(hws, roster, db_path, out_path, on_step=None, name_reader=None, answer_reader=None):
        seen["answer_reader"] = answer_reader
        return scan_grader.ScanBatchResult()

    monkeypatch.setattr("graderbot.scan_grader.mark_scan", fake_mark_scan)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.selectbox(key="grade_answer_source").set_value("EasyOCR").run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    next(b for b in at.button if b.label == "Grade").click().run()

    assert not at.exception
    assert isinstance(seen["answer_reader"], EasyOcrAnswerReader)
    assert seen["answer_reader"].allowlist == "0123456789."


def test_grade_tab_easyocr_extra_chars_widen_the_allowlist(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)
    monkeypatch.setenv("EASYOCR_SERVICE_URL", "http://localhost:8080")

    seen = {}

    def fake_mark_scan(hws, roster, db_path, out_path, on_step=None, name_reader=None, answer_reader=None):
        seen["answer_reader"] = answer_reader
        return scan_grader.ScanBatchResult()

    monkeypatch.setattr("graderbot.scan_grader.mark_scan", fake_mark_scan)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.selectbox(key="grade_answer_source").set_value("EasyOCR").run()
    at.text_input(key="grade_easyocr_extra_chars").set_value("xy").run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    next(b for b in at.button if b.label == "Grade").click().run()

    assert not at.exception
    assert seen["answer_reader"].allowlist == "0123456789.xy"


def test_grade_tab_errors_when_easyocr_service_url_is_not_set(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)
    monkeypatch.delenv("EASYOCR_SERVICE_URL", raising=False)
    calls = []
    monkeypatch.setattr(
        "graderbot.scan_grader.mark_scan",
        lambda *args, **kwargs: calls.append(1) or scan_grader.ScanBatchResult(),
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.selectbox(key="grade_answer_source").set_value("EasyOCR").run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    next(b for b in at.button if b.label == "Grade").click().run()

    assert not at.exception
    assert not calls
    assert any("EASYOCR_SERVICE_URL" in e.value for e in at.error)


def test_grade_tab_shows_per_page_names_and_flags_low_confidence(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_classroom(db_path)
    _set_env(monkeypatch, db_path)
    _patch_saved_classifier(monkeypatch, exists=False)

    result = scan_grader.ScanBatchResult(
        name_predictions=[
            scan_grader.NamePrediction("scan.pdf (page 1)", "ws_1", "Anna Smith", 1.0, "classifier"),
            scan_grader.NamePrediction("scan.pdf (page 2)", "ws_1", "Zeke Jones", 0.33, "classifier"),
        ]
    )
    monkeypatch.setattr(
        "graderbot.scan_grader.mark_scan", lambda *args, **kwargs: result
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    uploader = next(fu for fu in at.get("file_uploader") if "Student work" in fu.label)
    uploader.set_value(("scan.pdf", b"not-a-real-pdf", "application/pdf"))
    at.run()
    next(b for b in at.button if b.label == "Grade").click().run()

    assert not at.exception
    rows = at.dataframe[0].value.to_dict("records")
    assert {
        "Page": "scan.pdf (page 1)",
        "Worksheet": "ws_1",
        "Student": "Anna Smith",
        "Confidence": "100%",
        "Read by": "classifier",
    } in rows
    # Only the doubtful page is called out for a hand check.
    warnings = " ".join(w.value for w in at.warning)
    assert "Zeke Jones" in warnings
    assert "Anna Smith" not in warnings


def test_app_errors_when_bucket_not_configured(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _set_env(monkeypatch, db_path)
    monkeypatch.delenv("S3_BUCKET", raising=False)

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("S3_BUCKET" in err.value for err in at.error)


def test_dl_query_param_redirects_to_presigned_student_pdf(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    _seed_worksheet(db_path, public_id="ws_a1b2c3d4")
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.query_params["dl"] = "ws_a1b2c3d4"
    at.run()

    assert not at.exception
    markdown_texts = " ".join(md.value for md in at.markdown)
    assert 'http-equiv="refresh"' in markdown_texts
    assert "worksheet/student.pdf" in markdown_texts
    # The redirect short-circuits the rest of the app (st.stop()) -- the
    # gallery/tabs below it never render.
    assert not at.tabs


def test_dl_query_param_shows_error_for_unknown_public_id(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    storage.init_db(db_path).close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.query_params["dl"] = "ws_doesnotexist"
    at.run()

    assert not at.exception
    assert any("No worksheet found" in e.value for e in at.error)


def test_build_permanent_download_url_joins_base_and_public_id():
    assert (
        app.build_permanent_download_url("https://grader-bot.fly.dev", "ws_a1b2c3d4")
        == "https://grader-bot.fly.dev/?dl=ws_a1b2c3d4"
    )


def test_dl_query_param_shows_error_when_no_student_pdf(tmp_path, monkeypatch):
    db_path = tmp_path / "worksheets.sqlite3"
    conn = storage.init_db(db_path)
    storage.insert_worksheet(
        conn,
        storage.WorksheetRecord(
            prompt="p",
            tex_source="t",
            questions_json="[]",
            model="m",
            num_questions=1,
            public_id="ws_nopdf001",
            created_at="2026-07-18T00:00:00+00:00",
        ),
    )
    conn.close()
    _set_env(monkeypatch, db_path)

    at = AppTest.from_file(APP_PATH)
    at.query_params["dl"] = "ws_nopdf001"
    at.run()

    assert not at.exception
    assert any("no student PDF" in e.value for e in at.error)
