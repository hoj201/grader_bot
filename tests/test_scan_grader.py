import json

import numpy as np
import pymupdf
import pytest

from graderbot import scan_grader
from graderbot.models import Box, QuestionResult
from graderbot.name_reader import CLASSIFIER_SOURCE, OCR_SOURCE, NameGuess
from graderbot.scan_grader import grade_scans, mark_scan, results_by_student
from graderbot.storage import init_db, insert_worksheet, serialize_boxes
from tests.test_storage import _sample_record


@pytest.fixture
def db_with_two_worksheets(tmp_path):
    """A DB holding two worksheets (ws_1, ws_2), each with a name box, two
    question boxes, and a stored answer key."""
    db_path = tmp_path / "worksheets.sqlite3"
    conn = init_db(db_path)

    boxes = serialize_boxes(
        {
            "name": Box(0.1, 0.9, 0.4, 0.05),
            "q1": Box(0.1, 0.5, 0.3, 0.05),
            "q2": Box(0.1, 0.4, 0.3, 0.05),
        }
    )
    for public_id in ("ws_1", "ws_2"):
        insert_worksheet(
            conn,
            _sample_record(
                public_id=public_id,
                boxes_json=boxes,
                questions_json=json.dumps(
                    [
                        {"id": "q1", "text": "1+1", "answer": "2"},
                        {"id": "q2", "text": "2+2", "answer": "4"},
                    ]
                ),
            ),
        )
    conn.close()
    return db_path


@pytest.fixture
def patched_cv(monkeypatch):
    """Replaces the CV/OCR functions so grade_scans runs offline. Each scan's
    'image' is just its path string; the maps below drive decode/name/scoring."""
    id_by_path = {
        "alice.png": "ws_1",
        "bob.png": "ws_1",
        "carol.png": "ws_2",
        "blank.png": None,       # QR undecodable
        "ghost.png": "ws_gone",  # decodes but not in DB
    }
    name_by_path = {
        "alice.png": "Alice Smith",
        "bob.png": "Bob Jones",
        "carol.png": "Carol White",
    }
    grade_hw_calls = []

    # Each scan's 'image' is just its path string; load_scan_pages yields one
    # page per scan and rectify_to_canonical is an identity, so the path string
    # flows through unchanged into the decode/name/grade maps above.
    monkeypatch.setattr(scan_grader, "load_scan_pages", lambda path: [path])
    monkeypatch.setattr(scan_grader, "rectify_to_canonical", lambda image: image)
    monkeypatch.setattr(scan_grader, "read_worksheet_id", lambda image: id_by_path[image])

    class _StubOcrNameReader:
        """Stands in for the Tesseract reader `_grade_batch` builds by default,
        looking the name up by path instead of running OCR."""

        def __init__(self, roster):
            self.roster = roster

        def read_many(self, images, box):
            return [NameGuess(name_by_path[image], 1.0, OCR_SOURCE) for image in images]

    monkeypatch.setattr(scan_grader, "OcrNameReader", _StubOcrNameReader)

    def fake_grade_hw(answer_key, boxes, image, open_ended=None, answer_reader=None):
        grade_hw_calls.append(
            {
                "answer_key": answer_key,
                "boxes": boxes,
                "image": image,
                "open_ended": open_ended,
                "answer_reader": answer_reader,
            }
        )
        # A perfect paper: response == answer for every question box.
        return {
            qid: QuestionResult(answer=answer_key[qid], response=answer_key[qid], correct=True)
            for qid in boxes
        }

    monkeypatch.setattr(scan_grader, "grade_hw", fake_grade_hw)
    return grade_hw_calls


def test_grade_scans_groups_by_worksheet_id(db_with_two_worksheets, patched_cv):
    result = grade_scans(
        ["alice.png", "bob.png", "carol.png"],
        roster=["Alice Smith", "Bob Jones", "Carol White"],
        db_path=db_with_two_worksheets,
    )

    assert set(result.results_by_worksheet) == {"ws_1", "ws_2"}
    assert set(result.results_by_worksheet["ws_1"]) == {"Alice Smith", "Bob Jones"}
    assert set(result.results_by_worksheet["ws_2"]) == {"Carol White"}


def test_grade_scans_uses_stored_answer_key_and_excludes_name_box(
    db_with_two_worksheets, patched_cv
):
    grade_scans(["alice.png"], roster=["Alice Smith"], db_path=db_with_two_worksheets)

    call = patched_cv[0]
    assert call["answer_key"] == {"q1": "2", "q2": "4"}
    # The name box must not be graded as a question.
    assert set(call["boxes"]) == {"q1", "q2"}


def test_grade_scans_passes_open_ended_flags_to_grade_hw(tmp_path, patched_cv):
    # issue #65: a stored open-ended question must be flagged for grade_hw
    # so it never gets compared against its (empty) answer.
    db_path = tmp_path / "worksheets.sqlite3"
    conn = init_db(db_path)
    boxes = serialize_boxes(
        {
            "name": Box(0.1, 0.9, 0.4, 0.05),
            "q1": Box(0.1, 0.5, 0.3, 0.05),
            "q2": Box(0.1, 0.4, 0.3, 0.05),
        }
    )
    insert_worksheet(
        conn,
        _sample_record(
            public_id="ws_1",
            boxes_json=boxes,
            questions_json=json.dumps(
                [
                    {"id": "q1", "text": "1+1", "answer": "2"},
                    {"id": "q2", "text": "How do you feel about math?", "answer": "", "open_ended": True},
                ]
            ),
        ),
    )
    conn.close()

    grade_scans(["alice.png"], roster=["Alice Smith"], db_path=db_path)

    call = patched_cv[0]
    assert call["open_ended"] == {"q1": False, "q2": True}


def test_grade_scans_results_are_per_question(db_with_two_worksheets, patched_cv):
    result = grade_scans(["alice.png"], roster=["Alice Smith"], db_path=db_with_two_worksheets)

    student = result.results_by_worksheet["ws_1"]["Alice Smith"]
    assert student == {
        "q1": QuestionResult(answer="2", response="2", correct=True),
        "q2": QuestionResult(answer="4", response="4", correct=True),
    }


def test_grade_scans_collects_unreadable_scans(db_with_two_worksheets, patched_cv):
    result = grade_scans(["alice.png", "blank.png"], roster=["Alice Smith"], db_path=db_with_two_worksheets)

    assert result.unreadable == ["blank.png"]
    assert "Alice Smith" in result.results_by_worksheet["ws_1"]


def test_grade_scans_collects_unknown_worksheet_ids(db_with_two_worksheets, patched_cv):
    result = grade_scans(["ghost.png"], roster=[], db_path=db_with_two_worksheets)

    assert result.unknown_worksheets == {"ws_gone": ["ghost.png"]}
    assert result.results_by_worksheet == {}


def test_mark_scan_writes_one_page_per_graded_scan(
    db_with_two_worksheets, patched_cv, monkeypatch, tmp_path
):
    # render_marked_page runs on the identity-"rectified" path string, so stub it
    # out to a real page image; images_to_pdf then writes a genuine PDF.
    monkeypatch.setattr(
        scan_grader,
        "render_marked_page",
        lambda image, results, boxes: np.full((20, 20, 3), 255, dtype=np.uint8),
    )
    out_path = tmp_path / "marked.pdf"

    result = mark_scan(
        ["alice.png", "bob.png", "carol.png"],
        roster=["Alice Smith", "Bob Jones", "Carol White"],
        db_path=db_with_two_worksheets,
        out_path=out_path,
    )

    # Same grading summary as grade_scans, plus one PDF page per graded scan.
    assert set(result.results_by_worksheet["ws_1"]) == {"Alice Smith", "Bob Jones"}
    assert out_path.exists()
    assert pymupdf.open(out_path).page_count == 3


def test_results_by_student_transposes_to_display_shape(db_with_two_worksheets, patched_cv):
    result = grade_scans(
        ["alice.png", "carol.png"],
        roster=["Alice Smith", "Carol White"],
        db_path=db_with_two_worksheets,
    )

    by_student = results_by_student(result)

    assert set(by_student) == {"Alice Smith", "Carol White"}
    assert by_student["Alice Smith"]["ws_1"] == {
        "q1": QuestionResult(answer="2", response="2", correct=True),
        "q2": QuestionResult(answer="4", response="4", correct=True),
    }
    assert set(by_student["Carol White"]) == {"ws_2"}


def test_grade_scans_reports_progress_via_on_step(db_with_two_worksheets, patched_cv):
    messages = []
    grade_scans(
        ["alice.png"],
        roster=["Alice Smith"],
        db_path=db_with_two_worksheets,
        on_step=lambda msg, detail=None: messages.append(msg),
    )

    joined = "\n".join(messages)
    assert "decoded worksheet id=ws_1" in joined
    assert "graded Alice Smith [ocr 100%] -- 2/2 correct" in joined


def test_on_step_distinguishes_rectify_from_qr_failure(
    db_with_two_worksheets, patched_cv, monkeypatch
):
    # "skew.png" fails rectification; "blank.png" rectifies but its QR won't decode.
    real_rectify = scan_grader.rectify_to_canonical

    def rectify(image):
        if image == "skew.png":
            raise ValueError("no markers")
        return real_rectify(image)

    monkeypatch.setattr(scan_grader, "rectify_to_canonical", rectify)

    messages = []
    result = grade_scans(
        ["skew.png", "blank.png"],
        roster=[],
        db_path=db_with_two_worksheets,
        on_step=lambda msg, detail=None: messages.append(msg),
    )

    assert result.unreadable == ["skew.png", "blank.png"]
    joined = "\n".join(messages)
    assert "skew.png: could not rectify" in joined
    assert "blank.png: rectified OK, but the worksheet QR code could not be decoded" in joined


def test_mark_scan_reports_pdf_render_via_on_step(
    db_with_two_worksheets, patched_cv, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        scan_grader,
        "render_marked_page",
        lambda image, results, boxes: np.full((20, 20, 3), 255, dtype=np.uint8),
    )
    messages = []
    mark_scan(
        ["alice.png"],
        roster=["Alice Smith"],
        db_path=db_with_two_worksheets,
        out_path=tmp_path / "marked.pdf",
        on_step=lambda msg, detail=None: messages.append(msg),
    )

    assert any("Rendering marked-up PDF" in m for m in messages)
    assert any("Wrote marked-up PDF" in m for m in messages)


def test_mark_scan_writes_no_pdf_when_nothing_grades(
    db_with_two_worksheets, patched_cv, tmp_path
):
    out_path = tmp_path / "marked.pdf"

    result = mark_scan(
        ["blank.png"], roster=[], db_path=db_with_two_worksheets, out_path=out_path
    )

    assert result.unreadable == ["blank.png"]
    assert not out_path.exists()


# --------------------------------------------------------------------------
# Name reading (issue #58)


def test_grade_scans_records_a_name_prediction_per_page(
    db_with_two_worksheets, patched_cv
):
    result = grade_scans(
        ["alice.png", "bob.png", "carol.png"],
        roster=["Alice Smith", "Bob Jones", "Carol White"],
        db_path=db_with_two_worksheets,
    )

    by_page = {p.page: p for p in result.name_predictions}
    assert set(by_page) == {"alice.png", "bob.png", "carol.png"}
    assert by_page["alice.png"].name == "Alice Smith"
    assert by_page["alice.png"].worksheet_id == "ws_1"
    assert by_page["carol.png"].worksheet_id == "ws_2"
    assert all(p.source == OCR_SOURCE for p in result.name_predictions)


def test_name_predictions_keep_pages_the_results_dict_collapses(
    db_with_two_worksheets, monkeypatch, patched_cv
):
    """Two pages read as the same student overwrite each other in
    results_by_worksheet; the per-page predictions still show both, which is
    how such a misread gets noticed."""

    class _AlwaysAlice:
        def __init__(self, roster):
            pass

        def read_many(self, images, box):
            return [NameGuess("Alice Smith", 0.4, OCR_SOURCE) for _ in images]

    monkeypatch.setattr(scan_grader, "OcrNameReader", _AlwaysAlice)

    result = grade_scans(
        ["alice.png", "bob.png"], roster=["Alice Smith"], db_path=db_with_two_worksheets
    )

    assert set(result.results_by_worksheet["ws_1"]) == {"Alice Smith"}
    assert [p.page for p in result.name_predictions] == ["alice.png", "bob.png"]


def test_grade_scans_uses_an_injected_name_reader(db_with_two_worksheets, patched_cv):
    """Passing a reader bypasses OCR entirely -- this is how the Grade tab
    switches to the handwriting classifier."""
    seen = []

    class _ClassifierStub:
        def read_many(self, images, box):
            seen.append(list(images))
            return [NameGuess("Zoe Zhang", 0.67, CLASSIFIER_SOURCE, student_id=9) for _ in images]

    result = grade_scans(
        ["alice.png", "bob.png"],
        roster=["Alice Smith", "Bob Jones"],
        db_path=db_with_two_worksheets,
        name_reader=_ClassifierStub(),
    )

    assert set(result.results_by_worksheet["ws_1"]) == {"Zoe Zhang"}
    assert all(p.source == CLASSIFIER_SOURCE for p in result.name_predictions)
    assert result.name_predictions[0].confidence == pytest.approx(0.67)
    # Both pages of the worksheet group were read in a single batched call, so a
    # remote embedder is charged once rather than per page.
    assert seen == [["alice.png", "bob.png"]]


def test_mark_scan_forwards_the_name_reader(
    db_with_two_worksheets, patched_cv, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        scan_grader,
        "render_marked_page",
        lambda image, results, boxes: np.full((20, 20, 3), 255, dtype=np.uint8),
    )

    class _ClassifierStub:
        def read_many(self, images, box):
            return [NameGuess("Zoe Zhang", 1.0, CLASSIFIER_SOURCE) for _ in images]

    result = mark_scan(
        ["alice.png"],
        roster=["Alice Smith"],
        db_path=db_with_two_worksheets,
        out_path=tmp_path / "marked.pdf",
        name_reader=_ClassifierStub(),
    )

    assert set(result.results_by_worksheet["ws_1"]) == {"Zoe Zhang"}


def test_grade_scans_forwards_the_answer_reader(db_with_two_worksheets, patched_cv):
    """Passing a reader switches the answer-box OCR backend -- this is how the
    Grade tab switches to EasyOCR (issue #70)."""
    sentinel = object()

    grade_scans(
        ["alice.png"],
        roster=["Alice Smith"],
        db_path=db_with_two_worksheets,
        answer_reader=sentinel,
    )

    assert all(call["answer_reader"] is sentinel for call in patched_cv)
