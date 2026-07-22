import json

import numpy as np
import pymupdf
import pytest

from graderbot import scan_grader
from graderbot.core import Box, QuestionResult
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
    monkeypatch.setattr(
        scan_grader, "extract_name", lambda image, box, roster: name_by_path[image]
    )

    def fake_grade_hw(answer_key, boxes, image):
        grade_hw_calls.append({"answer_key": answer_key, "boxes": boxes, "image": image})
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


def test_mark_scan_writes_no_pdf_when_nothing_grades(
    db_with_two_worksheets, patched_cv, tmp_path
):
    out_path = tmp_path / "marked.pdf"

    result = mark_scan(
        ["blank.png"], roster=[], db_path=db_with_two_worksheets, out_path=out_path
    )

    assert result.unreadable == ["blank.png"]
    assert not out_path.exists()
