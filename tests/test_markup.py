import numpy as np
import pymupdf
import pytest

from graderbot import markup
from graderbot.models import Box, QuestionResult
from graderbot.markup import _CORRECT_COLOR, _WRONG_COLOR, render_marked_page, save_marked_pdf


@pytest.fixture
def page():
    # White canonical page, answer boxes on the left so marks/answers have room.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {
        "q1": Box(0.1, 0.7, 0.2, 0.1),
        "q2": Box(0.1, 0.4, 0.2, 0.1),
    }
    results = {
        "q1": QuestionResult(answer="3", response="3", correct=True),
        "q2": QuestionResult(answer="5", response="4", correct=False),
    }
    return image, results, boxes


def _has_pixel(image, color):
    return bool(np.any(np.all(image == np.array(color, dtype=np.uint8), axis=-1)))


def test_render_marked_page_draws_correct_and_wrong_marks(page):
    image, results, boxes = page

    marked = render_marked_page(image, results, boxes)

    assert _has_pixel(marked, _CORRECT_COLOR)  # a green check for q1
    assert _has_pixel(marked, _WRONG_COLOR)    # a red cross for q2


def test_render_marked_page_leaves_input_unmodified(page):
    image, results, boxes = page

    render_marked_page(image, results, boxes)

    assert np.all(image == 255)


def test_render_marked_page_writes_correct_answer_only_for_wrong(page, monkeypatch):
    image, results, boxes = page
    calls = []

    def fake_draw_answer(img, box_px, answer, font, text_size):
        calls.append({"box_px": box_px, "answer": answer})
        return img

    monkeypatch.setattr(markup, "_draw_answer", fake_draw_answer)

    render_marked_page(image, results, boxes)

    # Only the wrong question (q2, answer "5") should get its answer drawn.
    assert [c["answer"] for c in calls] == ["5"]


def test_render_marked_page_draws_note_for_wrong_answer(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.4, 0.2, 0.1)}
    results = {"q1": QuestionResult(answer=r"\frac{3}{8}", response=r"\frac{45}{120}", correct=False, note="simplify")}
    calls = []

    def fake_draw_note(img, note, origin, box_height, color):
        calls.append(note)

    monkeypatch.setattr(markup, "_draw_note", fake_draw_note)

    render_marked_page(image, results, boxes)

    assert calls == ["simplify"]


def test_render_marked_page_draws_no_note_when_absent(page, monkeypatch):
    image, results, boxes = page  # q2 is wrong but carries no note
    monkeypatch.setattr(
        markup, "_draw_note", lambda *a: pytest.fail("note drawn without one")
    )

    render_marked_page(image, results, boxes)


def test_render_marked_page_has_no_score_header():
    # Issue #66: the marked-up page must never stamp a score at the top.
    import inspect

    assert not hasattr(markup, "_draw_score_header")
    assert "draw_score" not in inspect.signature(render_marked_page).parameters


def test_render_marked_page_draws_nothing_for_a_blank_result(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.2, 0.1)}
    results = {"q1": QuestionResult(answer="7", response="", correct=False, blank=True)}
    for name in ("_draw_check", "_draw_cross", "_draw_answer", "_draw_note"):
        monkeypatch.setattr(
            markup, name, lambda *a, name=name, **k: pytest.fail(f"{name} drawn for a blank result")
        )

    marked = render_marked_page(image, results, boxes)

    assert np.all(marked == 255)


def test_render_marked_page_still_marks_non_blank_questions_when_mixed_with_blank(page):
    image, results, boxes = page
    boxes["q3"] = Box(0.1, 0.1, 0.2, 0.1)
    results["q3"] = QuestionResult(answer="9", response="", correct=False, blank=True)

    marked = render_marked_page(image, results, boxes)

    assert _has_pixel(marked, _CORRECT_COLOR)  # q1 still checked
    assert _has_pixel(marked, _WRONG_COLOR)  # q2 still crossed


def test_render_marked_page_draws_nothing_for_an_open_ended_result(monkeypatch):
    # issue #65: an open-ended question has no correct answer to check
    # against, so markup must treat it like a blank result -- draw nothing.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.2, 0.1)}
    results = {"q1": QuestionResult(answer="", response="I like fractions", correct=False, open_ended=True)}
    for name in ("_draw_check", "_draw_cross", "_draw_answer", "_draw_note"):
        monkeypatch.setattr(
            markup, name, lambda *a, name=name, **k: pytest.fail(f"{name} drawn for an open-ended result")
        )

    marked = render_marked_page(image, results, boxes)

    assert np.all(marked == 255)


def test_render_marked_page_still_marks_non_open_ended_questions_when_mixed_with_open_ended(page):
    image, results, boxes = page
    boxes["q3"] = Box(0.1, 0.1, 0.2, 0.1)
    results["q3"] = QuestionResult(answer="", response="I like fractions", correct=False, open_ended=True)

    marked = render_marked_page(image, results, boxes)

    assert _has_pixel(marked, _CORRECT_COLOR)  # q1 still checked
    assert _has_pixel(marked, _WRONG_COLOR)  # q2 still crossed


def test_save_marked_pdf_writes_single_page(page, tmp_path):
    image, results, boxes = page
    out_path = tmp_path / "marked.pdf"

    result = save_marked_pdf(image, results, boxes, out_path)

    assert result == out_path
    assert out_path.exists()
    assert pymupdf.open(out_path).page_count == 1
