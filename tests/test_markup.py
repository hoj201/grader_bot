import numpy as np
import pymupdf
import pytest

from graderbot import markup
from graderbot.imaging import box_pixel_rect
from graderbot.models import Box, QuestionResult
from graderbot.markup import render_marked_page, save_marked_pdf


@pytest.fixture
def page():
    # White canonical page, answer boxes on the left so there's room for the
    # correct answer to be written to their right.
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


def _fake_draw_answer(calls):
    def fake(img, box_px, answer, font, text_size):
        calls.append({"box_px": box_px, "answer": answer})
        return img

    return fake


def test_render_marked_page_writes_correct_answer_only_for_wrong(page, monkeypatch):
    image, results, boxes = page
    calls = []
    monkeypatch.setattr(markup, "_draw_answer", _fake_draw_answer(calls))

    render_marked_page(image, results, boxes)

    # Only the wrong question (q2, answer "5") should get its answer drawn.
    assert [c["answer"] for c in calls] == ["5"]


def test_render_marked_page_draws_nothing_for_a_correct_answer(monkeypatch):
    # Issue #71: no checkmark, no mark at all -- a correct answer gets no
    # markup whatsoever.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.2, 0.1)}
    results = {"q1": QuestionResult(answer="7", response="7", correct=True)}
    monkeypatch.setattr(
        markup, "_draw_answer", lambda *a, **k: pytest.fail("answer drawn for a correct result")
    )

    marked = render_marked_page(image, results, boxes)

    assert np.all(marked == 255)


def test_render_marked_page_writes_visible_answer_for_a_wrong_result(page):
    # No mocking here: the correct answer must actually be rendered onto
    # the page (in the computer font), not just requested.
    image, results, boxes = page

    marked = render_marked_page(image, results, boxes)

    assert np.any(marked != 255)


def test_render_marked_page_leaves_input_unmodified(page):
    image, results, boxes = page

    render_marked_page(image, results, boxes)

    assert np.all(image == 255)


def test_render_marked_page_writes_answer_close_to_the_box(page, monkeypatch):
    # Issue #71: the correct answer should be written as close to the
    # answer box as possible, not offset to make room for a mark.
    image, results, boxes = page
    calls = []
    monkeypatch.setattr(markup, "_draw_answer", _fake_draw_answer(calls))

    render_marked_page(image, results, boxes)

    x0, y0, x1, y1 = box_pixel_rect(boxes["q2"], 200, 200)
    ans_x0, ans_y0, ans_x1, ans_y1 = calls[0]["box_px"]
    box_height = y1 - y0
    assert (ans_y0, ans_y1) == (y0, y1)
    assert ans_x0 > x1
    assert ans_x0 - x1 < box_height


def test_render_marked_page_has_no_score_header():
    # Issue #66: the marked-up page must never stamp a score at the top.
    import inspect

    assert not hasattr(markup, "_draw_score_header")
    assert "draw_score" not in inspect.signature(render_marked_page).parameters


def test_render_marked_page_draws_nothing_for_a_blank_result(monkeypatch):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.2, 0.1)}
    results = {"q1": QuestionResult(answer="7", response="", correct=False, blank=True)}
    monkeypatch.setattr(
        markup, "_draw_answer", lambda *a, **k: pytest.fail("answer drawn for a blank result")
    )

    marked = render_marked_page(image, results, boxes)

    assert np.all(marked == 255)


def test_render_marked_page_still_marks_non_blank_questions_when_mixed_with_blank(page, monkeypatch):
    image, results, boxes = page
    boxes["q3"] = Box(0.1, 0.1, 0.2, 0.1)
    results["q3"] = QuestionResult(answer="9", response="", correct=False, blank=True)
    calls = []
    monkeypatch.setattr(markup, "_draw_answer", _fake_draw_answer(calls))

    render_marked_page(image, results, boxes)

    # q1 is correct (no markup) and q3 is blank (no markup); only q2 gets
    # its answer drawn.
    assert [c["answer"] for c in calls] == ["5"]


def test_render_marked_page_draws_nothing_for_an_open_ended_result(monkeypatch):
    # issue #65: an open-ended question has no correct answer to check
    # against, so markup must treat it like a blank result -- draw nothing.
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    boxes = {"q1": Box(0.1, 0.5, 0.2, 0.1)}
    results = {"q1": QuestionResult(answer="", response="I like fractions", correct=False, open_ended=True)}
    monkeypatch.setattr(
        markup, "_draw_answer", lambda *a, **k: pytest.fail("answer drawn for an open-ended result")
    )

    marked = render_marked_page(image, results, boxes)

    assert np.all(marked == 255)


def test_render_marked_page_still_marks_non_open_ended_questions_when_mixed_with_open_ended(page, monkeypatch):
    image, results, boxes = page
    boxes["q3"] = Box(0.1, 0.1, 0.2, 0.1)
    results["q3"] = QuestionResult(answer="", response="I like fractions", correct=False, open_ended=True)
    calls = []
    monkeypatch.setattr(markup, "_draw_answer", _fake_draw_answer(calls))

    render_marked_page(image, results, boxes)

    # q1 is correct (no markup) and q3 is open-ended (no markup); only q2
    # gets its answer drawn.
    assert [c["answer"] for c in calls] == ["5"]


def test_save_marked_pdf_writes_single_page(page, tmp_path):
    image, results, boxes = page
    out_path = tmp_path / "marked.pdf"

    result = save_marked_pdf(image, results, boxes, out_path)

    assert result == out_path
    assert out_path.exists()
    assert pymupdf.open(out_path).page_count == 1
