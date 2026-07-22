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


def test_render_marked_page_stamps_score_header(page, monkeypatch):
    image, results, boxes = page
    calls = []

    def fake_header(img, correct, total):
        calls.append((correct, total))

    monkeypatch.setattr(markup, "_draw_score_header", fake_header)

    render_marked_page(image, results, boxes)

    # One correct (q1), one wrong (q2) out of two questions.
    assert calls == [(1, 2)]


def test_render_marked_page_can_skip_score_header(page, monkeypatch):
    image, results, boxes = page
    monkeypatch.setattr(
        markup, "_draw_score_header", lambda *a: pytest.fail("score header drawn")
    )

    render_marked_page(image, results, boxes, draw_score=False)


def test_save_marked_pdf_writes_single_page(page, tmp_path):
    image, results, boxes = page
    out_path = tmp_path / "marked.pdf"

    result = save_marked_pdf(image, results, boxes, out_path)

    assert result == out_path
    assert out_path.exists()
    assert pymupdf.open(out_path).page_count == 1
