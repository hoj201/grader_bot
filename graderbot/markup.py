"""Render a marked-up feedback page from grading results.

Given a student's worksheet image (already in the canonical page frame, e.g.
via `graderbot.rectify_to_canonical`), the per-question grading results
(`graderbot.grade_hw` -> {question id: QuestionResult}), and the answer-box
locations, this draws a green check on correct answers and a red cross on wrong
ones, writing the correct answer beside each wrong box. The result can be saved
as a one-page PDF to hand back to the student.

This is a pure renderer: it takes an already-graded image + results + boxes and
does no database or OCR work of its own.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

from graderbot.imaging import box_pixel_rect
from graderbot.models import Box, QuestionResult
from graderbot.storage import image_to_pdf
from graderbot.worksheet_synth import _DEFAULT_FONT, _DEFAULT_TEXT_SIZE, _draw_answer

# Colors are in RGB order (the renderer works on RGB arrays).
_CORRECT_COLOR = (0, 150, 0)
_WRONG_COLOR = (200, 0, 0)
_MARK_FRACTION = 0.6  # mark size as a fraction of the answer box height


def _draw_check(image: np.ndarray, center: Tuple[int, int], size: int, color, thickness: int) -> None:
    cx, cy = center
    s = size / 2
    points = np.array(
        [
            (cx - s, cy),
            (cx - s / 3, cy + s * 0.8),
            (cx + s, cy - s * 0.8),
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)


def _draw_cross(image: np.ndarray, center: Tuple[int, int], size: int, color, thickness: int) -> None:
    cx, cy = center
    s = size // 2
    cv2.line(image, (cx - s, cy - s), (cx + s, cy + s), color, thickness)
    cv2.line(image, (cx - s, cy + s), (cx + s, cy - s), color, thickness)


def _draw_note(image: np.ndarray, note: str, origin: Tuple[int, int], box_height: int, color) -> None:
    """Writes a short feedback word (e.g. "simplify") at `origin`, sized
    relative to the answer box height. Modifies `image` in place."""
    scale = max(box_height / 60.0, 0.4)
    thickness = max(int(round(scale * 2)), 1)
    cv2.putText(image, note, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_marked_page(
    image: np.ndarray,
    results: Dict[str, QuestionResult],
    boxes: Dict[str, Box],
    font: str = _DEFAULT_FONT,
    text_size: int = _DEFAULT_TEXT_SIZE,
) -> np.ndarray:
    """Annotates a canonical-frame RGB `image` with per-question feedback and
    returns a new image (the input is left unmodified). For each question in
    `results` that has a box in `boxes` and was actually answered (not
    `result.blank`), draws a green check (correct) or red cross (incorrect)
    just right of the answer box; for wrong answers, writes the correct
    answer beside the mark. A blank box gets no markup at all -- no check,
    cross, answer, or note (issue #66)."""
    marked = np.ascontiguousarray(image).copy()
    height, width = marked.shape[:2]

    for qid, result in results.items():
        box = boxes.get(qid)
        if box is None or result.blank:
            continue

        x0, y0, x1, y1 = box_pixel_rect(box, width, height)
        box_height = y1 - y0
        mark_size = max(int(box_height * _MARK_FRACTION), 8)
        thickness = max(mark_size // 6, 2)
        margin = mark_size // 2

        mark_cx = min(x1 + margin + mark_size // 2, width - 1)
        mark_cy = (y0 + y1) // 2
        color = _CORRECT_COLOR if result.correct else _WRONG_COLOR

        if result.correct:
            _draw_check(marked, (mark_cx, mark_cy), mark_size, color, thickness)
        else:
            _draw_cross(marked, (mark_cx, mark_cy), mark_size, color, thickness)
            # Write the correct answer to the right of the mark, in a rect the
            # same size as the answer box (clamped to the page width).
            ans_x0 = mark_cx + mark_size
            ans_x1 = min(ans_x0 + (x1 - x0), width)
            if ans_x1 > ans_x0:
                marked = _draw_answer(marked, (ans_x0, y0, ans_x1, y1), result.answer, font, text_size)

            # A feedback nudge (e.g. "simplify") goes just below the box.
            if result.note:
                note_y = min(y1 + box_height, height - 1)
                _draw_note(marked, result.note, (ans_x0, note_y), box_height, color)

    return marked


def save_marked_pdf(
    image: np.ndarray,
    results: Dict[str, QuestionResult],
    boxes: Dict[str, Box],
    out_path: Union[str, Path],
    font: str = _DEFAULT_FONT,
    text_size: int = _DEFAULT_TEXT_SIZE,
) -> Path:
    """Renders the marked-up page (see `render_marked_page`) and writes it as a
    one-page PDF to `out_path`, returning the path."""
    marked_rgb = render_marked_page(image, results, boxes, font=font, text_size=text_size)
    # image_to_pdf expects BGR (it converts BGR->RGB internally before saving).
    marked_bgr = cv2.cvtColor(marked_rgb, cv2.COLOR_RGB2BGR)
    return image_to_pdf(marked_bgr, Path(out_path))
