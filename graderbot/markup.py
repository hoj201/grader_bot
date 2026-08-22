"""Render a marked-up feedback page from grading results.

Given a student's worksheet image (already in the canonical page frame, e.g.
via `graderbot.rectify_to_canonical`), the per-question grading results
(`graderbot.grade_hw` -> {question id: QuestionResult}), and the answer-box
locations, this writes the correct answer beside any wrong answer, in a
plain computer font as close to the answer box as possible. A correct
answer gets no markup at all -- issue #71 dropped the check/cross marks (and
the "simplify" feedback note from issue #38) in favor of simply showing
students the right answer when they get one wrong. The result can be saved
as a one-page PDF to hand back to the student.

This is a pure renderer: it takes an already-graded image + results + boxes and
does no database or OCR work of its own.
"""

from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np

from graderbot.imaging import box_pixel_rect
from graderbot.models import Box, QuestionResult
from graderbot.storage import image_to_pdf
from graderbot.worksheet_synth import _draw_answer

_REPO_ROOT = Path(__file__).resolve().parent.parent
# A plain computer (serif) font for showing the correct answer, distinct
# from worksheet_synth's handwriting font used to simulate student work.
_DEFAULT_FONT = str(_REPO_ROOT / "fonts" / "LiberationSerif-Regular.ttf")
_DEFAULT_TEXT_SIZE = 36
_ANSWER_MARGIN_FRACTION = 0.15  # gap before the answer, as a fraction of box height


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
    `result.blank`): a wrong answer gets the correct answer written in a
    plain computer font just right of the answer box; a correct answer gets
    no markup at all. A blank box gets no markup at all -- no answer written
    (issue #66). Same for an open-ended question (`result.open_ended`) -- it
    has no correct answer to check against, so nothing is drawn for it
    either (issue #65)."""
    marked = np.ascontiguousarray(image).copy()
    height, width = marked.shape[:2]

    for qid, result in results.items():
        box = boxes.get(qid)
        if box is None or result.blank or result.open_ended or result.correct:
            continue

        x0, y0, x1, y1 = box_pixel_rect(box, width, height)
        box_height = y1 - y0
        margin = max(int(box_height * _ANSWER_MARGIN_FRACTION), 4)

        # Write the correct answer just right of the box, in a rect the same
        # size as the answer box (clamped to the page width).
        ans_x0 = min(x1 + margin, width - 1)
        ans_x1 = min(ans_x0 + (x1 - x0), width)
        if ans_x1 > ans_x0:
            marked = _draw_answer(marked, (ans_x0, y0, ans_x1, y1), result.answer, font, text_size)

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
