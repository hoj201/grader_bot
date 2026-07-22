import re
from fractions import Fraction
from typing import Dict, LiteralString, Optional, Union

import numpy as np

from graderbot.models import Box, QuestionResult
from graderbot.ocr import read_box

_ANSWER_FRAC_PATTERN = re.compile(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_DECIMAL_PLACES = 3


def grade_hw(answer_key: Dict[LiteralString, LiteralString], boxes: Dict[LiteralString, Box], hw_image: np.ndarray) -> Dict[LiteralString, QuestionResult]:
    """Grades a single student's work, returning a per-question breakdown
    keyed by question id: for each box, the stored `answer`, the student's
    OCR'd `response`, and whether they match. This granularity is what a
    marked-up feedback PDF needs (issue #24)."""
    results: Dict[LiteralString, QuestionResult] = {}
    for qid, box in boxes.items():
        response = read_box(hw_image, box)
        answer = answer_key[qid]
        results[qid] = QuestionResult(
            answer=answer, response=response, correct=is_correct(response, answer)
        )
    return results


def _parse_answer(text: LiteralString) -> Optional[Union[Fraction, float]]:
    """Parses a LaTeX answer/response snippet into a `Fraction` (exact
    rationals - plain integers and `\\frac{a}{b}`, which `Fraction`
    automatically reduces so a fraction over 1 compares equal to that
    integer) or a `float` (decimals). Returns None if `text` is neither."""
    text = text.strip()

    frac_match = _ANSWER_FRAC_PATTERN.match(text)
    if frac_match:
        numerator, denominator = frac_match.groups()
        return Fraction(int(numerator), int(denominator))

    try:
        if "." in text:
            return float(text)
        return Fraction(int(text))
    except ValueError:
        return None


def is_correct(response: LiteralString, answer: LiteralString) -> bool:
    """Takes the LaTeX string for an answer and compares it to the submitted response by a student.
    If the answers are equal it is marked as correct."""
    response_value = _parse_answer(response)
    answer_value = _parse_answer(answer)
    if response_value is None or answer_value is None:
        return False

    if isinstance(response_value, float) or isinstance(answer_value, float):
        return round(float(response_value), _DECIMAL_PLACES) == round(float(answer_value), _DECIMAL_PLACES)

    return response_value == answer_value
