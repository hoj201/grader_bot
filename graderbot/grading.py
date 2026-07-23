import re
from fractions import Fraction
from math import gcd
from typing import Dict, Iterator, LiteralString, Optional, Tuple, Union

import numpy as np

from graderbot.models import Box, QuestionResult
from graderbot.ocr import read_box

_ANSWER_FRAC_PATTERN = re.compile(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_DECIMAL_PLACES = 3
_SIMPLIFY_NOTE = "simplify"


def grade_hw(answer_key: Dict[LiteralString, LiteralString], boxes: Dict[LiteralString, Box], hw_image: np.ndarray) -> Dict[LiteralString, QuestionResult]:
    """Grades a single student's work, returning a per-question breakdown
    keyed by question id: for each box, the stored `answer`, the student's
    OCR'd `response`, and whether they match. This granularity is what a
    marked-up feedback PDF needs (issue #24)."""
    results: Dict[LiteralString, QuestionResult] = {}
    for qid, box in boxes.items():
        response = read_box(hw_image, box)
        answer = answer_key[qid]
        correct, note = grade_response(response, answer)
        results[qid] = QuestionResult(
            answer=answer, response=response, correct=correct, note=note
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


def _iter_fraction_reinterpretations(text: LiteralString) -> Iterator[Tuple[Fraction, bool]]:
    """Yields `(value, reduced)` pairs for the fractions a garbled OCR `text`
    could plausibly represent (issue #39). Mathpix frequently transcribes a
    hand-drawn fraction bar as the digit `1` (so `10 / 21` becomes `10 1 21`)
    or keeps it as a literal `/`. We strip all whitespace and then treat every
    `/` as a forced bar and every `1` as an *optional* bar, enumerating both

      - simple fractions   num/den, and
      - mixed numbers      whole num/den  (the whole|num boundary was a
        space that whitespace-collapsing erased, so it is enumerated
        separately from the bar).

    `reduced` reports whether the *written* form was already in lowest terms,
    so callers can accept a reduced garble as correct while flagging an
    unsimplified one (e.g. `20 1 42` -> 10/21) as a "simplify" nudge."""
    text = text.strip()
    sign = 1
    if text.startswith("-"):
        sign, text = -1, text[1:]

    s = re.sub(r"\s+", "", text)
    if not s:
        return

    slashes = [i for i, c in enumerate(s) if c == "/"]
    if len(slashes) > 1:
        # More than one explicit bar is not a fraction we know how to read.
        return
    bar_positions = slashes if slashes else [i for i, c in enumerate(s) if c == "1"]

    for bar in bar_positions:
        left, right = s[:bar], s[bar + 1 :]
        if not left.isdigit() or not right.isdigit():
            continue
        den = int(right)
        if den <= 1:
            continue

        # Simple fraction num/den (improper is fine, e.g. 7/4).
        num = int(left)
        if num != 0:
            yield sign * Fraction(num, den), gcd(num, den) == 1

        # Mixed number whole num/den; split the pre-bar digits every way.
        for k in range(1, len(left)):
            whole, mnum = int(left[:k]), int(left[k:])
            if whole >= 1 and 0 < mnum < den:
                yield sign * (whole + Fraction(mnum, den)), gcd(mnum, den) == 1


def grade_response(response: LiteralString, answer: LiteralString) -> Tuple[bool, str]:
    """Grades a student `response` against the LaTeX `answer`, returning
    `(correct, note)`. `note` is normally "" but is set to a short feedback
    nudge ("simplify") when the response has the right *value* yet is written as
    a non-reduced fraction, so the marked-up page can tell the student what went
    wrong instead of only crossing it out (issue #38)."""
    answer_value = _parse_answer(answer)
    if answer_value is None:
        return False, ""

    response_value = _parse_answer(response)

    # A fraction written in non-reduced form (e.g. \frac{20}{42} for 10/21) is
    # marked wrong even when its value is right - the student must simplify. If
    # the value is right we add a "simplify" note; if it is also wrong, it is
    # just an ordinary wrong answer.
    frac_match = _ANSWER_FRAC_PATTERN.match(response.strip())
    if frac_match:
        numerator, denominator = (int(g) for g in frac_match.groups())
        if denominator != 0 and gcd(numerator, denominator) != 1:
            note = _SIMPLIFY_NOTE if response_value == answer_value else ""
            return False, note

    if response_value is not None:
        if isinstance(response_value, float) or isinstance(answer_value, float):
            close = round(float(response_value), _DECIMAL_PLACES) == round(float(answer_value), _DECIMAL_PLACES)
            return close, ""
        if response_value == answer_value:
            return True, ""

    # Fall back to tolerating mathpix's fraction garbling, but only when the
    # correct answer is genuinely a fraction (issue #39). A garble that reaches
    # the right value only through an unreduced written form (e.g. `20 1 42`)
    # is still wrong, but earns the same "simplify" nudge (issue #38).
    if isinstance(answer_value, Fraction) and answer_value.denominator != 1:
        reduced_flags = [reduced for value, reduced in _iter_fraction_reinterpretations(response) if value == answer_value]
        if any(reduced_flags):
            return True, ""
        if reduced_flags:
            return False, _SIMPLIFY_NOTE

    return False, ""


def is_correct(response: LiteralString, answer: LiteralString) -> bool:
    """Takes the LaTeX string for an answer and compares it to the submitted
    response by a student. If the answers are equal it is marked as correct.
    (Thin wrapper over `grade_response` for callers that only need the boolean.)"""
    return grade_response(response, answer)[0]
