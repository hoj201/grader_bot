import re
from fractions import Fraction
from math import gcd
from typing import Dict, Iterator, LiteralString, Optional, Tuple, Union

import numpy as np

from graderbot.answer_reader import AnswerReader, MathpixAnswerReader
from graderbot.imaging import _crop_box, is_blank
from graderbot.models import Box, QuestionResult
from graderbot.ocr import _BOX_INSET, OcrResult
from graderbot.response_candidates import generate_candidates, is_plain_numeric
from graderbot.response_scorer import ResponseScorer

_ANSWER_FRAC_PATTERN = re.compile(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_DECIMAL_PLACES = 3

# Where a scored response's `ocr_source`/`QuestionResult.ocr_source` comes
# from when `response_scorer` supplied the response instead of `answer_reader`.
_RESPONSE_SCORER_SOURCE = "response_scorer"


def grade_hw(
    answer_key: Dict[LiteralString, LiteralString],
    boxes: Dict[LiteralString, Box],
    hw_image: np.ndarray,
    open_ended: Optional[Dict[LiteralString, bool]] = None,
    answer_reader: Optional[AnswerReader] = None,
    response_scorer: Optional[ResponseScorer] = None,
) -> Dict[LiteralString, QuestionResult]:
    """Grades a single student's work, returning a per-question breakdown
    keyed by question id: for each box, the stored `answer`, the student's
    OCR'd `response`, and whether they match. A box with too little ink to
    have anything written in it is graded as blank without ever calling
    OCR (issue #66) -- `response` is `""`, `correct` is False, and `blank`
    is True so a marked-up page can skip drawing anything for it. This
    granularity is what a marked-up feedback PDF needs (issue #24).

    `open_ended` optionally maps a question id to whether it has no single
    correct answer (issue #65). Such a question still gets its response
    OCR'd (for a teacher to read later) unless the box is blank, but is
    never compared against `answer_key` -- its result always carries
    `correct=False, open_ended=True` so callers can tell "not graded" apart
    from "graded wrong" and skip it when scoring or marking up a page.

    `answer_reader` picks the OCR backend (issue #70); it defaults to
    `MathpixAnswerReader`, the pre-issue-#70 behavior. Each result also
    carries that backend's confidence and raw pre-repair text for the
    response, so a wrong answer can be traced back to what OCR actually
    saw -- see `answer_reader.AnswerReader`/`ocr.OcrResult`.

    `response_scorer` (issue #81) verifies the crop against the known
    answer plus a handful of OCR-confusable near-misses
    (`response_candidates.generate_candidates`) instead of transcribing it
    open-vocabulary, and takes over from `answer_reader` for a non-blank,
    non-open-ended question whose stored answer is plain numeric
    (`response_candidates.is_plain_numeric` -- v1 has no fraction support,
    see that module's docstring). `answer_reader` still handles every other
    box on the worksheet (fractions, open-ended questions), so passing
    `response_scorer` alongside the default `MathpixAnswerReader` is the
    normal way to use it, not an either/or choice. The response it picks is
    recorded with `ocr_source="response_scorer"` and its normalized
    candidate-vs-candidate score as `ocr_confidence`, so it shows up in the
    same debugging views a wrong OCR read does."""
    open_ended = open_ended or {}
    answer_reader = answer_reader if answer_reader is not None else MathpixAnswerReader()
    results: Dict[LiteralString, QuestionResult] = {}
    for qid, box in boxes.items():
        answer = answer_key[qid]
        is_open_ended = open_ended.get(qid, False)
        crop = _crop_box(hw_image, box, _BOX_INSET)
        if crop.size > 0 and is_blank(crop):
            results[qid] = QuestionResult(answer=answer, response="", correct=False, blank=True, open_ended=is_open_ended)
            continue
        if response_scorer is not None and not is_open_ended and is_plain_numeric(answer):
            scored = response_scorer.score(hw_image, box, generate_candidates(answer))
            ocr_result = OcrResult(
                text=scored.best,
                raw_text=scored.best,
                confidence=scored.scores[scored.best],
                source=_RESPONSE_SCORER_SOURCE,
            )
        else:
            ocr_result = answer_reader.read(hw_image, box)
        response = ocr_result.text
        if is_open_ended:
            results[qid] = QuestionResult(
                answer=answer,
                response=response,
                correct=False,
                open_ended=True,
                ocr_confidence=ocr_result.confidence,
                ocr_raw=ocr_result.raw_text,
                ocr_source=ocr_result.source,
            )
            continue
        correct = grade_response(response, answer)
        results[qid] = QuestionResult(
            answer=answer,
            response=response,
            correct=correct,
            ocr_confidence=ocr_result.confidence,
            ocr_source=ocr_result.source,
            ocr_raw=ocr_result.raw_text,
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
    so callers can accept a reduced garble as correct while still rejecting
    an unsimplified one (e.g. `20 1 42` -> 10/21)."""
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


def grade_response(response: LiteralString, answer: LiteralString) -> bool:
    """Grades a student `response` against the LaTeX `answer`, returning
    whether it is correct. A response that has the right *value* but is
    written as a non-reduced fraction (e.g. `\\frac{20}{42}` for `10/21`) is
    still graded wrong -- the student must simplify -- it just gets no special
    treatment beyond that (issue #71 removed the "simplify" feedback nudge
    from issue #38; a wrong answer is just wrong)."""
    answer_value = _parse_answer(answer)
    if answer_value is None:
        return False

    response_value = _parse_answer(response)

    # A fraction written in non-reduced form is wrong even when its value is
    # right - the student must simplify.
    frac_match = _ANSWER_FRAC_PATTERN.match(response.strip())
    if frac_match:
        numerator, denominator = (int(g) for g in frac_match.groups())
        if denominator != 0 and gcd(numerator, denominator) != 1:
            return False

    if response_value is not None:
        if isinstance(response_value, float) or isinstance(answer_value, float):
            return round(float(response_value), _DECIMAL_PLACES) == round(float(answer_value), _DECIMAL_PLACES)
        if response_value == answer_value:
            return True

    # Fall back to tolerating mathpix's fraction garbling, but only when the
    # correct answer is genuinely a fraction (issue #39). A garble that only
    # reaches the right value through an unreduced written form (e.g.
    # `20 1 42`) is still wrong.
    if isinstance(answer_value, Fraction) and answer_value.denominator != 1:
        reduced_flags = [reduced for value, reduced in _iter_fraction_reinterpretations(response) if value == answer_value]
        if any(reduced_flags):
            return True

    return False


def is_correct(response: LiteralString, answer: LiteralString) -> bool:
    """Takes the LaTeX string for an answer and compares it to the submitted
    response by a student. If the answers are equal it is marked as correct.
    (Thin wrapper over `grade_response` -- kept for callers that read better
    naming the boolean check explicitly.)"""
    return grade_response(response, answer)
