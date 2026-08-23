"""Generates plausible near-miss strings for a stored answer (issue #81's
closed-set verification approach): a `response_scorer.ResponseScorer`
doesn't decide "what does this crop say" the open-vocabulary way an OCR
backend does -- it scores a crop against a short list of *candidate*
strings and reports which one the crop most likely shows. This module
builds that candidate list, and doubles as the hard-negative generator used
to train the scorer itself (see the "Data pipeline" section of issue #81).

Scope (v1): plain numeric answers only -- integers, decimals, negative
numbers. This mirrors the domain `answer_reader.EasyOcrAnswerReader` and
`GoogleVisionAnswerReader` already commit to (neither reads fractions
either, and the Grade tab already tells users to fall back to Mathpix for
fraction worksheets). `grading.grade_hw` skips `response_scorer` entirely
for a `\\frac{a}{b}` answer, the same way it would for any answer
`is_plain_numeric` rejects -- extending this to fractions is future work
(issue #81, later phases), not attempted here.
"""

import re
from typing import List, Optional

import numpy as np

# Handwritten-digit confusions worth stress-testing a candidate list against
# -- pairs that look alike in sloppy/cursive digits. Not exhaustive, and
# deliberately digit-for-digit only (unlike issue #70's OCR misreads, which
# included letters -- irrelevant here since the scorer's vocabulary has no
# letters to confuse a digit with).
_DIGIT_CONFUSIONS = {
    "0": ("6", "8"),
    "1": ("7",),
    "2": ("7",),
    "3": ("8",),
    "4": ("9",),
    "5": ("6",),
    "6": ("0", "5"),
    "7": ("1", "2"),
    "8": ("0", "3"),
    "9": ("4",),
}

_PLAIN_NUMERIC_PATTERN = re.compile(r"-?\d+(\.\d+)?")

# A representative middle-school-worksheet answer pool: small whole numbers,
# some negatives, some one/two-decimal-place values. Not exhaustive -- just
# varied enough that a model trained on it doesn't overfit to one answer
# shape. Shared by two things that need the exact same domain distribution:
# `training/dataset.py`'s synthetic data generator, and
# `handwriting_sample_worksheets.py`'s "copy this number" worksheets used to
# harvest real (crop, text) training pairs (issue #81) -- keeping this in
# one place means the real and synthetic halves of training data cover the
# same answer shapes rather than drifting apart.
SAMPLE_ANSWER_POOL = tuple(
    [str(n) for n in range(0, 100)]
    + [str(-n) for n in range(1, 50)]
    + [f"{n}.{d}" for n in range(0, 20) for d in range(10)]
    + [f"{n}.{d}{e}" for n in range(0, 10) for d in range(10) for e in (0, 5)]
)


def is_plain_numeric(answer: str) -> bool:
    """True if `answer` is a plain (optionally signed, optionally decimal)
    number -- the only shape `generate_candidates` and `CnnResponseScorer`
    handle. False for a `\\frac{a}{b}` answer or anything else
    `grading._parse_answer` wouldn't understand either."""
    return bool(_PLAIN_NUMERIC_PATTERN.fullmatch(answer.strip()))


def _digit_swap_candidates(digits: str) -> List[str]:
    """One-digit-swapped variants of `digits`, one per (position, confusable)
    pair -- e.g. "16" -> ["76", "10", "15"] (1->7 at position 0; 6->0, 6->5
    at position 1)."""
    swapped = []
    for i, d in enumerate(digits):
        for alt in _DIGIT_CONFUSIONS.get(d, ()):
            swapped.append(digits[:i] + alt + digits[i + 1 :])
    return swapped


def generate_candidates(
    answer: str, rng: Optional[np.random.Generator] = None, max_candidates: int = 6
) -> List[str]:
    """Returns a short list of candidate strings a `ResponseScorer` should
    score a crop against for this `answer`: the answer itself first
    (always present, always first), followed by plausible near-misses --
    digit-confusion swaps, a misread/missing decimal point, and a
    missed/added sign -- deduplicated and capped at `max_candidates`
    (including the answer itself).

    `rng` only matters when there would be more candidates than
    `max_candidates`, in which case it subsamples which ones survive; pass a
    seeded `Generator` for reproducible training data, or leave it unset for
    runtime use (score every candidate the model gets, no need for the
    subsampling to be reproducible).

    Returns `[answer]` alone for anything `is_plain_numeric` rejects (e.g. a
    fraction) -- see module docstring."""
    answer = answer.strip()
    if not is_plain_numeric(answer):
        return [answer]

    sign = "-" if answer.startswith("-") else ""
    unsigned = answer[1:] if sign else answer

    candidates = set()
    if "." in unsigned:
        whole, frac = unsigned.split(".", 1)
        digits = whole + frac
        for swapped_digits in _digit_swap_candidates(digits):
            new_whole, new_frac = swapped_digits[: len(whole)], swapped_digits[len(whole) :]
            candidates.add(f"{sign}{new_whole}.{new_frac}" if new_frac else f"{sign}{new_whole}")
        # The decimal point itself misread one place over.
        if whole:
            candidates.add(f"{sign}{whole[:-1]}.{whole[-1]}{frac}")
        if len(frac) > 1:
            candidates.add(f"{sign}{whole}{frac[0]}.{frac[1:]}")
        # The decimal point missed entirely (read as nothing, or as a stray mark).
        candidates.add(f"{sign}{whole}{frac}")
    else:
        for swapped_digits in _digit_swap_candidates(unsigned):
            candidates.add(f"{sign}{swapped_digits}")
        # A decimal point hallucinated somewhere in an integer answer.
        for i in range(1, len(unsigned)):
            candidates.add(f"{sign}{unsigned[:i]}.{unsigned[i:]}")

    # Sign missed or hallucinated -- a faint/cramped "-" is easy to miss, and
    # stray marks near a box's left edge are easy to mistake for one.
    candidates.add(f"-{unsigned}" if not sign else unsigned)

    candidates.discard(answer)
    ordered = [answer] + sorted(candidates)
    if len(ordered) <= max_candidates:
        return ordered

    rng = rng if rng is not None else np.random.default_rng()
    kept = rng.choice(ordered[1:], size=max_candidates - 1, replace=False).tolist()
    return [answer] + kept
