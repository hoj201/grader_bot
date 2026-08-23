import base64
import difflib
import os
import re
from dataclasses import dataclass
from typing import List, LiteralString, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import requests
from dotenv import load_dotenv

from graderbot.imaging import _crop_box, crop_box_content_aware
from graderbot.models import Box

load_dotenv()

_NAME_OCR_UPSCALE = 6
_NAME_MATCH_CUTOFF = 0.4

_MATHPIX_TEXT_URL = "https://api.mathpix.com/v3/text"
# Fraction of the box's own width/height to inset the crop by, so the drawn
# border itself is excluded. Including the border causes Mathpix to read the
# box as an empty "checkbox" placeholder (`\square`) instead of OCR'ing its
# contents.
_BOX_INSET = 0.08
_MATH_DELIMITER_PATTERN = re.compile(
    r"^\s*(?:\$\$|\$|\\\(|\\\[)(.*?)(?:\$\$|\$|\\\)|\\\])\s*$", re.DOTALL
)
# Answers are always plain numbers or `\frac{a}{b}` - division is never
# written with a slash. A stray "/" or "\" (not starting a LaTeX command
# name) is therefore a misread of the numeral "1".
_STRAY_SLASH_PATTERN = re.compile(r"[/\\](?![a-zA-Z])")

# NOTE: Mathpix's `alphabets_allowed` request param (non-Latin *script*
# toggles - hi/zh/ja/ko/ru/th/ta/te/gu/bn/vi) looked like a fix for
# misreads, but real testing showed it also tightens Mathpix's internal
# confidence gating for handwritten math: it made Mathpix outright refuse
# (`"error": "image_no_content"`) on a clean, legible handwritten
# `\frac{3}{4}` that read correctly without it. This pipeline deliberately
# repairs Mathpix's garbled-but-present guesses downstream
# (_fix_stray_slashes, grading.py's _iter_fraction_reinterpretations) rather
# than relying on Mathpix's own confidence, so a stricter refusal-prone mode
# is a net loss here - don't re-add `alphabets_allowed` without re-verifying
# against real handwritten crops first.

# Answers are always plain numbers or `\frac{a}{b}` - a lone Greek letter
# can never be a legitimate answer, so any occurrence is a misread of a
# handwritten digit. Confirmed confusion pairs go here as they're observed
# (see the Mathpix call log, issue #1); start with the one that prompted
# this - a hand-drawn "2" (with its looped tail) read as "\alpha".
_GREEK_MISREAD_MAP = {
    r"\alpha": "2",
}
_GREEK_MISREAD_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _GREEK_MISREAD_MAP)
)


# Identifies which AnswerReader produced an OcrResult (issue #70), so a
# graded result can be traced back to the backend that read it once more
# than one is in play. See answer_reader.EASYOCR_SOURCE for the other one --
# defined there rather than here since only this module needs its own.
MATHPIX_SOURCE = "mathpix"


@dataclass(frozen=True)
class OcrResult:
    """One OCR backend's result for one answer box, kept around for
    debugging misreads (issue #70) instead of collapsing straight to a
    string:

    - `text`: the repaired text grading actually compares against, after
      Mathpix's `_strip_math_delimiters`/`_fix_stray_slashes`/`_fix_greek_misreads`
      (EasyOCR has no repair step yet, so its `text` and `raw_text` match).
    - `raw_text`: the backend's own text before any of that repair, so a
      wrong answer can be traced back to what it literally read.
    - `confidence`: the backend's self-reported confidence for the read
      (0-1), or `None` if it didn't report one.
    - `source`: which backend produced this (`MATHPIX_SOURCE` or
      `answer_reader.EASYOCR_SOURCE`), or `""` if unspecified.
    """

    text: str
    raw_text: str
    confidence: Optional[float]
    source: str = ""


def _tesseract_ocr_name(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(
        gray, None, fx=_NAME_OCR_UPSCALE, fy=_NAME_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC
    )
    return pytesseract.image_to_string(upscaled, config="--psm 7").strip()


def extract_name_scored(
    image: np.ndarray, box: Box, roster: List[LiteralString]
) -> Tuple[LiteralString, float]:
    """Like `extract_name`, but also returns how close the OCR text was to the
    roster name it matched, as a difflib similarity ratio in (0, 1].

    Cursive handwriting OCR is too unreliable to trust verbatim (e.g.
    Tesseract regularly misreads individual letters), but since students
    are drawn from a known, finite roster, fuzzy-matching the noisy OCR
    text against that roster resolves those misreadings in practice. The
    similarity is that match's quality, which the Grade tab surfaces so a weak
    match can be spotted by eye (issue #58). Returns `("", 0.0)` when nothing
    clears `_NAME_MATCH_CUTOFF`.
    """
    cropped = _crop_box(image, box, _BOX_INSET)
    ocr_text = _tesseract_ocr_name(cropped)
    matches = difflib.get_close_matches(ocr_text, roster, n=1, cutoff=_NAME_MATCH_CUTOFF)
    if not matches:
        return "", 0.0
    # Same sequence orientation difflib.get_close_matches itself uses
    # (seq1=candidate, seq2=query), so the score matches what it ranked on.
    return matches[0], difflib.SequenceMatcher(None, matches[0], ocr_text).ratio()


def extract_name(image: np.ndarray, box: Box, roster: List[LiteralString]) -> LiteralString:
    """Reads the handwritten name inside `box` on `image` (an already-loaded
    RGB numpy array, e.g. from `load_image_rgb`) and returns whichever name
    in `roster` it most closely matches, or `""` if none is close enough."""
    return extract_name_scored(image, box, roster)[0]


def _strip_math_delimiters(text: str) -> str:
    match = _MATH_DELIMITER_PATTERN.match(text)
    return match.group(1).strip() if match else text.strip()


def _fix_stray_slashes(text: str) -> str:
    return _STRAY_SLASH_PATTERN.sub("1", text)


def _fix_greek_misreads(text: str) -> str:
    return _GREEK_MISREAD_PATTERN.sub(lambda m: _GREEK_MISREAD_MAP[m.group(0)], text)


def _mathpix_ocr(image: np.ndarray) -> OcrResult:
    app_id = os.environ.get("MATHPIX_APP_ID")
    app_key = os.environ.get("MATHPIX_APP_KEY")
    if not app_id or not app_key:
        raise EnvironmentError(
            "MATHPIX_APP_ID and MATHPIX_APP_KEY must be set (e.g. in a .env file) to use read_box"
        )

    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Could not encode cropped box image")
    data_uri = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    response = requests.post(
        _MATHPIX_TEXT_URL,
        headers={
            "app_id": app_id,
            "app_key": app_key,
            "Content-type": "application/json",
        },
        json={"src": data_uri, "formats": ["text"], "rm_spaces": True},
    )
    response.raise_for_status()
    raw = response.json()
    raw_text = raw.get("text", "")
    text = _fix_greek_misreads(_fix_stray_slashes(_strip_math_delimiters(raw_text)))

    # Log the exact bytes posted plus the raw response for a future OCR
    # training set (issue #1). Self-gates on env config and is non-fatal.
    from graderbot.mathpix_log import log_mathpix_call

    log_mathpix_call(encoded.tobytes(), raw, text)

    return OcrResult(
        text=text, raw_text=raw_text, confidence=raw.get("confidence"), source=MATHPIX_SOURCE
    )


def read_box(image: np.ndarray, box: Box) -> OcrResult:
    """Reads the handwritten LaTeX answer inside `box` on `image` (an
    already-loaded RGB numpy array, e.g. from `load_image_rgb`). See
    `OcrResult` for what's returned beyond the repaired text."""
    cropped = crop_box_content_aware(image, box, fallback_inset=_BOX_INSET)
    return _mathpix_ocr(cropped)
