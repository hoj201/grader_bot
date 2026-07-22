import base64
import difflib
import os
import re
from typing import List, LiteralString

import cv2
import numpy as np
import pytesseract
import requests
from dotenv import load_dotenv

from graderbot.imaging import _crop_box
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


def _tesseract_ocr_name(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(
        gray, None, fx=_NAME_OCR_UPSCALE, fy=_NAME_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC
    )
    return pytesseract.image_to_string(upscaled, config="--psm 7").strip()


def extract_name(image: np.ndarray, box: Box, roster: List[LiteralString]) -> LiteralString:
    """Reads the handwritten name inside `box` on `image` (an already-loaded
    RGB numpy array, e.g. from `load_image_rgb`) and returns whichever name
    in `roster` it most closely matches.

    Cursive handwriting OCR is too unreliable to trust verbatim (e.g.
    Tesseract regularly misreads individual letters), but since students
    are drawn from a known, finite roster, fuzzy-matching the noisy OCR
    text against that roster resolves those misreadings in practice.
    """
    cropped = _crop_box(image, box, _BOX_INSET)
    ocr_text = _tesseract_ocr_name(cropped)
    matches = difflib.get_close_matches(ocr_text, roster, n=1, cutoff=_NAME_MATCH_CUTOFF)
    return matches[0] if matches else ""


def _strip_math_delimiters(text: str) -> str:
    match = _MATH_DELIMITER_PATTERN.match(text)
    return match.group(1).strip() if match else text.strip()


def _fix_stray_slashes(text: str) -> str:
    return _STRAY_SLASH_PATTERN.sub("1", text)


def _mathpix_ocr(image: np.ndarray) -> str:
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
    text = _fix_stray_slashes(_strip_math_delimiters(raw.get("text", "")))

    # Log the exact bytes posted plus the raw response for a future OCR
    # training set (issue #1). Self-gates on env config and is non-fatal.
    from graderbot.mathpix_log import log_mathpix_call

    log_mathpix_call(encoded.tobytes(), raw, text)

    return text


def read_box(image: np.ndarray, box: Box) -> str:
    """Reads the handwritten LaTeX answer inside `box` on `image` (an
    already-loaded RGB numpy array, e.g. from `load_image_rgb`)."""
    cropped = _crop_box(image, box, _BOX_INSET)
    return _mathpix_ocr(cropped)
