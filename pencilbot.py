import base64
import difflib
import os
import re
import tempfile
from typing import Dict, Tuple, List, LiteralString

from dataclasses import dataclass

import cv2
import fitz
import numpy as np
import pytesseract
import requests
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Box:
    x_lower_left: float # relative coordinate between 0 and 1
    y_lower_left: float # relative coordinate between 0 and 1
    width: float
    height: float

@dataclass(frozen=True)
class Score:
    correct: int
    attempted: int
    total_questions: int

_RED = (1, 0, 0)
_RED_TOLERANCE = 0.15

def grade_hw_stack(worksheet_fn: LiteralString, answer_key_fn: LiteralString, hws: List[LiteralString], roster: List[LiteralString]) -> Dict[LiteralString, Score]:
    boxes = extract_answer_boxes(worksheet_fn)
    answer_key_fn = align_document_image(answer_key_fn, worksheet_fn)
    answer_key = {qid: read_box(answer_key_fn, box) for qid, box in boxes.items()}
    scores = dict()
    for hw in hws:
        name = extract_name(hw, boxes["name"], roster)
        scores[name] = grade_hw(answer_key, boxes, hw)
    return scores


_NAME_OCR_UPSCALE = 6
_NAME_MATCH_CUTOFF = 0.4


def _tesseract_ocr_name(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(
        gray, None, fx=_NAME_OCR_UPSCALE, fy=_NAME_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC
    )
    return pytesseract.image_to_string(upscaled, config="--psm 7").strip()


def extract_name(hw_fn: LiteralString, box: Box, roster: List[LiteralString]) -> LiteralString:
    """Reads the handwritten name inside `box` on `hw_fn` and returns
    whichever name in `roster` it most closely matches.

    Cursive handwriting OCR is too unreliable to trust verbatim (e.g.
    Tesseract regularly misreads individual letters), but since students
    are drawn from a known, finite roster, fuzzy-matching the noisy OCR
    text against that roster resolves those misreadings in practice.
    """
    image = _load_image_rgb(hw_fn)
    cropped = _crop_box(image, box, _BOX_INSET)
    ocr_text = _tesseract_ocr_name(cropped)
    matches = difflib.get_close_matches(ocr_text, roster, n=1, cutoff=_NAME_MATCH_CUTOFF)
    return matches[0] if matches else ""

def grade_hw(answer_key: Dict[LiteralString, LiteralString], boxes: Dict[LiteralString, LiteralString], hw_fn: LiteralString) -> Score:
    responses = {qid: read_box(hw_fn, box) for qid, box in boxes.items()}
    correct, attempted = 0,0
    for qid, box in boxes.items():
        response = read_box(hw_fn, box)
        if response != "":
            attempted += 1
        answer = answer_key[qid]
        if is_correct(response, answer):
            correct += 1
    return Score(correct=correct, attempted=attempted, total_questions=len(answer_key))

def is_correct(response: LiteralString, answer: LiteralString) -> bool:
    """Takes the LaTeX string for an answer and compares it to the submitted response by a student.
    If the answers are equal it is marked as correct."""
    raise NotImplementedError

def _is_red(color) -> bool:
    if color is None or len(color) != 3:
        return False
    return all(abs(c - r) <= _RED_TOLERANCE for c, r in zip(color, _RED))


def extract_answer_boxes(worksheet_filename: str) -> Dict[str, Box]:
    """Takes the filename of a pdf and finds all the red outlined rectangles in the file.
    Inside each rectangle is text corresponding to an id, which is also read.
    The returned object is a dictionary that maps each id to it's corresponding box.
    """
    boxes: Dict[str, Box] = {}

    with fitz.open(worksheet_filename) as doc:
        for page in doc:
            page_width, page_height = page.rect.width, page.rect.height

            for drawing in page.get_drawings():
                if not _is_red(drawing.get("color")):
                    continue

                rect = drawing["rect"]

                words = page.get_text("words", clip=rect)
                if not words:
                    continue
                words.sort(key=lambda w: (w[5], w[6], w[7]))
                box_id = "".join(w[4] for w in words)

                boxes[box_id] = Box(
                    x_lower_left=rect.x0 / page_width,
                    y_lower_left=1 - rect.y1 / page_height,
                    width=rect.width / page_width,
                    height=rect.height / page_height,
                )

    return boxes


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


def _load_image_rgb(image_fn: str) -> np.ndarray:
    if os.path.splitext(image_fn)[1].lower() == ".pdf":
        return render_pdf_page_image(image_fn)
    image_bgr = cv2.imread(image_fn)
    if image_bgr is None:
        raise ValueError(f"Could not read image {image_fn}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _crop_box(image: np.ndarray, box: Box, inset: float) -> np.ndarray:
    image_height, image_width = image.shape[:2]

    left = box.x_lower_left + inset * box.width
    right = box.x_lower_left + box.width - inset * box.width
    top = 1 - box.y_lower_left - box.height + inset * box.height
    bottom = 1 - box.y_lower_left - inset * box.height

    x0, x1 = round(left * image_width), round(right * image_width)
    y0, y1 = round(top * image_height), round(bottom * image_height)
    return image[y0:y1, x0:x1]


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
    text = _strip_math_delimiters(response.json().get("text", ""))
    return _fix_stray_slashes(text)


def read_box(image_fn: str, box: Box) -> str:
    """Reads the handwritten LaTeX answer inside `box` from `image_fn` (a PDF or raster image)."""
    image = _load_image_rgb(image_fn)
    cropped = _crop_box(image, box, _BOX_INSET)
    return _mathpix_ocr(cropped)


_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
_MARKER_ID_TL, _MARKER_ID_TR, _MARKER_ID_BL, _MARKER_ID_BR = 0, 1, 2, 3
_REGISTRATION_MARKER_IDS = (_MARKER_ID_TL, _MARKER_ID_TR, _MARKER_ID_BL, _MARKER_ID_BR)
_WORKSHEET_RENDER_DPI = 150


def render_pdf_page_image(pdf_filename: str, dpi: int = _WORKSHEET_RENDER_DPI, page_index: int = 0) -> np.ndarray:
    """Rasterizes a single page of `pdf_filename` at `dpi` and returns it
    as an RGB numpy array."""
    with fitz.open(pdf_filename) as doc:
        page = doc[page_index]
        zoom = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        ).copy()


def _detect_marker_centers(image, source_name: str) -> Dict[int, Tuple[float, float]]:
    detector = cv2.aruco.ArucoDetector(_ARUCO_DICTIONARY, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)

    centers: Dict[int, Tuple[float, float]] = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            centers[int(marker_id)] = tuple(marker_corners[0].mean(axis=0))

    missing = [m for m in _REGISTRATION_MARKER_IDS if m not in centers]
    if missing:
        raise ValueError(
            f"Could not find registration marker(s) {missing} in {source_name}"
        )

    return centers


def align_document_image(image_filename: str, worksheet_filename: str):
    """Takes an image of a document with aruco markers and aligns it so
    that it maps onto a referene image.  A new temporary file is written
    and the filename is returned"""
    with fitz.open(worksheet_filename) as doc:
        page = doc[0]
        page_width_pt, page_height_pt = page.rect.width, page.rect.height

    reference_image = render_pdf_page_image(worksheet_filename)

    reference_centers = _detect_marker_centers(reference_image, worksheet_filename)

    photo = cv2.imread(image_filename)
    if photo is None:
        raise ValueError(f"Could not read image {image_filename}")
    photo_centers = _detect_marker_centers(photo, image_filename)

    photo_height, photo_width = photo.shape[:2]
    target_width = photo_width
    target_height = round(target_width * page_height_pt / page_width_pt)

    pixmap_height, pixmap_width = reference_image.shape[:2]
    scale_x = target_width / pixmap_width
    scale_y = target_height / pixmap_height

    src_points = np.array(
        [photo_centers[m] for m in _REGISTRATION_MARKER_IDS], dtype=np.float32
    )
    dst_points = np.array(
        [
            (reference_centers[m][0] * scale_x, reference_centers[m][1] * scale_y)
            for m in _REGISTRATION_MARKER_IDS
        ],
        dtype=np.float32,
    )

    homography, _ = cv2.findHomography(src_points, dst_points)
    aligned = cv2.warpPerspective(photo, homography, (target_width, target_height))

    suffix = os.path.splitext(image_filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        output_filename = tmp_file.name
    cv2.imwrite(output_filename, aligned)

    return output_filename