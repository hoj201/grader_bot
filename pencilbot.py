import os
import tempfile
from typing import Dict, Tuple, List, LiteralString

from dataclasses import dataclass

import cv2
import fitz
import numpy as np

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

def grade_hw_stack(worksheet_fn: LiteralString, answer_key_fn: LiteralString, hws: List[LiteralString]) -> Dict[LiteralString, Score]:
    boxes = extract_answer_boxes(worksheet_fn)
    answer_key_fn = align_document_image(answer_key_fn, worksheet_fn)
    answer_key = {qid: read_box(answer_key_fn, box) for qid, box in boxes.items()}
    scores = dict()
    for hw in hws:
        name = extract_name(hw)
        scores[name] = grade_hw(answer_key, boxes, hw)
    return scores


def extract_name(hw_fn: LiteralString) -> LiteralString:
    """Reads the name field of a worksheet using OCR"""
    raise NotImplementedError

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


def read_box(image_fn: str, box: Box) -> str:
    raise NotImplementedError


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