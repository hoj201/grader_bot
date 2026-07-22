from typing import Dict, Optional, Tuple

import cv2
import fitz
import numpy as np

from graderbot.imaging import _WORKSHEET_RENDER_DPI, render_pdf_page_image
from graderbot.worksheet_qr import decode_worksheet_id

_ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
_MARKER_ID_TL, _MARKER_ID_TR, _MARKER_ID_BL, _MARKER_ID_BR = 0, 1, 2, 3
_REGISTRATION_MARKER_IDS = (_MARKER_ID_TL, _MARKER_ID_TR, _MARKER_ID_BL, _MARKER_ID_BR)

# Canonical worksheet page geometry, mirroring gbworksheet.sty (letter paper,
# marker inset 0.15in, marker size 0.75in). Each corner marker's center sits
# inset + size/2 in from its page corner. Used to rectify a scan to a canonical
# page without a reference PDF (see rectify_to_canonical).
_PAGE_WIDTH_IN, _PAGE_HEIGHT_IN = 8.5, 11.0
_MARKER_INSET_IN = 0.15
_MARKER_SIZE_IN = 0.75


def read_worksheet_id(image: np.ndarray) -> Optional[str]:
    """Decodes the embedded worksheet id from the QR code on a scanned
    worksheet `image` (an RGB or BGR numpy array). Returns the id string, or
    `None` if no QR code is found. See issue #11 / `worksheet_qr`."""
    return decode_worksheet_id(image)


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


def align_document_image(image_filename: str, worksheet_filename: str) -> np.ndarray:
    """Takes an image of a document with aruco markers and aligns it so
    that it maps onto a reference image, returning the aligned image as
    an RGB numpy array."""
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

    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)


def _canonical_marker_centers(width_px: int, height_px: int, dpi: int) -> Dict[int, Tuple[float, float]]:
    offset = (_MARKER_INSET_IN + _MARKER_SIZE_IN / 2) * dpi
    return {
        _MARKER_ID_TL: (offset, offset),
        _MARKER_ID_TR: (width_px - offset, offset),
        _MARKER_ID_BL: (offset, height_px - offset),
        _MARKER_ID_BR: (width_px - offset, height_px - offset),
    }


def rectify_to_canonical(image: np.ndarray, dpi: int = _WORKSHEET_RENDER_DPI) -> np.ndarray:
    """Warps a scanned worksheet `image` (a numpy array with the four corner
    aruco markers) to a canonical letter-page frame, so that stored normalized
    box coordinates map directly onto it. Unlike `align_document_image`, this
    needs no reference PDF -- the destination marker centers come from the fixed
    worksheet geometry in gbworksheet.sty. Returns the rectified image in the
    same channel order it was given; exposed border areas are filled white."""
    source_centers = _detect_marker_centers(image, "scan")

    width_px = round(_PAGE_WIDTH_IN * dpi)
    height_px = round(_PAGE_HEIGHT_IN * dpi)
    canonical_centers = _canonical_marker_centers(width_px, height_px, dpi)

    src_points = np.array(
        [source_centers[m] for m in _REGISTRATION_MARKER_IDS], dtype=np.float32
    )
    dst_points = np.array(
        [canonical_centers[m] for m in _REGISTRATION_MARKER_IDS], dtype=np.float32
    )

    homography, _ = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(
        image, homography, (width_px, height_px), borderValue=(255, 255, 255)
    )
