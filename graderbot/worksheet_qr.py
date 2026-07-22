"""Worksheet identity via QR codes.

Every generated worksheet carries a stable public id, printed on the page as a
QR code. This lets a scan of student work be linked back to the worksheet it
came from (issue #11). This module handles minting the id, rendering it to a QR
PNG for embedding in the LaTeX, and decoding it back from a scanned image.

Generation uses the pure-Python ``qrcode`` package; decoding uses OpenCV's
built-in ``cv2.QRCodeDetector`` (no extra dependency).
"""

from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_H

_ID_PREFIX = "ws_"


def generate_worksheet_id() -> str:
    """Mints a short, unique, URL/filename-safe worksheet id.

    Mirrors the ``uuid4().hex[:8]`` convention already used for output-dir
    names in ``app.py``, with a ``ws_`` prefix to make the id recognizable.
    """
    return f"{_ID_PREFIX}{uuid4().hex[:8]}"


def render_qr_png(data: str, out_path: Union[str, Path]) -> Path:
    """Renders ``data`` to a QR-code PNG at ``out_path`` and returns the path.

    Uses high error correction (``ERROR_CORRECT_H``) and a quiet-zone border so
    the code stays decodable after being printed and scanned/photographed.
    """
    out_path = Path(out_path)
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    image.save(str(out_path))
    return out_path


def decode_worksheet_id(image: np.ndarray) -> Optional[str]:
    """Decodes a worksheet id from a QR code in ``image`` (a BGR or RGB numpy
    array). Returns the decoded string, or ``None`` if no QR code is found."""
    detector = cv2.QRCodeDetector()
    data, _points, _straight = detector.detectAndDecode(image)
    return data if data else None
