import os
from typing import List, Tuple

import cv2
import fitz
import numpy as np

from graderbot.models import Box

_WORKSHEET_RENDER_DPI = 150


def load_image_rgb(image_fn: str) -> np.ndarray:
    """Loads `image_fn` (a PDF or raster image) as an RGB numpy array."""
    if os.path.splitext(image_fn)[1].lower() == ".pdf":
        return render_pdf_page_image(image_fn)
    image_bgr = cv2.imread(image_fn)
    if image_bgr is None:
        raise ValueError(f"Could not read image {image_fn}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def box_pixel_rect(
    box: Box, image_width: int, image_height: int, inset: float = 0.0
) -> Tuple[int, int, int, int]:
    """Maps a `Box` (relative coords, origin bottom-left) to a pixel rectangle
    `(x0, y0, x1, y1)` on an `image_width` x `image_height` image (origin
    top-left, so the y axis is flipped). `inset` shrinks the rectangle inward by
    that fraction of the box's own width/height on each side."""
    left = box.x_lower_left + inset * box.width
    right = box.x_lower_left + box.width - inset * box.width
    top = 1 - box.y_lower_left - box.height + inset * box.height
    bottom = 1 - box.y_lower_left - inset * box.height

    x0, x1 = round(left * image_width), round(right * image_width)
    y0, y1 = round(top * image_height), round(bottom * image_height)
    return x0, y0, x1, y1


def _crop_box(image: np.ndarray, box: Box, inset: float) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x0, y0, x1, y1 = box_pixel_rect(box, image_width, image_height, inset)
    return image[y0:y1, x0:x1]


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


def load_pdf_pages_rgb(pdf_filename: str, dpi: int = _WORKSHEET_RENDER_DPI) -> List[np.ndarray]:
    """Rasterizes every page of `pdf_filename` at `dpi`, returning a list of RGB
    numpy arrays (one per page). Useful when a single uploaded PDF holds a whole
    pile of scanned worksheets, one per page."""
    with fitz.open(pdf_filename) as doc:
        page_count = doc.page_count
    return [render_pdf_page_image(pdf_filename, dpi, i) for i in range(page_count)]


def load_scan_pages(path: str, dpi: int = _WORKSHEET_RENDER_DPI) -> List[np.ndarray]:
    """Loads a scan source into a list of RGB page images: every page of a PDF,
    or a single-element list for a raster image."""
    if os.path.splitext(path)[1].lower() == ".pdf":
        return load_pdf_pages_rgb(path, dpi)
    return [load_image_rgb(path)]
