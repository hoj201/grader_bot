import os
from collections.abc import Sequence
from typing import Iterator, List, Tuple

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


class _LazyPdfPages(Sequence):
    """A `len()`-able, iterable view over a PDF's pages that rasterizes each
    page on demand rather than up front. A multi-page scan otherwise gets
    rasterized entirely into memory before the first page is even processed,
    which OOM'd fly.io's 1GB VM on a 15MB / dozens-of-page roster scan (issue
    #52); this keeps peak memory to whatever the caller holds onto per page."""

    def __init__(self, pdf_filename: str, dpi: int):
        self._pdf_filename = pdf_filename
        self._dpi = dpi
        with fitz.open(pdf_filename) as doc:
            self._count = doc.page_count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> np.ndarray:
        return render_pdf_page_image(self._pdf_filename, self._dpi, index)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self._count):
            yield self[i]


def load_pdf_pages_rgb(pdf_filename: str, dpi: int = _WORKSHEET_RENDER_DPI) -> Sequence[np.ndarray]:
    """Returns a lazy, `len()`-able sequence of RGB numpy arrays, one per page
    of `pdf_filename`, rasterized at `dpi` as each page is accessed. Useful
    when a single uploaded PDF holds a whole pile of scanned worksheets, one
    per page."""
    return _LazyPdfPages(pdf_filename, dpi)


def load_scan_pages(path: str, dpi: int = _WORKSHEET_RENDER_DPI) -> Sequence[np.ndarray]:
    """Loads a scan source into a lazy sequence of RGB page images: every page
    of a PDF (rasterized as accessed), or a single-element list for a raster
    image."""
    if os.path.splitext(path)[1].lower() == ".pdf":
        return load_pdf_pages_rgb(path, dpi)
    return [load_image_rgb(path)]
