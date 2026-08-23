import os
from collections.abc import Sequence
from typing import Iterator, List, Tuple

import cv2
import fitz
import numpy as np

from graderbot.models import Box

_WORKSHEET_RENDER_DPI = 150

# Guards against a PDF whose page geometry is wrong -- e.g. a scanning app
# that stuffs a photo's raw pixel dimensions into the page's point-based
# MediaBox instead of a true physical size (a 4284x5712px photo becomes a
# "59.5in x 79.3in" page). Rendering that at our normal dpi blows up to a
# multi-hundred-MB raster per page and OOM'd fly.io (issue #52). Our own
# generated worksheets are always near Letter size, so this never engages
# for them; it only clamps runaway external scans.
_MAX_RENDER_DIMENSION_PX = 2200


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


# Pixel margin (not a fraction of the box's own size, unlike `_BOX_INSET` in
# ocr.py) used only to skip past the printed border stroke before looking for
# ink. The border renders at a roughly constant few-pixel width regardless of
# how big the box itself is (issue #70 follow-up), so a *fractional* inset
# meant to clear it ends up far larger than necessary on a wide/short answer
# box -- e.g. 8% of a 180px-wide box is ~14px, enough to slice off a digit
# written flush against the left edge, while 8% of the box's own ~60px height
# is only ~5px, barely past the border. `crop_box_content_aware` below uses
# this small, size-independent margin instead. Measured directly off a real
# scan's rectified box borders: the top/bottom rule is a solid ~2px line
# (dark-pixel row fraction ~1.0 for 1-2 rows, then 0); the left/right rule is
# much fainter (row/col dark fraction ~0.05 throughout, likely rendering
# antialiasing rather than a real stroke). 3px clears the solid case with a
# 1px buffer, without re-introducing the old fractional inset's problem of
# eating real ink that sits close to an edge.
_BORDER_MARGIN_PX = 3
# Padding re-added around the detected ink's own bounding box, so a crop
# isn't shrink-wrapped pixel-tight to the ink (which starves OCR of the
# surrounding context it expects).
_CONTENT_PAD_PX = 6


def crop_box_content_aware(image: np.ndarray, box: Box, fallback_inset: float) -> np.ndarray:
    """Crops `box` on `image` to the actual ink inside it, instead of a fixed
    fraction of the box's own size (`_crop_box`'s `inset`). A fixed-fraction
    inset clips real handwriting whenever it's written close to a box edge --
    on a wide box this can slice off an entire leading digit (issue #70
    follow-up). This crops past the border by a small constant pixel margin
    (`_BORDER_MARGIN_PX`), finds the bounding box of ink (pixels darker than
    `_INK_THRESHOLD`) inside that, and returns that bounding box plus
    `_CONTENT_PAD_PX` padding, clamped so the border can't creep back in.

    Falls back to the legacy `_crop_box(image, box, fallback_inset)` when no
    ink is found (or the box is too small for the border margin) -- callers
    that check `is_blank`/`ink_fraction` on the result rely on that shape.
    """
    image_height, image_width = image.shape[:2]
    x0, y0, x1, y1 = box_pixel_rect(box, image_width, image_height, inset=0.0)
    bx0, by0 = x0 + _BORDER_MARGIN_PX, y0 + _BORDER_MARGIN_PX
    bx1, by1 = x1 - _BORDER_MARGIN_PX, y1 - _BORDER_MARGIN_PX
    if bx1 <= bx0 or by1 <= by0:
        return _crop_box(image, box, fallback_inset)

    interior = image[by0:by1, bx0:bx1]
    gray = cv2.cvtColor(interior, cv2.COLOR_RGB2GRAY)
    mask = gray < _INK_THRESHOLD
    if not mask.any():
        return _crop_box(image, box, fallback_inset)

    ys, xs = np.where(mask)
    h, w = interior.shape[:2]
    iy0, iy1 = max(0, ys.min() - _CONTENT_PAD_PX), min(h, ys.max() + 1 + _CONTENT_PAD_PX)
    ix0, ix1 = max(0, xs.min() - _CONTENT_PAD_PX), min(w, xs.max() + 1 + _CONTENT_PAD_PX)
    return interior[iy0:iy1, ix0:ix1]


# A crop with less than this fraction of dark pixels is treated as blank --
# i.e. nothing was written in it. Originally tuned for name-collection grid
# boxes (name_dataset.ingest_name_sheets); generalized here into the shared
# preliminary check every box-reading caller (grading, name reading) runs
# before its expensive step (Mathpix/Tesseract/embed+predict), so a blank box
# skips that work entirely (issue #66, #62).
#
# _INK_THRESHOLD used to be 128 (roughly "as dark as ink"), but real scanned
# pencil answers rarely get that dark -- graphite scans as mid-gray, not
# near-black. issue #74: three genuine answers, written in faint pencil, had
# almost none of their pixels below 128 (as little as 0.01% of the crop) even
# though a human reads them easily, so they read as blank and skipped Mathpix
# entirely. 200 still leaves a wide margin above scan-noise/shadow gray levels
# (measured at 0.0 for genuinely blank boxes in the same scan, even at this
# looser cutoff) while catching the lighter end of real handwriting.
_INK_THRESHOLD = 200
_BLANK_INK_FRACTION = 0.005


def ink_fraction(image: np.ndarray) -> float:
    """Fraction of `image`'s pixels darker than `_INK_THRESHOLD` (after
    converting to grayscale) -- a cheap proxy for how much was written on
    this crop, with no OCR/embedding call involved. `image` must be
    non-empty; a caller with a possibly zero-size crop (`_crop_box` can
    return one for a tiny/edge-case box) must check `image.size > 0` itself
    first, same as `is_blank` below."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.count_nonzero(gray < _INK_THRESHOLD)) / gray.size


def is_blank(image: np.ndarray, threshold: float = _BLANK_INK_FRACTION) -> bool:
    """True if `image` has too little ink to have anything meaningfully
    written on it. Same non-empty-input requirement as `ink_fraction`."""
    return ink_fraction(image) < threshold


def render_pdf_page_image(pdf_filename: str, dpi: int = _WORKSHEET_RENDER_DPI, page_index: int = 0) -> np.ndarray:
    """Rasterizes a single page of `pdf_filename` at `dpi` and returns it
    as an RGB numpy array. The effective dpi is clamped so neither side of
    the output exceeds `_MAX_RENDER_DIMENSION_PX`, in case the page's declared
    size is much larger than its real physical size."""
    with fitz.open(pdf_filename) as doc:
        page = doc[page_index]
        zoom = dpi / 72
        longer_side_pt = max(page.rect.width, page.rect.height)
        if longer_side_pt * zoom > _MAX_RENDER_DIMENSION_PX:
            zoom = _MAX_RENDER_DIMENSION_PX / longer_side_pt
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
