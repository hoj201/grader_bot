"""Ingest scanned name-collection sheets into a labelled handwriting dataset
(issue #2).

A name-collection sheet (see `name_worksheets` / `tex/name_collection_template`)
prints one student's name at the top and offers a grid of blank boxes the
student copies it into. This module turns a scan of such sheets into training
data for a per-student name classifier:

  rectify each page -> OCR the printed name (the label) -> crop each filled grid
  box -> upload the crop to S3 (content-addressed by sha256) -> record a
  HANDWRITING_SAMPLE row linking the name to the crop.

Box locations come from a one-off cv-mode render of the collection template
(`name_collection_boxes`), where the exemplar name box is exposed as
`printedname` alongside the grid boxes `name1..nameN`. Ingest self-gates on a
configured S3 bucket, mirroring `mathpix_log`.
"""

import difflib
import hashlib
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from graderbot.imaging import _crop_box, load_scan_pages
from graderbot.models import Box
from graderbot.name_worksheets import _fill_name_template
from graderbot.ocr import _BOX_INSET, _NAME_MATCH_CUTOFF, _tesseract_ocr_name
from graderbot.registration import rectify_to_canonical
from graderbot.storage import (
    HandwritingSampleRecord,
    handwriting_sample_exists,
    init_db,
    insert_handwriting_sample,
    slugify_title,
)
from graderbot.worksheet_boxes import extract_answer_boxes
from graderbot.worksheet_synth import latexmk_worksheet

PRINTED_NAME_BOX_ID = "printedname"

# A grid box with less than this fraction of dark pixels (after the border is
# inset away by `_BOX_INSET`) is treated as blank and skipped, so unused rows
# don't become empty "samples". Tune on real scans.
_INK_THRESHOLD = 128
_BLANK_INK_FRACTION = 0.005


def _log_bucket(bucket: Optional[str]) -> Optional[str]:
    """Resolve the destination bucket: an explicit argument wins, otherwise fall
    back to the `S3_BUCKET` env var (same convention as `mathpix_log`)."""
    return bucket or os.environ.get("S3_BUCKET")


def name_collection_boxes() -> Dict[str, Box]:
    """Return the `{box id: Box}` layout of a name-collection sheet by rendering
    the template in cv mode and extracting its red boxes. Yields `printedname`
    (the exemplar box) plus one entry per grid row (`name1..nameN`). The layout
    is identical for every student sheet (no per-sheet id is embedded), so
    callers can compute it once and reuse it across a whole scan."""
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = Path(tmp) / "name_collection.tex"
        # cv mode omits the exemplar name, so the name given here is irrelevant.
        tex_path.write_text(_fill_name_template(""))
        cv_pdf = latexmk_worksheet(str(tex_path), cv_mode=True)
        return extract_answer_boxes(cv_pdf)


def _read_printed_name(crop: np.ndarray, roster: Optional[List[str]]) -> str:
    """OCR the printed exemplar name from its crop. The name is set in a plain
    computer font, so Tesseract is reliable here; when a roster is given the
    result is snapped to the closest roster spelling (as `extract_name` does)."""
    text = _tesseract_ocr_name(crop)
    if roster:
        matches = difflib.get_close_matches(text, roster, n=1, cutoff=_NAME_MATCH_CUTOFF)
        return matches[0] if matches else ""
    return text


def _ink_fraction(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(np.count_nonzero(gray < _INK_THRESHOLD)) / gray.size


def ingest_name_sheets(
    scan_path: str,
    db_path: Path,
    bucket: Optional[str] = None,
    roster: Optional[List[str]] = None,
    boxes: Optional[Dict[str, Box]] = None,
    s3_client=None,
) -> List[HandwritingSampleRecord]:
    """Ingest a scan of filled-in name-collection sheets into the handwriting
    dataset and return the newly inserted `HandwritingSampleRecord`s.

    `scan_path` is a multi-page PDF (one sheet per page) or a single raster
    image. Each page is rectified, its printed name OCR'd as the label, and
    every non-blank grid box uploaded to S3 (keyed by sha256) with a
    HANDWRITING_SAMPLE row linking it to the name. Crops already present (same
    content hash) are skipped, so re-ingesting a scan is idempotent.

    Ingest is a no-op returning `[]` unless an S3 bucket is configured (via the
    `bucket` argument or the `S3_BUCKET` env var). `boxes` overrides the box
    layout (otherwise computed once via `name_collection_boxes`), and pages that
    lack the four registration markers are skipped with a warning.
    """
    bucket = _log_bucket(bucket)
    if not bucket:
        warnings.warn("ingest_name_sheets: no S3 bucket configured; skipping.")
        return []

    if boxes is None:
        boxes = name_collection_boxes()
    if PRINTED_NAME_BOX_ID not in boxes:
        raise ValueError(
            f"box layout is missing the '{PRINTED_NAME_BOX_ID}' box; "
            "re-render the name-collection template in cv mode"
        )
    grid_ids = sorted(box_id for box_id in boxes if box_id != PRINTED_NAME_BOX_ID)

    client = s3_client if s3_client is not None else None
    if client is None:
        from graderbot import storage

        client = storage._default_s3_client()

    conn = init_db(db_path)
    inserted: List[HandwritingSampleRecord] = []
    try:
        for page_index, page in enumerate(load_scan_pages(scan_path)):
            try:
                rectified = rectify_to_canonical(page)
            except ValueError as exc:
                warnings.warn(f"Skipping page {page_index}: {exc}")
                continue

            name = _read_printed_name(
                _crop_box(rectified, boxes[PRINTED_NAME_BOX_ID], _BOX_INSET), roster
            )
            if not name:
                warnings.warn(
                    f"Skipping page {page_index}: could not read the printed name."
                )
                continue

            for box_id in grid_ids:
                crop = _crop_box(rectified, boxes[box_id], _BOX_INSET)
                if crop.size == 0 or _ink_fraction(crop) < _BLANK_INK_FRACTION:
                    continue

                ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                if not ok:
                    warnings.warn(f"Could not encode {box_id} on page {page_index}.")
                    continue
                png_bytes = encoded.tobytes()
                sha256 = hashlib.sha256(png_bytes).hexdigest()
                if handwriting_sample_exists(conn, sha256):
                    continue

                key = f"handwriting/{slugify_title(name)}/{sha256}.png"
                client.put_object(
                    Bucket=bucket, Key=key, Body=png_bytes, ContentType="image/png"
                )
                record = HandwritingSampleRecord(
                    student_name=name,
                    box_id=box_id,
                    image_s3url=f"https://{bucket}.s3.amazonaws.com/{key}",
                    image_sha256=sha256,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                record.id = insert_handwriting_sample(conn, record)
                inserted.append(record)
    finally:
        conn.close()

    return inserted
