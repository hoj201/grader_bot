"""Ingest scanned name-collection sheets into a labelled handwriting dataset
(issue #2).

A name-collection sheet (see `name_worksheets` / `tex/name_collection_template`)
prints one student's name at the top and offers a grid of blank boxes the
student copies it into. This module turns a scan of such sheets into training
data for a per-student name classifier:

  rectify each page -> OCR the printed name (the label) -> resolve it to a
  STUDENT in the target classroom by an exact, case/whitespace-insensitive
  match (issue #43), creating one if no roster entry matches -> crop each
  filled grid box -> upload the crop to S3
  (content-addressed by sha256) -> record a NAME_IMAGES row linking the
  student to the crop.

Box locations come from a one-off cv-mode render of the collection template
(`name_collection_boxes`), where the exemplar name box is exposed as
`printedname` alongside the grid boxes `name1..nameN`. Ingest self-gates on a
configured S3 bucket, mirroring `mathpix_log`.
"""

import hashlib
import logging
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import cv2
import numpy as np

from graderbot.imaging import _crop_box, is_blank, load_scan_pages
from graderbot.models import Box
from graderbot.name_worksheets import _fill_name_template
from graderbot.ocr import _BOX_INSET, _tesseract_ocr_name
from graderbot.registration import rectify_to_canonical
from graderbot.scan_grader import OnStep, _print_step
from graderbot.storage import (
    NameImageRecord,
    StudentRecord,
    get_or_create_student,
    init_db,
    insert_name_image,
    list_students,
    name_image_exists,
)
from graderbot.worksheet_boxes import extract_answer_boxes
from graderbot.worksheet_synth import latexmk_worksheet

logger = logging.getLogger(__name__)

PRINTED_NAME_BOX_ID = "printedname"


class IngestResult(NamedTuple):
    """Result of `ingest_name_sheets`: the newly inserted records plus a
    human-readable reason for every page that was skipped, so callers (the
    Roster tab in particular) can show *why* a scan yielded fewer samples than
    expected instead of a bare count (issue #51)."""

    records: List[NameImageRecord]
    skipped: List[str]


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


def _read_printed_name(crop: np.ndarray) -> str:
    """OCR the printed exemplar name from its crop. The name is set in a
    plain computer font, so Tesseract is reliable here."""
    return _tesseract_ocr_name(crop)


def _display_name(student: StudentRecord) -> str:
    return f"{student.first_name} {student.last_name}".strip()


def _normalize_name(text: str) -> str:
    return " ".join(text.split()).casefold()


def _match_student(text: str, students: List[StudentRecord]) -> Optional[StudentRecord]:
    """Matches OCR'd `text` against each student's "first last" name or
    nickname, ignoring case and whitespace differences. The printed name is
    typeset (not handwritten), so OCR is reliable enough that an exact match
    is appropriate; fuzzy matching here previously let one page's OCR result
    collapse into an unrelated, similarly-spelled earlier page's auto-created
    student (issue #53). Returns the matching `StudentRecord`, or `None` if no
    student matches."""
    if not text or not students:
        return None
    candidates = {_normalize_name(_display_name(s)): s for s in students}
    for s in students:
        if s.nickname:
            candidates[_normalize_name(s.nickname)] = s
    return candidates.get(_normalize_name(text))


def _split_name(text: str) -> tuple[str, str]:
    """Splits raw OCR text into `(first_name, last_name)` for auto-creating a
    STUDENT with no roster match: the first token is the first name, the rest
    (if any) is the last name."""
    parts = text.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def ingest_name_sheets(
    scan_path: str,
    db_path: Path,
    classroom_id: int,
    bucket: Optional[str] = None,
    boxes: Optional[Dict[str, Box]] = None,
    s3_client=None,
    on_step: OnStep = _print_step,
) -> IngestResult:
    """Ingest a scan of filled-in name-collection sheets into the handwriting
    dataset and return an `IngestResult` of the newly inserted `NameImageRecord`s
    plus a reason string for every page that was skipped.

    `scan_path` is a multi-page PDF (one sheet per page) or a single raster
    image. Each page is rectified and its printed name OCR'd, then
    exactly matched (ignoring case/whitespace) against `classroom_id`'s roster
    (issue #43); an OCR'd name
    with no roster match auto-creates a new STUDENT in that classroom. Every
    non-blank grid box is uploaded to S3 (keyed by sha256) with a NAME_IMAGES
    row linking it to the resolved student. Crops already present (same
    content hash) are skipped, so re-ingesting a scan is idempotent.

    Ingest is a no-op returning `IngestResult([], [reason])` unless an S3
    bucket is configured (via the `bucket` argument or the `S3_BUCKET` env
    var). `boxes` overrides the box layout (otherwise computed once via
    `name_collection_boxes`), and pages that lack the four registration
    markers or a readable printed name are skipped, with the reason recorded
    in `IngestResult.skipped` (issue #51) as well as emitted as a warning.

    `on_step(msg)` streams per-page progress (page loaded, matched, or
    skipped); it's also logged via the module logger regardless of `on_step`,
    so `fly logs` shows progress even if the process dies mid-scan (issue #52).
    """
    bucket = _log_bucket(bucket)
    if not bucket:
        reason = "no S3 bucket configured; skipping."
        warnings.warn(f"ingest_name_sheets: {reason}")
        return IngestResult([], [reason])

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

    def step(msg: str) -> None:
        logger.info(msg)
        on_step(msg)

    conn = init_db(db_path)
    inserted: List[NameImageRecord] = []
    skipped: List[str] = []
    try:
        pages = load_scan_pages(scan_path)
        step(f"Loaded {len(pages)} page(s); starting ingest.")
        for page_index, page in enumerate(pages):
            try:
                rectified = rectify_to_canonical(page)
            except ValueError as exc:
                reason = f"page {page_index}: {exc}"
                warnings.warn(f"Skipping {reason}")
                skipped.append(reason)
                step(f"Skipping {reason}")
                continue

            text = _read_printed_name(
                _crop_box(rectified, boxes[PRINTED_NAME_BOX_ID], _BOX_INSET)
            )
            if not text:
                reason = f"page {page_index}: could not read the printed name."
                warnings.warn(f"Skipping {reason}")
                skipped.append(reason)
                step(f"Skipping {reason}")
                continue

            students = list_students(conn, classroom_id)
            student = _match_student(text, students)
            if student is None:
                first_name, last_name = _split_name(text)
                student = get_or_create_student(conn, classroom_id, first_name, last_name)
            step(f"page {page_index}: matched '{_display_name(student)}'.")

            for box_id in grid_ids:
                crop = _crop_box(rectified, boxes[box_id], _BOX_INSET)
                if crop.size == 0 or is_blank(crop):
                    continue

                ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                if not ok:
                    reason = f"page {page_index}: could not encode box {box_id}."
                    warnings.warn(f"Skipping {reason}")
                    skipped.append(reason)
                    continue
                png_bytes = encoded.tobytes()
                sha256 = hashlib.sha256(png_bytes).hexdigest()
                if name_image_exists(conn, sha256):
                    continue

                key = f"handwriting/{classroom_id}/{student.id}/{sha256}.png"
                client.put_object(
                    Bucket=bucket, Key=key, Body=png_bytes, ContentType="image/png"
                )
                record = NameImageRecord(
                    student_id=student.id,
                    box_id=box_id,
                    image_s3url=f"https://{bucket}.s3.amazonaws.com/{key}",
                    image_sha256=sha256,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                record.id = insert_name_image(conn, record)
                inserted.append(record)
    finally:
        conn.close()

    step(f"Done: ingested {len(inserted)} sample(s), skipped {len(skipped)} page(s).")
    return IngestResult(inserted, skipped)
