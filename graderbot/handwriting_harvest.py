"""Harvests trustworthy (crop, text) training pairs for the response-scorer
model (issue #81) from scans of "copy worksheets"
(`handwriting_sample_worksheets.py`): each answer box's true content is
known in advance -- it's literally what was printed for the student to copy
-- so every non-blank box crop can be recorded as a verified
`HANDWRITING_LABEL` row without ever needing OCR to guess at it. Unlike
`scripts/label_handwriting.py`'s Mathpix-seeded review pass, there is
nothing here for a human to confirm.

Reuses the same per-page rectify/decode/lookup path
`scan_grader._grade_batch` uses, but skips grading and OCR entirely -- the
box's stored `answer_key` entry *is* the label.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import cv2

from graderbot.imaging import crop_box_content_aware, is_blank, load_scan_pages
from graderbot.ocr import _BOX_INSET
from graderbot.registration import read_worksheet_id, rectify_to_canonical
from graderbot.storage import (
    HandwritingLabelRecord,
    deserialize_boxes,
    get_worksheet_by_public_id,
    handwriting_label_exists,
    init_db,
    insert_handwriting_label,
)
from graderbot.worksheetbot import OnStep, _print_step


@dataclass
class HarvestResult:
    """Outcome of `harvest_handwriting_labels`: the newly inserted
    `HandwritingLabelRecord`s plus a human-readable reason for every page
    that contributed nothing, mirroring `name_dataset.IngestResult`."""

    inserted: List[HandwritingLabelRecord] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def harvest_handwriting_labels(
    scan_path: Union[str, Path],
    db_path: Union[str, Path],
    bucket: Optional[str],
    s3_client=None,
    on_step: OnStep = _print_step,
) -> HarvestResult:
    """Harvests every non-blank answer box on the scanned "copy worksheet"
    pages in `scan_path` (a multi-page PDF or single raster image) as a
    `HANDWRITING_LABEL` row, labeled with that box's own stored answer --
    trustworthy by construction, since the worksheet printed that exact
    text for the student to copy (any other worksheet type would have an
    answer key that's the *correct answer to a problem*, not a
    known-in-advance transcription target, so running this against one
    would just harvest wrong labels for right-or-wrong-graded work; nothing
    here checks which kind of worksheet it is, so only point it at scans of
    `handwriting_sample_worksheets.py` output).

    Idempotent like `name_dataset.ingest_name_sheets`: a crop already
    stored (same content hash) is skipped rather than duplicated, so
    re-harvesting the same scan twice is safe. A no-op returning
    `HarvestResult(skipped=[reason])` if `bucket` isn't set.
    """
    if not bucket:
        reason = "no S3 bucket configured; skipping."
        return HarvestResult(skipped=[reason])

    client = s3_client
    if client is None:
        from graderbot import storage

        client = storage._default_s3_client()

    conn = init_db(Path(db_path))
    result = HarvestResult()
    try:
        pages = load_scan_pages(str(scan_path))
        on_step(f"Loaded {len(pages)} page(s).")
        for page_index, page in enumerate(pages):
            label = f"page {page_index}"
            try:
                rectified = rectify_to_canonical(page)
            except ValueError as exc:
                reason = f"{label}: could not rectify to canonical frame ({exc})."
                result.skipped.append(reason)
                on_step(reason)
                continue

            worksheet_id = read_worksheet_id(rectified)
            if not worksheet_id:
                reason = f"{label}: could not decode the worksheet QR code."
                result.skipped.append(reason)
                on_step(reason)
                continue

            record = get_worksheet_by_public_id(conn, worksheet_id)
            if record is None or not record.boxes_json or not record.questions_json:
                reason = f"{label}: worksheet id={worksheet_id} not found (or incompletely stored)."
                result.skipped.append(reason)
                on_step(reason)
                continue

            questions_data = json.loads(record.questions_json)
            answer_key = {q["id"]: q["answer"] for q in questions_data}
            boxes = deserialize_boxes(record.boxes_json)

            n_harvested = 0
            for qid, box in boxes.items():
                # Only box ids the worksheet's own answer key knows about --
                # excludes the header's "name" box, and anything open-ended
                # (store_worksheet never puts those in the answer key either).
                if qid not in answer_key:
                    continue
                crop = crop_box_content_aware(rectified, box, fallback_inset=_BOX_INSET)
                if crop.size == 0 or is_blank(crop):
                    continue

                ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                if not ok:
                    continue
                png_bytes = encoded.tobytes()
                sha256 = hashlib.sha256(png_bytes).hexdigest()
                if handwriting_label_exists(conn, sha256):
                    continue

                key = f"handwriting_labels/{worksheet_id}/{qid}/{sha256}.png"
                client.put_object(Bucket=bucket, Key=key, Body=png_bytes, ContentType="image/png")
                hw_record = HandwritingLabelRecord(
                    image_s3url=f"https://{bucket}.s3.amazonaws.com/{key}",
                    image_sha256=sha256,
                    text=answer_key[qid],
                    verified=True,  # ground truth by construction -- see docstring
                    source_mathpix_call_id=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                hw_record.id = insert_handwriting_label(conn, hw_record)
                result.inserted.append(hw_record)
                n_harvested += 1
            on_step(f"{label}: worksheet id={worksheet_id}, harvested {n_harvested} sample(s).")
    finally:
        conn.close()

    on_step(f"Done: harvested {len(result.inserted)} sample(s), skipped {len(result.skipped)} page(s).")
    return result
