"""Captures low/no-confidence name-box crops seen during grading, queuing
them for a human to label in the "Label names" tab (issue #92, part 1:
turning them into (image, student_name) pairs that feed the handwriting
classifier's training data). Self-gating and non-fatal by design, mirroring
`mathpix_log.py`: grading must never fail, or even slow down noticeably,
because a labelling capture failed.
"""

import hashlib
import os
import warnings
from datetime import datetime, timezone
from sqlite3 import Connection
from typing import Optional

import cv2
import numpy as np

from graderbot.imaging import is_blank

# A page whose name confidence falls below this is worth a human's review --
# the same threshold app.py's Grade tab uses to flag a doubtful read
# (_LOW_CONFIDENCE).
LOW_CONFIDENCE_THRESHOLD = 0.5


def _log_bucket(bucket: Optional[str]) -> Optional[str]:
    return bucket or os.environ.get("S3_BUCKET")


def maybe_capture_pending_name_label(
    conn: Connection,
    crop: np.ndarray,
    classroom_id: int,
    predicted_name: str,
    confidence: float,
    source: str,
    bucket: Optional[str] = None,
    s3_client=None,
) -> Optional[int]:
    """Queues `crop` (an already-cropped RGB name box, e.g. from
    `_crop_box(image, name_box, _BOX_INSET)`) as a PENDING_NAME_LABEL row if
    `confidence` is below `LOW_CONFIDENCE_THRESHOLD`. Returns the new row
    id, or `None` for a confident read, a blank crop, a crop already queued
    or already labeled some other way, or no S3 bucket configured. Any S3/DB
    error is warned about and swallowed rather than breaking grading.
    """
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return None
    if crop.size == 0 or is_blank(crop):
        return None

    resolved_bucket = _log_bucket(bucket)
    if not resolved_bucket:
        return None

    try:
        from graderbot import storage

        ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        png_bytes = encoded.tobytes()
        image_sha256 = hashlib.sha256(png_bytes).hexdigest()

        # Already usable as training data, or already queued -- nothing new
        # to capture (e.g. the same scan graded a second time).
        if storage.name_image_exists(conn, image_sha256):
            return None
        if storage.pending_name_label_exists(conn, image_sha256):
            return None

        client = s3_client if s3_client is not None else storage._default_s3_client()
        key = f"pending_name_labels/{classroom_id}/{image_sha256}.png"
        client.put_object(
            Bucket=resolved_bucket, Key=key, Body=png_bytes, ContentType="image/png"
        )

        record = storage.PendingNameLabelRecord(
            classroom_id=classroom_id,
            image_s3url=f"https://{resolved_bucket}.s3.amazonaws.com/{key}",
            image_sha256=image_sha256,
            predicted_name=predicted_name or None,
            confidence=confidence,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return storage.insert_pending_name_label(conn, record)
    except Exception as exc:  # noqa: BLE001 - capture must never break grading
        warnings.warn(f"Failed to capture pending name label: {exc}")
        return None
