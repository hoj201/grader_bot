"""Logs every Mathpix OCR call so we can compile a labelled dataset for a
future in-house OCR model (issue #1).

Each call stores the exact PNG posted to Mathpix in S3 (content-addressed by
sha256) and a MATHPIX_CALL row in the SQLite DB holding the S3 URL, the raw
Mathpix response JSON, and the parsed answer text.

Logging is opt-in and self-gating: it is a no-op unless a bucket is configured
via `MATHPIX_LOG_BUCKET` (falling back to `S3_BUCKET`). It is also non-fatal by
design -- any S3/DB error is warned about and swallowed so it can never break
OCR or grading. `storage` is imported lazily inside the function because
`storage` imports `graderbot`, and `graderbot._mathpix_ocr` calls into here.
"""

import hashlib
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DB_PATH = "worksheets.sqlite3"


def _log_bucket() -> Optional[str]:
    return os.environ.get("MATHPIX_LOG_BUCKET") or os.environ.get("S3_BUCKET")


def log_mathpix_call(
    image_png_bytes: bytes,
    response_json: dict,
    response_text: str,
    *,
    s3_client=None,
) -> Optional[int]:
    """Logs one Mathpix OCR call. Returns the new MATHPIX_CALL row id, or
    `None` if logging is disabled (no bucket configured) or failed.

    `image_png_bytes` must be the exact bytes posted to Mathpix so the stored
    image is perfectly aligned with `response_text` as a training label.
    """
    bucket = _log_bucket()
    if not bucket:
        return None

    try:
        from graderbot import storage

        image_sha256 = hashlib.sha256(image_png_bytes).hexdigest()
        key = f"mathpix/{image_sha256}.png"

        client = s3_client if s3_client is not None else storage._default_s3_client()
        client.put_object(
            Bucket=bucket, Key=key, Body=image_png_bytes, ContentType="image/png"
        )
        image_s3url = f"https://{bucket}.s3.amazonaws.com/{key}"

        db_path = Path(os.environ.get("WORKSHEETS_DB_PATH", _DEFAULT_DB_PATH))
        conn = storage.init_db(db_path)
        try:
            return storage.insert_mathpix_call(
                conn,
                image_s3url=image_s3url,
                image_sha256=image_sha256,
                response_json=json.dumps(response_json),
                response_text=response_text,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - logging must never break OCR
        warnings.warn(f"Failed to log Mathpix call: {exc}")
        return None
