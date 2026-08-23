"""CLI for issue #81's "dedicated labeling pass"
(`graderbot.storage.HandwritingLabelRecord`): walks unlabeled MATHPIX_CALL
rows one at a time, opens each crop (macOS `open`) for you to look at, and
asks you to confirm or correct Mathpix's own guess as the ground-truth
label. Seeding from Mathpix's existing guess means most crops are a
one-keystroke confirm, not a from-scratch transcription.

A one-off solo pass, not a repeated multi-user flow -- hence a CLI rather
than a Streamlit page (see issue #81's "Data pipeline" section).

Usage:
    poetry run python scripts/label_handwriting.py [--limit N]

Requires the same S3_BUCKET / AWS credentials / WORKSHEETS_DB_PATH env vars
app.py does (see README.md). Opens one Preview window per crop -- close each
as you go rather than letting them stack up across a long session.
"""

import argparse
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

from graderbot.storage import (
    HandwritingLabelRecord,
    handwriting_label_exists,
    init_db,
    insert_handwriting_label,
    parse_s3_url,
    unlabeled_mathpix_calls,
)

DB_PATH = Path(os.environ.get("WORKSHEETS_DB_PATH", "worksheets.sqlite3"))
BUCKET = os.environ.get("S3_BUCKET")


def _download_to_tempfile(s3_client, image_s3url: str) -> Path:
    bucket, key = parse_s3_url(image_s3url)
    suffix = Path(key).suffix or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    s3_client.download_file(bucket, key, path)
    return Path(path)


def _prompt_for_label(mathpix_guess: str) -> "str | None":
    """Returns the confirmed/corrected text, or `None` to skip this crop
    entirely (e.g. illegible, or not actually a plain numeric answer --
    `response_scorer`'s v1 scope has no use for anything else anyway)."""
    prompt = f"Mathpix read: {mathpix_guess!r}. Enter to accept, type a correction, or 's' to skip: "
    response = input(prompt).strip()
    if response.lower() == "s":
        return None
    return response if response else mathpix_guess


def main(limit: int) -> None:
    if not BUCKET:
        raise EnvironmentError("S3_BUCKET must be set (see README.md)")
    conn = init_db(DB_PATH)
    s3_client = boto3.client("s3")

    pending = unlabeled_mathpix_calls(conn, limit=limit)
    print(f"{len(pending)} unlabeled Mathpix call(s) to review.")

    for call_id, image_s3url, image_sha256, response_text in pending:
        if handwriting_label_exists(conn, image_sha256):
            # The same crop can show up under more than one MATHPIX_CALL row
            # (issue #1's dedupe-by-hash is per-call, not global) -- already
            # labeled via a different call, nothing new to review.
            continue
        local_path = _download_to_tempfile(s3_client, image_s3url)
        try:
            subprocess.run(["open", str(local_path)], check=False)
            text = _prompt_for_label(response_text or "")
            if text is None:
                print("  skipped.")
                continue
            insert_handwriting_label(
                conn,
                HandwritingLabelRecord(
                    image_s3url=image_s3url,
                    image_sha256=image_sha256,
                    text=text,
                    verified=True,
                    source_mathpix_call_id=call_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
            print(f"  labeled: {text!r}")
        finally:
            local_path.unlink(missing_ok=True)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    main(args.limit)
