"""Tests for ingesting scanned name-collection sheets into the handwriting
dataset (issue #2).

Rectification and printed-name OCR are exercised elsewhere (registration / ocr),
so these tests monkeypatch them to identity/fixed values and focus on the new
logic: cropping grid boxes, skipping blank ones, uploading crops to S3 by
sha256, recording HANDWRITING_SAMPLE rows, and deduping on re-ingest.
"""

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from graderbot import name_dataset
from graderbot.models import Box
from graderbot.storage import init_db, list_handwriting_samples

BUCKET = "grader-handwriting"
STUDENT = "Jane Doe"


def _boxes() -> dict[str, Box]:
    """A printed-name box plus three stacked grid boxes (relative coords,
    origin bottom-left)."""
    return {
        "printedname": Box(x_lower_left=0.1, y_lower_left=0.85, width=0.5, height=0.08),
        "name1": Box(x_lower_left=0.1, y_lower_left=0.70, width=0.5, height=0.08),
        "name2": Box(x_lower_left=0.1, y_lower_left=0.55, width=0.5, height=0.08),
        "name3": Box(x_lower_left=0.1, y_lower_left=0.40, width=0.5, height=0.08),
    }


def _page_with_marks(filled_ids, boxes, size=(1100, 850)) -> np.ndarray:
    """White page with a distinct dark mark drawn in each of `filled_ids` (each
    box's own id as text, so crops differ box-to-box), leaving the rest blank."""
    height, width = size
    page = np.full((height, width, 3), 255, np.uint8)
    for box_id in filled_ids:
        box = boxes[box_id]
        cx = int((box.x_lower_left + 0.5 * box.width) * width)
        cy = int((1 - box.y_lower_left - 0.5 * box.height) * height)
        cv2.putText(page, box_id, (cx - 70, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    return page


def _write_scan(page, path) -> str:
    cv2.imwrite(str(path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    return str(path)


@pytest.fixture
def _patched(monkeypatch):
    """Bypass rectification (identity) and printed-name OCR (fixed name)."""
    monkeypatch.setattr(name_dataset, "rectify_to_canonical", lambda page: page)
    monkeypatch.setattr(name_dataset, "_read_printed_name", lambda crop, roster: STUDENT)


def test_ingest_is_noop_when_no_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("S3_BUCKET", raising=False)

    with pytest.warns(UserWarning, match="no S3 bucket"):
        result = name_dataset.ingest_name_sheets("nonexistent.pdf", tmp_path / "db.sqlite3")

    assert result == []
    assert not (tmp_path / "db.sqlite3").exists()


@mock_aws
def test_ingest_uploads_crops_and_records_rows(tmp_path, _patched):
    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["name1", "name2"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"

    records = name_dataset.ingest_name_sheets(
        scan, db_path, bucket=BUCKET, boxes=boxes, s3_client=s3
    )

    # name3 is blank, so only the two filled boxes become samples.
    assert {r.box_id for r in records} == {"name1", "name2"}
    assert all(r.student_name == STUDENT for r in records)

    # Each crop is retrievable from S3 under its sha256 key (name in the path).
    for record in records:
        key = f"handwriting/Jane_Doe/{record.image_sha256}.png"
        assert record.image_s3url.endswith(key)
        s3.get_object(Bucket=BUCKET, Key=key)  # raises if missing

    conn = init_db(db_path)
    rows = list_handwriting_samples(conn)
    conn.close()
    assert {r.box_id for r in rows} == {"name1", "name2"}


@mock_aws
def test_ingest_is_idempotent_on_reingest(tmp_path, _patched):
    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["name1", "name2"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"

    first = name_dataset.ingest_name_sheets(scan, db_path, bucket=BUCKET, boxes=boxes, s3_client=s3)
    assert len(first) == 2

    # Re-ingesting the same scan inserts nothing new (crops dedupe by sha256).
    second = name_dataset.ingest_name_sheets(scan, db_path, bucket=BUCKET, boxes=boxes, s3_client=s3)
    assert second == []

    conn = init_db(db_path)
    rows = list_handwriting_samples(conn)
    conn.close()
    assert len(rows) == 2


@mock_aws
def test_ingest_skips_page_without_readable_name(tmp_path, monkeypatch):
    boxes = _boxes()
    monkeypatch.setattr(name_dataset, "rectify_to_canonical", lambda page: page)
    monkeypatch.setattr(name_dataset, "_read_printed_name", lambda crop, roster: "")
    scan = _write_scan(_page_with_marks(["name1"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    with pytest.warns(UserWarning, match="could not read the printed name"):
        records = name_dataset.ingest_name_sheets(
            scan, tmp_path / "db.sqlite3", bucket=BUCKET, boxes=boxes, s3_client=s3
        )

    assert records == []
