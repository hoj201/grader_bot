"""Tests for ingesting scanned name-collection sheets into the handwriting
dataset (issue #2/#43).

Rectification and printed-name OCR are exercised elsewhere (registration / ocr),
so these tests monkeypatch them to identity/fixed values and focus on the new
logic: cropping grid boxes, skipping blank ones, uploading crops to S3 by
sha256, resolving/creating STUDENT rows, recording NAME_IMAGES rows, and
deduping on re-ingest.
"""

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from graderbot import name_dataset
from graderbot.models import Box
from graderbot.storage import get_or_create_classroom, get_or_create_student, init_db, list_name_images

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
    monkeypatch.setattr(name_dataset, "_read_printed_name", lambda crop: STUDENT)


def _classroom_id(db_path, label="Room 101") -> int:
    conn = init_db(db_path)
    try:
        return get_or_create_classroom(conn, label).id
    finally:
        conn.close()


def test_ingest_is_noop_when_no_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("S3_BUCKET", raising=False)

    with pytest.warns(UserWarning, match="no S3 bucket"):
        result = name_dataset.ingest_name_sheets("nonexistent.pdf", tmp_path / "db.sqlite3", classroom_id=1)

    assert result == []


@mock_aws
def test_ingest_uploads_crops_and_records_rows(tmp_path, _patched):
    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["name1", "name2"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom_id = _classroom_id(db_path)

    records = name_dataset.ingest_name_sheets(
        scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3
    )

    # name3 is blank, so only the two filled boxes become samples.
    assert {r.box_id for r in records} == {"name1", "name2"}
    assert len({r.student_id for r in records}) == 1

    # Each crop is retrievable from S3 under its sha256 key (student id in the path).
    for record in records:
        key = f"handwriting/{classroom_id}/{record.student_id}/{record.image_sha256}.png"
        assert record.image_s3url.endswith(key)
        s3.get_object(Bucket=BUCKET, Key=key)  # raises if missing

    conn = init_db(db_path)
    rows = list_name_images(conn)
    conn.close()
    assert {r.box_id for r in rows} == {"name1", "name2"}


@mock_aws
def test_ingest_is_idempotent_on_reingest(tmp_path, _patched):
    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["name1", "name2"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom_id = _classroom_id(db_path)

    first = name_dataset.ingest_name_sheets(scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3)
    assert len(first) == 2

    # Re-ingesting the same scan inserts nothing new (crops dedupe by sha256).
    second = name_dataset.ingest_name_sheets(scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3)
    assert second == []

    conn = init_db(db_path)
    rows = list_name_images(conn)
    conn.close()
    assert len(rows) == 2


@mock_aws
def test_ingest_skips_page_without_readable_name(tmp_path, monkeypatch):
    boxes = _boxes()
    monkeypatch.setattr(name_dataset, "rectify_to_canonical", lambda page: page)
    monkeypatch.setattr(name_dataset, "_read_printed_name", lambda crop: "")
    scan = _write_scan(_page_with_marks(["name1"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom_id = _classroom_id(db_path)

    with pytest.warns(UserWarning, match="could not read the printed name"):
        records = name_dataset.ingest_name_sheets(
            scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3
        )

    assert records == []


@mock_aws
def test_ingest_matches_existing_roster_student(tmp_path, _patched):
    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["name1"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom_id = _classroom_id(db_path)
    conn = init_db(db_path)
    existing = get_or_create_student(conn, classroom_id, "Jane", "Doe")
    conn.close()

    records = name_dataset.ingest_name_sheets(
        scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3
    )

    assert all(r.student_id == existing.id for r in records)


@mock_aws
def test_ingest_matches_nickname(tmp_path, monkeypatch):
    boxes = _boxes()
    monkeypatch.setattr(name_dataset, "rectify_to_canonical", lambda page: page)
    monkeypatch.setattr(name_dataset, "_read_printed_name", lambda crop: "Janie")
    scan = _write_scan(_page_with_marks(["name1"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom_id = _classroom_id(db_path)
    conn = init_db(db_path)
    conn.execute(
        "UPDATE STUDENT SET nickname = 'Janie' WHERE id = ?",
        (get_or_create_student(conn, classroom_id, "Jane", "Doe").id,),
    )
    conn.commit()
    n_students_before = len(conn.execute("SELECT id FROM STUDENT").fetchall())
    conn.close()

    records = name_dataset.ingest_name_sheets(
        scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3
    )

    conn = init_db(db_path)
    n_students_after = len(conn.execute("SELECT id FROM STUDENT").fetchall())
    conn.close()
    assert n_students_after == n_students_before  # no new student was created
    assert len(records) == 1


@mock_aws
def test_ingest_auto_creates_student_on_no_roster_match(tmp_path, _patched):
    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["name1"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom_id = _classroom_id(db_path)

    records = name_dataset.ingest_name_sheets(
        scan, db_path, classroom_id, bucket=BUCKET, boxes=boxes, s3_client=s3
    )
    assert len(records) == 1

    from graderbot.storage import list_students

    conn = init_db(db_path)
    students = list_students(conn, classroom_id)
    conn.close()
    assert [(s.first_name, s.last_name) for s in students] == [("Jane", "Doe")]
