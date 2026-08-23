"""Tests for harvesting HANDWRITING_LABEL rows from scanned "copy
worksheets" (issue #81).

Rectification and QR decoding are exercised elsewhere (registration), so
these monkeypatch them to identity/a fixed id and focus on the new logic:
looking the worksheet up by id, cropping each answer box, skipping blank
ones, labeling every crop with its own stored answer (not OCR), uploading
by content hash, and deduping on re-harvest -- same shape as
test_name_dataset.py's ingest_name_sheets tests.
"""

import json

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from graderbot import handwriting_harvest
from graderbot.models import Box
from graderbot.storage import (
    WorksheetRecord,
    init_db,
    insert_worksheet,
    list_handwriting_labels,
    serialize_boxes,
)

BUCKET = "grader-handwriting-labels"
WORKSHEET_ID = "ws-copy-1"


def _boxes() -> dict[str, Box]:
    return {
        "name": Box(x_lower_left=0.1, y_lower_left=0.85, width=0.5, height=0.08),
        "hw1": Box(x_lower_left=0.1, y_lower_left=0.70, width=0.3, height=0.08),
        "hw2": Box(x_lower_left=0.1, y_lower_left=0.55, width=0.3, height=0.08),
        "hw3": Box(x_lower_left=0.1, y_lower_left=0.40, width=0.3, height=0.08),
    }


def _answer_key():
    return {"hw1": "16", "hw2": "-3.5", "hw3": "7"}


def _page_with_marks(filled_ids, boxes, size=(1100, 850)) -> np.ndarray:
    height, width = size
    page = np.full((height, width, 3), 255, np.uint8)
    for box_id in filled_ids:
        box = boxes[box_id]
        cx = int((box.x_lower_left + 0.5 * box.width) * width)
        cy = int((1 - box.y_lower_left - 0.5 * box.height) * height)
        cv2.putText(page, box_id, (cx - 60, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    return page


def _write_scan(page, path) -> str:
    cv2.imwrite(str(path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
    return str(path)


def _seed_worksheet(db_path, boxes, answer_key) -> None:
    conn = init_db(db_path)
    try:
        questions_json = json.dumps([{"id": qid, "text": "", "answer": ans} for qid, ans in answer_key.items()])
        record = WorksheetRecord(
            prompt="",
            tex_source="",
            questions_json=questions_json,
            model="handwriting_sample",
            num_questions=len(answer_key),
            public_id=WORKSHEET_ID,
            boxes_json=serialize_boxes(boxes),
        )
        insert_worksheet(conn, record)
    finally:
        conn.close()


@pytest.fixture
def _patched(monkeypatch):
    monkeypatch.setattr(handwriting_harvest, "rectify_to_canonical", lambda page: page)
    monkeypatch.setattr(handwriting_harvest, "read_worksheet_id", lambda page: WORKSHEET_ID)


def test_harvest_is_noop_when_no_bucket(tmp_path):
    result = handwriting_harvest.harvest_handwriting_labels(
        "nonexistent.pdf", tmp_path / "db.sqlite3", bucket=None
    )
    assert result.inserted == []
    assert result.skipped == ["no S3 bucket configured; skipping."]


@mock_aws
def test_harvest_uploads_crops_and_records_the_stored_answer_as_the_label(tmp_path, _patched):
    boxes = _boxes()
    answer_key = _answer_key()
    scan = _write_scan(_page_with_marks(["hw1", "hw2"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    _seed_worksheet(db_path, boxes, answer_key)

    result = handwriting_harvest.harvest_handwriting_labels(
        scan, db_path, bucket=BUCKET, s3_client=s3
    )

    assert len(result.inserted) == 2
    labeled_texts = sorted(r.text for r in result.inserted)
    assert labeled_texts == ["-3.5", "16"]
    assert all(r.verified for r in result.inserted)
    assert all(r.source_mathpix_call_id is None for r in result.inserted)

    conn = init_db(db_path)
    try:
        stored = list_handwriting_labels(conn)
    finally:
        conn.close()
    assert len(stored) == 2


@mock_aws
def test_harvest_skips_blank_boxes_and_the_name_box(tmp_path, _patched):
    boxes = _boxes()
    answer_key = _answer_key()
    # Only hw1 has ink; hw2/hw3 stay blank, and "name" isn't in answer_key
    # even though it's a real box on the sheet.
    scan = _write_scan(_page_with_marks(["hw1", "name"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    _seed_worksheet(db_path, boxes, answer_key)

    result = handwriting_harvest.harvest_handwriting_labels(
        scan, db_path, bucket=BUCKET, s3_client=s3
    )

    assert len(result.inserted) == 1
    assert result.inserted[0].text == "16"


@mock_aws
def test_harvest_is_idempotent_on_re_harvest(tmp_path, _patched):
    boxes = _boxes()
    answer_key = _answer_key()
    scan = _write_scan(_page_with_marks(["hw1"], boxes), tmp_path / "scan.png")

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    _seed_worksheet(db_path, boxes, answer_key)

    first = handwriting_harvest.harvest_handwriting_labels(scan, db_path, bucket=BUCKET, s3_client=s3)
    second = handwriting_harvest.harvest_handwriting_labels(scan, db_path, bucket=BUCKET, s3_client=s3)

    assert len(first.inserted) == 1
    assert len(second.inserted) == 0

    conn = init_db(db_path)
    try:
        stored = list_handwriting_labels(conn)
    finally:
        conn.close()
    assert len(stored) == 1


@mock_aws
def test_harvest_skips_a_page_whose_worksheet_id_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(handwriting_harvest, "rectify_to_canonical", lambda page: page)
    monkeypatch.setattr(handwriting_harvest, "read_worksheet_id", lambda page: "no-such-id")

    boxes = _boxes()
    scan = _write_scan(_page_with_marks(["hw1"], boxes), tmp_path / "scan.png")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    init_db(db_path).close()

    result = handwriting_harvest.harvest_handwriting_labels(scan, db_path, bucket=BUCKET, s3_client=s3)

    assert result.inserted == []
    assert len(result.skipped) == 1
    assert "no-such-id" in result.skipped[0]


def test_harvest_skips_a_page_that_fails_to_rectify(tmp_path, monkeypatch):
    def _raise(page):
        raise ValueError("markers not found")

    monkeypatch.setattr(handwriting_harvest, "rectify_to_canonical", _raise)
    scan = _write_scan(np.full((200, 200, 3), 255, np.uint8), tmp_path / "scan.png")
    db_path = tmp_path / "db.sqlite3"
    init_db(db_path).close()

    result = handwriting_harvest.harvest_handwriting_labels(scan, db_path, bucket=BUCKET)

    assert result.inserted == []
    assert len(result.skipped) == 1
    assert "could not rectify" in result.skipped[0]
