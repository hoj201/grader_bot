from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import numpy as np
import pytest
from moto import mock_aws

from graderbot.pending_name_capture import (
    LOW_CONFIDENCE_THRESHOLD,
    MIN_NAME_IMAGES_PER_STUDENT,
    maybe_capture_pending_name_label,
)
from graderbot.storage import (
    NameImageRecord,
    count_all_pending_name_labels,
    get_or_create_classroom,
    get_or_create_student,
    init_db,
    insert_name_image,
    parse_s3_url,
    pending_name_label_exists,
    random_pending_name_label_any,
)

_BUCKET = "graderbot-test-bucket"


def _blank_crop() -> np.ndarray:
    return np.full((40, 40, 3), 255, dtype=np.uint8)


def _written_crop() -> np.ndarray:
    # A mostly-white crop with a solid dark patch -- comfortably above
    # is_blank's ink-fraction threshold.
    crop = np.full((40, 40, 3), 255, dtype=np.uint8)
    crop[10:30, 10:30] = 0
    return crop


@pytest.fixture
def s3_client():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def test_maybe_capture_skips_a_confident_read(tmp_path, s3_client, monkeypatch):
    monkeypatch.setenv("WORKSHEETS_DB_PATH", str(tmp_path / "worksheets.sqlite3"))
    conn = init_db(tmp_path / "worksheets.sqlite3")

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "Alice Smith",
        LOW_CONFIDENCE_THRESHOLD, "ocr", bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is None
    assert count_all_pending_name_labels(conn) == 0


def test_maybe_capture_skips_a_blank_crop(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    result = maybe_capture_pending_name_label(
        conn, _blank_crop(), "", 0.0, "ocr",
        bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is None
    assert count_all_pending_name_labels(conn) == 0


def test_maybe_capture_skips_when_no_bucket_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    conn = init_db(tmp_path / "worksheets.sqlite3")

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "Alice Smith", 0.3, "ocr",
    )

    assert result is None
    assert count_all_pending_name_labels(conn) == 0


def test_maybe_capture_queues_a_low_confidence_crop(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "Alice Smith", 0.33, "classifier",
        bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is not None
    assert count_all_pending_name_labels(conn) == 1
    pending = random_pending_name_label_any(conn)
    assert pending.predicted_name == "Alice Smith"
    assert pending.confidence == pytest.approx(0.33)
    assert pending.source == "classifier"
    assert pending.classroom_id is None  # no classroom to tag it with (issue #109)

    bucket, key = parse_s3_url(pending.image_s3url)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    assert obj["Body"].read() != b""


def test_maybe_capture_treats_no_name_the_same_as_low_confidence(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "", 0.0, "ocr",
        bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is not None
    pending = random_pending_name_label_any(conn)
    assert pending.predicted_name is None  # "" is normalized to None


def test_maybe_capture_is_idempotent_for_the_same_crop(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    maybe_capture_pending_name_label(
        conn, _written_crop(), "Alice Smith", 0.33, "classifier",
        bucket=_BUCKET, s3_client=s3_client,
    )
    second = maybe_capture_pending_name_label(
        conn, _written_crop(), "Alice Smith", 0.33, "classifier",
        bucket=_BUCKET, s3_client=s3_client,
    )

    assert second is None
    assert count_all_pending_name_labels(conn) == 1


def test_maybe_capture_skips_a_crop_already_labeled_as_a_name_image(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    student = get_or_create_student(conn, classroom.id, "Anna", "Smith")

    # Compute the same content hash maybe_capture_pending_name_label would,
    # by capturing once, reading its hash, then discarding the pending row
    # and pre-labeling that same crop as a NAME_IMAGES row instead.
    first = maybe_capture_pending_name_label(
        conn, _written_crop(), "Anna Smith", 0.33, "classifier",
        bucket=_BUCKET, s3_client=s3_client,
    )
    pending = random_pending_name_label_any(conn)
    assert pending.id == first
    insert_name_image(
        conn,
        NameImageRecord(
            student_id=student.id,
            box_id="name",
            image_s3url=pending.image_s3url,
            image_sha256=pending.image_sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.execute("DELETE FROM PENDING_NAME_LABEL WHERE id = ?", (pending.id,))
    conn.commit()

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "Anna Smith", 0.33, "classifier",
        bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is None
    assert count_all_pending_name_labels(conn) == 0
    assert not pending_name_label_exists(conn, pending.image_sha256)


def test_maybe_capture_ignores_confidence_when_a_student_is_below_quota(tmp_path, s3_client):
    # issue #110: a classifier that has never seen "Ben Jones" can still
    # confidently misread his crop as someone it does know, so while any
    # student is below the quota, confidence must not gate capture.
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    get_or_create_student(conn, classroom.id, "Ben", "Jones")

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "Alice Smith",
        LOW_CONFIDENCE_THRESHOLD, "classifier", bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is not None
    assert count_all_pending_name_labels(conn) == 1


def test_maybe_capture_resumes_confidence_gating_once_quota_is_met(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    classroom = get_or_create_classroom(conn, "Room 101")
    student = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    for i in range(MIN_NAME_IMAGES_PER_STUDENT):
        insert_name_image(
            conn,
            NameImageRecord(
                student_id=student.id,
                box_id=f"name{i}",
                image_s3url=f"https://bucket.s3.amazonaws.com/img{i}.png",
                image_sha256=f"sha-{i}",
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    result = maybe_capture_pending_name_label(
        conn, _written_crop(), "Anna Smith",
        LOW_CONFIDENCE_THRESHOLD, "classifier", bucket=_BUCKET, s3_client=s3_client,
    )

    assert result is None
    assert count_all_pending_name_labels(conn) == 0


def test_maybe_capture_swallows_errors_and_warns(tmp_path, s3_client):
    conn = init_db(tmp_path / "worksheets.sqlite3")

    with patch("graderbot.storage._default_s3_client", side_effect=RuntimeError("boom")):
        with pytest.warns(UserWarning, match="Failed to capture pending name label"):
            result = maybe_capture_pending_name_label(
                conn, _written_crop(), "Alice Smith", 0.33, "ocr",
                bucket=_BUCKET,  # s3_client omitted, forces the failing default lookup
            )

    assert result is None
    assert count_all_pending_name_labels(conn) == 0
