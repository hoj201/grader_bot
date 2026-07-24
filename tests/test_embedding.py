"""Tests for handwriting embedding and the S3 vector collection (issue #2)."""

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from graderbot.embedding import (
    DEFAULT_COLLECTION_KEY,
    LocalEmbedder,
    RemoteEmbedder,
    load_vector_collection,
    vectorize_samples,
)
from graderbot.storage import (
    HandwritingSampleRecord,
    init_db,
    insert_handwriting_sample,
)

BUCKET = "grader-handwriting"


def _crop_with_text(text: str, size=(60, 200)) -> np.ndarray:
    height, width = size
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(img, text, (5, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def test_local_embedder_shape_and_unit_norm():
    embedder = LocalEmbedder(size=(16, 64))
    images = [_crop_with_text("a"), _crop_with_text("b")]

    vectors = embedder.embed(images)

    assert vectors.shape == (2, 16 * 64)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_local_embedder_is_deterministic_and_discriminative():
    embedder = LocalEmbedder()
    a1 = embedder.embed([_crop_with_text("Anna")])[0]
    a2 = embedder.embed([_crop_with_text("Anna")])[0]
    b = embedder.embed([_crop_with_text("Zeke")])[0]

    np.testing.assert_array_equal(a1, a2)
    # Same text embeds identically; different text is farther apart.
    assert np.linalg.norm(a1 - b) > np.linalg.norm(a1 - a2)


def test_local_embedder_handles_empty_batch():
    embedder = LocalEmbedder(size=(8, 8))
    assert embedder.embed([]).shape == (0, 64)


def test_remote_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="VOYAGE_API_KEY"):
        RemoteEmbedder()


def test_remote_embedder_handles_empty_batch(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = RemoteEmbedder()
    with patch("graderbot.embedding.requests.post") as mock_post:
        assert embedder.embed([]).shape == (0, 0)
        mock_post.assert_not_called()


def test_remote_embedder_calls_voyage_and_normalizes(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = RemoteEmbedder()
    images = [_crop_with_text("Anna"), _crop_with_text("Zeke")]

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [3.0, 4.0]}, {"embedding": [1.0, 0.0]}]
    }
    with patch("graderbot.embedding.requests.post", return_value=mock_response) as mock_post:
        vectors = embedder.embed(images)

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "voyage-multimodal-3"
    assert len(kwargs["json"]["inputs"]) == 2
    for entry in kwargs["json"]["inputs"]:
        image_b64 = entry["content"][0]["image_base64"]
        assert image_b64.startswith("data:image/png;base64,")

    mock_response.raise_for_status.assert_called_once()
    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(vectors[0], [0.6, 0.8], rtol=1e-5)


def _seed_sample(conn, s3_client, name: str, text: str) -> str:
    """Upload a crop to S3 and insert a HANDWRITING_SAMPLE row; return its sha."""
    crop = _crop_with_text(text)
    _, encoded = cv2.imencode(".png", crop)
    png = encoded.tobytes()
    sha = hashlib.sha256(png).hexdigest()
    key = f"handwriting/{name}/{sha}.png"
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=png, ContentType="image/png")
    insert_handwriting_sample(
        conn,
        HandwritingSampleRecord(
            student_name=name,
            box_id="name1",
            image_s3url=f"https://{BUCKET}.s3.amazonaws.com/{key}",
            image_sha256=sha,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return sha


@mock_aws
def test_vectorize_samples_builds_collection(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    _seed_sample(conn, s3, "Anna", "Anna")
    _seed_sample(conn, s3, "Zeke", "Zeke")
    conn.close()

    added = vectorize_samples(db_path, bucket=BUCKET, s3_client=s3)
    assert added == 2

    vectors, labels, shas = load_vector_collection(BUCKET, s3_client=s3)
    assert vectors.shape == (2, LocalEmbedder().dim)
    assert set(labels.tolist()) == {"Anna", "Zeke"}
    assert len(shas) == 2


@mock_aws
def test_vectorize_samples_appends_and_dedupes(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    _seed_sample(conn, s3, "Anna", "Anna")
    conn.close()

    assert vectorize_samples(db_path, bucket=BUCKET, s3_client=s3) == 1
    # Nothing new to embed on a second run.
    assert vectorize_samples(db_path, bucket=BUCKET, s3_client=s3) == 0

    # Add another sample; only the new one is embedded, prior vectors preserved.
    conn = init_db(db_path)
    _seed_sample(conn, s3, "Zeke", "Zeke")
    conn.close()
    assert vectorize_samples(db_path, bucket=BUCKET, s3_client=s3) == 1

    vectors, labels, _ = load_vector_collection(BUCKET, s3_client=s3)
    assert vectors.shape[0] == 2
    assert sorted(labels.tolist()) == ["Anna", "Zeke"]


def test_vectorize_samples_is_noop_without_bucket(monkeypatch, tmp_path):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    with pytest.warns(UserWarning, match="no S3 bucket"):
        assert vectorize_samples(tmp_path / "db.sqlite3") == 0


def test_load_vector_collection_missing_returns_empty():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        vectors, labels, shas = load_vector_collection(BUCKET, s3_client=s3)
    assert vectors.size == 0 and labels.size == 0 and shas.size == 0
    assert DEFAULT_COLLECTION_KEY  # sanity: constant exists
