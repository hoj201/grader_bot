"""Embed handwriting crops into fixed-length vectors and maintain the vector
collection on S3 (issue #2, phase 2).

Each name-classifier training sample is one student's own handwriting of their
own name, so the per-class pattern is highly consistent -- closer to template
matching than open-ended handwriting recognition. A lightweight, in-process
embedder (`LocalEmbedder`: normalize + resize + flatten) is therefore the
default and pulls in no heavy dependencies.

The `Embedder` protocol keeps the embedding step swappable: `RemoteEmbedder`
(issue #46) POSTs crops to the Voyage multimodal-3 API instead of running a
self-hosted model, without touching ingest, training, or the collection
format.

`vectorize_samples` embeds any NAME_IMAGES crops not already vectorized
(issue #43) and uploads each vector to its own `.npy` object on S3, recording
a NAME_EMBEDDINGS row per image -- one row/object per image, matching the KNN
classifier's need for one training vector per handwriting sample.
"""

import base64
import io
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from graderbot.storage import (
    NameEmbeddingRecord,
    init_db,
    insert_name_embedding,
    list_unembedded_name_images,
    parse_s3_url,
)

load_dotenv()

_VOYAGE_MULTIMODAL_URL = "https://api.voyageai.com/v1/multimodalembeddings"
_VOYAGE_MODEL = "voyage-multimodal-3"


class Embedder(Protocol):
    """Turns handwriting crops into fixed-length feature vectors."""

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        """Embed `images` (RGB uint8 arrays) into an `(n, d)` float32 array,
        one row per image. Every image must map to the same dimension `d`."""
        ...


class LocalEmbedder:
    """Default in-process embedder: convert each crop to grayscale, invert so
    ink is high, resize to a fixed `size`, flatten, and L2-normalize. Cheap,
    deterministic, CPU-only, and sufficient given the consistent per-student
    patterns."""

    def __init__(self, size: Tuple[int, int] = (32, 128)):
        # size is (height, width); names are far wider than tall.
        self.height, self.width = size

    @property
    def dim(self) -> int:
        return self.height * self.width

    def _feature(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        ink = 255.0 - gray.astype(np.float32)  # strokes high, paper ~0
        resized = cv2.resize(ink, (self.width, self.height), interpolation=cv2.INTER_AREA)
        vector = resized.reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._feature(image) for image in images]).astype(np.float32)


class RemoteEmbedder:
    """Embeds crops via the Voyage multimodal-3 API (issue #46) instead of a
    self-hosted model, per the decision to outsource vectorization to a
    pay-as-you-go embedding API rather than deploy and maintain a torch-based
    service."""

    def __init__(self, api_key: Optional[str] = None, model: str = _VOYAGE_MODEL):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "VOYAGE_API_KEY must be set (e.g. in a .env file) to use RemoteEmbedder"
            )
        self.model = model

    @staticmethod
    def _data_uri(image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image crop")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        inputs = [
            {"content": [{"type": "image_base64", "image_base64": self._data_uri(image)}]}
            for image in images
        ]
        response = requests.post(
            _VOYAGE_MULTIMODAL_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"inputs": inputs, "model": self.model},
        )
        response.raise_for_status()
        data = response.json()["data"]
        vectors = np.stack([item["embedding"] for item in data]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=vectors, where=norms > 0)


def _resolve_bucket(bucket: Optional[str]) -> Optional[str]:
    return bucket or os.environ.get("S3_BUCKET")


def _default_client(s3_client):
    if s3_client is not None:
        return s3_client
    from graderbot import storage

    return storage._default_s3_client()


def _download_image_rgb(image_s3url: str, client) -> np.ndarray:
    bucket, key = parse_s3_url(image_s3url)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image at {image_s3url}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _embedding_key(student_id: int, name_image_id: int) -> str:
    return f"name_embeddings/{student_id}/{name_image_id}.npy"


def _download_vector(embedding_s3url: str, client) -> np.ndarray:
    bucket, key = parse_s3_url(embedding_s3url)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return np.load(io.BytesIO(body), allow_pickle=False)


def load_training_vectors(
    db_path: Path, bucket: Optional[str] = None, s3_client=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every NAME_EMBEDDINGS row's vector from S3. Returns `(vectors,
    student_ids, name_image_ids)`: `vectors` is `(n, d)` float32,
    `student_ids`/`name_image_ids` are `(n,)` int arrays. This is the KNN
    classifier's training data source (issue #43)."""
    bucket = _resolve_bucket(bucket)
    client = _default_client(s3_client)
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT student_id, name_image_id, embedding_s3url FROM NAME_EMBEDDINGS"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    vectors = np.stack([_download_vector(row[2], client) for row in rows]).astype(np.float32)
    student_ids = np.array([row[0] for row in rows], dtype=np.int64)
    name_image_ids = np.array([row[1] for row in rows], dtype=np.int64)
    return vectors, student_ids, name_image_ids


def vectorize_samples(
    db_path: Path,
    bucket: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    s3_client=None,
) -> int:
    """Embed every NAME_IMAGES crop without a NAME_EMBEDDINGS row yet, upload
    each vector to its own `.npy` object on S3, and insert a NAME_EMBEDDINGS
    row. Returns the number of newly vectorized images.

    Idempotent via NAME_EMBEDDINGS.name_image_id (unique), checked through
    `list_unembedded_name_images`. No-ops (returns 0) if no S3 bucket is
    configured.
    """
    bucket = _resolve_bucket(bucket)
    if not bucket:
        warnings.warn("vectorize_samples: no S3 bucket configured; skipping.")
        return 0

    embedder = embedder if embedder is not None else LocalEmbedder()
    client = _default_client(s3_client)

    conn = init_db(db_path)
    try:
        unembedded = list_unembedded_name_images(conn)
        if not unembedded:
            return 0

        images = [_download_image_rgb(image.image_s3url, client) for image in unembedded]
        vectors = embedder.embed(images)

        for image, vector in zip(unembedded, vectors):
            key = _embedding_key(image.student_id, image.id)
            buffer = io.BytesIO()
            np.save(buffer, vector)
            client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            insert_name_embedding(
                conn,
                NameEmbeddingRecord(
                    student_id=image.student_id,
                    name_image_id=image.id,
                    embedding_s3url=f"https://{bucket}.s3.amazonaws.com/{key}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
    finally:
        conn.close()

    return len(unembedded)
