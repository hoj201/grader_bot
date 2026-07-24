"""Embed handwriting crops into fixed-length vectors and maintain the vector
collection on S3 (issue #2, phase 2).

Each name-classifier training sample is one student's own handwriting of their
own name, so the per-class pattern is highly consistent -- closer to template
matching than open-ended handwriting recognition. A lightweight, in-process
embedder (`LocalEmbedder`: normalize + resize + flatten) is therefore the
default and pulls in no heavy dependencies.

The `Embedder` protocol keeps the embedding step swappable: `RemoteEmbedder`
POSTs crops to the separately deployed `vectorizer_service` (DINOv2, issue
#46) without touching ingest, training, or the collection format. Switching
between embedders changes the vector dimension, so the S3 collection must be
rebuilt from scratch (not incrementally appended) whenever the embedder
changes.

`vectorize_samples` embeds any HANDWRITING_SAMPLE crops not already vectorized
and appends them to a single `.npz` collection on S3 (vectors + labels + the
source sha256s, which key the append so re-running is idempotent).
"""

import base64
import io
import os
import warnings
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from graderbot.storage import init_db, list_handwriting_samples, parse_s3_url

load_dotenv()

DEFAULT_COLLECTION_KEY = "name_vectors/collection.npz"


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


def _encode_png_b64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Could not encode crop image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class RemoteEmbedder:
    """Embedder backed by `vectorizer_service`, a separately deployed DINOv2
    service (issue #46). Keeps torch and other heavy ML dependencies out of
    the main graderbot app/deploy; see vectorizer_service/README.md."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or os.environ.get("VECTORIZER_SERVICE_URL")
        self.api_key = api_key or os.environ.get("VECTORIZER_API_KEY")
        if not self.url or not self.api_key:
            raise EnvironmentError(
                "VECTORIZER_SERVICE_URL and VECTORIZER_API_KEY must be set "
                "(e.g. in a .env file) to use RemoteEmbedder"
            )

    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        response = requests.post(
            self.url,
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            json={"images": [_encode_png_b64(image) for image in images]},
        )
        response.raise_for_status()
        return np.array(response.json()["vectors"], dtype=np.float32)


def _resolve_bucket(bucket: Optional[str]) -> Optional[str]:
    return bucket or os.environ.get("S3_BUCKET")


def _default_client(s3_client):
    if s3_client is not None:
        return s3_client
    from graderbot import storage

    return storage._default_s3_client()


def load_vector_collection(
    bucket: str, key: str = DEFAULT_COLLECTION_KEY, s3_client=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the `(vectors, labels, shas)` collection stored at `key`, or three
    empty arrays if it does not exist yet. `vectors` is `(n, d)` float32;
    `labels` and `shas` are `(n,)` unicode arrays."""
    client = _default_client(s3_client)
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 - a missing collection is the first-run case
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype="<U1"),
            np.empty((0,), dtype="<U64"),
        )
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        return data["vectors"], data["labels"], data["shas"]


def _download_image_rgb(image_s3url: str, client) -> np.ndarray:
    bucket, key = parse_s3_url(image_s3url)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image at {image_s3url}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def vectorize_samples(
    db_path: Path,
    bucket: Optional[str] = None,
    embedder: Optional[Embedder] = None,
    key: str = DEFAULT_COLLECTION_KEY,
    s3_client=None,
) -> int:
    """Embed every HANDWRITING_SAMPLE crop not already in the collection and
    append them to the `.npz` collection at `key` on S3. Returns the number of
    newly vectorized samples.

    Existing vectors are preserved and samples are keyed by their source
    sha256, so re-running only embeds new crops (idempotent). No-ops (returns
    0) if no S3 bucket is configured.
    """
    bucket = _resolve_bucket(bucket)
    if not bucket:
        warnings.warn("vectorize_samples: no S3 bucket configured; skipping.")
        return 0

    embedder = embedder if embedder is not None else LocalEmbedder()
    client = _default_client(s3_client)

    vectors, labels, shas = load_vector_collection(bucket, key, client)
    known = set(shas.tolist())

    conn = init_db(db_path)
    try:
        samples = list_handwriting_samples(conn)
    finally:
        conn.close()

    new_images, new_labels, new_shas = [], [], []
    for sample in samples:
        if sample.image_sha256 in known:
            continue
        new_images.append(_download_image_rgb(sample.image_s3url, client))
        new_labels.append(sample.student_name)
        new_shas.append(sample.image_sha256)
        known.add(sample.image_sha256)

    if not new_images:
        return 0

    new_vectors = embedder.embed(new_images)
    vectors = new_vectors if vectors.size == 0 else np.vstack([vectors, new_vectors])
    labels = np.concatenate([labels, np.array(new_labels)])
    shas = np.concatenate([shas, np.array(new_shas)])

    buffer = io.BytesIO()
    np.savez(buffer, vectors=vectors, labels=labels, shas=shas)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

    return len(new_images)
