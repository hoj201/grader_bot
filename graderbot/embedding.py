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


def crop_to_ink(
    image: np.ndarray, ink_threshold: int = 40, pad_frac: float = 0.08
) -> np.ndarray:
    """Crop `image` to the bounding box of its ink (dark strokes), with a small
    relative padding, so where a student wrote their name inside the crop box
    stops mattering (issue #56). A near-blank crop (no pixel darker than
    `ink_threshold` below white) is returned unchanged so downstream resize
    still has something to work with.

    Registration like this matters because `LocalEmbedder`'s raw-pixel
    representation is translation-sensitive: two samples of the same name
    written at different offsets inside the box land far apart in pixel space,
    which is exactly the kind of nuisance variation that drags down KNN."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    ink = 255 - gray.astype(np.int16)
    mask = ink > ink_threshold
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(round((y1 - y0) * pad_frac))
    pad_x = int(round((x1 - x0) * pad_frac))
    h, w = gray.shape[:2]
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    return image[y0:y1, x0:x1]


class LocalEmbedder:
    """Default in-process embedder: convert each crop to grayscale, invert so
    ink is high, resize to a fixed `size`, flatten, and L2-normalize. Cheap,
    deterministic, CPU-only, and sufficient given the consistent per-student
    patterns.

    `register=True` first crops each image to its ink bounding box
    (`crop_to_ink`) so the resize normalizes for where the name sits inside
    the box -- removing translation/scale nuisance variation that hurts the
    raw-pixel representation (issue #56)."""

    def __init__(self, size: Tuple[int, int] = (32, 128), register: bool = False):
        # size is (height, width); names are far wider than tall.
        self.height, self.width = size
        self.register = register

    @property
    def dim(self) -> int:
        return self.height * self.width

    def _feature(self, image: np.ndarray) -> np.ndarray:
        if self.register:
            image = crop_to_ink(image)
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


def hog_descriptor(
    gray: np.ndarray,
    win_size: Tuple[int, int] = (128, 64),
    cell: int = 8,
    block: int = 2,
    nbins: int = 9,
) -> np.ndarray:
    """Compute a Histogram of Oriented Gradients descriptor for a grayscale
    image (issue #56). opencv 5 dropped the Python `HOGDescriptor` binding and
    the project has no scikit-image, so this is a small self-contained
    implementation: resize to `win_size` (width, height), take Sobel
    gradients, soft-bin each pixel's unsigned orientation (0-180 deg) into
    `nbins`, accumulate per `cell`x`cell` cell, then L2-normalize overlapping
    `block`x`block`-cell windows and concatenate. Returns a flat float32
    vector of fixed length."""
    w, h = win_size
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0  # unsigned orientation

    bin_width = 180.0 / nbins
    pos = angle / bin_width
    lo = np.floor(pos).astype(np.int64)
    frac = pos - lo
    lo %= nbins
    hi = (lo + 1) % nbins

    n_cells_y, n_cells_x = h // cell, w // cell
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = yy // cell, xx // cell

    hist = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
    np.add.at(hist, (cy, cx, lo), magnitude * (1.0 - frac))
    np.add.at(hist, (cy, cx, hi), magnitude * frac)

    blocks = []
    eps = 1e-6
    for by in range(n_cells_y - block + 1):
        for bx in range(n_cells_x - block + 1):
            v = hist[by : by + block, bx : bx + block, :].reshape(-1)
            v = v / np.sqrt((v * v).sum() + eps * eps)
            blocks.append(v)
    return np.concatenate(blocks).astype(np.float32)


class HOGEmbedder:
    """Embed crops via a Histogram of Oriented Gradients descriptor (issue
    #56). HOG summarizes local stroke *orientation* over a grid of cells,
    which is far more robust to the small translations, thickness changes, and
    lighting differences between scans than `LocalEmbedder`'s raw pixels --
    the intuition being that a name's shape lives in its stroke directions,
    not in exactly which pixels are inked.

    Each crop is ink-registered (`crop_to_ink`) then resized to a fixed HOG
    window before the descriptor is computed and L2-normalized. CPU-only and
    dependency-free (see `hog_descriptor`)."""

    def __init__(
        self,
        win_size: Tuple[int, int] = (128, 64),
        register: bool = True,
    ):
        # win_size is (width, height); names are far wider than tall.
        self.win_size = win_size
        self.register = register

    @property
    def dim(self) -> int:
        w, h = self.win_size
        n_cells_y, n_cells_x, block, nbins = h // 8, w // 8, 2, 9
        n_blocks = (n_cells_y - block + 1) * (n_cells_x - block + 1)
        return n_blocks * block * block * nbins

    def _feature(self, image: np.ndarray) -> np.ndarray:
        if self.register:
            image = crop_to_ink(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        vector = hog_descriptor(gray, self.win_size)
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


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 6.0,
    max_shift_frac: float = 0.06,
    max_scale_frac: float = 0.06,
) -> np.ndarray:
    """Apply a small random rotation/translation/scale to `image`, simulating
    natural handwriting variation for classifier data augmentation (issue
    #56). Border fill matches blank paper (white)."""
    height, width = image.shape[:2]
    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    scale = 1.0 + rng.uniform(-max_scale_frac, max_scale_frac)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-max_shift_frac, max_shift_frac) * width
    matrix[1, 2] += rng.uniform(-max_shift_frac, max_shift_frac) * height
    border_value = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=border_value, flags=cv2.INTER_LINEAR
    )


def augment_images(
    images: List[np.ndarray], rng: np.random.Generator, n_augmentations: int = 4, **kwargs
) -> List[np.ndarray]:
    """Generate `n_augmentations` distorted copies of each image in `images`
    (issue #56). Returns a flat list of length `len(images) * n_augmentations`."""
    return [
        augment_image(image, rng, **kwargs) for image in images for _ in range(n_augmentations)
    ]


def load_training_images(
    db_path: Path, bucket: Optional[str] = None, s3_client=None
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Load every NAME_IMAGES row's raw RGB crop from S3. Returns `(images,
    student_ids, name_image_ids)`, mirroring `load_training_vectors` but with
    undecoded images instead of pre-computed vectors -- needed to re-embed
    augmented copies on the fly (issue #56)."""
    bucket = _resolve_bucket(bucket)
    client = _default_client(s3_client)
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT student_id, id, image_s3url FROM NAME_IMAGES"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return [], np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    images = [_download_image_rgb(row[2], client) for row in rows]
    student_ids = np.array([row[0] for row in rows], dtype=np.int64)
    name_image_ids = np.array([row[1] for row in rows], dtype=np.int64)
    return images, student_ids, name_image_ids


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
