"""Tests for DinoEmbedder (issue #46). Requires this project's own poetry
env (torch, transformers) -- run via `poetry install && poetry run pytest`
from vectorizer_service/. Not part of the main graderbot pytest run."""

import numpy as np
import pytest

from model import DinoEmbedder


def _blank_crop(size=(60, 200)) -> np.ndarray:
    height, width = size
    return np.full((height, width, 3), 255, np.uint8)


def _ink_crop(size=(60, 200)) -> np.ndarray:
    import cv2

    img = _blank_crop(size)
    cv2.putText(img, "Anna", (5, size[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


@pytest.fixture(scope="module")
def embedder():
    return DinoEmbedder()


def test_embed_output_shape(embedder):
    vectors = embedder.embed([_blank_crop(), _ink_crop()])
    assert vectors.shape == (2, embedder.dim)
    assert vectors.dtype == np.float32


def test_embed_is_deterministic(embedder):
    a1 = embedder.embed([_ink_crop()])[0]
    a2 = embedder.embed([_ink_crop()])[0]
    np.testing.assert_allclose(a1, a2, rtol=1e-5)


def test_embed_discriminates(embedder):
    blank = embedder.embed([_blank_crop()])[0]
    ink = embedder.embed([_ink_crop()])[0]
    assert np.linalg.norm(blank - ink) > 1e-3


def test_embed_handles_empty_batch(embedder):
    assert embedder.embed([]).shape == (0, embedder.dim)
