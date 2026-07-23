"""Tests for training and persisting the name classifier (issue #2)."""

import io

import boto3
import numpy as np
import pytest
from moto import mock_aws
from sklearn.neighbors import KNeighborsClassifier

from graderbot.name_classifier import (
    DEFAULT_CLASSIFIER_KEY,
    load_classifier,
    save_classifier,
    train_from_collection,
    train_name_classifier,
)

BUCKET = "grader-handwriting"


def _labelled_vectors():
    """Two well-separated clusters, five samples each, labelled by student."""
    rng = np.random.default_rng(0)
    anna = rng.normal(0.0, 0.01, size=(5, 8)) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
    zeke = rng.normal(0.0, 0.01, size=(5, 8)) + np.array([0, 0, 0, 0, 0, 0, 0, 1])
    vectors = np.vstack([anna, zeke]).astype(np.float32)
    labels = np.array(["Anna"] * 5 + ["Zeke"] * 5)
    return vectors, labels


def test_train_name_classifier_predicts_known_students():
    vectors, labels = _labelled_vectors()
    clf = train_name_classifier(vectors, labels)

    anna_like = np.array([[0.99, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    zeke_like = np.array([[0, 0, 0, 0, 0, 0, 0, 0.99]], dtype=np.float32)
    assert clf.predict(anna_like)[0] == "Anna"
    assert clf.predict(zeke_like)[0] == "Zeke"


def test_train_name_classifier_clamps_k_to_sample_count():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array(["A", "B"])
    # n_neighbors far exceeds the 2 samples; training must not raise.
    clf = train_name_classifier(vectors, labels, n_neighbors=25)
    assert clf.n_neighbors == 2


def test_train_name_classifier_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        train_name_classifier(np.empty((0, 4), dtype=np.float32), np.empty((0,)))


@mock_aws
def test_save_and_load_classifier_round_trip(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    vectors, labels = _labelled_vectors()
    clf = train_name_classifier(vectors, labels)

    key = "models/knn-v1.joblib"
    url = save_classifier(clf, BUCKET, key=key, s3_client=s3)
    assert url.endswith(key)

    loaded = load_classifier(BUCKET, key=key, s3_client=s3)
    assert isinstance(loaded, KNeighborsClassifier)
    # The reloaded model predicts identically to the original.
    np.testing.assert_array_equal(loaded.predict(vectors), clf.predict(vectors))


@mock_aws
def test_train_from_collection(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    # Store a vector collection the way embedding.vectorize_samples would.
    vectors, labels = _labelled_vectors()
    shas = np.array([f"sha{i}" for i in range(len(labels))])
    buffer = io.BytesIO()
    np.savez(buffer, vectors=vectors, labels=labels, shas=shas)
    s3.put_object(Bucket=BUCKET, Key="name_vectors/collection.npz", Body=buffer.getvalue())

    clf = train_from_collection(BUCKET, s3_client=s3)
    assert set(clf.classes_) == {"Anna", "Zeke"}
    assert DEFAULT_CLASSIFIER_KEY  # sanity: constant exists
