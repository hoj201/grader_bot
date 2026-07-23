"""Train a per-student name classifier over handwriting vectors and persist it
to S3 (issue #2, phases 3-4).

The dataset is tiny (~10 samples per student) and each class is one student's own
handwriting of their own name, so a k-nearest-neighbours classifier over the
embedding vectors is a good fit. The trained model is serialized with joblib and
stored in S3 at a caller-chosen key so grading can fetch it later.
"""

import io
from typing import Optional

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from graderbot.embedding import DEFAULT_COLLECTION_KEY, load_vector_collection

DEFAULT_CLASSIFIER_KEY = "name_classifier/knn.joblib"
_DEFAULT_N_NEIGHBORS = 3


def train_name_classifier(
    vectors: np.ndarray, labels: np.ndarray, n_neighbors: int = _DEFAULT_N_NEIGHBORS
) -> KNeighborsClassifier:
    """Fit a KNN classifier mapping embedding vectors to student names. `k` is
    clamped to the number of samples so tiny datasets still train."""
    if len(vectors) == 0:
        raise ValueError("cannot train a classifier on an empty vector collection")
    k = min(n_neighbors, len(vectors))
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(vectors, labels)
    return classifier


def train_from_collection(
    bucket: str,
    collection_key: str = DEFAULT_COLLECTION_KEY,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    s3_client=None,
) -> KNeighborsClassifier:
    """Load the vector collection from S3 and train a classifier on it."""
    vectors, labels, _ = load_vector_collection(bucket, collection_key, s3_client)
    return train_name_classifier(vectors, labels, n_neighbors)


def _default_client(s3_client):
    if s3_client is not None:
        return s3_client
    from graderbot import storage

    return storage._default_s3_client()


def save_classifier(
    classifier: KNeighborsClassifier,
    bucket: str,
    key: str = DEFAULT_CLASSIFIER_KEY,
    s3_client=None,
) -> str:
    """Serialize `classifier` with joblib and store it in S3 at `key`. Returns
    the S3 URL. `key` is a parameter so different classifiers/versions can live
    side by side."""
    client = _default_client(s3_client)
    buffer = io.BytesIO()
    joblib.dump(classifier, buffer)
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def load_classifier(
    bucket: str, key: str = DEFAULT_CLASSIFIER_KEY, s3_client=None
) -> KNeighborsClassifier:
    """Fetch and deserialize the classifier stored at `key` in S3."""
    client = _default_client(s3_client)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return joblib.load(io.BytesIO(body))
