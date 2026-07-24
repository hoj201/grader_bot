"""Train a per-student name classifier over handwriting vectors and persist it
to S3 (issue #2, phases 3-4).

The dataset is tiny (~10 samples per student) and each class is one student's own
handwriting of their own name, so a k-nearest-neighbours classifier over the
embedding vectors is a good fit. The trained model is serialized with joblib and
stored in S3 at a caller-chosen key so grading can fetch it later.
"""

import io
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from graderbot.embedding import load_training_vectors

DEFAULT_CLASSIFIER_KEY = "name_classifier/knn.joblib"
_DEFAULT_N_NEIGHBORS = 3


def train_name_classifier(
    vectors: np.ndarray, labels: np.ndarray, n_neighbors: int = _DEFAULT_N_NEIGHBORS
) -> KNeighborsClassifier:
    """Fit a KNN classifier mapping embedding vectors to student ids. `k` is
    clamped to the number of samples so tiny datasets still train."""
    if len(vectors) == 0:
        raise ValueError("cannot train a classifier on an empty vector collection")
    k = min(n_neighbors, len(vectors))
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(vectors, labels)
    return classifier


def train_from_db(
    db_path: Union[str, Path],
    bucket: str,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    s3_client=None,
) -> KNeighborsClassifier:
    """Load per-image embeddings from NAME_EMBEDDINGS (issue #43) and train a
    classifier keyed by student_id."""
    vectors, student_ids, _ = load_training_vectors(Path(db_path), bucket, s3_client)
    return train_name_classifier(vectors, student_ids, n_neighbors)


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
