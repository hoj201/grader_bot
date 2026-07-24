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
    loo_cross_validate,
    loo_cross_validate_from_db,
    save_classifier,
    train_from_db,
    train_name_classifier,
)
from graderbot.storage import (
    NameEmbeddingRecord,
    NameImageRecord,
    get_or_create_classroom,
    get_or_create_student,
    init_db,
    insert_name_embedding,
    insert_name_image,
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
def test_train_from_db(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    anna = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    zeke = get_or_create_student(conn, classroom.id, "Zeke", "Jones")

    # Store per-image vectors/rows the way embedding.vectorize_samples would.
    vectors, labels = _labelled_vectors()
    for i, (vector, label) in enumerate(zip(vectors, labels)):
        student = anna if label == "Anna" else zeke
        image_id = insert_name_image(
            conn,
            NameImageRecord(
                student_id=student.id,
                box_id="name1",
                image_s3url=f"https://{BUCKET}.s3.amazonaws.com/img{i}.png",
                image_sha256=f"sha{i}",
            ),
        )
        key = f"name_embeddings/{student.id}/{image_id}.npy"
        buffer = io.BytesIO()
        np.save(buffer, vector)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
        insert_name_embedding(
            conn,
            NameEmbeddingRecord(
                student_id=student.id,
                name_image_id=image_id,
                embedding_s3url=f"https://{BUCKET}.s3.amazonaws.com/{key}",
            ),
        )
    conn.close()

    clf = train_from_db(db_path, BUCKET, s3_client=s3)
    assert set(clf.classes_.tolist()) == {anna.id, zeke.id}
    assert DEFAULT_CLASSIFIER_KEY  # sanity: constant exists


def test_loo_cross_validate_scores_well_separated_clusters_perfectly():
    vectors, labels = _labelled_vectors()
    accuracy, insufficient, confusion = loo_cross_validate(vectors, labels)

    assert accuracy == {"Anna": 1.0, "Zeke": 1.0}
    assert insufficient == []
    # Perfect separation: every held-out fold predicts its own true label.
    assert confusion == {"Anna": {"Anna": 5}, "Zeke": {"Zeke": 5}}


def test_loo_cross_validate_flags_singleton_class_as_insufficient():
    vectors, labels = _labelled_vectors()
    # A third student with only one sample: too few to leave one out and
    # still have a training example of their own handwriting.
    lonely = np.array([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    vectors = np.vstack([vectors, lonely])
    labels = np.append(labels, "Lonely")

    accuracy, insufficient, confusion = loo_cross_validate(vectors, labels)

    assert insufficient == ["Lonely"]
    assert "Lonely" not in accuracy
    assert set(accuracy) == {"Anna", "Zeke"}
    assert "Lonely" not in confusion


def test_loo_cross_validate_caps_folds_per_class():
    # A single prolific student with far more samples than the fold cap:
    # only `max_folds_per_class` of their samples should be held out and
    # scored, not all of them, so evaluation cost doesn't grow with their
    # sample count.
    rng = np.random.default_rng(1)
    vectors = rng.normal(0.0, 0.01, size=(30, 8)) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
    vectors = vectors.astype(np.float32)
    labels = np.array(["Anna"] * 30)

    accuracy, insufficient, confusion = loo_cross_validate(
        vectors, labels, max_folds_per_class=5
    )

    assert insufficient == []
    assert sum(confusion["Anna"].values()) == 5


@mock_aws
def test_loo_cross_validate_from_db(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    anna = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    zeke = get_or_create_student(conn, classroom.id, "Zeke", "Jones")

    vectors, labels = _labelled_vectors()
    for i, (vector, label) in enumerate(zip(vectors, labels)):
        student = anna if label == "Anna" else zeke
        image_id = insert_name_image(
            conn,
            NameImageRecord(
                student_id=student.id,
                box_id="name1",
                image_s3url=f"https://{BUCKET}.s3.amazonaws.com/img{i}.png",
                image_sha256=f"sha{i}",
            ),
        )
        key = f"name_embeddings/{student.id}/{image_id}.npy"
        buffer = io.BytesIO()
        np.save(buffer, vector)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
        insert_name_embedding(
            conn,
            NameEmbeddingRecord(
                student_id=student.id,
                name_image_id=image_id,
                embedding_s3url=f"https://{BUCKET}.s3.amazonaws.com/{key}",
            ),
        )
    conn.close()

    accuracy, insufficient, confusion = loo_cross_validate_from_db(db_path, BUCKET, s3_client=s3)

    assert set(accuracy) == {anna.id, zeke.id}
    assert accuracy[anna.id] == 1.0
    assert accuracy[zeke.id] == 1.0
    assert insufficient == []
    assert confusion == {anna.id: {anna.id: 5}, zeke.id: {zeke.id: 5}}
