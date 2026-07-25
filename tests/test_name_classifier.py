"""Tests for training and persisting the name classifier (issue #2)."""

import io

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from graderbot.embedding import LocalEmbedder
from graderbot.name_classifier import (
    DEFAULT_CLASSIFIER_KEY,
    classifier_key,
    load_classifier,
    logistic_regression_factory,
    loo_cross_validate,
    loo_cross_validate_augmented,
    loo_cross_validate_from_db,
    save_classifier,
    train_classroom_classifier,
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


def _name_crop(text: str, x_offset: int = 0, size=(60, 200)) -> np.ndarray:
    """A crop of handwritten-looking `text`, with `x_offset` nudging its
    position -- stands in for the natural variation between two real
    writings of the same name."""
    height, width = size
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(img, text, (5 + x_offset, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


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


def test_train_name_classifier_uses_custom_classifier_factory():
    vectors, labels = _labelled_vectors()
    clf = train_name_classifier(
        vectors, labels, classifier_factory=logistic_regression_factory(C=0.01)
    )

    assert isinstance(clf, LogisticRegression)
    anna_like = np.array([[0.99, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    zeke_like = np.array([[0, 0, 0, 0, 0, 0, 0, 0.99]], dtype=np.float32)
    assert clf.predict(anna_like)[0] == "Anna"
    assert clf.predict(zeke_like)[0] == "Zeke"


def test_logistic_regression_factory_builds_regularized_model():
    factory = logistic_regression_factory(C=0.05, max_iter=500)
    model = factory(n_samples=10)
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.05
    assert model.max_iter == 500


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


def _store_vector(conn, s3, student_id: int, vector: np.ndarray, tag: str) -> None:
    """Insert a NAME_IMAGES + NAME_EMBEDDINGS pair with `vector` on S3, exactly
    as embedding.vectorize_samples would."""
    image_id = insert_name_image(
        conn,
        NameImageRecord(
            student_id=student_id,
            box_id="name1",
            image_s3url=f"https://{BUCKET}.s3.amazonaws.com/img{tag}.png",
            image_sha256=f"sha{tag}",
        ),
    )
    key = f"name_embeddings/{student_id}/{image_id}.npy"
    buffer = io.BytesIO()
    np.save(buffer, vector)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    insert_name_embedding(
        conn,
        NameEmbeddingRecord(
            student_id=student_id,
            name_image_id=image_id,
            embedding_s3url=f"https://{BUCKET}.s3.amazonaws.com/{key}",
        ),
    )


def test_classifier_key_is_per_classroom():
    assert classifier_key(7) == "name_classifier/7.joblib"
    assert classifier_key(7) != classifier_key(8)


# An embedder whose dim matches the 8-wide vectors `_labelled_vectors` builds,
# so train_classroom_classifier's dim filter keeps them.
def _eight_dim_embedder():
    return LocalEmbedder(size=(2, 4))


@mock_aws
def test_train_classroom_classifier_saves_and_reports(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    anna = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    zeke = get_or_create_student(conn, classroom.id, "Zeke", "Jones")
    # A roster member who never handed in a name sheet, and one who handed in
    # a single sample -- the two "insufficient data" cases the UI warns about.
    get_or_create_student(conn, classroom.id, "Nora", "None")
    solo = get_or_create_student(conn, classroom.id, "Solo", "One")

    vectors, labels = _labelled_vectors()
    for i, (vector, label) in enumerate(zip(vectors, labels)):
        _store_vector(conn, s3, (anna if label == "Anna" else zeke).id, vector, str(i))
    _store_vector(conn, s3, solo.id, vectors[0], "solo")
    conn.close()

    report = train_classroom_classifier(
        db_path, BUCKET, classroom.id, embedder=_eight_dim_embedder(), s3_client=s3
    )

    assert report.n_samples == 11
    assert report.n_students == 3
    assert report.embedding_dim == 8
    assert report.discarded_wrong_dim == 0
    assert report.students_with_no_samples == ["Nora None"]
    assert report.students_with_one_sample == ["Solo One"]
    assert report.s3_url.endswith(classifier_key(classroom.id))

    # The model actually landed at the per-classroom key and predicts student ids.
    loaded = load_classifier(BUCKET, classifier_key(classroom.id), s3_client=s3)
    assert set(loaded.classes_.tolist()) == {anna.id, zeke.id, solo.id}
    assert loaded.predict(vectors[:1])[0] == anna.id


@mock_aws
def test_train_classroom_classifier_ignores_other_classrooms(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    room_a = get_or_create_classroom(conn, "Room A")
    room_b = get_or_create_classroom(conn, "Room B")
    anna = get_or_create_student(conn, room_a.id, "Anna", "Smith")
    other = get_or_create_student(conn, room_b.id, "Zeke", "Jones")
    vectors, _ = _labelled_vectors()
    _store_vector(conn, s3, anna.id, vectors[0], "a")
    _store_vector(conn, s3, other.id, vectors[-1], "b")
    conn.close()

    report = train_classroom_classifier(
        db_path, BUCKET, room_a.id, embedder=_eight_dim_embedder(), s3_client=s3
    )

    assert report.n_samples == 1
    loaded = load_classifier(BUCKET, classifier_key(room_a.id), s3_client=s3)
    assert loaded.classes_.tolist() == [anna.id]


@mock_aws
def test_train_classroom_classifier_skips_other_embedders_vectors(tmp_path):
    """Vectors left behind by a previous embedder are excluded, and counted."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    anna = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    vectors, _ = _labelled_vectors()
    _store_vector(conn, s3, anna.id, vectors[0], "good")
    _store_vector(conn, s3, anna.id, np.zeros(4096, dtype=np.float32), "stale")
    conn.close()

    report = train_classroom_classifier(
        db_path, BUCKET, classroom.id, embedder=_eight_dim_embedder(), s3_client=s3
    )

    assert report.n_samples == 1
    assert report.discarded_wrong_dim == 1


@mock_aws
def test_train_classroom_classifier_without_usable_vectors_raises(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    get_or_create_student(conn, classroom.id, "Anna", "Smith")
    conn.close()

    with pytest.raises(ValueError, match="no 8-dimensional handwriting embeddings"):
        train_classroom_classifier(
            db_path, BUCKET, classroom.id, embedder=_eight_dim_embedder(), s3_client=s3
        )


def test_loo_cross_validate_scores_well_separated_clusters_perfectly():
    vectors, labels = _labelled_vectors()
    accuracy, insufficient, confusion = loo_cross_validate(vectors, labels)

    assert accuracy == {"Anna": 1.0, "Zeke": 1.0}
    assert insufficient == []
    # Perfect separation: every held-out fold predicts its own true label.
    assert confusion == {"Anna": {"Anna": 5}, "Zeke": {"Zeke": 5}}


def test_loo_cross_validate_accepts_classifier_factory():
    vectors, labels = _labelled_vectors()
    # Note: C=0.01 here would be *too* strong for this fixture's feature
    # scale -- the (unpenalized) intercept would dominate the near-zero
    # coefficients and the model would degenerate to a single-class
    # prediction. logistic_regression_factory's own default C=1.0 avoids that.
    accuracy, insufficient, confusion = loo_cross_validate(
        vectors, labels, classifier_factory=logistic_regression_factory()
    )

    assert accuracy == {"Anna": 1.0, "Zeke": 1.0}
    assert insufficient == []
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


def test_loo_cross_validate_augmented_scores_well_separated_images_perfectly():
    embedder = LocalEmbedder()
    # Same crop repeated per student, like real per-image collection samples
    # -- an x_offset per sample would make same-student raw-pixel distance
    # exceed cross-student distance for this simplistic embedder, unrelated
    # to what augmentation is meant to test here.
    images = [_name_crop("Anna") for _ in range(4)] + [_name_crop("Zeke") for _ in range(4)]
    labels = np.array(["Anna"] * 4 + ["Zeke"] * 4)

    accuracy, insufficient, confusion = loo_cross_validate_augmented(
        images, labels, embedder, n_augmentations=2, random_state=0
    )

    assert insufficient == []
    assert accuracy == {"Anna": 1.0, "Zeke": 1.0}
    assert confusion == {"Anna": {"Anna": 4}, "Zeke": {"Zeke": 4}}


def test_loo_cross_validate_augmented_flags_singleton_class_as_insufficient():
    embedder = LocalEmbedder()
    images = [_name_crop("Anna", x) for x in (0, 2)] + [_name_crop("Zeke")]
    labels = np.array(["Anna", "Anna", "Zeke"])

    accuracy, insufficient, confusion = loo_cross_validate_augmented(
        images, labels, embedder, n_augmentations=2, random_state=0
    )

    assert insufficient == ["Zeke"]
    assert "Zeke" not in accuracy


@mock_aws
def test_loo_cross_validate_augmented_from_db(tmp_path):
    from graderbot.name_classifier import loo_cross_validate_augmented_from_db
    from graderbot.storage import NameImageRecord, get_or_create_classroom, insert_name_image

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    anna = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    zeke = get_or_create_student(conn, classroom.id, "Zeke", "Jones")

    for student, text in ((anna, "Anna"), (zeke, "Zeke")):
        for i in range(4):
            crop = _name_crop(text)
            _, encoded = cv2.imencode(".png", crop)
            png = encoded.tobytes()
            sha = f"{text}-{i}"
            key = f"handwriting/{classroom.id}/{student.id}/{sha}.png"
            s3.put_object(Bucket=BUCKET, Key=key, Body=png, ContentType="image/png")
            insert_name_image(
                conn,
                NameImageRecord(
                    student_id=student.id,
                    box_id="name1",
                    image_s3url=f"https://{BUCKET}.s3.amazonaws.com/{key}",
                    image_sha256=sha,
                ),
            )
    conn.close()

    accuracy, insufficient, confusion = loo_cross_validate_augmented_from_db(
        db_path, BUCKET, LocalEmbedder(), n_augmentations=2, random_state=0, s3_client=s3
    )

    assert insufficient == []
    assert accuracy == {anna.id: 1.0, zeke.id: 1.0}
