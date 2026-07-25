"""Tests for reading the student name off a scanned page (issue #58)."""

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

from graderbot import name_classifier, ocr
from graderbot.models import Box
from graderbot.name_reader import (
    CLASSIFIER_SOURCE,
    OCR_SOURCE,
    ClassifierNameReader,
    NameGuess,
    OcrNameReader,
)
from graderbot.name_classifier import classifier_key, train_name_classifier
from graderbot.storage import get_or_create_classroom, get_or_create_student, init_db

BUCKET = "grader-handwriting"

# The whole page is the name box, so a crop is just the (inset) page itself.
FULL_BOX = Box(x_lower_left=0.0, y_lower_left=0.0, width=1.0, height=1.0)


def _page(text: str = "Anna Smith") -> np.ndarray:
    img = np.full((120, 400, 3), 255, np.uint8)
    cv2.putText(img, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


class _FakeEmbedder:
    """Embeds each crop to a fixed vector, recording how many batches it saw --
    the point being that a whole worksheet group costs one call, not one per
    page (a remote embedder charges per request)."""

    def __init__(self, vectors, dim=2):
        self._vectors = vectors
        self.dim = dim
        self.calls = 0

    def embed(self, images):
        self.calls += 1
        return np.asarray(self._vectors[: len(images)], dtype=np.float32)


# --------------------------------------------------------------------------
# OcrNameReader


def test_ocr_name_reader_returns_roster_match_and_score(monkeypatch):
    # Tesseract misreads a letter, as it does on real handwriting.
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Anna Smtih")
    reader = OcrNameReader(["Anna Smith", "Zeke Jones"])

    [guess] = reader.read_many([_page()], FULL_BOX)

    assert guess.name == "Anna Smith"
    assert guess.source == OCR_SOURCE
    assert 0.0 < guess.confidence <= 1.0
    assert guess.student_id is None


def test_ocr_name_reader_reports_no_match_as_zero_confidence(monkeypatch):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "qqqqqqqq")
    reader = OcrNameReader(["Anna Smith", "Zeke Jones"])

    [guess] = reader.read_many([_page()], FULL_BOX)

    assert guess == NameGuess(name="", confidence=0.0, source=OCR_SOURCE)


def test_ocr_name_reader_scores_an_exact_read_higher_than_a_garbled_one(monkeypatch):
    reader = OcrNameReader(["Anna Smith"])
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Anna Smith")
    [exact] = reader.read_many([_page()], FULL_BOX)
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Arna Smth")
    [garbled] = reader.read_many([_page()], FULL_BOX)

    assert exact.confidence == 1.0
    assert garbled.confidence < exact.confidence


def test_ocr_name_reader_returns_one_guess_per_page(monkeypatch):
    monkeypatch.setattr(ocr, "_tesseract_ocr_name", lambda image: "Anna Smith")
    reader = OcrNameReader(["Anna Smith"])

    guesses = reader.read_many([_page(), _page(), _page()], FULL_BOX)

    assert [g.name for g in guesses] == ["Anna Smith"] * 3


# --------------------------------------------------------------------------
# ClassifierNameReader


def _fitted_classifier():
    """KNN over two well-separated 2-d clusters, labelled by student id."""
    vectors = np.array(
        [[1.0, 0.0], [0.98, 0.02], [1.02, 0.0], [0.0, 1.0], [0.02, 0.98], [0.0, 1.02]],
        dtype=np.float32,
    )
    labels = np.array([11, 11, 11, 22, 22, 22])
    return train_name_classifier(vectors, labels)


def test_classifier_name_reader_maps_ids_to_display_names():
    embedder = _FakeEmbedder([[1.0, 0.0], [0.0, 1.0]])
    reader = ClassifierNameReader(
        _fitted_classifier(), {11: "Anna Smith", 22: "Zeke Jones"}, embedder
    )

    guesses = reader.read_many([_page(), _page()], FULL_BOX)

    assert [g.name for g in guesses] == ["Anna Smith", "Zeke Jones"]
    assert [g.student_id for g in guesses] == [11, 22]
    assert all(g.source == CLASSIFIER_SOURCE for g in guesses)


def test_classifier_name_reader_embeds_the_whole_batch_in_one_call():
    embedder = _FakeEmbedder([[1.0, 0.0]] * 4)
    reader = ClassifierNameReader(_fitted_classifier(), {11: "Anna Smith"}, embedder)

    reader.read_many([_page()] * 4, FULL_BOX)

    assert embedder.calls == 1


def test_classifier_name_reader_confidence_reflects_neighbour_agreement():
    """A vector sitting in one cluster gets unanimous neighbours; one equidistant
    between clusters does not, and that is what flags a page for a hand check."""
    embedder = _FakeEmbedder([[1.0, 0.0], [0.5, 0.5]])
    reader = ClassifierNameReader(
        _fitted_classifier(), {11: "Anna Smith", 22: "Zeke Jones"}, embedder
    )

    clear, ambiguous = reader.read_many([_page(), _page()], FULL_BOX)

    assert clear.confidence == 1.0
    assert ambiguous.confidence < 1.0


def test_classifier_name_reader_falls_back_to_the_raw_id_for_unknown_students():
    """A student deleted since training still gets predicted; showing the bare
    id makes the stale model visible instead of looking like an unread name."""
    embedder = _FakeEmbedder([[0.0, 1.0]])
    reader = ClassifierNameReader(_fitted_classifier(), {11: "Anna Smith"}, embedder)

    [guess] = reader.read_many([_page()], FULL_BOX)

    assert guess.name == "22"
    assert guess.student_id == 22


def test_classifier_name_reader_handles_empty_batch():
    embedder = _FakeEmbedder([])
    reader = ClassifierNameReader(_fitted_classifier(), {}, embedder)

    assert reader.read_many([], FULL_BOX) == []
    assert embedder.calls == 0


def test_classifier_name_reader_rejects_a_mismatched_embedder():
    """The model was trained on 2-d vectors; a 4096-d embedder means someone
    changed NAME_EMBEDDER without retraining."""
    with pytest.raises(ValueError, match="trained with a different embedder"):
        ClassifierNameReader(_fitted_classifier(), {}, _FakeEmbedder([], dim=4096))


def test_classifier_name_reader_uses_predict_proba_when_absent():
    """An estimator with no predict_proba offers no basis to doubt itself, so
    nothing is flagged rather than everything."""

    class _NoProba:
        n_features_in_ = 2

        def predict(self, vectors):
            return np.array([11] * len(vectors))

    reader = ClassifierNameReader(_NoProba(), {11: "Anna Smith"}, _FakeEmbedder([[1.0, 0.0]]))

    [guess] = reader.read_many([_page()], FULL_BOX)

    assert guess.confidence == 1.0


# --------------------------------------------------------------------------
# ClassifierNameReader.from_classroom


def _seed_classroom(db_path):
    conn = init_db(db_path)
    classroom = get_or_create_classroom(conn, "Room 101")
    anna = get_or_create_student(conn, classroom.id, "Anna", "Smith")
    zeke = get_or_create_student(conn, classroom.id, "Zeke", "Jones")
    conn.close()
    return classroom, anna, zeke


@mock_aws
def test_from_classroom_loads_the_saved_model_and_roster(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom, anna, zeke = _seed_classroom(db_path)

    vectors = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]], dtype=np.float32)
    clf = train_name_classifier(vectors, np.array([anna.id, anna.id, zeke.id, zeke.id]))
    name_classifier.save_classifier(clf, BUCKET, classifier_key(classroom.id), s3_client=s3)

    reader = ClassifierNameReader.from_classroom(
        db_path, classroom.id, BUCKET, embedder=_FakeEmbedder([[1.0, 0.0]]), s3_client=s3
    )

    assert reader is not None
    assert reader.names_by_id == {anna.id: "Anna Smith", zeke.id: "Zeke Jones"}
    [guess] = reader.read_many([_page()], FULL_BOX)
    assert guess.name == "Anna Smith"


@mock_aws
def test_from_classroom_returns_none_when_untrained(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom, _, _ = _seed_classroom(db_path)

    assert (
        ClassifierNameReader.from_classroom(db_path, classroom.id, BUCKET, s3_client=s3)
        is None
    )


@mock_aws
def test_from_classroom_reads_only_its_own_classrooms_model(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    db_path = tmp_path / "db.sqlite3"
    classroom, anna, zeke = _seed_classroom(db_path)
    conn = init_db(db_path)
    other = get_or_create_classroom(conn, "Room 202")
    conn.close()

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    clf = train_name_classifier(vectors, np.array([anna.id, zeke.id]))
    name_classifier.save_classifier(clf, BUCKET, classifier_key(classroom.id), s3_client=s3)

    # Room 101 has a model; Room 202 does not, and must not borrow it.
    assert (
        ClassifierNameReader.from_classroom(
            db_path, classroom.id, BUCKET, embedder=_FakeEmbedder([[1.0, 0.0]]), s3_client=s3
        )
        is not None
    )
    assert (
        ClassifierNameReader.from_classroom(db_path, other.id, BUCKET, s3_client=s3) is None
    )
