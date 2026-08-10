"""Read the student name off a scanned worksheet's name box (issue #58,
phase 5 of issue #2).

Two interchangeable strategies sit behind one `NameReader` protocol:

- `OcrNameReader` -- Tesseract plus a fuzzy match against the roster. This is
  what grading has always done, and remains the better choice when students
  print their names legibly (or their initials in large caps).
- `ClassifierNameReader` -- embeds the crop and asks the per-classroom
  handwriting classifier trained by `name_classifier.train_classroom_classifier`.
  This is the point of the whole embedding/training pipeline: identify cursive
  a student writes the same way every week, which OCR reads badly.

Which one grading uses is a runtime choice made in the Grade tab, because real
per-roster accuracy is not known ahead of time.

Both readers take a whole batch of pages and one shared name `Box`, so a remote
embedder embeds every page of a worksheet group in a single API call rather than
one call per page. Both crop with `_crop_box(..., _BOX_INSET)` -- the same crop
`name_dataset` takes when building the training set, and the collection-sheet
boxes are deliberately the same size as the header name box, so a crop taken
here is geometrically comparable to the crops the classifier was trained on.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import numpy as np

from graderbot.embedding import Embedder, default_embedder
from graderbot.imaging import _crop_box, is_blank
from graderbot.models import Box
from graderbot.ocr import _BOX_INSET, extract_name_scored
from graderbot.storage import init_db, list_students

logger = logging.getLogger(__name__)

CLASSIFIER_SOURCE = "classifier"
OCR_SOURCE = "ocr"


@dataclass
class NameGuess:
    """One page's identification of a student.

    `confidence` is only comparable within a `source`: for OCR it's a difflib
    similarity, for the classifier it's the estimator's own class probability
    (with KNN k=3 that's a coarse 0/⅓/⅔/1). It is meant for spotting the weak
    reads on a page-by-page table, not as a calibrated probability.
    """

    name: str  # "" when nothing matched
    confidence: float
    source: str
    student_id: Optional[int] = None


class NameReader(Protocol):
    """Identifies the student who wrote on each of a batch of rectified pages."""

    def read_many(self, images: List[np.ndarray], box: Box) -> List[NameGuess]:
        """Read the name inside `box` on every image, returning one `NameGuess`
        per image in the same order."""
        ...


class OcrNameReader:
    """Tesseract + fuzzy roster match -- the pre-issue-#58 behavior, kept as a
    first-class choice rather than a fallback."""

    def __init__(self, roster: List[str]):
        self.roster = roster

    def read_many(self, images: List[np.ndarray], box: Box) -> List[NameGuess]:
        guesses = []
        for image in images:
            crop = _crop_box(image, box, _BOX_INSET)
            if crop.size > 0 and is_blank(crop):
                guesses.append(NameGuess(name="", confidence=0.0, source=OCR_SOURCE))
                continue
            name, score = extract_name_scored(image, box, self.roster)
            guesses.append(NameGuess(name=name, confidence=score, source=OCR_SOURCE))
        return guesses


class ClassifierNameReader:
    """Predicts the student from the handwriting itself, using a classroom's
    trained classifier over embedded name crops."""

    def __init__(
        self,
        classifier,
        names_by_id: Dict[int, str],
        embedder: Optional[Embedder] = None,
    ):
        self.classifier = classifier
        self.names_by_id = names_by_id
        self.embedder = embedder if embedder is not None else default_embedder()
        expected = getattr(classifier, "n_features_in_", None)
        if expected is not None and expected != self.embedder.dim:
            raise ValueError(
                f"classifier expects {expected}-dimensional vectors but the "
                f"configured embedder produces {self.embedder.dim}; the model was "
                "trained with a different embedder than NAME_EMBEDDER now selects "
                "-- retrain it, or set NAME_EMBEDDER back"
            )

    @classmethod
    def from_classroom(
        cls,
        db_path: Union[str, Path],
        classroom_id: int,
        bucket: str,
        embedder: Optional[Embedder] = None,
        s3_client=None,
    ) -> Optional["ClassifierNameReader"]:
        """Build a reader from the classroom's saved model, or return `None` if
        no model has been trained for it yet. Raises `ValueError` if a model
        exists but was trained with a different embedder."""
        # Imported here rather than at module scope: name_classifier pulls in
        # sklearn/joblib, which grading shouldn't pay for on the OCR path.
        from graderbot import name_classifier

        key = name_classifier.classifier_key(classroom_id)
        try:
            classifier = name_classifier.load_classifier(bucket, key, s3_client)
        except Exception:
            # boto3 raises a client-specific NoSuchKey/ClientError here; any
            # failure to fetch means "no usable model", which the caller
            # reports rather than silently grading with the wrong strategy.
            # Logged because a credentials/network failure looks identical to
            # an untrained classroom from the outside.
            logger.warning("could not load classifier %s", key, exc_info=True)
            return None

        conn = init_db(Path(db_path))
        try:
            students = list_students(conn, classroom_id)
        finally:
            conn.close()
        names_by_id = {s.id: f"{s.first_name} {s.last_name}".strip() for s in students}
        return cls(classifier, names_by_id, embedder)

    def read_many(self, images: List[np.ndarray], box: Box) -> List[NameGuess]:
        if not images:
            return []
        crops = [_crop_box(image, box, _BOX_INSET) for image in images]

        # A blank name box never reaches the embedder/classifier (issue #66)
        # -- without this check the classifier has no "blank" class to fall
        # back on and confidently assigns it to whichever student is nearest
        # in embedding space (issue #62). Only the non-blank crops are
        # embedded, still in one batched call so a remote embedder is billed
        # once per group rather than once per page.
        non_blank_indices = [
            i for i, crop in enumerate(crops) if not (crop.size > 0 and is_blank(crop))
        ]
        guesses = [
            NameGuess(name="", confidence=0.0, source=CLASSIFIER_SOURCE) for _ in images
        ]
        if not non_blank_indices:
            return guesses

        vectors = self.embedder.embed([crops[i] for i in non_blank_indices])
        predictions = self.classifier.predict(vectors)
        confidences = self._confidences(vectors, predictions)
        for i, student_id, confidence in zip(non_blank_indices, predictions, confidences):
            guesses[i] = NameGuess(
                # A predicted id missing from the roster means the model is
                # stale (student deleted since training); show the raw id
                # rather than a blank, so the mismatch is visible on the page
                # table instead of looking like an unreadable name.
                name=self.names_by_id.get(int(student_id), str(student_id)),
                confidence=confidence,
                source=CLASSIFIER_SOURCE,
                student_id=int(student_id),
            )
        return guesses

    def _confidences(self, vectors: np.ndarray, predictions: np.ndarray) -> List[float]:
        """The estimator's own probability for the class it picked. Estimators
        without `predict_proba` report 1.0 -- they offer no basis to doubt
        their own prediction, so nothing gets flagged rather than everything."""
        if not hasattr(self.classifier, "predict_proba"):
            return [1.0] * len(predictions)
        proba = self.classifier.predict_proba(vectors)
        classes = list(self.classifier.classes_)
        return [float(row[classes.index(prediction)]) for row, prediction in zip(proba, predictions)]
