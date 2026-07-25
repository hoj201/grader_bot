"""Train a per-student name classifier over handwriting vectors and persist it
to S3 (issue #2, phases 3-4).

The dataset is tiny (~10 samples per student) and each class is one student's own
handwriting of their own name, so a k-nearest-neighbours classifier over the
embedding vectors is a good fit. The trained model is serialized with joblib and
stored in S3 at a caller-chosen key so grading can fetch it later.
"""

import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from graderbot.embedding import Embedder, augment_images, load_training_images, load_training_vectors

DEFAULT_CLASSIFIER_KEY = "name_classifier/knn.joblib"
_DEFAULT_N_NEIGHBORS = 3
# LOO-CV cost scales with training-set size on every fold, so an unbounded
# per-class fold count would make evaluation blow up as a prolific student
# (or a large roster) accumulates samples (issue #55 follow-up). Capping
# folds per class keeps each student's own contribution to runtime bounded.
_DEFAULT_MAX_FOLDS_PER_CLASS = 20

# A factory builds an *unfitted* estimator given the training-set size (KNN
# needs this to clamp k; other estimators can ignore it). Keeping the
# classifier pluggable lets loo_cross_validate compare candidates (e.g. KNN
# vs. a heavily-regularized logistic regression) instead of assuming KNN
# (issue #56).
ClassifierFactory = Callable[[int], BaseEstimator]


def _default_classifier_factory(n_neighbors: int) -> ClassifierFactory:
    return lambda n_samples: KNeighborsClassifier(n_neighbors=min(n_neighbors, n_samples))


def logistic_regression_factory(C: float = 1.0, max_iter: int = 1000) -> ClassifierFactory:
    """Factory for a multinomial logistic regression classifier, for
    comparing against the default KNN via `loo_cross_validate` (issue #56).
    `C` is inverse regularization strength -- smaller values mean stronger
    regularization. Pushing `C` too low makes the (unpenalized) intercept
    dominate the near-zero coefficients and the model degenerates to
    predicting a single class, so sweep `C` empirically (e.g. via
    `loo_cross_validate`) rather than assuming smaller is better; the default
    here is sklearn's own unregularized-ish `C=1.0` starting point."""
    return lambda n_samples: LogisticRegression(C=C, max_iter=max_iter)


def train_name_classifier(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    classifier_factory: Optional[ClassifierFactory] = None,
) -> BaseEstimator:
    """Fit a classifier mapping embedding vectors to student ids. Defaults to
    KNN with `k` clamped to the number of samples so tiny datasets still
    train; pass `classifier_factory` to swap in a different estimator
    (issue #56)."""
    if len(vectors) == 0:
        raise ValueError("cannot train a classifier on an empty vector collection")
    factory = classifier_factory or _default_classifier_factory(n_neighbors)
    classifier = factory(len(vectors))
    classifier.fit(vectors, labels)
    return classifier


def train_from_db(
    db_path: Union[str, Path],
    bucket: str,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    classifier_factory: Optional[ClassifierFactory] = None,
    s3_client=None,
) -> BaseEstimator:
    """Load per-image embeddings from NAME_EMBEDDINGS (issue #43) and train a
    classifier keyed by student_id."""
    vectors, student_ids, _ = load_training_vectors(Path(db_path), bucket, s3_client)
    return train_name_classifier(vectors, student_ids, n_neighbors, classifier_factory)


def _select_fold_indices(
    labels: np.ndarray, max_folds_per_class: int, rng: np.random.Generator
) -> Tuple[List[int], List]:
    """Pick which sample indices get held out in leave-one-out CV: every
    sample of a class with >=2 members, subsampled down to
    `max_folds_per_class` per class (issue #55). Returns
    `(fold_indices, insufficient_labels)`."""
    counts = Counter(labels.tolist())
    insufficient = sorted(label for label, count in counts.items() if count < 2)

    fold_indices: List[int] = []
    for label, count in counts.items():
        if count < 2:
            continue
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) > max_folds_per_class:
            class_indices = rng.choice(class_indices, size=max_folds_per_class, replace=False)
        fold_indices.extend(class_indices.tolist())
    return fold_indices, insufficient


def loo_cross_validate(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    max_folds_per_class: int = _DEFAULT_MAX_FOLDS_PER_CLASS,
    random_state: int = 0,
    classifier_factory: Optional[ClassifierFactory] = None,
) -> Tuple[Dict, List, Dict]:
    """Leave-one-out cross-validation of the classifier, scored per class
    (issue #55).

    For each held-out sample, a classifier is trained on every other sample
    and asked to predict it. A class needs at least 2 samples for this to be
    meaningful -- with only 1, the held-out sample's own class is absent from
    its training fold, so the model can never predict it correctly no matter
    how well-separated the handwriting is. Those classes are reported
    separately as `insufficient` rather than folded into accuracy as an
    automatic miss.

    A class with more than `max_folds_per_class` samples has its held-out
    folds randomly subsampled down to that many (seeded by `random_state`),
    so one student accumulating many samples doesn't blow up evaluation cost
    -- otherwise every additional sample means another full retrain-and-predict
    pass.

    `classifier_factory` swaps the estimator (defaults to KNN), so different
    candidates can be compared on the same folds (issue #56).

    Returns `(accuracy, insufficient, confusion)`:
    - `accuracy`: label -> fraction of correct held-out predictions.
    - `insufficient`: labels with fewer than 2 samples.
    - `confusion`: true label -> {predicted label: count}, over the same
      held-out folds used for `accuracy` (predicted labels may include
      insufficient/other classes that were never themselves held out).
    """
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    fold_indices, insufficient = _select_fold_indices(labels, max_folds_per_class, rng)

    correct: Counter = Counter()
    total: Counter = Counter()
    confusion: Dict = defaultdict(Counter)
    all_indices = np.arange(len(vectors))
    for i in fold_indices:
        label = labels[i]
        train_idx = all_indices[all_indices != i]
        classifier = train_name_classifier(
            vectors[train_idx], labels[train_idx], n_neighbors, classifier_factory
        )
        prediction = classifier.predict(vectors[i : i + 1])[0]
        total[label] += 1
        confusion[label][prediction] += 1
        if prediction == label:
            correct[label] += 1

    accuracy = {label: correct[label] / total[label] for label in total}
    return accuracy, insufficient, {label: dict(preds) for label, preds in confusion.items()}


def loo_cross_validate_from_db(
    db_path: Union[str, Path],
    bucket: str,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    max_folds_per_class: int = _DEFAULT_MAX_FOLDS_PER_CLASS,
    random_state: int = 0,
    classifier_factory: Optional[ClassifierFactory] = None,
    s3_client=None,
) -> Tuple[Dict, List, Dict]:
    """Load per-image embeddings from NAME_EMBEDDINGS and run `loo_cross_validate`
    keyed by student_id."""
    vectors, student_ids, _ = load_training_vectors(Path(db_path), bucket, s3_client)
    return loo_cross_validate(
        vectors, student_ids, n_neighbors, max_folds_per_class, random_state, classifier_factory
    )


def loo_cross_validate_augmented(
    images: List[np.ndarray],
    labels: np.ndarray,
    embedder: Embedder,
    n_augmentations: int = 4,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    max_folds_per_class: int = _DEFAULT_MAX_FOLDS_PER_CLASS,
    random_state: int = 0,
    classifier_factory: Optional[ClassifierFactory] = None,
) -> Tuple[Dict, List, Dict]:
    """Leave-one-out cross-validation with data augmentation (issue #56): each
    fold's *training* images (never the held-out one, to avoid leaking a
    distorted copy of the test sample into its own training fold) are
    expanded with `n_augmentations` randomly distorted copies apiece before
    embedding and fitting. Otherwise identical to `loo_cross_validate` --
    same fold selection, same `(accuracy, insufficient, confusion)` return
    shape -- so it can be compared directly against the unaugmented run."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    fold_indices, insufficient = _select_fold_indices(labels, max_folds_per_class, rng)

    correct: Counter = Counter()
    total: Counter = Counter()
    confusion: Dict = defaultdict(Counter)
    all_indices = np.arange(len(images))
    for i in fold_indices:
        label = labels[i]
        train_idx = all_indices[all_indices != i]
        train_images = [images[j] for j in train_idx]
        train_labels = labels[train_idx]

        augmented = augment_images(train_images, rng, n_augmentations)
        augmented_labels = np.repeat(train_labels, n_augmentations)
        all_train_images = train_images + augmented
        all_train_labels = np.concatenate([train_labels, augmented_labels])

        train_vectors = embedder.embed(all_train_images)
        classifier = train_name_classifier(
            train_vectors, all_train_labels, n_neighbors, classifier_factory
        )
        held_out_vector = embedder.embed([images[i]])
        prediction = classifier.predict(held_out_vector)[0]

        total[label] += 1
        confusion[label][prediction] += 1
        if prediction == label:
            correct[label] += 1

    accuracy = {label: correct[label] / total[label] for label in total}
    return accuracy, insufficient, {label: dict(preds) for label, preds in confusion.items()}


def loo_cross_validate_augmented_from_db(
    db_path: Union[str, Path],
    bucket: str,
    embedder: Embedder,
    n_augmentations: int = 4,
    n_neighbors: int = _DEFAULT_N_NEIGHBORS,
    max_folds_per_class: int = _DEFAULT_MAX_FOLDS_PER_CLASS,
    random_state: int = 0,
    classifier_factory: Optional[ClassifierFactory] = None,
    s3_client=None,
) -> Tuple[Dict, List, Dict]:
    """Load raw NAME_IMAGES crops and run `loo_cross_validate_augmented` keyed
    by student_id (issue #56)."""
    images, student_ids, _ = load_training_images(Path(db_path), bucket, s3_client)
    return loo_cross_validate_augmented(
        images,
        student_ids,
        embedder,
        n_augmentations,
        n_neighbors,
        max_folds_per_class,
        random_state,
        classifier_factory,
    )


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
