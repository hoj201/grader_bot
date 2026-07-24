"""Pure helpers for the embedding visualizer tab (issue #48): reduce
handwriting-sample embeddings to 3D via t-SNE and join them with student
names for plotting. No S3/Streamlit dependency, so this is straightforward
to test."""

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from graderbot.storage import StudentRecord

_COLUMNS = ["x", "y", "z", "student_id", "student_name", "name_image_id"]
_N_DIMS = 3
# Below this many samples, t-SNE's PCA-based init (and perplexity) become
# unreliable/erroring, so fall back to plain PCA.
_MIN_SAMPLES_FOR_TSNE = 4


def _pca_project(vectors: np.ndarray, n_dims: int) -> np.ndarray:
    n = vectors.shape[0]
    if n < 2:
        return np.zeros((n, n_dims), dtype=np.float32)
    n_components = min(n_dims, vectors.shape[1])
    coords = PCA(n_components=n_components).fit_transform(vectors)
    if n_components < n_dims:
        coords = np.pad(coords, ((0, 0), (0, n_dims - n_components)))
    return coords


def project_3d(vectors: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Reduce `(n, d)` embedding vectors to `(n, 3)` via t-SNE. Falls back
    to PCA when there are too few samples for t-SNE to run meaningfully."""
    n = vectors.shape[0]
    if n < _MIN_SAMPLES_FOR_TSNE:
        return _pca_project(vectors, _N_DIMS)

    perplexity = min(30.0, (n - 1) / 3)
    coords = TSNE(
        n_components=_N_DIMS,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
    ).fit_transform(vectors)
    return coords.astype(np.float32)


def build_scatter_df(
    vectors: np.ndarray,
    student_ids: np.ndarray,
    name_image_ids: np.ndarray,
    students: Iterable[StudentRecord],
) -> pd.DataFrame:
    """Project `vectors` to 3D and join in student display names, keeping
    only rows whose `student_id` is one of `students` (i.e. scoped to a
    single classroom)."""
    names_by_id = {
        s.id: f"{s.first_name} {s.last_name}".strip() for s in students
    }
    keep = np.array([sid in names_by_id for sid in student_ids], dtype=bool)

    if not keep.any():
        return pd.DataFrame(columns=_COLUMNS)

    coords = project_3d(vectors[keep])
    kept_student_ids = student_ids[keep]
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "student_id": kept_student_ids,
            "student_name": [names_by_id[sid] for sid in kept_student_ids],
            "name_image_id": name_image_ids[keep],
        }
    )
