import numpy as np
import pytest

from graderbot.embedding_viz import build_scatter_df, project_3d
from graderbot.storage import StudentRecord


def test_project_3d_shape():
    vectors = np.random.default_rng(0).normal(size=(10, 4096)).astype(np.float32)
    coords = project_3d(vectors)
    assert coords.shape == (10, 3)


def test_project_3d_separates_distinct_clusters():
    # 30 points/cluster mirrors real scale (~10 samples/student x ~30
    # students/class); t-SNE needs enough points per cluster for its
    # perplexity-based neighborhood structure to actually separate them --
    # a handful of points is too few to reliably preserve cluster identity.
    rng = np.random.default_rng(0)
    n = 30
    cluster_a = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (n, 1)) + rng.normal(scale=0.01, size=(n, 5))
    cluster_b = np.tile([0.0, 1.0, 0.0, 0.0, 0.0], (n, 1)) + rng.normal(scale=0.01, size=(n, 5))
    vectors = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    coords = project_3d(vectors)
    centroid_a = coords[:n].mean(axis=0)
    centroid_b = coords[n:].mean(axis=0)
    mean_within = np.mean(
        [np.linalg.norm(coords[i] - centroid_a) for i in range(n)]
        + [np.linalg.norm(coords[i] - centroid_b) for i in range(n, 2 * n)]
    )
    dist_between_centroids = np.linalg.norm(centroid_a - centroid_b)
    assert dist_between_centroids > mean_within


def test_project_3d_falls_back_to_pca_for_few_samples():
    vectors = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    coords = project_3d(vectors)
    assert coords.shape == (3, 3)


def test_project_3d_single_vector():
    vectors = np.zeros((1, 8), dtype=np.float32)
    coords = project_3d(vectors)
    assert coords.shape == (1, 3)


def test_project_3d_empty():
    vectors = np.empty((0, 0), dtype=np.float32)
    coords = project_3d(vectors)
    assert coords.shape == (0, 3)


def _students():
    return [
        StudentRecord(id=1, classroom_id=1, first_name="Ada", last_name="Lovelace"),
        StudentRecord(id=2, classroom_id=1, first_name="Alan", last_name="Turing"),
    ]


def test_build_scatter_df_joins_names_and_filters_to_classroom():
    vectors = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    student_ids = np.array([1, 2, 999], dtype=np.int64)  # 999 not in this classroom
    name_image_ids = np.array([10, 11, 12], dtype=np.int64)

    df = build_scatter_df(vectors, student_ids, name_image_ids, _students())

    assert len(df) == 2
    assert set(df["student_name"]) == {"Ada Lovelace", "Alan Turing"}
    assert list(df.columns) == ["x", "y", "z", "student_id", "student_name", "name_image_id"]


def test_build_scatter_df_empty_vectors():
    vectors = np.empty((0, 2), dtype=np.float32)
    student_ids = np.empty((0,), dtype=np.int64)
    name_image_ids = np.empty((0,), dtype=np.int64)

    df = build_scatter_df(vectors, student_ids, name_image_ids, _students())

    assert len(df) == 0
    assert list(df.columns) == ["x", "y", "z", "student_id", "student_name", "name_image_id"]
