"""`ctc_log_prob` is checked against brute-force path enumeration (same
technique the paused `handwriting-ctc-match` branch used in
`tests/test_handwriting_match.py`) rather than trusted on faith -- the log-
space recursion has enough index bookkeeping (the "skip a blank between two
different symbols" case in particular) that a silent off-by-one would be
easy to miss otherwise."""

import itertools
import json
import sys
import types

import numpy as np
import pytest

from graderbot.models import Box
from graderbot.response_scorer import (
    VOCAB,
    CnnResponseScorer,
    ScoredResponse,
    ctc_log_prob,
    prepare_crop_for_model,
    score_text_candidates,
)


def _brute_force_ctc_prob(probs: np.ndarray, labels, blank: int = 0) -> float:
    T, C = probs.shape
    total = 0.0
    for path in itertools.product(range(C), repeat=T):
        path_prob = 1.0
        for t, c in enumerate(path):
            path_prob *= probs[t, c]
        decoded = []
        prev = None
        for c in path:
            if c != prev and c != blank:
                decoded.append(c)
            prev = c
        if decoded == list(labels):
            total += path_prob
    return total


def _random_log_probs(T: int, C: int, seed: int):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, C))
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs, np.log(probs)


@pytest.mark.parametrize("labels,seed", [([1], 0), ([1, 2], 1), ([1, 1], 2), ([2, 1, 2], 3)])
def test_ctc_log_prob_matches_brute_force_enumeration(labels, seed):
    probs, log_probs = _random_log_probs(T=5, C=3, seed=seed)
    expected = _brute_force_ctc_prob(probs, labels)
    actual = np.exp(ctc_log_prob(log_probs, labels))
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_ctc_log_prob_returns_neg_inf_when_too_few_timesteps():
    log_probs = np.log(np.full((2, 3), 1 / 3))
    assert ctc_log_prob(log_probs, [1, 2, 1]) == float("-inf")


def test_ctc_log_prob_handles_the_empty_label_sequence():
    # No symbols to emit: the only valid alignment is "blank" at every step.
    log_probs = np.log(np.full((3, 2), 1 / 2))
    expected = (1 / 2) ** 3
    assert np.exp(ctc_log_prob(log_probs, [])) == pytest.approx(expected)


def test_score_text_candidates_normalizes_to_one():
    C = len(VOCAB) + 1
    log_probs = np.log(np.full((6, C), 1 / C))
    result = score_text_candidates(log_probs, ["1", "2", "3"])
    assert sum(result.scores.values()) == pytest.approx(1.0)


def test_score_text_candidates_picks_the_dominant_candidate():
    T, C = 4, len(VOCAB) + 1
    nine_index = VOCAB.index("9") + 1
    log_probs = np.full((T, C), np.log(1e-6))
    log_probs[:, nine_index] = np.log(1 - 1e-6 * (C - 1))

    result = score_text_candidates(log_probs, ["1", "9", "3"])

    assert result.best == "9"
    assert result.scores["9"] > 0.9


def test_score_text_candidates_falls_back_to_first_candidate_when_none_scoreable():
    C = len(VOCAB) + 1
    log_probs = np.log(np.full((2, C), 1 / C))  # too few timesteps for either 3-char candidate

    result = score_text_candidates(log_probs, ["123", "456"])

    assert result.best == "123"
    assert result.scores["123"] == pytest.approx(0.5)
    assert result.scores["456"] == pytest.approx(0.5)


def test_prepare_crop_for_model_resizes_and_normalizes():
    crop = np.full((80, 200, 3), 255, dtype=np.uint8)
    prepared = prepare_crop_for_model(crop, target_height=32)
    assert prepared.shape[:3] == (1, 1, 32)
    assert prepared.dtype == np.float32
    assert prepared.min() >= -1.0 and prepared.max() <= 1.0


def test_cnn_response_scorer_raises_when_no_model_files_exist(tmp_path):
    scorer = CnnResponseScorer(model_dir=tmp_path)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    box = Box(x_lower_left=0.1, y_lower_left=0.1, width=0.2, height=0.1)
    with pytest.raises(FileNotFoundError):
        scorer.score(image, box, ["1"])


def test_cnn_response_scorer_scores_using_a_fake_onnx_session(tmp_path, monkeypatch):
    (tmp_path / "vocab.json").write_text(json.dumps({"vocab": VOCAB}))
    (tmp_path / "weights.onnx").write_bytes(b"stand-in -- the fake session below never reads this file")

    T, C = 4, len(VOCAB) + 1

    class _FakeInput:
        name = "input"

    class _FakeSession:
        def __init__(self, path):
            self.path = path

        def get_inputs(self):
            return [_FakeInput()]

        def run(self, output_names, feed):
            assert "input" in feed
            return [np.full((1, T, C), np.log(1 / C), dtype=np.float32)]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    scorer = CnnResponseScorer(model_dir=tmp_path)
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = Box(x_lower_left=0.2, y_lower_left=0.2, width=0.3, height=0.2)
    result = scorer.score(image, box, ["1", "2"])

    assert isinstance(result, ScoredResponse)
    assert result.best in ("1", "2")
    assert sum(result.scores.values()) == pytest.approx(1.0)


def test_cnn_response_scorer_loads_session_lazily_only_once(tmp_path, monkeypatch):
    (tmp_path / "vocab.json").write_text(json.dumps({"vocab": VOCAB}))
    (tmp_path / "weights.onnx").write_bytes(b"stand-in")

    load_count = {"n": 0}

    class _FakeInput:
        name = "input"

    class _FakeSession:
        def __init__(self, path):
            load_count["n"] += 1

        def get_inputs(self):
            return [_FakeInput()]

        def run(self, output_names, feed):
            return [np.full((1, 3, len(VOCAB) + 1), np.log(1 / (len(VOCAB) + 1)), dtype=np.float32)]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    scorer = CnnResponseScorer(model_dir=tmp_path)
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = Box(x_lower_left=0.2, y_lower_left=0.2, width=0.3, height=0.2)
    scorer.score(image, box, ["1"])
    scorer.score(image, box, ["2"])

    assert load_count["n"] == 1
