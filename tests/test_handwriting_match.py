"""Pure-logic tests for graderbot/handwriting_match.py.

None of these tests load the real ONNX model or touch the network -- they
exercise the CTC forward algorithm, vocabulary parsing, and preprocessing in
isolation, monkeypatching the model/vocab where a real one would otherwise be
needed. See tests/test_handwriting_match_integration.py for real-model tests,
gated on the model file / onnxruntime actually being available.
"""

import itertools

import numpy as np
import pytest

from graderbot import handwriting_match as hm


# ---------------------------------------------------------------------------
# _ctc_log_prob: the forward algorithm itself
# ---------------------------------------------------------------------------


def _brute_force_log_prob(probs: np.ndarray, target_ids, blank: int) -> float:
    """Reference implementation: literally enumerate every length-T path over
    the vocabulary, collapse repeats then drop blanks (the CTC decoding rule),
    and sum the probability of every path that collapses to `target_ids`.
    Only tractable for tiny T/vocab -- exactly what it's used for here."""
    T, C = probs.shape
    total = 0.0
    for path in itertools.product(range(C), repeat=T):
        collapsed = [k for k, _ in itertools.groupby(path)]
        collapsed = [k for k in collapsed if k != blank]
        if collapsed == list(target_ids):
            p = 1.0
            for t, k in enumerate(path):
                p *= probs[t, k]
            total += p
    return np.log(total) if total > 0 else -np.inf


@pytest.mark.parametrize(
    "target_ids",
    [
        [],
        [1],
        [2],
        [1, 2],
        [1, 1],  # adjacent repeat -- must force a blank between them
        [2, 1, 2],
    ],
)
def test_ctc_log_prob_matches_brute_force_enumeration(target_ids):
    rng = np.random.default_rng(0)
    T, C = 4, 3
    logits = rng.normal(size=(T, C))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    log_probs = np.log(probs)

    expected = _brute_force_log_prob(probs, target_ids, blank=0)
    actual = hm._ctc_log_prob(log_probs, target_ids, blank=0)

    if expected == -np.inf:
        assert actual == -np.inf
    else:
        assert actual == pytest.approx(expected, abs=1e-6)


def test_ctc_log_prob_is_minus_inf_when_sequence_too_short_for_target():
    # 1 timestep cannot possibly emit 2 distinct non-blank labels.
    log_probs = np.log(np.array([[0.5, 0.3, 0.2]]))
    assert hm._ctc_log_prob(log_probs, [1, 2], blank=0) == -np.inf


def test_ctc_log_prob_ambiguous_1_vs_slash_gives_partial_credit_to_both():
    """The motivating case: a timestep genuinely torn between two labels
    should score both candidate readings with real, non-negligible, and
    non-dominant-over-the-other probability -- not push one to ~0."""
    # vocab: 0=blank, 1='1', 2='/'
    # A single ambiguous frame split 50/50 between '1' and '/'.
    log_probs = np.log(
        np.array(
            [
                [0.1, 0.45, 0.45],
                [0.8, 0.1, 0.1],
            ]
        )
    )
    p_one = np.exp(hm._ctc_log_prob(log_probs, [1], blank=0))
    p_slash = np.exp(hm._ctc_log_prob(log_probs, [2], blank=0))
    assert p_one == pytest.approx(p_slash, abs=1e-9)
    assert 0.3 < p_one < 0.5


# ---------------------------------------------------------------------------
# vocabulary parsing
# ---------------------------------------------------------------------------


def test_load_symbols_parses_syms_txt(tmp_path):
    syms_path = tmp_path / "syms.txt"
    syms_path.write_text("<ctc> 0\na 1\nb 2\n")
    symbol_to_id, id_to_symbol = hm._parse_symbols(syms_path)
    assert symbol_to_id == {"<ctc>": 0, "a": 1, "b": 2}
    assert id_to_symbol == ["<ctc>", "a", "b"]


def test_load_symbols_requires_blank_at_index_zero(tmp_path):
    syms_path = tmp_path / "syms.txt"
    syms_path.write_text("a 0\n<ctc> 1\n")
    with pytest.raises(ValueError, match="<ctc>"):
        hm._parse_symbols(syms_path)


def test_encode_candidate_maps_known_characters():
    symbol_to_id = {"<ctc>": 0, "1": 1, "/": 2, "2": 3}
    assert hm._encode_candidate("1/2", symbol_to_id) == [1, 2, 3]


def test_encode_candidate_rejects_out_of_vocabulary_character():
    symbol_to_id = {"<ctc>": 0, "1": 1}
    with pytest.raises(ValueError, match="'x'"):
        hm._encode_candidate("1x", symbol_to_id)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_crop_resizes_to_target_height_keeping_aspect_ratio():
    crop = np.full((64, 256, 3), 255, dtype=np.uint8)
    batch = hm.preprocess_crop(crop)
    assert batch.shape[0] == 1  # batch dim
    assert batch.shape[1] == 1  # channel dim (grayscale)
    assert batch.shape[2] == hm._TARGET_HEIGHT
    assert batch.shape[3] == pytest.approx(256 * hm._TARGET_HEIGHT / 64, abs=1)


def test_preprocess_crop_rejects_empty_image():
    with pytest.raises(ValueError):
        hm.preprocess_crop(np.zeros((0, 0, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# match_probability, with the model/vocab faked out
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stands in for the onnxruntime.InferenceSession: emits fixed
    (T, N, C) log-probabilities regardless of input, so match_probability's
    plumbing (not the real model) is what's under test."""

    def __init__(self, log_probs: np.ndarray):
        self._log_probs = log_probs

    def run(self, output_names, input_feed):
        return [self._log_probs]


def test_match_probability_prefers_the_correct_candidate(monkeypatch):
    # vocab: 0=blank, 1='1', 2='2'
    symbol_to_id = {"<ctc>": 0, "1": 1, "2": 2}
    id_to_symbol = ["<ctc>", "1", "2"]
    # Two confident frames both strongly favoring label '1'.
    log_probs = np.log(
        np.array(
            [[[0.05, 0.9, 0.05]], [[0.05, 0.9, 0.05]]], dtype=np.float64
        )
    )
    monkeypatch.setattr(hm, "_load_symbols", lambda: (symbol_to_id, id_to_symbol))
    monkeypatch.setattr(hm, "_load_session", lambda: _FakeSession(log_probs))

    crop = np.full((32, 64, 3), 255, dtype=np.uint8)
    p_right = hm.match_probability(crop, "1")
    p_wrong = hm.match_probability(crop, "2")
    assert p_right > p_wrong


def test_match_probability_is_a_valid_probability(monkeypatch):
    symbol_to_id = {"<ctc>": 0, "1": 1}
    id_to_symbol = ["<ctc>", "1"]
    log_probs = np.log(np.array([[[0.5, 0.5]]], dtype=np.float64))
    monkeypatch.setattr(hm, "_load_symbols", lambda: (symbol_to_id, id_to_symbol))
    monkeypatch.setattr(hm, "_load_session", lambda: _FakeSession(log_probs))

    crop = np.full((32, 64, 3), 255, dtype=np.uint8)
    p = hm.match_probability(crop, "1")
    assert 0.0 <= p <= 1.0
