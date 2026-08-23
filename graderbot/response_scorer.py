"""Closed-set handwriting-to-answer verification (issue #81): scores an
answer-box crop against a short list of candidate strings and reports which
one the crop most likely shows, instead of transcribing it open-vocabulary
the way `answer_reader`'s OCR backends do. This is what lets grading use
context an OCR backend doesn't have -- the correct answer, plus a handful of
plausible OCR-confusable near-misses (`response_candidates.generate_candidates`)
-- to resolve ambiguous ink instead of guessing blind.

Two layers:

- `ctc_log_prob` / `score_text_candidates`: pure numpy, no ML framework
  dependency. Implements the CTC forward algorithm's log-space alpha
  recursion to compute how well a per-timestep probability grid (a trained
  CRNN's output) explains a given label sequence, then softmax-normalizes
  those scores *across the candidate set* rather than thresholding any one
  of them absolutely -- raw CTC probabilities are small by construction
  (see `handwriting-ctc-match-paused`'s lesson from the paused issue #73
  spike), so only relative candidate-vs-candidate comparisons are
  meaningful.
- `CnnResponseScorer`: the `ResponseScorer` this module exists to provide,
  wrapping a trained CRNN exported to ONNX (`onnxruntime`, not torch --
  torch has no Intel-macOS wheel for this project's Python version, and
  onnxruntime does; see the training/ directory's own docstrings). The
  model itself is trained offline in `training/`, never in this module or
  at grading time.

Scope (v1): plain numeric answers only, matching `response_candidates`'s
vocabulary (`VOCAB`) -- digits, `.`, `-`. A `\\frac{a}{b}` answer should
never reach `CnnResponseScorer.score`; `grading.grade_hw` skips this scorer
for those and falls back to `answer_reader`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import cv2
import numpy as np

from graderbot.imaging import crop_box_content_aware
from graderbot.models import Box
from graderbot.ocr import _BOX_INSET

# CTC blank is index 0; every other symbol is `VOCAB[i]` at label index
# `i + 1`. Kept intentionally tiny (13 symbols) compared to a general OCR
# model's alphabet -- see `response_candidates`'s module docstring for why
# this scope is enough for what this scorer is asked to verify.
VOCAB = "0123456789.-"
_BLANK = 0

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "response_scorer"
_WEIGHTS_FILENAME = "weights.onnx"
_VOCAB_FILENAME = "vocab.json"

# Must match training/model.py's expected input height -- the CRNN's conv
# stack is only valid for a fixed input height (its stride collapses height
# to 1 row of features by design), so a crop is always resized to this
# height (aspect-ratio preserved) before the network ever sees it. Width is
# left free since the model's RNN/CTC head is inherently variable-length.
MODEL_INPUT_HEIGHT = 32

_NEG_INF = -np.inf


@dataclass
class ScoredResponse:
    """The outcome of scoring one crop against a candidate list.

    `best` is the candidate with the highest `scores` entry (ties broken by
    candidate order, i.e. in favor of whichever came first in the list
    passed to `score_text_candidates`). `scores` is softmax-normalized
    *across the candidate set* -- it sums to 1 over `scores.values()`, and
    is only meaningful as a relative comparison between candidates, not as
    a calibrated probability that `best` is actually correct. `log_probs`
    carries the raw (un-normalized) CTC log-probability behind each score,
    for debugging a surprising pick the way `QuestionResult.ocr_raw` lets a
    wrong OCR read be traced back to what OCR actually saw.
    """

    best: str
    scores: Dict[str, float]
    log_probs: Dict[str, float]


def ctc_log_prob(log_probs: np.ndarray, labels: List[int], blank: int = _BLANK) -> float:
    """Log-probability that the label sequence `labels` (indices into
    `log_probs`'s columns; must not itself contain `blank`) explains
    `log_probs`, a `(T, C)` array of per-timestep log-probabilities from a
    CTC model (each row a log-softmax over `C` classes, i.e.
    `np.exp(log_probs[t]).sum() == 1`).

    Implements the standard CTC forward algorithm: `labels` is expanded to
    `ext = [blank, labels[0], blank, labels[1], blank, ..., labels[-1],
    blank]` (length `2*len(labels) + 1`), and `alpha[t, s]` accumulates the
    log-probability, summed over every valid alignment, of the first `t+1`
    timesteps explaining the first `s+1` symbols of `ext`. The two
    "sources" `alpha[t, s]` can come from are staying on the same expanded
    symbol (`alpha[t-1, s]`) or advancing one (`alpha[t-1, s-1]`); advancing
    two (`alpha[t-1, s-2]`) is also valid when it skips over a blank
    between two *different* symbols (CTC's repeated-symbol rule -- skipping
    the blank between two of the *same* symbol would silently merge them
    into one output symbol instead of two, so that transition is excluded).

    Returns `-inf` if `log_probs` has fewer timesteps than `labels` has
    symbols -- there is no way to align them at all."""
    T = log_probs.shape[0]
    L = len(labels)
    if T < L:
        return _NEG_INF

    ext: List[int] = [blank]
    for label in labels:
        ext.append(label)
        ext.append(blank)
    S = len(ext)

    alpha = np.full((T, S), _NEG_INF)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        prev = alpha[t - 1]
        same_or_prev = prev.copy()
        same_or_prev[1:] = np.logaddexp(prev[1:], prev[:-1])
        if S > 2:
            skip_eligible = np.array(
                [ext[s] != blank and ext[s] != ext[s - 2] for s in range(2, S)]
            )
            skip_scores = np.where(skip_eligible, prev[:-2], _NEG_INF)
            same_or_prev[2:] = np.logaddexp(same_or_prev[2:], skip_scores)
        alpha[t] = same_or_prev + log_probs[t, ext]

    if S == 1:
        return float(alpha[T - 1, 0])
    return float(np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2]))


def _encode(text: str, vocab: str) -> Optional[List[int]]:
    """Maps `text` to label indices into `vocab` (offset by 1 for the
    reserved blank at index 0), or `None` if `text` contains a character
    `vocab` has no symbol for -- e.g. a stray letter, or (in the v1 scope)
    a `/` from a fraction candidate that slipped through."""
    indices = []
    for ch in text:
        pos = vocab.find(ch)
        if pos == -1:
            return None
        indices.append(pos + 1)
    return indices


def score_text_candidates(
    log_probs: np.ndarray, candidates: List[str], vocab: str = VOCAB, blank: int = _BLANK
) -> ScoredResponse:
    """Scores one crop's model output `log_probs` (`(T, C)`, see
    `ctc_log_prob`) against every string in `candidates`, softmax-normalizing
    the resulting CTC log-probabilities across the candidate set (not
    absolutely -- see module docstring) into `ScoredResponse.scores`.

    A candidate that can't be encoded in `vocab` (`_encode` returns `None`)
    or that `log_probs` is too short to possibly emit (`ctc_log_prob`
    returns `-inf`) gets a raw score of `-inf` and a normalized score of
    `0.0` -- it is never picked as `best` unless every candidate is equally
    unscoreable, in which case scores fall back to uniform so `best` is
    simply the first candidate (by convention, the stored answer -- see
    `response_candidates.generate_candidates`)."""
    raw: List[float] = []
    for candidate in candidates:
        labels = _encode(candidate, vocab)
        raw.append(ctc_log_prob(log_probs, labels, blank) if labels is not None else _NEG_INF)

    log_values = np.array(raw)
    finite = log_values[np.isfinite(log_values)]
    if finite.size == 0:
        normalized = np.full(len(candidates), 1.0 / len(candidates))
    else:
        shifted = log_values - finite.max()
        with np.errstate(over="ignore"):
            exp = np.where(np.isfinite(log_values), np.exp(shifted), 0.0)
        normalized = exp / exp.sum()

    best_index = int(np.argmax(normalized))
    return ScoredResponse(
        best=candidates[best_index],
        scores={c: float(p) for c, p in zip(candidates, normalized)},
        log_probs={c: float(v) for c, v in zip(candidates, log_values)},
    )


def prepare_crop_for_model(crop: np.ndarray, target_height: int = MODEL_INPUT_HEIGHT) -> np.ndarray:
    """Resizes `crop` (RGB, any size) to `target_height` px tall (aspect
    ratio preserved) and normalizes it to a `(1, 1, target_height, W)`
    float32 array in `[-1, 1]` -- the exact preprocessing `training/dataset.py`
    trains the CRNN on. Inference and training must agree on this
    pixel-for-pixel, or the exported ONNX graph silently sees
    out-of-distribution input it was never trained to handle."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale = target_height / height
    resized_width = max(1, int(round(width * scale)))
    resized = cv2.resize(gray, (resized_width, target_height), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return normalized[None, None, :, :]


class ResponseScorer(Protocol):
    """Scores an answer-box crop against a candidate list. Modeled on
    `name_reader.NameReader` rather than `answer_reader.AnswerReader`:
    scoring needs the candidate set, which only `grading.grade_hw` can build
    (it alone has the stored answer key), so this protocol takes it as a
    parameter instead of being handed a pre-built prompt like `AnswerReader.read`."""

    def score(self, image: np.ndarray, box: Box, candidates: List[str]) -> ScoredResponse:
        """Score the crop inside `box` on `image` (an already-loaded RGB
        numpy array) against `candidates` (see `response_candidates.generate_candidates`;
        `candidates[0]` is conventionally the stored answer)."""
        ...


def model_files_exist(model_dir: Optional[Union[str, Path]] = None) -> bool:
    """Whether a trained model is present at `model_dir` (defaults to
    `_DEFAULT_MODEL_DIR`) -- lets a caller like the Grade tab check before
    offering `CnnResponseScorer` as an option, rather than only finding out
    on the first `score()` call's `FileNotFoundError`."""
    model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
    return (model_dir / _WEIGHTS_FILENAME).is_file() and (model_dir / _VOCAB_FILENAME).is_file()


class CnnResponseScorer:
    """`ResponseScorer` backed by a CRNN trained offline in `training/` and
    exported to ONNX. Loads its `onnxruntime` session and vocabulary lazily
    on first `score()` call -- imported inside `_ensure_loaded` rather than
    at module scope, and the model files read from disk there rather than
    in `__init__`, so constructing (but never calling) a `CnnResponseScorer`
    costs nothing extra on a grading run that ends up using a different
    `response_scorer`/none at all. Same lazy pattern as
    `name_reader.ClassifierNameReader.from_classroom`'s `name_classifier`
    import.
    """

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        self._session = None
        self._vocab: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort  # see class docstring for why this isn't a module-level import

        weights_path = self.model_dir / _WEIGHTS_FILENAME
        vocab_path = self.model_dir / _VOCAB_FILENAME
        if not weights_path.is_file() or not vocab_path.is_file():
            raise FileNotFoundError(
                f"No trained response-scorer model found at {self.model_dir} "
                "-- run training/train.py then training/export_onnx.py first "
                "(see issue #81)."
            )
        self._vocab = json.loads(vocab_path.read_text())["vocab"]
        self._session = ort.InferenceSession(str(weights_path))

    def score(self, image: np.ndarray, box: Box, candidates: List[str]) -> ScoredResponse:
        self._ensure_loaded()
        crop = crop_box_content_aware(image, box, fallback_inset=_BOX_INSET)
        network_input = prepare_crop_for_model(crop)
        input_name = self._session.get_inputs()[0].name
        (log_probs,) = self._session.run(None, {input_name: network_input})
        # (1, T, C) -> (T, C): batch size is always 1 here, one crop per call.
        return score_text_candidates(log_probs[0], candidates, vocab=self._vocab)
