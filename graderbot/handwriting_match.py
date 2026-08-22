"""Scores a handwriting-crop image against a candidate text string using a
pretrained CRNN's raw per-timestep character probabilities, via the CTC
forward algorithm -- rather than decoding to one string and doing an exact
compare. This is what lets an answer-box read be "70% likely '1', 30% likely
'/'" instead of committing to one and being flatly wrong, which is the
project's actual, recurring Mathpix failure mode (see `grading._SIMPLIFY_NOTE`
and `_iter_fraction_reinterpretations`'s docstring for the "1" vs "/" garble
this is meant to be a probabilistic alternative to).

Model: `Teklia/pylaia-iam` (MIT licensed) from the Hugging Face Hub -- a
LaiaCRNN (4 conv layers + a 3-layer bidirectional LSTM) pretrained on the IAM
handwriting dataset. Its 81-label vocabulary includes '1' and '/' as distinct
labels, which is exactly the ambiguity this module exists to score
probabilistically instead of resolving with a hard decode.

Why ONNX Runtime, not PyTorch: the `pylaia` package that defines the real
model architecture pins `torch<1.14`/Python<3.11, and even a bare modern
`torch` has no macOS-x86_64 wheel for Python 3.13+ at all (Apple Silicon-only
from 2.6 onward) -- it cannot be installed on every developer machine this
project runs on. The checkpoint was converted to ONNX once, offline, inside a
throwaway container that *did* have the real `pylaia`/old-torch installed
(see `scripts/convert_pylaia_to_onnx.py`), and only the resulting
`models/pylaia_iam/weights.onnx` + `syms.txt` are used at runtime, via
`onnxruntime` -- which has no such platform gap and is a much lighter
dependency besides.

Because we don't have `torch.nn.functional.ctc_loss` available at runtime,
`_ctc_log_prob` is a small hand-written implementation of the CTC forward
algorithm (log-space alpha recursion) instead -- see its docstring.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO_ROOT / "models" / "pylaia_iam"
_ONNX_PATH = _MODEL_DIR / "weights.onnx"
_SYMS_PATH = _MODEL_DIR / "syms.txt"

_TARGET_HEIGHT = 128  # matches the height PyLaia's own IAM model was trained at
_BLANK_INDEX = 0  # syms.txt reserves index 0 for the CTC blank ("<ctc>")
_BLANK_SYMBOL = "<ctc>"
_INPUT_NAME = "input"  # must match the name baked in by the conversion script


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def _parse_symbols(path: Path) -> Tuple[Dict[str, int], List[str]]:
    """Parses a PyLaia-style `syms.txt` (`"<symbol> <index>"` per line, in any
    order) into a symbol->id dict and an index-ordered list of symbols.
    Raises `ValueError` if index 0 isn't the blank symbol `<ctc>` -- the CTC
    math below assumes `blank == 0`."""
    symbol_to_id: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            symbol, index_str = line.rsplit(" ", 1)
            symbol_to_id[symbol] = int(index_str)
    id_to_symbol = [""] * len(symbol_to_id)
    for symbol, index in symbol_to_id.items():
        id_to_symbol[index] = symbol
    if id_to_symbol[_BLANK_INDEX] != _BLANK_SYMBOL:
        raise ValueError(
            f"expected symbol index {_BLANK_INDEX} to be the blank symbol "
            f"{_BLANK_SYMBOL!r}, got {id_to_symbol[_BLANK_INDEX]!r} -- "
            f"check {path}"
        )
    return symbol_to_id, id_to_symbol


@lru_cache(maxsize=1)
def _load_symbols() -> Tuple[Dict[str, int], List[str]]:
    return _parse_symbols(_SYMS_PATH)


def _encode_candidate(candidate: str, symbol_to_id: Dict[str, int]) -> List[int]:
    """Maps a plain-glyph candidate string (e.g. `"10/21"`) to vocabulary ids.
    Raises `ValueError` naming the first out-of-vocabulary character -- callers
    must pass literal characters a student would write, not LaTeX markup
    (e.g. `"10/21"`, not `"\\frac{10}{21}"`)."""
    ids = []
    for char in candidate:
        if char not in symbol_to_id:
            raise ValueError(
                f"character {char!r} in candidate {candidate!r} is not in the "
                f"model's vocabulary"
            )
        ids.append(symbol_to_id[char])
    return ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_session() -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(
        str(_ONNX_PATH), providers=["CPUExecutionProvider"]
    )


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Converts an RGB (or already-grayscale) crop into the `(1, 1, H, W)`
    float32 batch the model expects, mirroring PyLaia's own training-time
    `ToImageTensor` transform exactly (verified against its real source):
    grayscale, resized to a fixed height of `_TARGET_HEIGHT` keeping the
    original aspect ratio (Lanczos resampling), then **inverted** -- PyLaia
    trains with ink bright and background dark (`ToImageTensor(invert=True)`
    is its default), the opposite of a raw scan -- before scaling to `[0, 1]`.
    Skipping the inversion would silently feed the model upside-down
    contrast and degrade every prediction without erroring."""
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError("preprocess_crop received an empty crop")
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape[:2]
    target_width = max(1, round(width * _TARGET_HEIGHT / height))
    resized = cv2.resize(
        gray, (target_width, _TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4
    )
    inverted = 255 - resized
    normalized = inverted.astype(np.float32) / 255.0
    return normalized[np.newaxis, np.newaxis, :, :]


# ---------------------------------------------------------------------------
# CTC forward algorithm
# ---------------------------------------------------------------------------


def _ctc_log_prob(log_probs: np.ndarray, target_ids: List[int], blank: int = _BLANK_INDEX) -> float:
    """Computes `log P(target_ids | log_probs)` by summing over every valid
    frame-to-label alignment, via the standard CTC forward (alpha-recursion)
    algorithm in log-space.

    `log_probs` is `(T, C)`: one row per timestep, log-softmax over the `C`
    vocabulary symbols (index `blank` is the CTC blank). `target_ids` is the
    candidate label sequence, blank-free.

    The label sequence is expanded to `l' = blank, l[0], blank, l[1], ..., blank`
    (length `2*len(target_ids) + 1`); `alpha[t, s]` is the total log-probability
    of every length-`t+1` alignment that is consistent with `l'[:s+1]` and ends
    on `l'[s]`. Three transitions feed `alpha[t, s]`: staying on `l'[s]`
    (`alpha[t-1, s]`), advancing from `l'[s-1]` (`alpha[t-1, s-1]`), or skipping
    the blank between two *different* non-blank labels (`alpha[t-1, s-2]`) --
    that skip is exactly what's disallowed when `l'[s] == l'[s-2]`, which is
    why a repeated label needs an explicit blank between its two occurrences.
    The final answer sums the two positions a complete alignment could end on:
    the last label, or the trailing blank after it."""
    T = log_probs.shape[0]
    L = len(target_ids)
    ext_len = 2 * L + 1
    ext_labels = [blank] * ext_len
    for i, label in enumerate(target_ids):
        ext_labels[2 * i + 1] = label

    if T == 0:
        return 0.0 if L == 0 else -np.inf

    alpha = np.full((T, ext_len), -np.inf)
    alpha[0, 0] = log_probs[0, blank]
    if ext_len > 1:
        alpha[0, 1] = log_probs[0, ext_labels[1]]

    for t in range(1, T):
        prev = alpha[t - 1]
        for s in range(ext_len):
            acc = prev[s]
            if s - 1 >= 0:
                acc = np.logaddexp(acc, prev[s - 1])
            if s - 2 >= 0 and ext_labels[s] != blank and ext_labels[s] != ext_labels[s - 2]:
                acc = np.logaddexp(acc, prev[s - 2])
            alpha[t, s] = acc + log_probs[t, ext_labels[s]]

    if ext_len == 1:
        return float(alpha[T - 1, 0])
    return float(np.logaddexp(alpha[T - 1, ext_len - 1], alpha[T - 1, ext_len - 2]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_log_probability(crop: np.ndarray, candidate: str) -> float:
    """`log P(candidate | crop)` under the pretrained handwriting model."""
    symbol_to_id, _ = _load_symbols()
    target_ids = _encode_candidate(candidate, symbol_to_id)
    batch = preprocess_crop(crop)
    session = _load_session()
    output = session.run(None, {_INPUT_NAME: batch})[0]  # (T, N, C)
    log_probs = output[:, 0, :]
    return _ctc_log_prob(log_probs, target_ids, blank=_BLANK_INDEX)


def match_probability(crop: np.ndarray, candidate: str) -> float:
    """`P(candidate | crop)`, i.e. `exp(match_log_probability(...))` clipped
    to `[0, 1]` to absorb harmless floating-point overshoot at the edges."""
    return float(np.clip(np.exp(match_log_probability(crop, candidate)), 0.0, 1.0))
