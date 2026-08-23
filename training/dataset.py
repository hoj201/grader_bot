"""Synthetic training data for the response-scorer CRNN (issue #81): wraps
`graderbot.answer_glyph_synth`'s fast renderer, generating an endless stream
of `(crop, text)` pairs -- the true answer text most of the time, and a hard
negative from `graderbot.response_candidates` the rest, so the model gets
explicit signal on the confusions it's meant to resolve rather than only
implicit signal from correct-reads-only supervision.

`graderbot.answer_glyph_synth`/`graderbot.response_candidates`/
`graderbot.response_scorer` are themselves plain numpy/cv2 code with no
torch dependency, so they import cleanly here despite `training/` being a
torch-only venv distinct from the main project's (see `model.py`'s
docstring) -- this file is the one place in `training/` that needs both
worlds at once, which is why `training/requirements.txt` also carries
`graderbot`'s own (non-torch) dependencies.
"""

import random
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from graderbot.answer_glyph_synth import generate_training_sample
from graderbot.response_candidates import SAMPLE_ANSWER_POOL, generate_candidates
from graderbot.response_scorer import VOCAB, prepare_crop_for_model

# Kept as a local alias (not a re-export) so eval.py's existing `from
# dataset import _SAMPLE_ANSWER_POOL` keeps working -- the canonical
# definition lives in graderbot.response_candidates.SAMPLE_ANSWER_POOL, also
# shared by handwriting_sample_worksheets.py's real-data harvesting (issue
# #81), so the synthetic and real halves of training data cover the same
# answer shapes.
_SAMPLE_ANSWER_POOL: List[str] = list(SAMPLE_ANSWER_POOL)

# Fraction of samples drawn from a near-miss (response_candidates) instead
# of the true answer -- teaches the model to actually distinguish "9" from
# "4", not just to recognize "9" whenever it sees a 9.
_HARD_NEGATIVE_RATE = 0.4


def encode_text(text: str, vocab: str = VOCAB) -> List[int]:
    """Maps `text` to CTC label indices, offset by 1 for the reserved blank
    at index 0. Mirrors `graderbot.response_scorer._encode` (a private
    helper there, so duplicated rather than imported)."""
    return [vocab.index(ch) + 1 for ch in text]


class ResponseGlyphDataset(IterableDataset):
    """Infinite stream of `(image, label_indices, text)` examples. `image`
    is a `(1, INPUT_HEIGHT, W)` float32 tensor
    (`prepare_crop_for_model`'s output with the batch dim dropped);
    `label_indices` is the CTC target; `text` is kept alongside for
    logging/eval, not used in training itself."""

    def __init__(
        self,
        answers: Optional[List[str]] = None,
        hard_negative_rate: float = _HARD_NEGATIVE_RATE,
        seed: Optional[int] = None,
    ):
        self.answers = answers if answers is not None else list(_SAMPLE_ANSWER_POOL)
        self.hard_negative_rate = hard_negative_rate
        self._seed = seed

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, List[int], str]]:
        # Each worker (DataLoader's num_workers > 0) gets its own iterator
        # call, so re-seeding here (rather than once in __init__) is what
        # keeps workers from all drawing the identical sequence of samples.
        seed = self._seed if self._seed is not None else random.SystemRandom().randrange(2**32)
        rng = np.random.default_rng(seed)
        py_random = random.Random(seed)
        while True:
            answer = py_random.choice(self.answers)
            if py_random.random() < self.hard_negative_rate:
                candidates = generate_candidates(answer, rng=rng)
                text = py_random.choice(candidates[1:]) if len(candidates) > 1 else answer
            else:
                text = answer
            image = generate_training_sample(text, rng=rng)
            network_input = prepare_crop_for_model(image)[0]  # drop batch dim -> (1, H, W)
            yield torch.from_numpy(network_input), encode_text(text), text
