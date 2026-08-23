"""Evaluates a trained response-scorer checkpoint (`train.py`'s output,
before or after `export_onnx.py`) the way
`graderbot.name_classifier.loo_cross_validate` evaluates the handwriting
classifier: reports overall and per-answer-type accuracy on freshly
generated synthetic held-out data, so a bad model is caught before it's ever
offered as an option in the Grade tab.

    cd training && python eval.py checkpoint.pt

Real-scan evaluation -- comparing this synthetic-only accuracy against
labeled real crops, the number issue #73's paused PyLaia spike lacked and
that would have caught its domain mismatch sooner -- needs the
`HANDWRITING_LABEL` storage table and labeling pass from issue #81's "Data
pipeline" section. That table doesn't exist yet, so this only reports the
synthetic side for now; extending this script to also read real labels is
the natural next step once there's a labeled corpus to read.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from dataset import _SAMPLE_ANSWER_POOL
from model import ResponseScorerCRNN

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graderbot.answer_glyph_synth import generate_training_sample  # noqa: E402
from graderbot.response_candidates import generate_candidates  # noqa: E402
from graderbot.response_scorer import prepare_crop_for_model, score_text_candidates  # noqa: E402


def _answer_type(answer: str) -> str:
    if "." in answer:
        return "decimal"
    if answer.startswith("-"):
        return "negative_integer"
    return "integer"


def evaluate(checkpoint_path: Path, n_samples: int = 500, seed: int = 0) -> Dict[str, object]:
    model = ResponseScorerCRNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(seed)
    correct = 0
    by_type = defaultdict(lambda: [0, 0])  # answer_type -> [n_correct, n_total]

    for _ in range(n_samples):
        answer = str(rng.choice(_SAMPLE_ANSWER_POOL))
        candidates = generate_candidates(answer, rng=rng)
        image = generate_training_sample(answer, rng=rng)
        network_input = prepare_crop_for_model(image)[0]

        with torch.no_grad():
            log_probs = model(torch.from_numpy(network_input).unsqueeze(0)).numpy()[0]
        result = score_text_candidates(log_probs, candidates)

        answer_type = _answer_type(answer)
        by_type[answer_type][1] += 1
        if result.best == answer:
            correct += 1
            by_type[answer_type][0] += 1

    return {
        "n_samples": n_samples,
        "accuracy": correct / n_samples,
        "per_type_accuracy": {t: n_correct / n_total for t, (n_correct, n_total) in by_type.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.n_samples)
    print(f"accuracy: {report['accuracy']:.1%} (n={report['n_samples']})")
    for answer_type, accuracy in sorted(report["per_type_accuracy"].items()):
        print(f"  {answer_type}: {accuracy:.1%}")
