"""Exports a trained checkpoint (`train.py`'s output) to the ONNX graph
`graderbot.response_scorer.CnnResponseScorer` loads
(`models/response_scorer/weights.onnx`), plus the `vocab.json` describing
its label alphabet. Also runs a numerical-parity check -- the exported
ONNX graph's output against the source torch model's, on random sample
input -- mirroring the ~2e-5 max-diff check the paused `handwriting-ctc-match`
branch already validated this export path with (see the
`handwriting-ctc-match-paused` memory note); raises if the two disagree by
more than a loose tolerance, so a broken export is never silently shipped.

    cd training && python export_onnx.py checkpoint.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import VOCAB, ResponseScorerCRNN

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT_DIR = _REPO_ROOT / "models" / "response_scorer"
_PARITY_TOLERANCE = 1e-3


def export(checkpoint_path: Path, out_dir: Path = _DEFAULT_OUT_DIR, sample_width: int = 240) -> Path:
    model = ResponseScorerCRNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "weights.onnx"
    dummy = torch.zeros(1, 1, 32, sample_width)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["log_probs"],
        # Batch size and crop width must stay dynamic -- CnnResponseScorer
        # always calls with batch size 1, but crop width varies with every
        # answer box's own aspect ratio.
        dynamic_axes={"input": {0: "batch", 3: "width"}, "log_probs": {0: "batch", 1: "time"}},
        opset_version=17,
    )
    (out_dir / "vocab.json").write_text(json.dumps({"vocab": VOCAB}))

    _check_parity(model, onnx_path, sample_width)
    print(f"exported {onnx_path}")
    return onnx_path


def _check_parity(model: torch.nn.Module, onnx_path: Path, sample_width: int) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    session = ort.InferenceSession(str(onnx_path))
    max_diff = 0.0
    for _ in range(5):
        sample = rng.uniform(-1, 1, size=(1, 1, 32, sample_width)).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(sample)).numpy()
        (onnx_out,) = session.run(None, {"input": sample})
        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    print(f"onnx/torch max abs diff over 5 samples: {max_diff:.2e}")
    if max_diff > _PARITY_TOLERANCE:
        raise RuntimeError(
            f"ONNX export parity check failed: max diff {max_diff:.2e} exceeds "
            f"tolerance {_PARITY_TOLERANCE:.0e} -- do not ship this export."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()
    export(args.checkpoint, args.out_dir)
