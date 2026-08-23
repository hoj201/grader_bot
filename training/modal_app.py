"""Runs response-scorer training (issue #81) on Modal's infra instead of
locally -- this dev machine (Intel Mac, Python 3.13) has no torch wheel
available at all (confirmed directly: `pip install torch` fails outright),
so local training isn't an option the way it is for a Linux/CUDA machine.
Mirrors `easyocr_service/modal_app.py`'s image-build pattern, but this is a
one-off job (`modal run`), not a deployed service (`modal deploy`) -- there
is nothing here that should run 24/7.

One-time setup (per Modal workspace), if not already done for
`easyocr_service`:

    poetry run modal setup

Run training + export in one go, writing the checkpoint and the exported
ONNX model (downloaded back to `models/response_scorer/` in this repo) --
NOT run automatically by anything in this codebase; a human decides when a
training run is worth its compute cost:

    poetry run modal run training/modal_app.py --steps 20000

Add `--gpu` for a GPU container (faster, costs more) -- CPU is the default
since this model is small enough (well under 1M parameters) that CPU
training is plausible for a smoke run.
"""

import pathlib
import sys

import modal

_TRAINING_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _TRAINING_DIR.parent

if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")  # cv2 runtime libs, same as easyocr_service
    .pip_install_from_requirements(str(_TRAINING_DIR / "requirements.txt"))
    .add_local_python_source("model", "dataset", "train", "export_onnx")
    # graderbot itself (answer_glyph_synth, response_candidates,
    # response_scorer, ...) is imported by dataset.py/eval.py -- ship the
    # whole package into the image rather than trying to cherry-pick files.
    .add_local_dir(str(_REPO_ROOT / "graderbot"), remote_path="/root/graderbot")
    .add_local_dir(str(_REPO_ROOT / "fonts"), remote_path="/root/fonts")
)

app = modal.App("graderbot-response-scorer-training", image=image)

# Persists checkpoints across runs/containers so a training run can be
# resumed or re-exported without starting over -- same rationale as
# easyocr_service's model-weights Volume.
_checkpoint_volume = modal.Volume.from_name("response-scorer-checkpoints", create_if_missing=True)


@app.function(
    volumes={"/checkpoints": _checkpoint_volume},
    timeout=6 * 60 * 60,
)
def run_training(steps: int = 20000, batch_size: int = 64, lr: float = 1e-3) -> bytes:
    import train
    import export_onnx
    from pathlib import Path

    checkpoint_path = Path("/checkpoints/checkpoint.pt")
    train.train(steps, batch_size, checkpoint_path, lr)
    _checkpoint_volume.commit()

    onnx_dir = Path("/checkpoints/response_scorer")
    onnx_path = export_onnx.export(checkpoint_path, onnx_dir)
    _checkpoint_volume.commit()
    return onnx_path.read_bytes()


@app.local_entrypoint()
def main(steps: int = 20000, batch_size: int = 64, lr: float = 1e-3):
    """Runs training remotely and writes the exported ONNX model back into
    this repo's `models/response_scorer/weights.onnx` (and copies
    `vocab.json` alongside it) so `CnnResponseScorer` can pick it up
    immediately."""
    from graderbot.response_scorer import VOCAB
    import json

    weights_bytes = run_training.remote(steps, batch_size, lr)

    out_dir = _REPO_ROOT / "models" / "response_scorer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.onnx").write_bytes(weights_bytes)
    (out_dir / "vocab.json").write_text(json.dumps({"vocab": VOCAB}))
    print(f"wrote {out_dir / 'weights.onnx'}")
