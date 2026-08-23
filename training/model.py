"""Small CRNN for issue #81's closed-set response verifier: a few conv+pool
blocks collapse a `(1, INPUT_HEIGHT, W)` crop down to a sequence of column
features, a bidirectional LSTM reads that sequence, and a linear +
log-softmax head emits one (blank + digit/`.`/`-`) log-probability
distribution per timestep -- the `(T, C)` grid
`graderbot.response_scorer.ctc_log_prob` scores candidate strings against.
Well under 1M parameters, trained with CTC loss (`train.py`) and exported to
ONNX (`export_onnx.py`) so the *trained* model can be used without torch at
all (see `graderbot/response_scorer.py`'s module docstring).

Kept in this isolated `training/` directory rather than `graderbot/` itself
-- same reasoning as `easyocr_service/`: torch has no Intel-macOS wheel for
this project's Python version, so it must never become a
`pyproject.toml` dependency of the main project. Training runs offline
(locally on a machine with a torch wheel, or on Modal -- see
`modal_app.py`), never at grading time.
"""

import torch
import torch.nn as nn

# Must match graderbot/response_scorer.py's VOCAB + MODEL_INPUT_HEIGHT --
# duplicated (not imported) so this file has no import-time dependency on
# graderbot's own dependencies (fitz, pytesseract, ...) beyond what's
# actually needed to define the network; dataset.py is the one file here
# that needs both worlds and pays that cost.
VOCAB = "0123456789.-"
NUM_CLASSES = len(VOCAB) + 1  # +1 for the CTC blank at index 0
INPUT_HEIGHT = 32


class ResponseScorerCRNN(nn.Module):
    """`x`: `(N, 1, INPUT_HEIGHT, W)` grayscale, normalized to `[-1, 1]`
    (`graderbot.response_scorer.prepare_crop_for_model`'s output). Returns
    `(N, T, NUM_CLASSES)` log-probabilities; `T` is `W` downsampled 4x by
    the conv stack's two width-halving pools."""

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # H,W: 32,W -> 16,W/2
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 8,W/4
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),  # -> 4,W/4 (width untouched)
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (4, 3), padding=(0, 1)), nn.ReLU(),  # collapse height 4 -> 1
        )
        self.rnn = nn.LSTM(64, rnn_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.head = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # (N, C, 1, T)
        features = features.squeeze(2).permute(0, 2, 1)  # (N, T, C)
        rnn_out, _ = self.rnn(features)  # (N, T, 2*rnn_hidden)
        logits = self.head(rnn_out)  # (N, T, num_classes)
        return torch.log_softmax(logits, dim=-1)
