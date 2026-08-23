"""Trains graderbot's response-scorer CRNN (issue #81) with CTC loss on
synthetic answer-box crops (`graderbot.answer_glyph_synth`) plus deliberate
near-misses (`graderbot.response_candidates`), and saves a checkpoint
`export_onnx.py` turns into the `graderbot/models/response_scorer/weights.onnx`
artifact `graderbot.response_scorer.CnnResponseScorer` loads. Torch-only --
run inside this directory's own venv (`requirements.txt`), never inside the
main project's (see `model.py`'s docstring for why).

    cd training && pip install -r requirements.txt
    python train.py --steps 20000 --out checkpoint.pt

Or on Modal (`modal_app.py` mirrors `easyocr_service/modal_app.py`'s
pattern) for compute this repo's own dev machine (Intel Mac, no torch wheel
available at all) can't provide locally.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ResponseGlyphDataset
from model import ResponseScorerCRNN


def _collate(batch):
    """Right-pads every crop in the batch to the widest one's width, with
    -1.0 (the "blank/white" end of `prepare_crop_for_model`'s `[-1, 1]`
    normalization, not 0 which sits mid-range on that scale) -- so a
    narrower sample's tail is genuinely background, not a mid-gray smear
    the conv stack would have to learn to ignore."""
    images, labels, texts = zip(*batch)
    height = images[0].shape[1]
    max_width = max(img.shape[-1] for img in images)
    padded = torch.full((len(images), 1, height, max_width), -1.0)
    for i, img in enumerate(images):
        padded[i, :, :, : img.shape[-1]] = img
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat([torch.tensor(l, dtype=torch.long) for l in labels])
    return padded, targets, label_lengths, texts


def train(steps: int, batch_size: int, out_path: Path, lr: float = 1e-3, log_every: int = 200) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResponseScorerCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # zero_infinity: a batch containing a candidate longer than the padded
    # width allows (T < label length) would otherwise blow up the whole
    # batch's loss to inf/nan; this drops just that sample's contribution
    # instead, which matters more here than usual since hard-negative
    # candidates can occasionally run a character or two longer than the
    # true answer they're a near-miss of.
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    loader = DataLoader(ResponseGlyphDataset(), batch_size=batch_size, collate_fn=_collate)

    model.train()
    for step, (images, targets, label_lengths, texts) in enumerate(loader):
        if step >= steps:
            break
        images = images.to(device)
        log_probs = model(images)  # (N, T, C)
        input_lengths = torch.full((images.shape[0],), log_probs.shape[1], dtype=torch.long)

        loss = ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, label_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"step {step}/{steps}  loss {loss.item():.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("checkpoint.pt"))
    args = parser.parse_args()
    train(args.steps, args.batch_size, args.out, args.lr)
