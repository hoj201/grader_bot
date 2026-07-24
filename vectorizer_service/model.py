"""DINOv2 embedding of handwritten student-name crops (issue #46).

Kept separate from graderbot's `Embedder` protocol on purpose: this module is
the only place in the repo that imports torch, so it stays isolated in its
own poetry environment (see pyproject.toml) and is never imported by the main
graderbot app. `modal_app.py` wraps `DinoEmbedder` behind an authenticated
HTTP endpoint; `graderbot.embedding.RemoteEmbedder` is the client for it.
"""

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_MODEL_NAME = "facebook/dinov2-small"


class DinoEmbedder:
    """Loads DINOv2 once and embeds RGB uint8 crops into L2-normalized
    feature vectors (the [CLS] token of the last hidden state)."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str = "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.inference_mode()
    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        pil_images = [Image.fromarray(image) for image in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        normalized = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32)
