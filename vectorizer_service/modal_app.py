"""Modal deployment of the DINOv2 vectorizer service (issue #46).

    modal serve vectorizer_service/modal_app.py   # local dev, real GPU
    modal deploy vectorizer_service/modal_app.py  # persistent endpoint

The image is built from this directory's own Dockerfile so local Docker
builds and the Modal-hosted container stay identical (one source of truth
for the environment, per the issue #46 decision to keep this fully separate
from the main graderbot app/deploy).
"""

import base64
import os

import modal
import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse

app = modal.App("graderbot-vectorizer")

image = modal.Image.from_dockerfile("Dockerfile")


@app.cls(image=image, gpu="T4", secrets=[modal.Secret.from_name("vectorizer-api-key")])
class Vectorizer:
    @modal.enter()
    def load_model(self):
        from model import DinoEmbedder

        self.embedder = DinoEmbedder(device="cuda")

    @modal.fastapi_endpoint(method="POST")
    async def embed(self, request: Request):
        import cv2

        api_key = request.headers.get("X-Api-Key")
        if not api_key or api_key != os.environ["VECTORIZER_API_KEY"]:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)

        body = await request.json()
        images = []
        for encoded in body.get("images", []):
            raw = base64.b64decode(encoded)
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        vectors = self.embedder.embed(images)
        return JSONResponse({"vectors": vectors.tolist(), "dim": self.embedder.dim})
