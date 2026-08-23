"""Modal deployment for the EasyOCR sidecar (issue #70).

Serves the exact same FastAPI app as `main.py` (imported, not reimplemented)
on Modal's infra instead of a self-hosted container, so it's reachable in
production without running Docker somewhere 24/7. `docker compose up easyocr`
remains the local-dev path -- see main.py's docstring; both paths build the
same CPU-only-torch + EasyOCR image.

One-time setup (per Modal workspace) -- see the README for the fuller
walkthrough, in particular why the key has to be captured into a shell
variable first rather than generated inline (`modal secret create` sends it
to Modal but never prints it back out for you to reuse in `.env`):

    poetry run modal setup                                  # browser auth
    EASYOCR_API_KEY=$(openssl rand -hex 32); echo "$EASYOCR_API_KEY"
    poetry run modal secret create easyocr-api-key EASYOCR_API_KEY=$EASYOCR_API_KEY

Deploy / redeploy:

    poetry run modal deploy easyocr_service/modal_app.py

That prints a URL ending in `-web.modal.run`. Set in `.env` (and wherever the
Streamlit app actually runs):

    EASYOCR_SERVICE_URL=https://<printed-url>
    EASYOCR_API_KEY=<same value you put in the easyocr-api-key secret>

EasyOcrAnswerReader sends EASYOCR_API_KEY as an X-Api-Key header; main.py's
_check_api_key rejects requests missing/mismatching it once the secret is
present in the container's environment (i.e. only on Modal -- the local
docker-compose container never gets this secret, so it stays unauthenticated
there, which is fine since it's only reachable on localhost).
"""

import pathlib
import sys

import modal

# `pip_install_from_requirements` resolves its path relative to the process's
# CWD at deploy time, not this file's location -- resolve it ourselves so
# `modal deploy easyocr_service/modal_app.py` works the same from any CWD
# (the README/docstring above both invoke it from the repo root).
_SERVICE_DIR = pathlib.Path(__file__).parent

# `add_local_python_source("main")` below needs to find main.py via sys.path
# at definition time -- make sure this directory is on it regardless of the
# CWD `modal deploy` was invoked from.
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

image = (
    modal.Image.debian_slim(python_version="3.13")
    # libgl1/libglib2.0-0: runtime libs cv2 needs even in headless-ish
    # EasyOCR use (same as the Dockerfile).
    .apt_install("libgl1", "libglib2.0-0")
    # CPU-only torch/torchvision from PyTorch's own wheel index -- this
    # service always runs with gpu=False (see main.py), and the default
    # PyPI torch wheel drags in several GB of unused CUDA libraries.
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(_SERVICE_DIR / "requirements.txt"))
    .add_local_python_source("main")
)

app = modal.App("graderbot-easyocr", image=image)

# Caches EasyOCR's downloaded model weights (~70MB) across cold starts. Model
# *loading* is already deferred to first request (see main.py's _get_reader),
# but without this Volume every scale-from-zero container would also pay a
# fresh model *download* on top of that, every single time.
_model_cache = modal.Volume.from_name("easyocr-models", create_if_missing=True)


@app.function(
    volumes={"/root/.EasyOCR": _model_cache},
    secrets=[modal.Secret.from_name("easyocr-api-key")],
    scaledown_window=300,
    timeout=60,
)
@modal.asgi_app()
def web():
    from main import app as fastapi_app

    return fastapi_app
