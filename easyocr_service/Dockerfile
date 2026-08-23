FROM python:3.13-slim

# libgl1/libglib2.0-0: runtime libs cv2 needs even in headless-ish EasyOCR use
# (EasyOCR's own opencv-python import fails without them on a slim base).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch/torchvision from PyTorch's own CPU wheel index -- this
# service always runs with gpu=False (see main.py), and the default PyPI
# torch wheel drags in several GB of CUDA libraries that would never be
# used, just to sit in the image. Installed before requirements.txt so pip
# sees the constraint already satisfied and doesn't pull the CUDA build.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py test_main.py ./

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
