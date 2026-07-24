# CUDA-enabled base since this runs on a Modal T4 GPU; also used unchanged
# for local `docker build`/dev parity (see modal_app.py).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN pip install --no-cache-dir poetry==1.8.3 \
    && poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root --no-interaction

COPY . .
