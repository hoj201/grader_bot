# vectorizer_service

DINOv2 embedding service for handwritten student-name crops (issue #46).
Deployed separately from the main `graderbot` app so torch and other heavy ML
dependencies never land in graderbot's fly.io deploy. Consumed by
`graderbot.embedding.RemoteEmbedder`.

This is its own poetry project — do not `import` anything from here into
`graderbot`, and don't add `graderbot`'s dependencies here.

**TODO**: this service and its Modal deployment are untested against a live
Modal account/GPU — `vectorizer_service/tests/test_model.py` hasn't been run
(needs `poetry install` here, which pulls real torch/DINOv2 weights) and
`modal deploy`/`modal serve` haven't been exercised. Set up a Modal account
and the `vectorizer-api-key` secret, then run through the "Local development"
and "Deploying" steps below end-to-end.

## Local development

```
cd vectorizer_service
poetry install
poetry run pytest              # model shape/determinism tests (needs a real DINOv2 download)
```

## Running against a real Modal GPU without a full deploy

```
modal serve modal_app.py
```

Prints an ephemeral HTTPS URL. Point graderbot's `.env` at it:

```
VECTORIZER_SERVICE_URL=<the modal serve URL>
VECTORIZER_API_KEY=<matches the vectorizer-api-key Modal secret below>
```

## Deploying

One-time: create the shared-secret Modal secret used for endpoint auth:

```
modal secret create vectorizer-api-key VECTORIZER_API_KEY=<a random value>
```

Then:

```
modal deploy modal_app.py
```

Set the same `VECTORIZER_SERVICE_URL` (the deployed endpoint URL) and
`VECTORIZER_API_KEY` in graderbot's production environment (fly.io secrets).

## Local Docker build (parity check)

```
docker build -t vectorizer-service .
```

`modal_app.py` builds its Modal image from this same `Dockerfile`
(`modal.Image.from_dockerfile`), so this is the same environment Modal runs.

## Migrating the vector collection after a model change

Switching embedders changes the vector dimension. `graderbot.embedding.vectorize_samples`
dedupes/appends by sha and will not detect a dimension change on its own, so after
deploying this service for the first time (or changing `_MODEL_NAME` in `model.py`):

1. Delete/move aside the existing `name_vectors/collection.npz` object on S3.
2. Re-run `vectorize_samples(db_path, embedder=RemoteEmbedder())` against all
   existing samples to rebuild the collection at the new dimension.
3. Retrain the classifier (`graderbot.name_classifier.train_from_collection`).
