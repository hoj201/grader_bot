import pytest

from graderbot.storage import _default_s3_client


@pytest.fixture(autouse=True)
def _reset_default_s3_client_cache():
    """`storage._default_s3_client` is process-wide `@lru_cache`'d (issue:
    Label Names tab slowness -- constructing a fresh boto3 client on every
    call was the dominant per-click cost). Without this, a client built
    under one test's `@mock_aws` context (or a monkeypatched AWS_REGION)
    could leak into a later, unrelated test via the cache."""
    _default_s3_client.cache_clear()
    yield
    _default_s3_client.cache_clear()
