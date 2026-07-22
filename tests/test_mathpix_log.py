import json

import boto3
import pytest
from moto import mock_aws

from graderbot.mathpix_log import log_mathpix_call
from graderbot.storage import init_db

_PNG_BYTES = b"\x89PNG\r\n\x1a\n-fake-png-bytes"
_RESPONSE = {"text": "$12$", "confidence": 0.99, "request_id": "abc"}


def _clear_bucket_env(monkeypatch):
    monkeypatch.delenv("MATHPIX_LOG_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)


def test_log_mathpix_call_is_noop_when_no_bucket(monkeypatch, tmp_path):
    _clear_bucket_env(monkeypatch)
    monkeypatch.setenv("WORKSHEETS_DB_PATH", str(tmp_path / "worksheets.sqlite3"))

    assert log_mathpix_call(_PNG_BYTES, _RESPONSE, "12") is None
    # Nothing should have been written to disk.
    assert not (tmp_path / "worksheets.sqlite3").exists()


@mock_aws
def test_log_mathpix_call_uploads_image_and_records_row(monkeypatch, tmp_path):
    _clear_bucket_env(monkeypatch)
    bucket = "mathpix-logs"
    db_path = tmp_path / "worksheets.sqlite3"
    monkeypatch.setenv("MATHPIX_LOG_BUCKET", bucket)
    monkeypatch.setenv("WORKSHEETS_DB_PATH", str(db_path))

    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=bucket)

    row_id = log_mathpix_call(_PNG_BYTES, _RESPONSE, "12", s3_client=s3_client)
    assert row_id is not None

    conn = init_db(db_path)
    rows = conn.execute(
        "SELECT image_s3url, image_sha256, response_json, response_text FROM MATHPIX_CALL"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    image_s3url, image_sha256, response_json, response_text = rows[0]
    assert image_s3url == f"https://{bucket}.s3.amazonaws.com/mathpix/{image_sha256}.png"
    assert response_text == "12"
    assert json.loads(response_json) == _RESPONSE

    # The exact bytes should be retrievable from S3 under the sha256 key.
    obj = s3_client.get_object(Bucket=bucket, Key=f"mathpix/{image_sha256}.png")
    assert obj["Body"].read() == _PNG_BYTES


def test_log_mathpix_call_is_non_fatal_on_failure(monkeypatch, tmp_path):
    _clear_bucket_env(monkeypatch)
    monkeypatch.setenv("MATHPIX_LOG_BUCKET", "mathpix-logs")
    monkeypatch.setenv("WORKSHEETS_DB_PATH", str(tmp_path / "worksheets.sqlite3"))

    class _BoomClient:
        def put_object(self, **kwargs):
            raise RuntimeError("s3 is down")

    with pytest.warns(UserWarning, match="Failed to log Mathpix call"):
        result = log_mathpix_call(_PNG_BYTES, _RESPONSE, "12", s3_client=_BoomClient())
    assert result is None


def test_init_db_creates_mathpix_call_table(tmp_path):
    conn = init_db(tmp_path / "worksheets.sqlite3")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(MATHPIX_CALL)")}
    conn.close()

    assert columns == {
        "id",
        "image_s3url",
        "image_sha256",
        "response_json",
        "response_text",
        "created_at",
    }
