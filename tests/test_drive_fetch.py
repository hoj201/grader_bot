"""Tests for graderbot.drive_fetch (issue #102): Drive file ID parsing, OAuth
access-token refresh, and file download -- all mocked at the `requests`
boundary, same style as tests/test_answer_reader.py. No real Google API call
is needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from graderbot import drive_fetch
from graderbot.drive_fetch import (
    DriveAccessError,
    DriveAuthError,
    DriveFile,
    DriveFileTooLargeError,
    DriveUnsupportedFileError,
    InvalidDriveUrlError,
    extract_file_id,
    fetch_drive_file,
    get_access_token,
)

FILE_ID = "1AbCDefGhijKLmnop"


# --- extract_file_id --------------------------------------------------------

def test_extract_file_id_from_file_d_view_url():
    url = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"
    assert extract_file_id(url) == FILE_ID


def test_extract_file_id_from_open_id_url():
    url = f"https://drive.google.com/open?id={FILE_ID}"
    assert extract_file_id(url) == FILE_ID


def test_extract_file_id_from_uc_id_url():
    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    assert extract_file_id(url) == FILE_ID


def test_extract_file_id_from_bare_id():
    assert extract_file_id(FILE_ID) == FILE_ID


def test_extract_file_id_strips_whitespace():
    assert extract_file_id(f"  {FILE_ID}  ") == FILE_ID


def test_extract_file_id_raises_on_garbage():
    with pytest.raises(InvalidDriveUrlError):
        extract_file_id("not a url")


def test_extract_file_id_raises_on_empty_string():
    with pytest.raises(InvalidDriveUrlError):
        extract_file_id("   ")


# --- get_access_token --------------------------------------------------------

def test_get_access_token_requires_env_vars(monkeypatch):
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_REFRESH_TOKEN", raising=False)

    with pytest.raises(EnvironmentError):
        get_access_token()


def test_get_access_token_uses_env_vars_when_args_omitted(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "env-cid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "env-csecret")
    monkeypatch.setenv("GOOGLE_OAUTH_REFRESH_TOKEN", "env-rtoken")
    response = MagicMock(status_code=200)
    response.json.return_value = {"access_token": "tok123"}

    with patch("graderbot.drive_fetch.requests.post", return_value=response) as mock_post:
        token = get_access_token()

    assert token == "tok123"
    assert mock_post.call_args.kwargs["data"] == {
        "client_id": "env-cid",
        "client_secret": "env-csecret",
        "refresh_token": "env-rtoken",
        "grant_type": "refresh_token",
    }


def test_get_access_token_posts_to_token_endpoint_and_returns_token():
    response = MagicMock(status_code=200)
    response.json.return_value = {"access_token": "tok123"}

    with patch("graderbot.drive_fetch.requests.post", return_value=response) as mock_post:
        token = get_access_token(client_id="cid", client_secret="csecret", refresh_token="rtoken")

    assert token == "tok123"
    args, kwargs = mock_post.call_args
    assert args[0] == "https://oauth2.googleapis.com/token"
    assert kwargs["data"] == {
        "client_id": "cid",
        "client_secret": "csecret",
        "refresh_token": "rtoken",
        "grant_type": "refresh_token",
    }


def test_get_access_token_raises_drive_auth_error_on_non_200():
    response = MagicMock(status_code=400)
    response.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Token has been expired or revoked.",
    }

    with patch("graderbot.drive_fetch.requests.post", return_value=response):
        with pytest.raises(DriveAuthError, match="invalid_grant"):
            get_access_token(client_id="cid", client_secret="csecret", refresh_token="rtoken")


# --- fetch_drive_file --------------------------------------------------------

def _metadata_response(name="scan.pdf", mime_type="application/pdf", size=None, status_code=200):
    response = MagicMock(status_code=status_code)
    body = {"name": name, "mimeType": mime_type}
    if size is not None:
        body["size"] = str(size)
    response.json.return_value = body
    return response


def _content_response(content=b"%PDF-1.4 fake", status_code=200):
    return MagicMock(status_code=status_code, content=content)


def test_fetch_drive_file_downloads_bytes_on_success():
    meta = _metadata_response()
    content = _content_response()

    with patch("graderbot.drive_fetch.requests.get", side_effect=[meta, content]) as mock_get:
        result = fetch_drive_file(FILE_ID, "tok123")

    assert result == DriveFile(file_id=FILE_ID, filename="scan.pdf", content=b"%PDF-1.4 fake")
    for call in mock_get.call_args_list:
        assert call.kwargs["headers"] == {"Authorization": "Bearer tok123"}


def test_fetch_drive_file_raises_invalid_url_before_any_request():
    with patch("graderbot.drive_fetch.requests.get") as mock_get:
        with pytest.raises(InvalidDriveUrlError):
            fetch_drive_file("not a url", "tok123")
    mock_get.assert_not_called()


def test_fetch_drive_file_raises_access_error_on_404():
    meta = _metadata_response(status_code=404)
    meta.json.return_value = {"error": {"message": "File not found"}}

    with patch("graderbot.drive_fetch.requests.get", return_value=meta):
        with pytest.raises(DriveAccessError, match="File not found"):
            fetch_drive_file(FILE_ID, "tok123")


def test_fetch_drive_file_raises_access_error_on_403():
    meta = _metadata_response(status_code=403)
    meta.json.return_value = {"error": {"message": "The user does not have sufficient permissions"}}

    with patch("graderbot.drive_fetch.requests.get", return_value=meta):
        with pytest.raises(DriveAccessError, match="sufficient permissions"):
            fetch_drive_file(FILE_ID, "tok123")


def test_fetch_drive_file_raises_auth_error_on_401():
    meta = _metadata_response(status_code=401)
    meta.json.return_value = {"error": {"message": "Invalid Credentials"}}

    with patch("graderbot.drive_fetch.requests.get", return_value=meta):
        with pytest.raises(DriveAuthError, match="401"):
            fetch_drive_file(FILE_ID, "tok123")


def test_fetch_drive_file_raises_unsupported_for_google_native_mimetype():
    meta = _metadata_response(name="Essay", mime_type="application/vnd.google-apps.document")

    with patch("graderbot.drive_fetch.requests.get", return_value=meta) as mock_get:
        with pytest.raises(DriveUnsupportedFileError, match="Essay"):
            fetch_drive_file(FILE_ID, "tok123")
    # No point downloading alt=media for a file type that can't be served
    # that way -- only the metadata call should have happened.
    assert mock_get.call_count == 1


def test_fetch_drive_file_raises_too_large():
    meta = _metadata_response(size=drive_fetch.MAX_DOWNLOAD_BYTES + 1)

    with patch("graderbot.drive_fetch.requests.get", return_value=meta) as mock_get:
        with pytest.raises(DriveFileTooLargeError):
            fetch_drive_file(FILE_ID, "tok123")
    assert mock_get.call_count == 1


def test_fetch_drive_file_allows_file_at_the_size_limit():
    meta = _metadata_response(size=drive_fetch.MAX_DOWNLOAD_BYTES)
    content = _content_response()

    with patch("graderbot.drive_fetch.requests.get", side_effect=[meta, content]):
        result = fetch_drive_file(FILE_ID, "tok123")

    assert result.content == b"%PDF-1.4 fake"
