"""Fetch bytes for restricted-access Google Drive files (issue #102).

Single-user by design: the app owner runs `scripts/google_drive_oauth_setup.py`
once, locally, to complete an interactive OAuth consent flow and print a
long-lived refresh token (`GOOGLE_OAUTH_REFRESH_TOKEN`). Every grading run
then silently exchanges that refresh token for a short-lived access token and
downloads whichever Drive files were pasted into the Grade tab -- no
per-visitor "Sign in with Google" flow, no per-user token storage.

Like `GoogleVisionAnswerReader` (see `graderbot/answer_reader.py`), this talks
to Google directly over `requests` -- no `google-api-python-client`/
`google-auth` runtime dependency. (The one-time setup script is the only
place `google-auth-oauthlib` is needed, and that's a dev-only dependency.)
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files/{file_id}"

_CLIENT_ID_ENV = "GOOGLE_OAUTH_CLIENT_ID"
_CLIENT_SECRET_ENV = "GOOGLE_OAUTH_CLIENT_SECRET"
_REFRESH_TOKEN_ENV = "GOOGLE_OAUTH_REFRESH_TOKEN"

# Native Google Docs/Sheets/Slides/Drawings have no fixed binary -- they can
# only be read via files.export (with a target mimeType), not alt=media.
# Recognizing this mimeType prefix up front lets fetch_drive_file raise a
# specific, actionable error instead of a confusing 403 from the Drive API.
_GOOGLE_NATIVE_MIME_PREFIX = "application/vnd.google-apps."

# A generous cap, well above any multi-page scanned worksheet, so a bad link
# can't hang the app streaming an enormous file into memory (fetch_drive_file
# loads the whole response body at once, same as st.file_uploader's
# .getvalue() already does for uploads).
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024  # 30 MB

# Order matters: URL-shaped patterns are tried before treating the whole
# input as a bare ID, so a malformed URL doesn't accidentally get "matched"
# by grabbing a nonsense substring.
_FILE_ID_URL_PATTERNS = [
    re.compile(r"/file/d/([a-zA-Z0-9_-]{10,})"),  # .../file/d/<id>/view
    re.compile(r"/document/d/([a-zA-Z0-9_-]{10,})"),  # a Docs share URL
    re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})"),  # ...?id=<id> (open?id=, uc?id=)
]
_BARE_FILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{10,}$")


class DriveFetchError(Exception):
    """Base class for all Google Drive fetch failures. Always constructed
    with a message that's already safe to show verbatim (e.g. via
    Streamlit's st.error)."""


class InvalidDriveUrlError(DriveFetchError):
    """Could not find a Drive file ID in the given string."""


class DriveAuthError(DriveFetchError):
    """The stored OAuth credentials are missing/invalid/expired -- almost
    always fixed by re-running scripts/google_drive_oauth_setup.py."""


class DriveAccessError(DriveFetchError):
    """The Drive API rejected a request for a specific file: not found, or
    not shared with the authenticated account."""


class DriveUnsupportedFileError(DriveFetchError):
    """The file is a native Google Doc/Sheet/Slide/Drawing, which has no
    fixed binary to download via `alt=media`."""


class DriveFileTooLargeError(DriveFetchError):
    """The file is bigger than MAX_DOWNLOAD_BYTES."""


@dataclass
class DriveFile:
    file_id: str
    filename: str
    content: bytes


def extract_file_id(url_or_id: str) -> str:
    """Parse a Google Drive file ID out of a pasted URL or bare ID.

    Raises InvalidDriveUrlError (naming the offending input) if nothing
    matches.
    """
    candidate = (url_or_id or "").strip()
    if not candidate:
        raise InvalidDriveUrlError("Empty Google Drive link.")
    for pattern in _FILE_ID_URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    if _BARE_FILE_ID_PATTERN.match(candidate):
        return candidate
    raise InvalidDriveUrlError(f"Could not find a Google Drive file ID in {url_or_id!r}.")


def _google_error_detail(response: requests.Response) -> str:
    """Best-effort pull of Google's JSON error body into a short string;
    falls back to the raw response text if the body isn't JSON."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    if "error_description" in body:  # token endpoint's error shape
        return f"{body.get('error', '')}: {body['error_description']}".strip(": ")
    error = body.get("error")
    if isinstance(error, dict):  # Drive API's error shape
        return error.get("message", str(error))
    if error is not None:
        return str(error)
    return response.text


def get_access_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
    timeout: float = 15.0,
) -> str:
    """Exchange the stored refresh token for a short-lived access token via
    POST https://oauth2.googleapis.com/token.

    Reads GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET/
    GOOGLE_OAUTH_REFRESH_TOKEN from the environment when args are omitted
    (same arg-or-env-var convention as GoogleVisionAnswerReader).

    Raises EnvironmentError if any of the three are unset -- the same
    exception type app.py already catches for GoogleVisionAnswerReader, so
    render_grade() can handle both the same way. Raises DriveAuthError if
    the token endpoint rejects the refresh token (expired/revoked/bad
    client secret).
    """
    client_id = client_id or os.environ.get(_CLIENT_ID_ENV)
    client_secret = client_secret or os.environ.get(_CLIENT_SECRET_ENV)
    refresh_token = refresh_token or os.environ.get(_REFRESH_TOKEN_ENV)
    if not (client_id and client_secret and refresh_token):
        raise EnvironmentError(
            f"{_CLIENT_ID_ENV}, {_CLIENT_SECRET_ENV}, and {_REFRESH_TOKEN_ENV} must "
            "all be set (e.g. in a .env file) to grade from Google Drive links. "
            "Run scripts/google_drive_oauth_setup.py once to generate them."
        )
    response = requests.post(
        _TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise DriveAuthError(
            "Could not refresh the Google Drive access token "
            f"({response.status_code}): {_google_error_detail(response)}. The "
            "stored refresh token may have expired or been revoked -- re-run "
            "scripts/google_drive_oauth_setup.py."
        )
    return response.json()["access_token"]


def _raise_for_drive_status(response: requests.Response, url_or_id: str) -> None:
    if response.status_code == 200:
        return
    if response.status_code == 401:
        raise DriveAuthError(
            "Google Drive rejected the access token (401) while fetching "
            f"{url_or_id!r}. Re-run scripts/google_drive_oauth_setup.py if this "
            "keeps happening."
        )
    if response.status_code in (403, 404):
        # Drive intentionally returns 404 (not 403) for "exists but not
        # shared with you", to avoid leaking whether a file exists -- so a
        # 404 here can mean either "wrong link" or "right link, wrong
        # account". Say both rather than guessing.
        raise DriveAccessError(
            f"Could not access Google Drive file {url_or_id!r} "
            f"({response.status_code}): {_google_error_detail(response)}. Check "
            "that the link is correct and that the file is shared with the "
            "Google account grading is authenticated as."
        )
    raise DriveAccessError(
        f"Google Drive request for {url_or_id!r} failed "
        f"({response.status_code}): {_google_error_detail(response)}."
    )


def fetch_drive_file(url_or_id: str, access_token: str, timeout: float = 30.0) -> DriveFile:
    """Download one Drive file's bytes.

    Does a metadata GET first (name/mimeType/size) so an unsupported native
    Google Doc or an oversized file gets reported with a specific,
    actionable message rather than either an opaque Drive error or the
    whole file being downloaded and discarded.
    """
    file_id = extract_file_id(url_or_id)
    headers = {"Authorization": f"Bearer {access_token}"}
    file_url = _DRIVE_FILES_URL.format(file_id=file_id)

    meta_response = requests.get(
        file_url, headers=headers, params={"fields": "name,mimeType,size"}, timeout=timeout
    )
    _raise_for_drive_status(meta_response, url_or_id)
    metadata = meta_response.json()
    name = metadata.get("name") or file_id
    mime_type = metadata.get("mimeType", "")

    if mime_type.startswith(_GOOGLE_NATIVE_MIME_PREFIX):
        raise DriveUnsupportedFileError(
            f"'{name}' is a native Google Doc/Sheet/Slide, not a scanned PDF or "
            "image -- download or export it to PDF from Drive and upload that "
            "file with the file picker above instead."
        )

    size = metadata.get("size")
    if size is not None and int(size) > MAX_DOWNLOAD_BYTES:
        raise DriveFileTooLargeError(
            f"'{name}' is {int(size) / 1_048_576:.1f} MB, over the "
            f"{MAX_DOWNLOAD_BYTES / 1_048_576:.0f} MB limit for a Drive-fetched scan."
        )

    content_response = requests.get(file_url, headers=headers, params={"alt": "media"}, timeout=timeout)
    _raise_for_drive_status(content_response, url_or_id)

    return DriveFile(file_id=file_id, filename=name, content=content_response.content)
