"""One-time, local, interactive setup for Google Drive grading (issue #102).

Run this ONCE on your own machine (never on fly.io) to produce a long-lived
refresh token that the deployed app then uses silently on every grading run.
It is not imported by any graderbot/* production code and needs no test
coverage -- it's the thing being set up, not application logic.

Prerequisites (see the Grade tab / README docs for the full walkthrough):
  1. In Google Cloud Console (any personal Google account -- this does NOT
     need to be the school account, and does NOT need Workspace admin
     rights), create a project, enable the "Google Drive API", configure an
     OAuth consent screen (External, scope
     https://www.googleapis.com/auth/drive.readonly, with the school account
     added as a Test user), and create an OAuth client ID of type
     "Desktop app". Note its Client ID and Client Secret.
  2. Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in your `.env`
     (or answer the prompts below).

Then run:
    poetry run python scripts/google_drive_oauth_setup.py

A browser window opens for you to log into the Google account whose Drive
files you want to grade from (the school account). If Google shows
"Access blocked -- contact your admin" instead of a normal consent screen,
that means the school's Workspace has restricted third-party OAuth apps:
screenshot that page, send it to the school's IT/Workspace admin, ask them
to allow this app under Admin Console -> Security -> API Controls -> App
Access Control, and re-run this script once they've done so. There is
nothing this script can do about that screen itself.

On success, this prints a GOOGLE_OAUTH_REFRESH_TOKEN to copy into `.env`
(local) and `fly secrets set` (production). Re-run this script whenever
that refresh token stops working -- e.g. the 7-day expiry Google applies
while the OAuth consent screen is in "Testing" publishing status (see
README.md for the Testing-vs-Production tradeoff), or if it's revoked.

Requires `google-auth-oauthlib`, a dev-only dependency
(`poetry install --with dev`) -- it is never imported by the deployed app,
which talks to Google over plain `requests` (see graderbot/drive_fetch.py).
"""

import os

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _get_or_prompt(env_var: str, prompt: str) -> str:
    value = os.environ.get(env_var)
    if value:
        return value
    value = input(f"{prompt}: ").strip()
    if not value:
        raise SystemExit(f"{env_var} is required.")
    return value


def main() -> None:
    load_dotenv()
    client_id = _get_or_prompt("GOOGLE_OAUTH_CLIENT_ID", "Google OAuth Client ID")
    client_secret = _get_or_prompt("GOOGLE_OAUTH_CLIENT_SECRET", "Google OAuth Client Secret")

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

    print(
        "\nOpening a browser to sign in -- log in with the Google account "
        "whose Drive files you want to grade from (e.g. your school account).\n"
        "If you land on an 'Access blocked' page instead of a consent screen, "
        "see this script's docstring for what to do next.\n"
    )
    credentials = flow.run_local_server(port=0)

    if not credentials.refresh_token:
        raise SystemExit(
            "Google did not return a refresh token. This usually means the "
            "account already granted this app consent once before without "
            "prompting for a new refresh token -- revoke this app's access at "
            "https://myaccount.google.com/permissions and re-run this script."
        )

    print("\nSuccess! Add these to your .env (and `fly secrets set` in production):\n")
    print(f"GOOGLE_OAUTH_CLIENT_ID={client_id}")
    print(f"GOOGLE_OAUTH_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_OAUTH_REFRESH_TOKEN={credentials.refresh_token}")


if __name__ == "__main__":
    main()
