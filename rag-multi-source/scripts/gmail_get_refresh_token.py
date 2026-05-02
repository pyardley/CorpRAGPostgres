"""
One-shot helper to obtain a Gmail OAuth2 refresh token.

Usage
-----
1. Place your OAuth desktop-app credentials JSON at the repo root as
   ``client_secret.json`` (the file Google downloads from
   APIs & Services → Credentials → "Download JSON").
2. Run this script from the repo root with the project's venv active:

       python scripts/gmail_get_refresh_token.py

   A browser window will open; sign in as the mailbox owner and grant
   the read-only Gmail permission. Once you click "Allow", the script
   prints two things to the terminal:

     * the **refresh token** (paste into the "Refresh Token" credential
       field in the Streamlit sidebar, alongside your Client ID + Secret)
     * the full **token.json** blob (paste into the "Full token.json
       (optional)" sidebar field if you'd rather use a single string;
       this also embeds the client id/secret/scope/token_uri so it's
       slightly more robust if you ever rotate keys).

Notes
-----
* Project must have the Gmail API enabled (APIs & Services → Library →
  Gmail API → Enable).
* While the OAuth consent screen is in *Testing* mode, refresh tokens
  issued to non-Workspace-internal accounts expire after 7 days. Rerun
  this script when that happens, or publish the app for verification
  if you need long-lived tokens.
* The script uses the loopback flow on a random localhost port; make
  sure ``http://localhost`` is listed under your OAuth client's
  authorised redirect URIs (it already is in the JSON you pasted).
* SCOPES is intentionally limited to ``gmail.readonly``. The ingestor
  in :mod:`app.ingestion.email_ingestor` requests the same scope, so
  Google won't prompt the user to re-consent on first ingest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same scope set the GmailClient uses — keep them in sync.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-secret",
        default="client_secret.json",
        help="Path to the OAuth client JSON downloaded from Cloud Console "
        "(default: ./client_secret.json relative to repo root).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to also write the resulting token.json to disk. "
        "If omitted, the blob is only printed to stdout.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port for the OAuth loopback redirect (default: random).",
    )
    args = parser.parse_args()

    # Late import so a missing dependency yields a clear, actionable
    # message rather than a stack trace at module load time.
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "ERROR: google-auth-oauthlib is not installed in this venv.\n"
            "Install it with:  pip install google-auth-oauthlib",
            file=sys.stderr,
        )
        return 2

    secret_path = Path(args.client_secret).resolve()
    if not secret_path.is_file():
        print(
            f"ERROR: OAuth client file not found at {secret_path}\n"
            "Download it from Google Cloud Console → APIs & Services → "
            "Credentials → your OAuth 2.0 Client ID → Download JSON, and "
            "save it to that path (or pass --client-secret <path>).",
            file=sys.stderr,
        )
        return 2

    print(f"Using OAuth client file: {secret_path}")
    flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)

    # ``access_type=offline`` + ``prompt=consent`` together force Google
    # to issue a refresh token even if the user has consented before —
    # without ``prompt=consent`` you only get a refresh token on the very
    # first consent, which is a common gotcha when re-running this script.
    creds = flow.run_local_server(
        port=args.port,
        prompt="consent",
        access_type="offline",
        open_browser=True,
        authorization_prompt_message=(
            "Opening browser for Google sign-in. Approve the "
            "gmail.readonly scope, then return here…"
        ),
        success_message=(
            "Authentication complete — you can close this browser tab."
        ),
    )

    if not creds.refresh_token:
        # Belt-and-braces: prompt=consent should guarantee one, but if
        # Google has changed behaviour we want a clear error.
        print(
            "ERROR: Google did not return a refresh_token. This usually "
            "means the OAuth client type isn't 'Desktop app', or the "
            "consent screen settings are unusual. Re-check the client "
            "type in Cloud Console and try again.",
            file=sys.stderr,
        )
        return 1

    token_blob = creds.to_json()

    print()
    print("=" * 72)
    print("REFRESH TOKEN (paste into 'Refresh Token' field in the sidebar):")
    print("=" * 72)
    print(creds.refresh_token)
    print()
    print("=" * 72)
    print("FULL token.json (paste into 'Full token.json (optional)' instead,")
    print("if you'd rather use a single field):")
    print("=" * 72)
    # Pretty-print so the user can eyeball it before pasting.
    parsed = json.loads(token_blob)
    print(json.dumps(parsed, indent=2))
    print()

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(token_blob, encoding="utf-8")
        try:
            os.chmod(out_path, 0o600)
        except OSError:
            # Windows doesn't honour POSIX modes — best effort only.
            pass
        print(f"Wrote token.json to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
