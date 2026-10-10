"""Authorize one Rabt account with its own Google account.

    uv run python -m rabt_integrations.google_auth huda
    uv run python -m rabt_integrations.google_auth awaiz

Opens Google's consent page in the browser. The person signs in themselves; no password is ever
seen by Rabt. The token is saved only if the Google account matches RABT_GOOGLE_EMAIL_<ID>, so one
person cannot accidentally connect their mailbox to the other person's identity.
"""

from __future__ import annotations

import json
import os
import sys

from rabt_integrations.accounts import get_account
from rabt_integrations.config import load_settings
from rabt_integrations.google_api import SCOPES, GoogleGateway, token_path


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m rabt_integrations.google_auth <awaiz|huda>")
        return 2
    settings = load_settings()
    account = get_account(argv[0])
    if not settings.google_client_secret_file.is_file():
        print(f"OAuth client file not found: {settings.google_client_secret_file}")
        return 1
    expected = account.google_email
    if not expected:
        print(f"Set RABT_GOOGLE_EMAIL_{account.account_id.upper()} in .env first.")
        return 1

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(settings.google_client_secret_file), SCOPES)
    print(f"Sign in as {expected} ({account.display_name}) in the browser window that opens.")
    creds = flow.run_local_server(port=0, login_hint=expected, prompt="consent", open_browser=True)

    data = json.loads(creds.to_json())
    path = token_path(settings, account.account_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)

    gateway = GoogleGateway(settings)
    try:
        actual = gateway.profile_email(account.account_id)
    except Exception as exc:
        path.unlink(missing_ok=True)
        print(f"Token check failed, not saved: {exc}")
        return 1
    if actual != expected:
        path.unlink(missing_ok=True)
        print(f"Refused: signed in as {actual}, but {account.account_id} must use {expected}. Token deleted.")
        return 1
    data["rabt_email"] = actual
    path.write_text(json.dumps(data), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    print(f"Connected {account.account_id} -> {actual}. Token stored at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
