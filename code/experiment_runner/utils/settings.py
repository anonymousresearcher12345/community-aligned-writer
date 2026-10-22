"""Deployment settings; import has no file, client, or network side effects."""
import json
import os

CONFIG_PATH = os.environ.get("CA_CONFIG_PATH", "")
XURL_BIN = os.environ.get("XURL_BIN") or "xurl"
API_ACCOUNTS = [name.strip() for name in os.environ.get("CA_API_ACCOUNTS", "").split(",")
                if name.strip()]


def load_account_config():
    if not CONFIG_PATH.strip():
        raise ValueError("Set CA_CONFIG_PATH to your private account configuration JSON.")
    with open(CONFIG_PATH) as stream:
        return json.load(stream)


def require_accounts():
    if not API_ACCOUNTS:
        raise ValueError("Set CA_API_ACCOUNTS to comma-separated account names from CA_CONFIG_PATH.")
