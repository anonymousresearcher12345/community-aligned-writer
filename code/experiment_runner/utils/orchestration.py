"""Account selection, eligible-post fetching, and submission retries.

The fetch ledger and experiment logs live in code/outputs/.
Random assignment is performed by the experiment runner in main.py.
"""

import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .settings import API_ACCOUNTS, require_accounts

from .cnapi.submit_note import submit_note
from .cnapi.xurl_util import XurlError
from .data_models import PostWithContext
from .util import check_for_unsupported_media_in_post_with_context, switch_account

# Experiment outputs stay outside the runner package.
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
TWEET_IDS_FILE = OUTPUTS_DIR / "tweet_ids.txt"


available_api_accounts: List[str] = API_ACCOUNTS.copy()


class NoAvailableAccountsError(Exception):
    """Raised when all API accounts have reached their writing limit."""
    pass


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not TWEET_IDS_FILE.exists():
        TWEET_IDS_FILE.touch()


# ---------------------------------------------------------------------------
# Submission with account rotation
# ---------------------------------------------------------------------------

def submit_note_with_retry(
    proposed_note,
    current_account: str,
) -> dict:
    """Submit a note, rotating accounts on rate-limit/admission/dup/service errors.

    Parameters
    ----------
    proposed_note : ProposedMisleadingNote
        The note to submit.
    current_account : str
        Account to try first.

    Returns
    -------
    dict
        ``{"api_account": str, "note_final_status": "success"}`` on success.

    Raises
    ------
    NoAvailableAccountsError
        When the available pool is empty.
    XurlError
        For non-retryable submission errors.
    """
    global available_api_accounts

    print(f"Starting with account {current_account}...")

    while True:
        if current_account not in available_api_accounts:
            if not available_api_accounts:
                raise NoAvailableAccountsError(
                    "All API accounts have reached their writing limit"
                )
            current_account = random.choice(available_api_accounts)

        print(f"Switching to account {current_account}...")
        switch_account(current_account)

        try:
            submit_note(note=proposed_note, test_mode=False, verbose_if_failed=True)
            return {"api_account": current_account, "note_final_status": "success"}

        except XurlError as e:
            error_message = e.detail

            if error_message == "Failed to create a note. You’ve reached your daily limit for writing notes.":
                print(f"Account {current_account} hit daily limit. Removing.")
                if current_account in available_api_accounts:
                    available_api_accounts.remove(current_account)
                if not available_api_accounts:
                    raise NoAvailableAccountsError(
                        "All API accounts have reached their writing limit"
                    )
                current_account = random.choice(available_api_accounts)
                print(f"Retrying with {current_account} ({len(available_api_accounts)} left)")

            elif error_message == (
                "Your AI Note Writer must earn admission to be able to set test_mode to false, "
                "please see https://communitynotes.x.com/guide/en/api/overview for more details."
            ):
                print(f"Account {current_account} not admitted. Removing.")
                if current_account in available_api_accounts:
                    available_api_accounts.remove(current_account)
                if not available_api_accounts:
                    raise NoAvailableAccountsError(
                        "All API accounts have not earned admission"
                    )
                current_account = random.choice(available_api_accounts)
                print(f"Retrying with {current_account} ({len(available_api_accounts)} left)")

            elif error_message == "Failed to create a note. User already created a note for this post.":
                print(f"Account {current_account} already noted this post. Trying another.")
                others = [a for a in available_api_accounts if a != current_account]
                if not others:
                    raise NoAvailableAccountsError(
                        "No remaining accounts to try after duplicate-note error"
                    )
                current_account = random.choice(others)
                continue

            elif error_message == "Service Unavailable":
                print(f"Account {current_account} got Service Unavailable. Backing off.")
                others = [a for a in available_api_accounts if a != current_account]
                if not others:
                    raise NoAvailableAccountsError(
                        "No remaining accounts to try after service-unavailable error"
                    )
                time.sleep(2)
                current_account = random.choice(others)
                continue

            else:
                print(f"Failed to submit note: {e}.")
                raise


def fetch_and_record(num_posts: int) -> List[PostWithContext]:
    """Fetch from the small feed and record previously unseen post IDs.

    The first configured API account is used for fetching. Submission-account
    selection is handled separately. Existing deduplication and media filtering
    apply to every fetched post.
    """
    from .cnapi.get_api_eligible_posts import get_posts_eligible_for_notes

    require_accounts()
    ensure_outputs_dir()

    tweet_ids_seen = set()
    with open(TWEET_IDS_FILE) as f:
        for line in f:
            tid = line.strip().split("\t")[0]
            if tid:
                tweet_ids_seen.add(tid)

    fetch_account = API_ACCOUNTS[0]
    switch_account(fetch_account)
    print(f"Getting recent posts eligible for notes ({num_posts} per page) "
          f"from the small feed via account {fetch_account!r}")
    fetched_posts = get_posts_eligible_for_notes(max_results=num_posts)

    fetch_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    eligible_posts: List[PostWithContext] = []
    for pwc in fetched_posts:
        pid = str(pwc.post.post_id)
        if pid in tweet_ids_seen:
            continue

        tweet_ids_seen.add(pid)
        with open(TWEET_IDS_FILE, "a") as f:
            f.write(f"{pid}\t{fetch_ts}\tsmall\n")

        if check_for_unsupported_media_in_post_with_context(pwc):
            print(f"Post {pid}: unsupported media in referenced post; skipping.")
            continue

        eligible_posts.append(pwc)

    print(f"Found {len(eligible_posts)} new posts eligible for notes")
    if eligible_posts:
        print(f"  Eligible Post IDs: {', '.join(p.post.post_id for p in eligible_posts)}\n")
    return eligible_posts
