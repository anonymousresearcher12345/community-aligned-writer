"""Randomized field-study runner: shared evidence retrieval, then the baseline
and CA writers on every post, with only the assigned writer able to submit.

One batch:
    0. Fetch the small feed of posts eligible for notes, drop posts seen in
       earlier batches, and drop posts whose quoted or replied-to post has
       non-photo media. The writers' post context adds image descriptions
       and any quoted or replied-to post text.
For each remaining post:
    1. Retrieve evidence with Grok. The first retrieval gates the post; if it
       is usable, the remaining ``n_research - 1`` retrievals run in parallel.
    2. Assign the post to ``baseline`` or ``ca`` with probability 1/2.
    3. Run both writers in random order on the shared evidence:
         - baseline: write one note (one retry on API or length/citation
           failure); an LLM judge decides whether to submit it.
         - ca: write 10 candidates, score each with the rating model, and
           submit the best one passing the thresholds and consensus rule
           (see ``ca.py``); otherwise abstain.
       Each writer logs its own outcomes file.
    4. If the assigned writer decided to submit, infer misleading-reason tags,
       pick a random API account, and submit (unless ``--dry-run``).
    5. Append a per-post record to ``code/outputs/rct_pipeline.jsonl``.
"""

import os
import sys


import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if __package__:
    from . import baseline
    from .utils import orchestration
    from . import ca
    from .utils.data_models import ProposedMisleadingNote
    from .utils.logger import append_jsonl
    from .utils.orchestration import (
        API_ACCOUNTS,
        NoAvailableAccountsError,
        ensure_outputs_dir,
        fetch_and_record,
        submit_note_with_retry,
    )
    from .retrieval import evidence_retrieval_multi
    from .utils.util import get_misleading_tags, get_post_with_context_info_for_prompt
else:
    import baseline
    from utils import orchestration
    import ca
    from utils.data_models import ProposedMisleadingNote
    from utils.logger import append_jsonl
    from utils.orchestration import (
        API_ACCOUNTS,
        NoAvailableAccountsError,
        ensure_outputs_dir,
        fetch_and_record,
        submit_note_with_retry,
    )
    from retrieval import evidence_retrieval_multi
    from utils.util import get_misleading_tags, get_post_with_context_info_for_prompt


OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
RCT_LOG_FILE = OUTPUTS_DIR / "rct_pipeline.jsonl"
RETRIEVAL_LOG_FILE = OUTPUTS_DIR / "retrieval.jsonl"
BASELINE_LOG_FILE = OUTPUTS_DIR / "baseline_outcomes.jsonl"

# Experiment stop point: once baseline_outcomes.jsonl holds this many records the
# cycle exits before fetching anything, and the process exits with
# EXIT_TARGET_REACHED so run.sh can stop. Checked only at cycle
# start, so a cycle already under way runs its whole batch and the final count
# overshoots the target by up to one batch.
TARGET_BASELINE_RECORDS = int(os.environ.get("STOP_AT_BASELINE_RECORDS", "4500"))
EXIT_TARGET_REACHED = 42


def _baseline_record_count() -> int:
    """Count the baseline-arm datapoints accumulated so far.

    Returns
    -------
    int
        Number of non-blank lines in ``baseline_outcomes.jsonl``; 0 if the file
        does not exist yet.
    """
    try:
        with open(BASELINE_LOG_FILE, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def _extract_final_note_text(writer_arm: str, writer_record: dict) -> Optional[str]:
    """Pull the writer's final note text from its outcome record."""
    if writer_arm == "baseline":
        if writer_record.get("decision") == "WRITE NOTE":
            return writer_record.get("note_or_explanation") or None
        return None
    # ca
    return writer_record.get("final_note_text") or None


def _run_writer(
    arm: str, post_id: str, post_text: str, raw_post_text: str, evidences: list
) -> Optional[dict]:
    """Run one arm's writer over the shared researches. Returns record or None."""
    writer = baseline if arm == "baseline" else ca
    try:
        return writer.write_note(post_id=post_id, post_text=post_text,
                                 raw_post_text=raw_post_text, evidences=evidences)
    except Exception as e:
        print(f"Writer error ({arm}) for post {post_id}: {e}")
        return None


def _rct_worker(
    post_with_context,
    dry_run: bool,
    run_id: str,
    n_research: int = 1,
) -> dict:
    """Run the full RCT pipeline for a single post and return its log record."""
    post_id = post_with_context.post.post_id
    post_text = get_post_with_context_info_for_prompt(post_with_context)

    record = {
        "post_id": post_id,
        "run_id": run_id,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "post_text": post_text,
        "selected_arm": None,
        "baseline_submission_decision": None,
        "ca_submission_decision": None,
        "final_note_text": None,
        "submitted": False,
        "submit_account": None,
        "submit_status": None,
        "n_research": None,
        "dry_run": dry_run,
    }

    # Shared evidence retrieval, used by both arms. With n_research > 1 the first
    # research gates the post and the rest run in parallel; writers then sample.
    print(f"\n{'-' * 20}\nPost: {post_id}\n{'-' * 20}")
    print(f"Post {post_id}: running shared evidence retrieval (n={n_research})...")
    evidences = evidence_retrieval_multi(post_with_context, n=n_research)
    if not evidences:
        record["submit_status"] = "no_evidence"
        print(f"Post {post_id}: retrieval returned no usable evidence. Skipping.")
        append_jsonl(RCT_LOG_FILE, record)
        return record

    # Persist the shared evidence: one record per research, all sharing post_id.
    # research_idx 0 is always the gating research, so readers that keep the
    # first row per post_id use the gating evidence.
    record["n_research"] = len(evidences)
    retrieval_run_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    for research_idx, evidence in enumerate(evidences):
        append_jsonl(RETRIEVAL_LOG_FILE, {
            "post_id": post_id,
            "run_id": run_id,
            "run_at": retrieval_run_at,
            "research_idx": research_idx,
            "n_research": len(evidences),
            "audience_reaction": evidence.audience_reaction,
            "video_description": getattr(evidence, "video_description", "") or "",
            "retrieved_sources": [s.model_dump() for s in evidence.retrieved_sources],
            "note_request_suggestions": post_with_context.note_request_suggestions or [],
            "suggested_source_links_with_counts": post_with_context.suggested_source_links_with_counts or [],
        })

    # Randomize the arm only after retrieval succeeds.
    selected_arm = random.choice(["baseline", "ca"])
    record["selected_arm"] = selected_arm
    print(f"Post {post_id}: selected_arm={selected_arm}")

    # Run both writers in a randomized order; each logs its own outcomes file.
    run_order = ["baseline", "ca"]
    random.shuffle(run_order)
    writer_records: dict = {}
    for arm in run_order:
        print(f"Post {post_id}: running {arm} writer...")
        writer_records[arm] = _run_writer(
            arm, post_id, post_text, post_with_context.post.text, evidences
        )

    for arm in ("baseline", "ca"):
        wr = writer_records.get(arm)
        record[f"{arm}_submission_decision"] = (
            int(wr.get("submission_decision", 0)) if wr else None
        )

    selected_record = writer_records.get(selected_arm)
    if selected_record is None:
        record["submit_status"] = "writer_error"
        append_jsonl(RCT_LOG_FILE, record)
        return record

    if int(selected_record.get("submission_decision", 0)) != 1:
        record["submit_status"] = "abstain_writer"
        print(f"Post {post_id}: {selected_arm} writer did not approve submission. Skipping.")
        append_jsonl(RCT_LOG_FILE, record)
        return record

    note_text = _extract_final_note_text(selected_arm, selected_record)
    record["final_note_text"] = note_text
    if not note_text or not note_text.strip() or len(note_text.strip()) < 10:
        record["submit_status"] = "writer_error: empty_note_with_submit_decision_1"
        append_jsonl(RCT_LOG_FILE, record)
        return record

    if not orchestration.available_api_accounts:
        record["submit_status"] = "no_available_account"
        print(f"Post {post_id}: no available accounts; skipping submission.")
        append_jsonl(RCT_LOG_FILE, record)
        return record

    # Misleading tags from the richer post-with-context description
    try:
        tags = get_misleading_tags(
            post_with_context_description=post_text,
            note_text=note_text,
            retries=1,
        )
    except Exception as e:
        print(f"Tag inference failed for post {post_id}: {e}")
        record["submit_status"] = f"tag_error: {e}"
        append_jsonl(RCT_LOG_FILE, record)
        return record

    proposed = ProposedMisleadingNote(
        post_id=post_id,
        note_text=note_text,
        misleading_tags=tags,
    )

    chosen_account = random.choice(orchestration.available_api_accounts)

    if dry_run:
        record["submitted"] = False
        record["submit_account"] = chosen_account
        record["submit_status"] = "abstain_dry_run"
        print(f"Post {post_id}: dry-run, would submit via {chosen_account}.")
        append_jsonl(RCT_LOG_FILE, record)
        return record

    try:
        result = submit_note_with_retry(proposed_note=proposed, current_account=chosen_account)
        record["submitted"] = True
        record["submit_account"] = result["api_account"]
        record["submit_status"] = "success"
        print(f"Post {post_id}: submitted via {result['api_account']}.")
    except NoAvailableAccountsError as e:
        record["submit_status"] = "no_available_account"
        print(f"Post {post_id}: {e}")
    except Exception as e:
        record["submit_status"] = f"submit_failed: {e}"
        print(f"Post {post_id}: submission failed: {e}")

    append_jsonl(RCT_LOG_FILE, record)
    return record


def main(
    num_posts: int,
    dry_run: bool,
    max_process: Optional[int] = None,
    n_research: int = 1,
) -> int:
    """Run one RCT batch. Returns the number of successful submissions."""
    count = _baseline_record_count()
    if count >= TARGET_BASELINE_RECORDS:
        print(f"Stop point reached: {count} >= {TARGET_BASELINE_RECORDS} baseline records. Skipping cycle.")
        return 0

    orchestration.available_api_accounts = API_ACCOUNTS.copy()
    ensure_outputs_dir()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print(f"Initializing with {len(orchestration.available_api_accounts)} API accounts (run_id={run_id})")

    eligible_posts = fetch_and_record(num_posts)
    if not eligible_posts:
        print("No posts to process.")
        return 0

    if max_process is not None and len(eligible_posts) > max_process:
        print(f"Sampling {max_process} of {len(eligible_posts)} eligible posts to process.")
        eligible_posts = random.sample(eligible_posts, max_process)

    success_count = 0
    for pwc in eligible_posts:
        try:
            rec = _rct_worker(
                post_with_context=pwc,
                dry_run=dry_run,
                run_id=run_id,
                n_research=n_research,
            )
            if rec.get("submit_status") == "success":
                success_count += 1
        except Exception as e:
            print(f"\n*ERROR processing post {pwc.post.post_id}: {e}*\nSkipping.\n")
            continue
        time.sleep(1)

    print(f"\nDone. Successful submissions: {success_count}/{len(eligible_posts)}")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RCT batch: shared-retrieval baseline vs ca writers.")
    parser.add_argument("--num-posts", type=int, default=100,
                        help="Max posts per retrieval page (capped at 100). At 100, "
                             "pagination follows next_token through the whole feed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline but skip the actual submit_note API call.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for the coin-flip RNG (for testing).")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch and record post ids; skip writer/scoring/submit.")
    parser.add_argument("--max-process", type=int, default=None,
                        help="If set, randomly sample down to this many eligible posts per fetch.")
    parser.add_argument("--n-research", type=int, default=1,
                        help="Number of independent Grok researches per post. The first "
                             "runs alone as a gate (a post with no usable evidence is "
                             "skipped without further calls); the rest run in parallel. "
                             "Writers sample one research per note.")
    args = parser.parse_args()

    if args.n_research < 1:
        parser.error("--n-research must be >= 1")

    if args.seed is not None:
        random.seed(args.seed)

    if args.fetch_only:
        ensure_outputs_dir()
        fetch_and_record(args.num_posts)
    else:
        main(
            num_posts=args.num_posts,
            dry_run=args.dry_run,
            max_process=args.max_process,
            n_research=args.n_research,
        )

    if _baseline_record_count() >= TARGET_BASELINE_RECORDS:
        print(f"Target of {TARGET_BASELINE_RECORDS} baseline records reached; signalling runner stop.")
        sys.exit(EXIT_TARGET_REACHED)
