"""CA writer: generate N candidate notes with Gemini, each with a randomly
sampled style instruction (``INSTRUCTION_PLUGINS``) and evidence sample, score
them with simulated community feedback, and apply the submission rule:

* Each WRITE NOTE candidate is checked for length/citations and scored by the
  rating model on the raw post text, giving a predicted helpfulness (note
  intercept) and factor.
* Consensus: submit only when at least ``MIN_WRITE_NOTE_CONSENSUS`` candidates
  produced a nonempty, scored note. This accounts for candidate abstentions,
  which no individual note score reflects.
* Selection: among candidates that pass the check and either
  (``helpfulness > HELPFULNESS_THRESHOLD`` with ``|factor| < FACTOR_THRESHOLD``)
  or (``HELPFULNESS_SOFT_THRESHOLD < helpfulness <= HELPFULNESS_THRESHOLD`` with
  ``|factor| < FACTOR_SOFT_THRESHOLD``), submit the one with the highest
  helpfulness. If none qualify, abstain.

Logs one summary per post (``ca_outcomes.jsonl``) and one line per candidate
(``ca_candidates.jsonl``).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

import asyncio
import concurrent.futures
import random
from datetime import datetime, timezone
from typing import List, Optional

if __package__:
    from . import gemini_writer
    from .utils.helpfulness_score import predict_helpfulness
    from .utils.logger import append_jsonl
else:
    import gemini_writer
    from utils.helpfulness_score import predict_helpfulness
    from utils.logger import append_jsonl


HELPFULNESS_THRESHOLD = 0.35
HELPFULNESS_SOFT_THRESHOLD = 0.3
FACTOR_SOFT_THRESHOLD = 0.35
FACTOR_THRESHOLD = 0.5
N_CANDIDATES = 10
# Minimum number of candidates that must choose WRITE NOTE before anything may be
# submitted; 1 disables the gate.
MIN_WRITE_NOTE_CONSENSUS = 2
MAX_CONCURRENCY = 10

OUTPUTS_DIR = os.path.join(_HERE, "..", "outputs")
LOG_PATH = os.path.join(OUTPUTS_DIR, "ca_outcomes.jsonl")
CAND_LOG_PATH = os.path.join(OUTPUTS_DIR, "ca_candidates.jsonl")

_api_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_loop_id: Optional[int] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Module-level semaphore bound to the running loop (rebinds on loop change)."""
    global _api_semaphore, _semaphore_loop_id
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _api_semaphore is None or _semaphore_loop_id != loop_id:
        _api_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        _semaphore_loop_id = loop_id
    return _api_semaphore


def _run_coro(coro):
    """Run a coroutine synchronously.

    If the current thread has a running event loop, execute the coroutine in a
    worker thread. Otherwise, create an event loop and run it to completion.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def _generate_all(
    post_id: str,
    post_text: str,
    evidences: list,
    n: int,
) -> List[dict]:
    """Generate ``n`` candidates concurrently, each with a random plugin.

    Each candidate independently samples one research from ``evidences``, so the
    candidate pool spans retrieval variance as well as writing-style variance.
    The sampled research's citation allowlist travels back with the candidate,
    since a candidate may only cite the sources it was actually shown.
    """
    client = gemini_writer.make_client()
    sem = _get_semaphore()
    plugin_keys = list(gemini_writer.INSTRUCTION_PLUGINS.keys())

    async def _one(run_idx: int) -> dict:
        key = random.choice(plugin_keys)
        guidance = gemini_writer.INSTRUCTION_PLUGINS[key]
        research_idx = random.randrange(len(evidences))
        sources, audience_reaction, video_description, allowed_urls = (
            gemini_writer.unpack_evidence(evidences[research_idx])
        )
        async with sem:
            try:
                parsed = await gemini_writer.generate_one_note(
                    client=client,
                    post_id=post_id,
                    post_text=post_text,
                    sources=sources,
                    audience_reaction=audience_reaction,
                    selected_guidance=guidance,
                    video_description=video_description,
                )
                error = None
            except Exception as e:
                parsed = None
                error = f"{type(e).__name__}: {e}"
        return {
            "run_idx": run_idx,
            "selected_guidance": key,
            "research_idx": research_idx,
            "allowed_urls": allowed_urls,
            "parsed": parsed,
            "error": error,
        }

    return await asyncio.gather(*[_one(i) for i in range(n)])


def _passes_submission_rule(c: dict) -> bool:
    """Submission rule: two-tier helpfulness/factor gate.

    ``(h > 0.35 AND |factor| < 0.50) OR (0.3 < h <= 0.35 AND |factor| < 0.35)``.
    """
    if not c["pass_check"]:
        return False
    if c["factor"] is None:
        return False
    h = c["helpfulness"]
    if h > HELPFULNESS_THRESHOLD:
        return abs(c["factor"]) < FACTOR_THRESHOLD
    return (
        HELPFULNESS_SOFT_THRESHOLD < h
        and abs(c["factor"]) < FACTOR_SOFT_THRESHOLD
    )


def write_note(
    post_id: str,
    post_text: str,
    raw_post_text: str,
    evidences,
    n: int = N_CANDIDATES,
    log_path: str = LOG_PATH,
    cand_log_path: str = CAND_LOG_PATH,
) -> dict:
    """Run the CA writer for a single post.

    Parameters
    ----------
    post_id : str
        X post id.
    post_text : str
        Prompt-ready post-with-context text, used for generation.
    raw_post_text : str
        The post's own text, used for rating-model scoring to match the
        training input.
    evidences : RetrievalResponse or list of RetrievalResponse
        Shared retrieval output (``audience_reaction`` + ``retrieved_sources``).
        When several researches are supplied, each candidate samples one
        independently.
    n : int
        Number of candidate notes to generate (default ``N_CANDIDATES``).

    Returns
    -------
    dict
        Summary record with ``submission_decision`` (1 only when at least
        ``MIN_WRITE_NOTE_CONSENSUS`` candidates chose WRITE NOTE *and* one of them
        passes the check + score filter) and ``final_note_text``. The summary is
        appended to ``log_path`` and one line per candidate to ``cand_log_path``.
    """
    run_at = datetime.now(timezone.utc).isoformat()
    evidences = gemini_writer.as_evidence_list(evidences)

    raw = _run_coro(_generate_all(post_id, post_text, evidences, n))

    candidates: List[dict] = []
    for item in raw:
        parsed = item["parsed"]
        cand = {
            "post_id": post_id,
            "run_at": run_at,
            "run_idx": item["run_idx"],
            "selected_guidance": item["selected_guidance"],
            "research_idx": item["research_idx"],
            "n_research": len(evidences),
            "decision": None,
            "note_or_explanation": None,
            "pass_check": None,
            "helpfulness": None,
            "factor": None,
            "error": item["error"],
        }
        if parsed is not None:
            cand["decision"] = parsed.decision
            if parsed.decision == "WRITE NOTE" and parsed.note_text:
                cand["note_or_explanation"] = parsed.note_text
                # Allowlist of the research this candidate actually saw.
                ok, _ = gemini_writer.check_note(parsed.note_text, item["allowed_urls"])
                cand["pass_check"] = ok
                # Score the produced note regardless of whether it passes the check.
                h = predict_helpfulness(note_text=parsed.note_text, post_text=raw_post_text)
                cand["helpfulness"] = h["score"]
                cand["factor"] = h["factor"]
            elif parsed.decision == "ABSTAIN":
                cand["note_or_explanation"] = parsed.no_note_explanation or ""
        candidates.append(cand)

    for cand in candidates:
        append_jsonl(cand_log_path, cand)

    write_note_cands = [
        c for c in candidates
        if c["decision"] == "WRITE NOTE" and c["note_or_explanation"]
        and c["helpfulness"] is not None
    ]
    passing = [c for c in write_note_cands if _passes_submission_rule(c)]
    # Lone-writer candidates are not submitted even when they clear the score gate.
    consensus_ok = len(write_note_cands) >= MIN_WRITE_NOTE_CONSENSUS

    submission_decision = 0
    final_note_text: Optional[str] = None
    chosen_run_idx: Optional[int] = None
    chosen_research_idx: Optional[int] = None
    if passing and consensus_ok:
        chosen = max(passing, key=lambda c: c["helpfulness"])
        submission_decision = 1
        final_note_text = chosen["note_or_explanation"]
        chosen_run_idx = chosen["run_idx"]
        chosen_research_idx = chosen["research_idx"]

    record = {
        "post_id": post_id,
        "run_at": run_at,
        "submission_decision": submission_decision,
        "final_note_text": final_note_text,
        "chosen_run_idx": chosen_run_idx,
        "chosen_research_idx": chosen_research_idx,
        "n_research": len(evidences),
        "n_write_note": len(write_note_cands),
        "n_passing": len(passing),
    }
    append_jsonl(log_path, record)
    return record
