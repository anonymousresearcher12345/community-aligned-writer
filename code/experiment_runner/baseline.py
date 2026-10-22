"""Baseline writer: write one note with Gemini over shared evidence, then
run an LLM submission gate to decide whether to submit it.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

import asyncio
import random
from datetime import datetime, timezone
from typing import Optional, Tuple

from google.genai import types
from pydantic import BaseModel

if __package__:
    from . import gemini_writer
    from .utils.logger import append_jsonl
else:
    import gemini_writer
    from utils.logger import append_jsonl


OUTPUTS_DIR = os.path.join(_HERE, "..", "outputs")
LOG_PATH = os.path.join(OUTPUTS_DIR, "baseline_outcomes.jsonl")

MAX_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# LLM submission gate
# ---------------------------------------------------------------------------

class SubmissionDecision(BaseModel):
    submit: bool
    reasoning: str


JUDGE_PROMPT = """You are deciding whether a drafted Community Note should be \
SUBMITTED to X/Twitter. 

Submit the note (submit = true) only if BOTH conditions hold:
- The note effectively explains why the post is misleading and/or provides important context.
- The note is likely to be found HELPFUL by raters with different viewpoints.

Do NOT submit the note (submit = false) if ANY of these hold:
- The post is not misleading, or is mainly opinion, emotion, personal experience, vague political \
rhetoric, satire, a joke, or otherwise lacks a specific factual claim.
- The note is likely to be found UNHELPFUL by people with different viewpoints.

Below is the post, the drafted note, and the note writer's own thinking process.

POST CONTENT:
\"\"\"
{post_text}
\"\"\"

DRAFTED NOTE:
\"\"\"
{note_text}
\"\"\"

NOTE WRITER'S THINKING:
\"\"\"
{thinking}
\"\"\"

Decide whether to submit. Respond with:
- `submit`: true to submit the note, false to withhold it.
- `reasoning`: a brief explanation (1-2 sentences) of your decision.
"""


def build_gate_prompt(post_text: str, note_text: str, thinking: str) -> str:
    return JUDGE_PROMPT.format(
        post_text=post_text or "(no post text available)",
        note_text=note_text or "(empty note)",
        thinking=thinking or "(no thinking provided)",
    )


def decide_submit(post_text: str, note_text: str, thinking: str) -> Tuple[bool, str]:
    """LLM submission gate. Returns ``(submit, reasoning)``.

    On model/parse failure, defaults to withholding (``False``) and returns the
    error string as the reasoning.
    """
    client = gemini_writer.make_client()
    prompt = build_gate_prompt(post_text, note_text, thinking)
    try:
        resp = client.models.generate_content(
            model=gemini_writer.MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                response_mime_type="application/json",
                response_schema=SubmissionDecision,
            ),
        )
        parsed: Optional[SubmissionDecision] = resp.parsed
        if parsed is None and resp.text:
            parsed = SubmissionDecision.model_validate_json(resp.text)
        if parsed is None:
            return False, "gate_error: empty response"
        return bool(parsed.submit), parsed.reasoning or ""
    except Exception as e:
        return False, f"gate_error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _note_or_explanation(resp: gemini_writer.WriterResponse) -> str:
    if resp.decision == "WRITE NOTE":
        return resp.note_text or ""
    return resp.no_note_explanation or ""


def write_note(
    post_id: str,
    post_text: str,
    raw_post_text: str,
    evidences,
    log_path: str = LOG_PATH,
) -> dict:
    """Run the baseline writer for a single post.

    Generates one note with up to one retry (a retry is triggered when the API
    call fails or when a WRITE NOTE output fails the length/citation check). For
    an accepted WRITE NOTE the LLM submission gate decides submit/withhold.

    Parameters
    ----------
    post_id : str
        X post id.
    post_text : str
        Prompt-ready post-with-context text.
    raw_post_text : str
        The post's own text. Unused by the baseline;
        accepted so both writers share one interface.
    evidences : RetrievalResponse or list of RetrievalResponse
        Shared retrieval output (``audience_reaction`` + ``retrieved_sources``).
        When several researches are supplied, one is sampled uniformly at random
        and used for this note; the retry loop keeps that same research.

    Returns
    -------
    dict
        Record with ``decision``, ``note_or_explanation``, ``submission_decision``
        (1 only when WRITE NOTE, passes the check, and the gate approves),
        ``thinking``, ``pass_check``, ``gate_submit``, ``gate_reasoning``,
        and the ``research_idx``/``n_research`` provenance of the
        sampled research. Also appended to the outcomes log.
    """
    run_at = datetime.now(timezone.utc).isoformat()
    evidences = gemini_writer.as_evidence_list(evidences)
    research_idx = random.randrange(len(evidences))
    sources, audience_reaction, video_description, allowed_urls = (
        gemini_writer.unpack_evidence(evidences[research_idx])
    )

    record = {
        "post_id": post_id,
        "run_at": run_at,
        "research_idx": research_idx,
        "n_research": len(evidences),
    }
    client = gemini_writer.make_client()

    parsed: Optional[gemini_writer.WriterResponse] = None
    pass_check: Optional[bool] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            parsed = asyncio.run(
                gemini_writer.generate_one_note(
                    client=client,
                    post_id=post_id,
                    post_text=post_text,
                    sources=sources,
                    audience_reaction=audience_reaction,
                    selected_guidance="",
                    video_description=video_description,
                )
            )
        except Exception as e:
            parsed = None
            record["fail_reason"] = f"generation_error: {type(e).__name__}: {e}"
            print(f"Baseline generation failed (attempt {attempt}) for post {post_id}: {e}")
            continue

        if parsed is None:
            record["fail_reason"] = "generation returned None"
            continue

        if parsed.decision == "ABSTAIN":
            pass_check = None
            break

        # WRITE NOTE: validate length + citations.
        ok, reason = gemini_writer.check_note(parsed.note_text or "", allowed_urls)
        pass_check = ok
        if ok:
            break
        print(f"Baseline note check failed (attempt {attempt}) for post {post_id}: {reason}")
        record["fail_reason"] = f"check_failed: {reason}"

    if parsed is None:
        record.update({
            "decision": "ABSTAIN",
            "note_or_explanation": "",
            "submission_decision": 0,
            "thinking": "",
            "pass_check": None,
            "gate_submit": None,
            "gate_reasoning": None,
        })
        append_jsonl(log_path, record)
        return record

    note_or_expl = _note_or_explanation(parsed)
    gate_submit: Optional[bool] = None
    gate_reasoning: Optional[str] = None
    submission_decision = 0


    if parsed.decision == "WRITE NOTE" and parsed.note_text:
        if pass_check:
            # The gate judges against the same post context the writer saw,
            # including the video description.
            gate_post_text = post_text
            if video_description:
                gate_post_text += (
                    f"\n\nDescription of the video attached to the post:\n{video_description}"
                )
            gate_submit, gate_reasoning = decide_submit(
                post_text=gate_post_text,
                note_text=parsed.note_text,
                thinking=parsed.thinking,
            )
            submission_decision = 1 if gate_submit else 0

    record.update({
        "decision": parsed.decision,
        "note_or_explanation": note_or_expl,
        "submission_decision": submission_decision,
        "thinking": parsed.thinking,
        "pass_check": pass_check,
        "gate_submit": gate_submit,
        "gate_reasoning": gate_reasoning,
    })
    append_jsonl(log_path, record)
    return record
