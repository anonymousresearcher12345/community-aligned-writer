"""Shared Gemini note-writing helpers for the (baseline, ca) pair.

The seed prompt, the instruction plugins, the structured ``WriterResponse`` schema,
the evidence formatter, the prompt builder, and a single async note-generation call.
Both ``baseline.py`` and ``ca.py`` import from here so the generation surface is
identical across arms.
"""

import os


import random
from typing import Literal, Optional

import dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
if __package__:
    from .utils.url_evaluator import extract_urls
    from .utils.util import length_check
else:
    from utils.url_evaluator import extract_urls
    from utils.util import length_check


dotenv.load_dotenv()

MODEL = "gemini-3.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")


class WriterResponse(BaseModel):
    thinking: str
    decision: Literal["WRITE NOTE", "ABSTAIN"]
    note_text: Optional[str] = None
    no_note_explanation: Optional[str] = None


INSTRUCTION_PLUGINS = {
    "none": "",
    "minimalist": ("- Write the SHORTEST possible note that still makes the core point clearly. "
            "Every word must earn its place. Aim for well under 100 characters while remaining readable."),
    "direct_refutation": ("- Write the note as a clear and direct refutation or correction. "
               "State the correction bluntly and confidently without hedging, softening, or extra politeness."),
    "acknowledge_and_correct": ("- Write the note by first briefly acknowledging what the post gets right or why it seems plausible, "
                                "then clearly correct the misleading part."),
    "explanatory": ("- Write the note in a helpful and explanatory tone. Focus on explaining the issue in plain English."
                    "Priortize clarity and understanding over blunt correction."),
    "punchy": ("- Write a punchy note by making it sharp and impactful while staying factual and neutral. "),
    "multi_perspective": ("- Write the note from multiple perspectives. Examine the claim from distinct angles before delivering the correction, "
                          "then provide a balanced and coherent note.")
}


SEED_PROMPT = """You are an expert Community Notes writer. A research step has already analyzed the given X post \
below and gathered evidence. Your task is to investigate the X post, decide whether you should write a community \
note to explain why the post is misleading, and either write that note or explain why a note is not needed.

Step 1: Investigate the post.
Read the post text below carefully and identify the factual claim(s) or implication the post makes.

POST CONTENT:
\"\"\"
{post_text}
\"\"\"

{video_section}{audience_reaction_section}
Step 2: Review and analyze the collected evidence.
The following sources have been retrieved for you via web search. Treat these as your only evidence base. Do NOT search \
the web yourself. Prioritize primary, original, and reliable sources over social-media commentary, partisan commentary, \
opinion pieces, and secondhand summaries. You can only use evidence and cite URLs that appear in this list. 

Analyze the evidence to identify which sources provide clear and direct information about whether the post’s specific \
claim is true, false, or misleading. Distinguish direct evidence from sources that merely discuss the same topic or provide \
related context. Note any conflicting evidence.


RETRIEVED EVIDENCE:
{evidence_block}

Step 3: Decide whether to write a note.
Write a note if the post is misleading, the issue is based on a specific checkable factual claim(s)\
or implication, and the correction is directly support by reliable evidence. The note should cite evidence \
that directly and clearly resolves the post's issue.

Do not write if the post is accurate, or mainly opinion, emotion, personal experience, vague political rhetoric, or \
otherwise lacks a specific factual claim. Do not write if a post is mainly perceived as satire or joke. \
Do not write if the post is not perceived as misleading or if there is no sufficient evidence.

Step 4: Produce the output.
If you decide to write a note, set `decision` to "WRITE NOTE":
- Write a clear note explaining why **the post** is misleading.
- The note must be at most 280 characters. Each cited URL counts as 1 character.
- Cite at least one URL as evidence. Include cited URLs exactly as provided in the retrieved evidence, not as indexes, footnotes, or parenthetical references.
- Never make up statements that are not supported by the URL you cited.
- Write factually and helpfully, so people across different viewpoints could find the note helpful. Notes that sound like they are defending one side are less likely to be helpful. 
- Put the note in `note_text` and leave `no_note_explanation` empty.
{additional_guidance}

If you decide not to write a note, set `decision` to "ABSTAIN":
- Briefly explain why the decision is to abstain, within 280 characters. Each cited URL counts as 1 character.
- Cite at least one URL as evidence from the retrieved evidence. Include cited URLs exactly as provided, not as indexes, footnotes, or parenthetical references.
- Put the explanation in `no_note_explanation` and leave `note_text` empty.

Always include a brief `thinking` field summarizing how you reached your decision (a short paragraph, \
not a transcript), including an explicit assessment of whether the note would be rated HELPFUL across \
different perspectives.

Ensure the length of the note or no-note explanation is within the 280-character limit. Each cited URL counts as 1 character.

As your response, always include:
- `thinking`: a brief summary of how you reached your decision and note/no-note explanation writing, \
including an explicit assessment of whether the note would be rated HELPFUL across different perspectives.
- `decision`: either "WRITE NOTE" or "ABSTAIN"
- `note_text`: the Community Note if writing one, otherwise empty
- `no_note_explanation`: the abstention explanation if not writing a note, otherwise empty
"""


def make_client() -> genai.Client:
    """Create a Gemini client using the GEMINI_API_KEY from the environment."""
    return genai.Client(api_key=API_KEY)


def evidence_urls(evidence) -> list[str]:
    """Trusted citation allowlist for a note: the retrieved evidence source URLs."""
    return [s.url for s in evidence.retrieved_sources]


def as_evidence_list(evidence) -> list:
    """Normalize a single RetrievalResponse or a list of them into a list."""
    if isinstance(evidence, (list, tuple)):
        return list(evidence)
    return [evidence]


def unpack_evidence(evidence) -> tuple[list[dict], str, str, list[str]]:
    """Split one research into the pieces the writers need.

    Parameters
    ----------
    evidence : RetrievalResponse
        A single research (not a list).

    Returns
    -------
    (sources, audience_reaction, video_description, allowed_urls)
        ``sources`` is the prompt-ready list of dicts; ``allowed_urls`` is the
        citation allowlist for exactly this research.
    """
    return (
        [s.model_dump() for s in evidence.retrieved_sources],
        evidence.audience_reaction,
        getattr(evidence, "video_description", "") or "",
        evidence_urls(evidence),
    )


def check_note(note_text: str, allowed_urls: list[str]) -> tuple[bool, str]:
    """Validate a WRITE NOTE output: length limit + strict citation allowlist.

    A note passes only if it is within the 280-char limit (each URL counts as 1),
    cites at least one URL, and every cited URL appears in ``allowed_urls`` (the
    retrieved evidence). Any cited URL outside that set fails the check.

    Returns
    -------
    (ok, reason) : tuple[bool, str]
    """
    ok_len, reason_len = length_check(note_text or "")
    if not ok_len:
        return False, f"length: {reason_len}"
    urls = extract_urls(note_text or "")
    if not urls:
        return False, "no URLs cited"
    allowed = set(allowed_urls)
    bad = [u for u in urls if u not in allowed]
    if bad:
        return False, f"cited URLs not in retrieved evidence: {bad}"
    return True, ""


def format_evidence(sources: list[dict]) -> str:
    if not sources:
        return "(no sources retrieved)"
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(
            f"[{i}] URL: {s.get('url', '')}\n"
            f"    Query: {s.get('retrieval_query', '')}\n"
            f"    Content: {s.get('content', '')}"
        )
    return "\n\n".join(parts)


def build_prompt(
    post_id: str,
    post_text: str,
    sources: list[dict],
    audience_reaction: str,
    selected_guidance: str = "",
    video_description: str = "",
) -> str:
    # Randomize evidence order per call to de-bias position effects in the prompt.
    shuffled_sources = list(sources)
    random.shuffle(shuffled_sources)
    # Optional sections, each included only when its content is non-empty.
    audience_reaction_section = ""
    if audience_reaction:
        audience_reaction_section = (
            "\nAUDIENCE REACTION (how the audience reacts to the post and whether they perceive it as satire or a joke):\n"
            f"{audience_reaction}\n"
        )
    video_section = ""
    if video_description:
        video_section = (
            "\nVIDEO IN THE POST (a brief description of the video attached to the post):\n"
            f"{video_description}\n"
        )
    return SEED_PROMPT.format(
        post_id=post_id,
        post_text=post_text,
        audience_reaction_section=audience_reaction_section,
        video_section=video_section,
        evidence_block=format_evidence(shuffled_sources),
        additional_guidance=selected_guidance,
    )


async def generate_one_note(
    client: genai.Client,
    post_id: str,
    post_text: str,
    sources: list[dict],
    audience_reaction: str,
    selected_guidance: str = "",
    video_description: str = "",
) -> Optional[WriterResponse]:
    """Run one Gemini note-writing call with structured output.

    Returns the parsed ``WriterResponse`` or ``None`` if the call/parse fails.
    """
    prompt = build_prompt(
        post_id, post_text, sources, audience_reaction, selected_guidance, video_description
    )
    resp = await client.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
            response_schema=WriterResponse,
        ),
    )
    parsed: Optional[WriterResponse] = resp.parsed
    if parsed is None and resp.text:
        parsed = WriterResponse.model_validate_json(resp.text)
    return parsed
