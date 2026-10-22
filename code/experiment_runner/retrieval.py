"""Evidence retrieval shared by both writers.

Each retrieval is one Grok call with web and X search tools and structured
output, returning sources, an audience-reaction summary, and a description of
any video in the post. ``evidence_retrieval_multi`` runs up to ``n``
independent retrievals per post.
"""

import concurrent.futures
import os


from typing import List, Optional

from pydantic import BaseModel
from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search, x_search

if __package__:
    from .utils.data_models import PostWithContext
    from .utils.util import format_user_suggested_inputs
else:
    from utils.data_models import PostWithContext
    from utils.util import format_user_suggested_inputs

import dotenv
dotenv.load_dotenv()


GROK_MODEL = "grok-4.3"
MAX_TRIALS = 1


_x_client: Optional[Client] = None


def _get_x_client() -> Client:
    """Create the xAI client on first use so imports do not require a key."""
    global _x_client
    if _x_client is None:
        _x_client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=180)
    return _x_client


RETRIEVAL_PROMPT="""You are an evidence collector for a Community Notes pipeline. Your job is to gather raw material about an X post so that downstream writers can analyze the post and decide whether to write a Community Note. You do NOT make judgements or recommend what a note should say. Your job is retrieval only.

The post id is {post_id}.

Step 1: Retrieve the post and immediate context.

Find the post on X using the post id. Read the post text and examine any images, videos, links, quoted posts, replies, or other surrounding context needed to understand what the post is about. Retrieve the threads and comments under the post and examine them to understand how the audience are interpreting it, and whether the post appears to be satire or a joke to the audience.

If the post includes media, inspect it. Notice any visible text, people, places, and whether it appears misattributed, edited, or AI-generated. If the post contains a video, watch it and write a brief neutral description of what the video shows: what happens on screen, what is said or claimed (spoken or in captions), any legible on-screen text, any identifiable people, places, or events, and whether the video appears edited, misattributed, taken out of context, or AI-generated. Downstream writers cannot watch the video, so this description is their only access to its content.

Step 2: Search for relevant external evidence.

Run multiple web and X searches to gather sources that downstream writers could use to understand and address the post. When feasible, run at least 7 distinct searches across different angles. Do not stop at the first plausible result, but also do not include unrelated sources merely to fill the output.

Cover several angles:
- Search directly for the topic, people, events, locations, dates, media, or specific factual statements in the post.
- Search for primary or original sources referenced or implied by the post (e.g., if the post discusses a study, paper, official statement, or news event, look for the original).
- Search for additional context surrounding the topic that may or may not be reflected in the post.
- Pay attention to when the post was created and prioritize recent evidence over outdated ones when recency matters.
- If the post includes media that may be misattributed or edited, search for the original or prior appearances of the media.

Deliberately seek a broad evidence base. Include sources that appear to support the post as well as sources that appear to contradict, complicate, or add context to it. For contested topics, include sources across relevant political and ideological perspectives. The goal is to give downstream writers enough material to reach their own conclusion, not to pre-select evidence pointing one direction.

Prioritize primary, recent, original, and reliable sources (official statements, primary documents, reputable news outlets, peer-reviewed sources) over partisan commentary, opinion pieces, or secondhand summaries. Use commentary or opinion sources only when they provide relevant context that is not available elsewhere, and include a range of perspectives when doing so.

Do not use Grok replies or other chatbot answers as sources.

{user_suggested_inputs_section}
Step 3: Fetch and record sources.

For each useful source found, record:
- `url`: the exact URL, verbatim.
- `content`: the relevant content frm the source: a one-sentence neutral description of the source, followed by the relevant excerpt or passage. Include enough surrounding context that a downstream writer can understand what the source says without re-fetching it. For long sources, extract the relevant section rather than the entire page.
- `retrieval_query`: the search query that surfaced this source.

Do not editorialize about the source. Do not write that a source supports, contradicts, debunks, confirms, or proves the post. Just record what the source says.

If after thorough searching across multiple angles you find no relevant sources, return an empty `retrieved_sources` list. Do not include unrelated sources just to fill the output. An empty `retrieved_sources` list is a valid output and signals to downstream writers that external evidence could not be found.

Step 4: Produce the output.

Return a structured object with three fields:

- `audience_reaction`: a brief statement of how the audience (from the comments) react to the post and whether they perceive the post as satire / a joke.
- `video_description`: a brief neutral description of the video in the post from Step 1. Use an empty string if the post contains no video or the video could not be accessed.
- `retrieved_sources`: a list of {{url, content, retrieval_query}} objects.

Do not include any other fields. Do not include a summary, assessment, recommendation, or notes about the retrieval process. Your output is raw material for downstream writers, not an analysis."""


class RetrievedSource(BaseModel):
    url: str
    content: str
    retrieval_query: str


class RetrievalResponse(BaseModel):
    audience_reaction: str
    # Brief neutral description of the video in the post; "" when no video.
    video_description: str = ""
    retrieved_sources: List[RetrievedSource]


def _build_user_suggested_inputs_section(post_with_context: PostWithContext) -> str:
    inputs_block = format_user_suggested_inputs(post_with_context)
    if not inputs_block:
        return ""
    return f"""
Additional context from X users (user-suggested, NOT pre-vetted):

The following information was provided by X users who requested a Community Note on this post. Treat these as user-proposed leads, not as authoritative or pre-verified inputs. You must critically evaluate every URL and explanation below — verify the URL is reachable, credible, and that its content actually substantiates what the user implies; check whether explanations are factually grounded or just opinion/partisan framing.

{inputs_block}

How to use this information:
- Investigate each suggested URL. If a URL is legitimate, credible, and directly relevant to the verifiable claims in the post, record it as one of your `retrieved_sources` with your OWN neutral factual description of what it says (do not parrot the user's framing).
- Skip suggested URLs that are broken, irrelevant, opinion/partisan commentary, or do not actually support what users claim they show.
- Treat user-proposed explanations as leads for what to search and verify, not as facts. Do not record an explanation itself as a source.
- Do not let the suggestions narrow your retrieval — still search for primary sources, opposing perspectives, and sources users did not suggest.
"""


def _call_grok(
    prompt: str,
    response_format=RetrievalResponse,
    max_trials: int = MAX_TRIALS,
    verbose: bool = False,
) -> Optional[RetrievalResponse]:
    """Run one Grok call with web/x search tools and structured output.

    Parameters
    ----------
    prompt : str
        The fully formatted user prompt to send to Grok.
    response_format : type[BaseModel], optional
        Pydantic model class used as the structured response schema.
    max_trials : int, optional
        Maximum number of request attempts.
    verbose : bool, optional
        If True, print the prompt before sending.

    Returns
    -------
    Optional[BaseModel]
        The parsed Pydantic response, or ``None`` if all trials failed.
    """
    if verbose:
        print("===== prompt =====")
        print(prompt)
        print("==================")

    trial = 0
    while trial < max_trials:
        try:
            chat = _get_x_client().chat.create(
                model=GROK_MODEL,
                tools=[
                    web_search(enable_image_understanding=True),
                    x_search(
                        enable_image_understanding=True,
                        enable_video_understanding=True,
                    ),
                ],
                response_format=response_format,
            )
            chat.append(user(prompt))
            response = chat.sample()
            parsed = response_format.model_validate_json(response.content)
            # Hard filter: drop Grok reply sources, which are not reliable.
            sources = getattr(parsed, "retrieved_sources", None)
            if sources is not None:
                parsed.retrieved_sources = [
                    s
                    for s in sources
                    if not s.url.startswith("https://x.com/grok/status")
                ]
            # Diagnostics: the call succeeded but the model recorded no sources.
            # Surface the raw retrieval signals (only available here, not in the
            # pipeline) to make "no usable evidence" debuggable.
            if getattr(parsed, "retrieved_sources", None) == []:
                print("Retrieval succeeded but returned 0 retrieved_sources.")
                tool_usage = getattr(response, "server_side_tool_usage", None)
                print(f"  server_side_tool_usage: {tool_usage}")
                citations = list(getattr(response, "citations", []) or [])
                print(f"  citations ({len(citations)}): {citations}")
                print(f"  audience_reaction: {getattr(parsed, 'audience_reaction', '')!r}")
                print(f"  video_description: {getattr(parsed, 'video_description', '')!r}")
                reasoning = getattr(response, "reasoning_content", "") or ""
                if reasoning:
                    snippet = reasoning if len(reasoning) <= 1000 else reasoning[:1000] + "..."
                    print(f"  reasoning_content (truncated): {snippet}")
            return parsed
        except Exception as e:
            print(f"Grok retrieval call failed on trial {trial + 1}/{max_trials}: {e}")
            trial += 1

    print("Grok retrieval call failed after max trials.")
    return None


def evidence_retrieval(
    post_with_context: PostWithContext, verbose: bool = False
) -> Optional[RetrievalResponse]:
    """Retrieve external evidence relevant to an X post.

    Formats ``RETRIEVAL_PROMPT`` with the post's id and any user-suggested
    source links / note-request explanations (as un-vetted leads), then runs
    a Grok call with web_search and x_search tools, returning the structured
    list of retrieved sources.

    Parameters
    ----------
    post_with_context : PostWithContext
        The post to investigate, including any user-suggested source links
        and note-request explanations used to build the suggested-inputs
        section (empty section when the post has none).
    verbose : bool, optional
        If True, print the prompt before sending.

    Returns
    -------
    Optional[RetrievalResponse]
        A ``RetrievalResponse`` whose ``retrieved_sources`` list may be
        empty when no relevant sources were found. ``None`` is returned
        only when all retry attempts fail.
    """
    prompt = RETRIEVAL_PROMPT.format(
        post_id=post_with_context.post.post_id,
        user_suggested_inputs_section=_build_user_suggested_inputs_section(
            post_with_context
        ),
    )
    return _call_grok(
        prompt=prompt,
        response_format=RetrievalResponse,
        max_trials=MAX_TRIALS,
        verbose=verbose,
    )


def _is_usable(evidence: Optional[RetrievalResponse]) -> bool:
    """A research is usable only if the call succeeded and it found >=1 source."""
    return evidence is not None and bool(evidence.retrieved_sources)


def evidence_retrieval_multi(
    post_with_context: PostWithContext,
    n: int = 1,
    verbose: bool = False,
) -> List[RetrievalResponse]:
    """Run up to ``n`` independent evidence retrievals for one post.

    The first retrieval runs alone as a gate. If it yields no usable evidence,
    no further retrieval calls are made for the post.
    Once it succeeds, the remaining ``n - 1`` retrievals run concurrently in a
    thread pool. Each call is fully independent, so the researches differ only
    through Grok's own sampling and search variance.

    Parameters
    ----------
    post_with_context : PostWithContext
        The post to investigate, passed unchanged to ``evidence_retrieval``.
    n : int, optional
        Number of retrievals to attempt (default 1).
    verbose : bool, optional
        If True, print each prompt before sending.

    Returns
    -------
    list of RetrievalResponse
        The usable researches, gating research first, in a stable order. Empty
        when the gating research produced no usable evidence; otherwise between
        1 and ``n`` entries, since individual parallel calls may fail or return
        no sources and are simply dropped.
    """
    post_id = post_with_context.post.post_id

    first = evidence_retrieval(post_with_context, verbose=verbose)
    if not _is_usable(first):
        return []
    if n <= 1:
        return [first]

    # Order the futures by submission (not completion) so research_idx is stable.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n - 1) as pool:
        futures = [
            pool.submit(evidence_retrieval, post_with_context, verbose)
            for _ in range(n - 1)
        ]
        rest = []
        for f in futures:
            try:
                rest.append(f.result())
            except Exception as e:
                print(f"Grok retrieval thread failed for post {post_id}: {e}")
                rest.append(None)

    evidences = [first] + [e for e in rest if _is_usable(e)]
    print(f"Post {post_id}: {len(evidences)}/{n} researches returned usable evidence.")
    return evidences
