"""Utility functions for the experiment."""

import json
import subprocess
import html
from .data_models import Post, PostWithContext, MisleadingTag
from openai import OpenAI
from typing import List
import dotenv
from .url_evaluator import extract_urls

dotenv.load_dotenv()
from .settings import XURL_BIN, load_account_config

_oai_client = None


def _get_oai_client() -> OpenAI:
    """Create the OpenAI client on first use so imports do not require a key."""
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI()
    return _oai_client


# ========== Account Management ==========

def switch_account(account_name: str):
    configs = load_account_config()
    xurl_cmd = XURL_BIN
    subprocess.run([xurl_cmd, "auth", "clear", "--all"], check=True, capture_output=True)
    config = configs[account_name]
    auth_cmd = [
        xurl_cmd, "auth", "oauth1",
        "--consumer-key", config["X_API_KEY"],
        "--consumer-secret", config["X_API_KEY_SECRET"],
        "--access-token", config["X_ACCESS_TOKEN"],
        "--token-secret", config["X_ACCESS_TOKEN_SECRET"]
    ]
    subprocess.run(auth_cmd, check=True, capture_output=True)
    print(f"✓ Switched to account: {account_name}")

# ========== checking for unsupported media ==========
def _check_for_unsupported_media(post: Post) -> bool:
    """Check if the post contains unsupported media types."""
    for media in post.media:
        if media.media_type not in ["photo"]:
            return True
    return False

def check_for_unsupported_media_in_post_with_context(post_with_context: PostWithContext) -> bool:
    """Check referenced posts for unsupported media; the target post may contain video."""

    if post_with_context.quoted_post and _check_for_unsupported_media(post_with_context.quoted_post):
        return True
    if post_with_context.in_reply_to_post and _check_for_unsupported_media(post_with_context.in_reply_to_post):
        return True
    return False


# ========== Image Processing ==========
def _describe_image(image_url: str) -> str:
    prompt = """
Describe the image. Include the following details if they are present and recognizable. No need to mention if they are not present:
- A brief caption("A red-brick lighthouse on a rocky shoreline at dusk")
- Any legible text (OCR)
- Any recognizable logos, landmarks, and geographical features that help with landmark matching
- Any faces/celebrities you can confidently identify.
Be brief. Stay neutral.

Only respond with the image description, no other text.
"""
    try:
        response = _get_oai_client().chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url","image_url": {"url": image_url}}],
            }],
        )

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error describing image: {e}") from e

def _summarize_images_for_post(post: Post) -> str:
    """Summarize all images in a post."""
    images_summary = ""
    for i, media in enumerate(post.media):
        if media.media_type == "photo":
            image_description = _describe_image(media.url)
            images_summary += f"Image {i+1}: {image_description}\n"
    return images_summary


# ========== Text Processing ==========
def unescape(text: str) -> str:
  """Remove layers of HTML escaping so the text matches natural language."""
  return html.unescape(html.unescape(text)) if isinstance(text, str) else text


# ========== Note validation (length) ==========

def length_check(note_text: str, limit: int = 280) -> tuple:
    """Check note length where each URL counts as 1 character.

    Returns
    -------
    (ok, reason) : tuple[bool, str]
        ``ok`` is True iff the URL-collapsed length is <= ``limit``.
    """
    if note_text is None:
        return False, "note_text is None"
    collapsed = note_text
    for u in extract_urls(note_text):
        collapsed = collapsed.replace(u, "U", 1)
    n = len(collapsed)
    if (n <= limit) and (n >= 3):
        return True, ""
    return False, f"length {n} outside [3, {limit}] (each URL counted as 1)"


def get_post_with_context_info_for_prompt(
    post_with_context: PostWithContext) -> str:
    """Describe the post, its images, and any quoted/replied-to post for prompts."""

    post_text = unescape(post_with_context.post.text)
    created_at = post_with_context.post.created_at.strftime("%Y-%m-%d")
    description = f"""The target post (posted on {created_at}):```\n{post_text}\n```"""

    # Add image summaries to the post context supplied to the writers.
    media = post_with_context.post.media or []
    if any(m.media_type == "photo" for m in media):
        images_summary = _summarize_images_for_post(post_with_context.post)
        if images_summary:
            description += f"""Summary of images in the post:```\n{images_summary}\n```"""
    if post_with_context.quoted_post:
        quoted_post_text = unescape(post_with_context.quoted_post.text)
        description += f"""The post of interest had quoted (referenced) another post. Here is the quoted post's text:```\n{quoted_post_text}\n```"""

        quoted_images_summary = _summarize_images_for_post(post_with_context.quoted_post)
        if quoted_images_summary is not None and len(quoted_images_summary) > 0:
            description += f"""Summary of images in the quoted post:```\n{quoted_images_summary}\n```"""

    if post_with_context.in_reply_to_post:
        replied_to_post_text = unescape(post_with_context.in_reply_to_post.text)
        description += f"""The post of interest was a reply to another post. Here is the replied-to post's text:```\n{replied_to_post_text}\n```"""

        replied_to_images_summary = _summarize_images_for_post(post_with_context.in_reply_to_post)
        if replied_to_images_summary is not None and len(replied_to_images_summary) > 0:
            description += f"""Summary of images in the replied-to post:```\n{replied_to_images_summary}\n```"""

    return description


def format_user_suggested_inputs(post_with_context: PostWithContext) -> str:
    """Format suggested_source_links_with_counts + note_request_suggestions
    into a prompt-ready string. Returns '' when both fields are empty."""
    suggested_urls = post_with_context.suggested_source_links_with_counts or []
    note_request_suggestions = post_with_context.note_request_suggestions or []

    if not suggested_urls and not note_request_suggestions:
        return ""

    parts = []

    if suggested_urls:
        url_lines = [
            f"- {item.get('url', '')} (suggested by {item.get('count', '?')} user(s))"
            for item in suggested_urls
        ]
        parts.append("Suggested source URLs:\n" + "\n".join(url_lines))
    else:
        parts.append("Suggested source URLs: (none)")

    if note_request_suggestions:
        sugg_lines = []
        for item in note_request_suggestions:
            suggestion = item.get("suggestion", "")
            source_link = item.get("source_link") or "no associated URL"
            sugg_lines.append(f'- "{suggestion}" (associated URL: {source_link})')
        parts.append("User-proposed explanations:\n" + "\n".join(sugg_lines))
    else:
        parts.append("User-proposed explanations: (none)")

    return "\n\n".join(parts)


def get_misleading_tags(
    post_with_context_description: str, note_text: str, retries: int = 3
) -> List[MisleadingTag]:
    misleading_why_tags_prompt = f"""Below will be a post on X, and a proposed community note that \
        adds additional context to the potentially misleading post. \
        Your task will be to identify which of the following tags apply to the post and note. \
        You may choose as many tags as apply, but you must choose at least one. \
        You must respond in valid JSON format, with a list of which of the following options apply:
        - "factual_error":  # the post contains a factual error
        - "manipulated_media":  # the post contains manipulated/fake/out-of-context media
        - "outdated_information":  # the post contains outdated information
        - "missing_important_context":  # the post is missing important context
        - "disputed_claim_as_fact":  # including unverified claims
        - "misinterpreted_satire":  # the post is satire that may likely be misinterpreted as fact
        - "other":  # the post contains other misleading reasons

        Example valid JSON response:
        {{
            "misleading_tags": ["factual_error", "outdated_information", "missing_important_context"]
        }}

        OTHER = 0
        FACTUAL_ERROR = 1
        MANIPULATED_MEDIA = 2
        OUTDATED_INFORMATION = 3
        MISSING_IMPORTANT_CONTEXT = 4
        DISPUTED_CLAIM_AS_FACT = 5
        MISINTERPRETED_SATIRE = 6

        The post and note are as follows:

        {post_with_context_description}

        Proposed community note:
        ```
        {note_text}
        ```
        """

    input = [{"role": "user","content": [{"type": "input_text", "text": misleading_why_tags_prompt}]}]
    while retries > 0:
        try:
            response = _get_oai_client().responses.create(
                model="gpt-5-mini-2025-08-07",
                instructions="You are a professional fact checker.",
                input=input,
            )
            misleading_why_tags_str = response.output_text
            misleading_why_tags = json.loads(misleading_why_tags_str)["misleading_tags"]
            break
        except Exception as e:
            retries -= 1
            if retries == 0:
                print(f"Failed to get misleading tags: {e}")
                return [MisleadingTag.missing_important_context]

    return [MisleadingTag(tag) for tag in misleading_why_tags]
