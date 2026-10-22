import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

from ..cnapi.xurl_util import run_xurl

from ..data_models import Media, Post, PostWithContext


MAX_PAGES = 180


def _fetch_posts_eligible_page(
    max_results: int = 100,
    pagination_token: Optional[str] = None,
) -> dict:
    """Fetch one small-feed page using the experiment's test_mode=true query.

    The small feed uses the API default, so post_selection is omitted.
    """
    path = (
        "/2/notes/search/posts_eligible_for_notes"
        "?test_mode=true"
        f"&max_results={max_results}"
        "&tweet.fields=author_id,created_at,referenced_tweets,media_metadata,note_tweet,suggested_source_links_with_counts,note_request_suggestions"
        "&expansions=attachments.media_keys,referenced_tweets.id,referenced_tweets.id.attachments.media_keys"
        "&media.fields=alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width,variants"
    )
    if pagination_token:
        path += f"&pagination_token={quote(pagination_token, safe='')}"
    return run_xurl(["xurl", path], verbose_if_failed=True)

def _parse_individual_post(item: Dict, media_by_key: Dict[str, Dict]) -> Post:
    media_objs: List[Media] = []
    media_keys = item.get("attachments", {}).get("media_keys", [])

    for key in media_keys:
        if key in media_by_key:
            media_obj = Media(**media_by_key[key])
            media_objs.append(media_obj)

    text = item["text"]
    note_tweet_text = item.get("note_tweet", {}).get("text", "")
    if note_tweet_text:
        text = note_tweet_text

    post = Post(
        post_id=item["id"],
        author_id=item["author_id"],
        created_at=datetime.fromisoformat(
            item["created_at"].replace("Z", "+00:00")
        ),
        text=text,
        media=media_objs,
    )
    return post


def _parse_posts_eligible_response(resp: Dict) -> List[PostWithContext]:
    """
    Convert an eligible-post API response into `PostWithContext` objects.

    For more details, see: https://docs.x.com/x-api/community-notes/introduction
    Args:
        resp: The raw response from the API, as a dictionary.
    Returns:
        A list of `PostWithContext` objects with available media and referenced posts.
    """
    includes_media = resp.get("includes", {}).get("media", [])
    media_by_key = {m["media_key"]: m for m in includes_media}

    includes_posts = resp.get("includes", {}).get("tweets", [])
    posts_by_id = {t["id"]: t for t in includes_posts}

    # rename type field to media_type to avoid name conflict with type
    for media_obj in media_by_key.values():
        media_obj["media_type"] = media_obj.pop("type")

    posts: List[Post] = []
    for item in resp.get("data", []):
        post = _parse_individual_post(item, media_by_key)

        # User suggestions attached to the note request (un-vetted)
        note_request_suggestions = item.get("note_request_suggestions", [])  # [{"suggestion_id":str, "suggestion":str, "source_link":str|None}]
        suggested_source_links_with_counts = item.get("suggested_source_links_with_counts", [])  # [{"url":str, "count":str}]

        # Handle quoted and in-reply-to posts ("referenced_tweets")
        quoted_post = None
        in_reply_to_post = None
        if 'referenced_tweets' in item:
            for ref in item["referenced_tweets"]:
                referenced_post_id = ref["id"]
                if referenced_post_id not in posts_by_id:
                    print(f"For post {post.post_id}, referenced post {referenced_post_id} not found in posts_by_id; skipping.")
                    continue
                referenced_post_item = posts_by_id[referenced_post_id]
                referenced_post = _parse_individual_post(referenced_post_item, media_by_key)

                if ref["type"] == "quoted":
                    assert quoted_post is None, "Multiple quoted posts found in a single post"
                    quoted_post = referenced_post
                elif ref["type"] == "replied_to":
                    assert in_reply_to_post is None, "Multiple in-reply-to posts found in a single post"
                    in_reply_to_post = referenced_post
                else:
                    raise ValueError(f"Unknown referenced tweet type: {ref['type']} (expected 'quoted' or 'replied_to')")

        postWithContext = PostWithContext(
            post=post,
            quoted_post=quoted_post,
            in_reply_to_post=in_reply_to_post,
            note_request_suggestions=note_request_suggestions,
            suggested_source_links_with_counts=suggested_source_links_with_counts,
        )
        posts.append(postWithContext)

    return posts


def get_posts_eligible_for_notes(max_results: int = 100) -> List[PostWithContext]:
    """Fetch the small feed, following pagination for page sizes of 100.

    Smaller requests fetch one page. Full-page requests continue until the
    API runs out of pages or MAX_PAGES is reached.
    """
    all_posts: List[PostWithContext] = []
    pagination_token: Optional[str] = None

    # Pagination only works with full pages: the API returns a usable
    # next_token only when the page size is 100, so smaller requests fetch a
    # single page. max_results is the per-page size, capped at the API max.
    paginate = max_results >= 100
    if not paginate:
        print(f"max_results={max_results} < 100; pagination disabled, fetching one page.")
    max_results = min(max_results, 100)

    for page_idx in range(MAX_PAGES):
        try:
            resp = _fetch_posts_eligible_page(
                max_results=max_results,
                pagination_token=pagination_token,
            )
        except Exception as e:
            # Propagate first-page failures; retain previously fetched posts if a
            # subsequent page fails.
            if page_idx == 0:
                raise
            print(f"Pagination stopped at page {page_idx + 1}: {e}. "
                  f"Proceeding with {len(all_posts)} posts gathered so far.")
            break

        page_posts = _parse_posts_eligible_response(resp)
        all_posts.extend(page_posts)
        print(f"Fetched page {page_idx + 1}: {len(page_posts)} posts (cumulative {len(all_posts)})")

        if not paginate:
            break

        pagination_token = resp.get("meta", {}).get("next_token")
        if not pagination_token:
            break

        time.sleep(1)
    else:
        print(f"Reached MAX_PAGES={MAX_PAGES} cap; stopping pagination.")

    return all_posts
