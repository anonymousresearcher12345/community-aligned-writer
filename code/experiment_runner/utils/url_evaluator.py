"""URL extraction for note validation."""
from typing import List
import re

_URL_RE = re.compile(r"https?://[^\s)\]\"'<>]+")
_URL_TRAILING_STRIP = ".,;:!?\")]}'>"


def extract_urls(text: str) -> List[str]:
    """Return URLs found in `text` (http/https), in order of appearance.

    Trailing punctuation that is unlikely to be part of a URL is stripped.
    """
    if not text:
        return []
    return [m.rstrip(_URL_TRAILING_STRIP) for m in _URL_RE.findall(text)]
