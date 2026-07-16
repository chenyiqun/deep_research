from __future__ import annotations

import html
import re
from urllib.parse import urlsplit, urlunsplit


# URLs in Chinese prose are frequently followed by full-width punctuation.  A
# whitespace-only URL regex will otherwise consume the rest of the sentence.
URL_RE = re.compile(r"https?://[^\s<>\[\]{}\"'`\)）】》」』，。；：！？]+", re.IGNORECASE)
MARKDOWN_URL_RE = re.compile(r"\[[^\]\n]*\]\((https?://[^\s\)]+)\)", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}>，。；：！？）】》」』"


def clean_url(url: str) -> str:
    """Remove prose punctuation without rewriting the source URL itself."""

    value = html.unescape(str(url or "").strip()).rstrip(TRAILING_URL_PUNCTUATION)
    return value


def canonicalize_url(url: str) -> str:
    """Return a comparison key for a supplied/cited HTTP URL.

    Path and query are preserved exactly.  Only scheme/host case, default ports,
    fragments, and surrounding prose punctuation are normalized.
    """

    value = clean_url(url)
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except Exception:
        return value
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return value
    host = parsed.hostname.lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return value
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path, parsed.query, ""))


def extract_urls(text: str) -> list[str]:
    """Extract ordered, de-duplicated URLs from Markdown or multilingual prose."""

    source = str(text or "")
    matches: list[tuple[int, str]] = []
    for match in MARKDOWN_URL_RE.finditer(source):
        matches.append((match.start(1), clean_url(match.group(1))))
    for match in URL_RE.finditer(source):
        matches.append((match.start(), clean_url(match.group(0))))
    matches.sort(key=lambda item: item[0])

    output: list[str] = []
    seen: set[str] = set()
    for _, url in matches:
        key = canonicalize_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(url)
    return output
