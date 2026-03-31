"""Shared utility functions for text matching and verification."""
import os
import re
import math
from typing import List

import tiktoken
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "minimax-m2.7-highspeed")
DEFAULT_VERIFY_MODEL = os.getenv("DEFAULT_VERIFY_MODEL", DEFAULT_LLM_MODEL)
DEFAULT_CONTEXT1_BASE_URL = os.getenv("CONTEXT1_BASE_URL", "https://context1.pkking.computer")
DEFAULT_CONTEXT1_HOSTNAME = os.getenv("CONTEXT1_HOSTNAME", "context1.pkking.computer")
DEFAULT_FRONTIER_WS_URL = os.getenv("FRONTIER_WS_URL", "ws://127.0.0.1:4500")

# Token counting
_tiktoken_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_tiktoken_encoder.encode(text))


def parse_tag(content: str, tag: str) -> str | None:
    """Parse a single XML tag from content."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
    return match.group(1).strip() if match else None


def get_anthropic_client():
    """Get an Anthropic-compatible client, preferring MiniMax when configured."""
    minimax_api_key = os.getenv("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        return Anthropic(api_key=minimax_api_key, base_url=base_url)

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        return Anthropic(api_key=anthropic_api_key)

    raise RuntimeError(
        "No LLM provider configured. Set MINIMAX_API_KEY for the default MiniMax path "
        "or ANTHROPIC_API_KEY for the Anthropic fallback."
    )


def get_embedding_client():
    """Get embedding client."""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_context1_client():
    """Get the Context-1 REST client."""
    from .context1_client import Context1Client
    return Context1Client()


def get_context1_base_url() -> str:
    """Get the base URL for the hosted Context-1 service."""
    return os.getenv("CONTEXT1_BASE_URL", DEFAULT_CONTEXT1_BASE_URL)


def get_context1_hostname() -> str:
    """Get the public hostname for the hosted Context-1 service."""
    return os.getenv("CONTEXT1_HOSTNAME", DEFAULT_CONTEXT1_HOSTNAME)


def get_frontier_ws_url() -> str:
    """Get the local websocket URL for the frontier reasoning model."""
    return os.getenv("FRONTIER_WS_URL", DEFAULT_FRONTIER_WS_URL)


def strip_links(text: str) -> str:
    """Remove hyperlinks from text while preserving link text."""
    if not text:
        return ""
    result = text
    result = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", result)
    result = re.sub(r"<a\s[^>]*>([^<]*)</a>", r"\1", result, flags=re.IGNORECASE)
    result = re.sub(r"https?://[^\s<>\[\]\"']+", " ", result)
    result = re.sub(r"ftp://[^\s<>\[\]\"']+", " ", result)
    result = re.sub(r"  +", " ", result)
    return result


def normalize_for_matching(text: str) -> str:
    """Normalize text for matching by handling unicode variants and whitespace."""
    if not text:
        return ""
    result = text.lower()
    result = re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u2035]", "'", result)
    result = re.sub(r"[\u201C\u201D\u201E\u201F\u2033\u2036]", '"', result)
    result = re.sub(r"[\u2013\u2014]", "-", result)
    result = result.replace("\u2026", "...")
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"[^\w\s'\-]", "", result)
    return result.strip()


def text_contains_quote(content: str, quote: str) -> bool:
    """Check if content contains the quote using multiple matching strategies."""
    if not content or not quote:
        return False
    quote = quote.strip()
    if not quote:
        return False

    if quote.lower() in content.lower():
        return True

    normalized_content = re.sub(r"\s+", " ", content.lower())
    normalized_quote = re.sub(r"\s+", " ", quote.lower())
    if normalized_quote in normalized_content:
        return True

    content_no_links = strip_links(content)
    if quote.lower() in content_no_links.lower():
        return True

    normalized_content_no_links = re.sub(r"\s+", " ", content_no_links.lower())
    if normalized_quote in normalized_content_no_links:
        return True

    fully_normalized_content = normalize_for_matching(content_no_links)
    fully_normalized_quote = normalize_for_matching(quote)
    if fully_normalized_quote in fully_normalized_content:
        return True

    quote_words = [w for w in fully_normalized_quote.split() if len(w) > 3]
    if len(quote_words) >= 3:
        words_to_check = quote_words[:5]
        pattern = r".*?".join(re.escape(w) for w in words_to_check)
        if re.search(pattern, fully_normalized_content):
            return True

    return False


def count_matching_quotes(quotes: List[str], content: str) -> int:
    """Count how many quotes are found in the content."""
    return sum(1 for q in quotes if text_contains_quote(content, q))


def min_required_matches(total: int) -> int:
    """Calculate minimum required matches (2/3 rounded up)."""
    if total == 0:
        return 0
    return math.ceil(total * 2 / 3)


def parse_quotes(content: str, tag: str) -> List[str] | None:
    """Parse multiple <q> tags from within a parent tag.

    Returns None if the content explicitly indicates no relevant quotes.
    Returns an empty list if the tag is not found.
    """
    parent_match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
    if parent_match:
        parent_content = parent_match.group(1).strip()
        if parent_content.lower() == 'none':
            return None
        q_matches = re.findall(r'<q>(.*?)</q>', parent_content, re.DOTALL)
        quotes = [q.strip() for q in q_matches if q.strip().lower() != 'none']
        return quotes
    return []
