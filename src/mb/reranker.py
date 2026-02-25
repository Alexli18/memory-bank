"""LLM-based reranker for search results.

Uses Ollama chat to score search candidates by relevance to a query,
providing a second-pass filter after vector cosine similarity.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from mb.models import SearchResult

if TYPE_CHECKING:
    from mb.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a relevance judge. Given a search query and a list of text snippets, "
    "rate each snippet's relevance to the query on a scale of 0 to 10.\n"
    "0 = completely irrelevant, 10 = perfectly relevant.\n"
    "Respond ONLY with JSON: {\"scores\": [<int>, ...]}\n"
    "The scores array must have exactly one integer per snippet, in order."
)


def _build_user_prompt(query: str, candidates: list[SearchResult]) -> str:
    """Build the user prompt with query and numbered snippets."""
    lines = [f"Query: {query}", "", "Snippets:"]
    for i, c in enumerate(candidates):
        snippet = c.text[:300].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)


def _parse_scores(raw: dict | str, expected_count: int) -> list[float]:
    """Parse and normalize LLM scores from 0-10 to 0.0-1.0.

    Handles:
    - dict with "scores" key
    - string that can be parsed as JSON
    - Clamps values to [0, 10] before normalizing
    """
    if isinstance(raw, str):
        raw = json.loads(raw)

    if not isinstance(raw, dict) or "scores" not in raw:
        raise ValueError(f"Expected dict with 'scores' key, got: {type(raw)}")

    scores = raw["scores"]
    if not isinstance(scores, list):
        raise ValueError(f"Expected list of scores, got: {type(scores)}")

    if len(scores) != expected_count:
        raise ValueError(
            f"Expected {expected_count} scores, got {len(scores)}"
        )

    result: list[float] = []
    for s in scores:
        val = float(s)
        val = max(0.0, min(10.0, val))  # clamp
        result.append(val / 10.0)  # normalize to 0.0-1.0
    return result


def rerank(
    query: str,
    candidates: list[SearchResult],
    ollama_client: OllamaClient,
    top_k: int = 5,
) -> list[SearchResult]:
    """Rerank search candidates using LLM relevance scoring.

    On any error (Ollama unavailable, timeout, parse failure), falls back
    to the original vector-based ordering with a warning log.

    Args:
        query: The search query.
        candidates: Search results from vector search.
        ollama_client: OllamaClient instance.
        top_k: Number of results to return.

    Returns:
        Reranked list of SearchResult (top_k items).
    """
    if not candidates:
        return []

    from mb.ollama_client import OllamaError

    try:
        user_prompt = _build_user_prompt(query, candidates)
        response = ollama_client.chat(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            as_json=True,
            temperature=0.0,
        )
        scores = _parse_scores(response, len(candidates))
    except (OllamaError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Reranker failed, falling back to vector scores: %s", exc)
        return candidates[:top_k]

    # Build reranked results with LLM scores
    scored = []
    for candidate, llm_score in zip(candidates, scores):
        reranked = SearchResult(
            chunk_id=candidate.chunk_id,
            session_id=candidate.session_id,
            index=candidate.index,
            text=candidate.text,
            ts_start=candidate.ts_start,
            ts_end=candidate.ts_end,
            token_estimate=candidate.token_estimate,
            quality_score=candidate.quality_score,
            score=llm_score,
            artifact_type=candidate.artifact_type,
        )
        scored.append(reranked)

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
