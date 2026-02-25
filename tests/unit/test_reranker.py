"""Tests for mb.reranker — LLM-based search result reranking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mb.models import SearchResult
from mb.ollama_client import OllamaNotRunningError, OllamaTimeoutError
from mb.reranker import _parse_scores, rerank


def _result(
    chunk_id: str = "c1",
    text: str = "some text",
    score: float = 0.5,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        session_id="s1",
        index=0,
        text=text,
        ts_start=0.0,
        ts_end=1.0,
        token_estimate=10,
        quality_score=0.8,
        score=score,
    )


class TestRerank:
    def test_reorders_by_llm_score(self) -> None:
        """Candidates are reordered by LLM relevance scores."""
        candidates = [
            _result(chunk_id="a", text="low relevance", score=0.9),
            _result(chunk_id="b", text="high relevance", score=0.8),
            _result(chunk_id="c", text="medium relevance", score=0.7),
        ]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"scores": [2, 9, 5]}

        result = rerank("test query", candidates, mock_client, top_k=3)

        assert [r.chunk_id for r in result] == ["b", "c", "a"]

    def test_normalizes_scores_0_10_to_0_1(self) -> None:
        """LLM scores 0-10 are normalized to 0.0-1.0."""
        candidates = [_result(chunk_id="a")]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"scores": [7]}

        result = rerank("query", candidates, mock_client, top_k=1)

        assert result[0].score == pytest.approx(0.7)

    def test_respects_top_k(self) -> None:
        """Only top_k results are returned."""
        candidates = [
            _result(chunk_id="a"),
            _result(chunk_id="b"),
            _result(chunk_id="c"),
        ]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"scores": [8, 3, 5]}

        result = rerank("query", candidates, mock_client, top_k=2)

        assert len(result) == 2
        assert result[0].chunk_id == "a"
        assert result[1].chunk_id == "c"

    def test_fallback_on_ollama_not_running(self) -> None:
        """Falls back to original ordering when Ollama is down."""
        candidates = [_result(chunk_id="a"), _result(chunk_id="b")]
        mock_client = MagicMock()
        mock_client.chat.side_effect = OllamaNotRunningError("down")

        result = rerank("query", candidates, mock_client, top_k=2)

        assert [r.chunk_id for r in result] == ["a", "b"]

    def test_fallback_on_ollama_timeout(self) -> None:
        """Falls back to original ordering on timeout."""
        candidates = [_result(chunk_id="a"), _result(chunk_id="b")]
        mock_client = MagicMock()
        mock_client.chat.side_effect = OllamaTimeoutError("timeout")

        result = rerank("query", candidates, mock_client, top_k=2)

        assert [r.chunk_id for r in result] == ["a", "b"]

    def test_fallback_on_parse_error(self) -> None:
        """Falls back when LLM returns unparseable response."""
        candidates = [_result(chunk_id="a")]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"not_scores": [1]}

        result = rerank("query", candidates, mock_client, top_k=1)

        assert len(result) == 1
        assert result[0].chunk_id == "a"

    def test_fallback_on_wrong_score_count(self) -> None:
        """Falls back when LLM returns wrong number of scores."""
        candidates = [_result(chunk_id="a"), _result(chunk_id="b")]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"scores": [5]}  # expected 2

        result = rerank("query", candidates, mock_client, top_k=2)

        assert len(result) == 2
        # Original order preserved
        assert [r.chunk_id for r in result] == ["a", "b"]

    def test_empty_candidates(self) -> None:
        """Empty input returns empty output."""
        mock_client = MagicMock()
        assert rerank("query", [], mock_client) == []

    def test_prompt_contains_query_and_snippets(self) -> None:
        """Verify the prompt sent to the LLM contains query and snippets."""
        candidates = [
            _result(chunk_id="a", text="alpha content"),
            _result(chunk_id="b", text="beta content"),
        ]
        mock_client = MagicMock()
        mock_client.chat.return_value = {"scores": [5, 5]}

        rerank("architecture design", candidates, mock_client, top_k=2)

        call_kwargs = mock_client.chat.call_args
        user_prompt = call_kwargs[1].get("user_prompt") or call_kwargs[0][0]
        assert "architecture design" in user_prompt
        assert "alpha content" in user_prompt
        assert "beta content" in user_prompt
        assert "[0]" in user_prompt
        assert "[1]" in user_prompt


class TestParseScores:
    def test_valid_dict(self) -> None:
        result = _parse_scores({"scores": [0, 5, 10]}, 3)
        assert result == [0.0, 0.5, 1.0]

    def test_clamping(self) -> None:
        """Scores outside 0-10 are clamped."""
        result = _parse_scores({"scores": [-5, 15]}, 2)
        assert result == [0.0, 1.0]

    def test_string_scores(self) -> None:
        """String values in scores list are converted to float."""
        result = _parse_scores({"scores": ["3", "7"]}, 2)
        assert result == [pytest.approx(0.3), pytest.approx(0.7)]

    def test_missing_scores_key(self) -> None:
        with pytest.raises(ValueError, match="scores"):
            _parse_scores({"wrong": [1]}, 1)

    def test_wrong_count(self) -> None:
        with pytest.raises(ValueError, match="Expected 3 scores"):
            _parse_scores({"scores": [1, 2]}, 3)

    def test_string_input_parsed_as_json(self) -> None:
        """String input is parsed as JSON first."""
        result = _parse_scores('{"scores": [5]}', 1)
        assert result == [0.5]
