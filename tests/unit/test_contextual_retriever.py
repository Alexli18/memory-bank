"""Tests for ContextualRetriever: episode-aware and failure-aware retrieval."""

from __future__ import annotations

import json


from mb.graph import EpisodeType
from mb.retriever import ContextualRetriever
from mb.store import NdjsonStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session(
    storage: NdjsonStorage,
    session_id: str,
    command: list[str],
    started_at: float,
    ended_at: float,
    exit_code: int,
    chunk_texts: list[str],
) -> None:
    """Write a session with meta + chunks to storage."""
    s_dir = storage.root / "sessions" / session_id
    s_dir.mkdir(parents=True)
    (s_dir / "meta.json").write_text(
        json.dumps({
            "session_id": session_id,
            "command": command,
            "cwd": "/tmp",
            "started_at": started_at,
            "ended_at": ended_at,
            "exit_code": exit_code,
        }) + "\n"
    )
    lines = []
    for i, text in enumerate(chunk_texts):
        lines.append(json.dumps({
            "chunk_id": f"{session_id}-{i}",
            "session_id": session_id,
            "index": i,
            "text": text,
            "ts_start": started_at + i,
            "ts_end": started_at + i + 1,
            "token_estimate": len(text) // 4,
            "quality_score": 0.8,
        }))
    (s_dir / "chunks.jsonl").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# retrieve_around_failure
# ---------------------------------------------------------------------------


class TestRetrieveAroundFailure:
    def test_returns_chunks_from_adjacent_sessions(self, storage: NdjsonStorage):
        # Build session (success)
        _create_session(storage, "s1", ["make"], 100.0, 110.0, 0, ["Build output line 1 with details"])
        # Test session (failure)
        _create_session(storage, "s2", ["pytest"], 111.0, 120.0, 1, ["FAILED test_foo - assertion error details"])
        # Debug session (follows failure)
        _create_session(storage, "s3", ["python", "-m", "pdb", "x.py"], 121.0, 130.0, 0, ["Debug session output data"])

        retriever = ContextualRetriever()
        chunks = retriever.retrieve_around_failure(storage, "s2")

        # Should include chunks from the failed session itself plus neighbors
        session_ids = {c.session_id for c in chunks}
        assert "s2" in session_ids
        # Adjacent sessions should be included
        assert "s1" in session_ids or "s3" in session_ids

    def test_returns_empty_for_nonexistent_session(self, storage: NdjsonStorage):
        retriever = ContextualRetriever()
        chunks = retriever.retrieve_around_failure(storage, "nonexistent")
        assert chunks == []

    def test_returns_only_failed_session_when_no_neighbors(self, storage: NdjsonStorage):
        _create_session(storage, "s1", ["pytest"], 100.0, 110.0, 1, ["Test failure output details here"])

        retriever = ContextualRetriever()
        chunks = retriever.retrieve_around_failure(storage, "s1")
        assert len(chunks) >= 1
        assert all(c.session_id == "s1" for c in chunks)


# ---------------------------------------------------------------------------
# retrieve_by_episode
# ---------------------------------------------------------------------------


class TestRetrieveByEpisode:
    def test_filters_by_episode_type(self, storage: NdjsonStorage):
        _create_session(storage, "s1", ["make"], 100.0, 110.0, 0, ["Build output from make command"])
        _create_session(storage, "s2", ["pytest"], 111.0, 120.0, 0, ["Test results from pytest run"])
        _create_session(storage, "s3", ["make", "install"], 121.0, 130.0, 0, ["Install output from make install"])

        retriever = ContextualRetriever()
        chunks = retriever.retrieve_by_episode(storage, EpisodeType.TEST)

        # Only chunks from TEST sessions
        session_ids = {c.session_id for c in chunks}
        assert "s2" in session_ids
        assert "s1" not in session_ids
        assert "s3" not in session_ids

    def test_returns_empty_for_no_matching_episodes(self, storage: NdjsonStorage):
        _create_session(storage, "s1", ["make"], 100.0, 110.0, 0, ["Build output from make command"])

        retriever = ContextualRetriever()
        chunks = retriever.retrieve_by_episode(storage, EpisodeType.DEPLOY)
        assert chunks == []

    def test_returns_chunks_from_multiple_sessions(self, storage: NdjsonStorage):
        _create_session(storage, "s1", ["pytest"], 100.0, 110.0, 0, ["Test run 1 output"])
        _create_session(storage, "s2", ["pytest", "-v"], 111.0, 120.0, 1, ["Test run 2 output"])

        retriever = ContextualRetriever()
        chunks = retriever.retrieve_by_episode(storage, EpisodeType.TEST)

        session_ids = {c.session_id for c in chunks}
        assert "s1" in session_ids
        assert "s2" in session_ids

    def test_max_chunks_limit(self, storage: NdjsonStorage):
        # Create sessions with many chunks
        texts = [f"Test output line {i} with enough content" for i in range(20)]
        _create_session(storage, "s1", ["pytest"], 100.0, 110.0, 0, texts)

        retriever = ContextualRetriever(max_chunks=5)
        chunks = retriever.retrieve_by_episode(storage, EpisodeType.TEST)
        assert len(chunks) <= 5

    def test_filters_claude_session_by_content(self, storage: NdjsonStorage):
        """Claude sessions are classified by content, not command."""
        _create_session(
            storage, "s1", ["claude"], 100.0, 110.0, 0,
            ["Running pytest test_foo PASSED test_bar FAILED"],
        )
        _create_session(
            storage, "s2", ["claude"], 111.0, 120.0, 0,
            ["Update README documentation and CHANGELOG"],
        )

        retriever = ContextualRetriever()
        test_chunks = retriever.retrieve_by_episode(storage, EpisodeType.TEST)
        docs_chunks = retriever.retrieve_by_episode(storage, EpisodeType.DOCS)

        test_sids = {c.session_id for c in test_chunks}
        docs_sids = {c.session_id for c in docs_chunks}

        assert "s1" in test_sids
        assert "s2" not in test_sids
        assert "s2" in docs_sids
        assert "s1" not in docs_sids
