"""Tests for mb.retriever — RecencyRetriever with mock Storage."""

from __future__ import annotations

import json
from pathlib import Path

from mb.retriever import RecencyRetriever
from mb.store import NdjsonStorage
from mb.models import Chunk


def _make_storage(tmp_path: Path) -> NdjsonStorage:
    """Create a minimal NdjsonStorage for tests."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return NdjsonStorage(tmp_path)


def _write_chunks(root: Path, session_id: str, chunks: list[dict]) -> None:
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    with (session_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


class TestRecencyRetriever:
    def test_returns_chunks_sorted_by_recency(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "session_id": "s1", "text": "first chunk with enough text here", "ts_start": 0.0, "ts_end": 1.0, "quality_score": 0.8},
            {"chunk_id": "s1-1", "session_id": "s1", "text": "second chunk with enough text here", "ts_start": 2.0, "ts_end": 3.0, "quality_score": 0.8},
        ])

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert len(result) == 2
        assert all(isinstance(c, Chunk) for c in result)
        # Most recent first
        assert result[0].chunk_id == "s1-1"
        assert result[1].chunk_id == "s1-0"

    def test_filters_low_quality(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "text": "good readable text that is long enough", "ts_end": 1.0, "quality_score": 0.8},
            {"chunk_id": "s1-1", "text": "---...!!!---...!!!---...!!!---", "ts_end": 2.0, "quality_score": 0.0},
        ])

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert len(result) == 1
        assert result[0].chunk_id == "s1-0"

    def test_filters_short_text(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "text": "This chunk has enough text to pass the filter", "ts_end": 1.0, "quality_score": 0.8},
            {"chunk_id": "s1-1", "text": "tiny", "ts_end": 2.0, "quality_score": 1.0},
        ])

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert len(result) == 1
        assert result[0].chunk_id == "s1-0"

    def test_respects_max_excerpts(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        chunks = [
            {"chunk_id": f"s1-{i}", "session_id": "s1", "text": f"chunk number {i} with enough text to pass filter", "ts_end": float(i), "quality_score": 0.8}
            for i in range(50)
        ]
        _write_chunks(root, "s1", chunks)

        retriever = RecencyRetriever(max_excerpts=5)
        result = retriever.retrieve(storage)

        assert len(result) == 5
        # Should be the 5 most recent
        ts_ends = [c.ts_end for c in result]
        assert ts_ends == [49.0, 48.0, 47.0, 46.0, 45.0]

    def test_empty_storage(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert result == []

    def test_multiple_sessions_merged(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "session_id": "s1", "text": "data from session one that passes filters", "ts_end": 1.0, "quality_score": 0.8},
        ])
        _write_chunks(root, "s2", [
            {"chunk_id": "s2-0", "session_id": "s2", "text": "data from session two that passes filters", "ts_end": 5.0, "quality_score": 0.8},
        ])

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert len(result) == 2
        assert result[0].session_id == "s2"  # ts_end=5.0 first
        assert result[1].session_id == "s1"  # ts_end=1.0 second

    def test_backward_compat_no_quality_score(self, tmp_path: Path) -> None:
        """Chunks without quality_score field get scored on the fly."""
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "text": "hello world readable content that passes both filters", "ts_end": 1.0},
            {"chunk_id": "s1-1", "text": "......!!!......!!!......!!!...", "ts_end": 2.0},
        ])

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert len(result) == 1
        assert result[0].chunk_id == "s1-0"

    def test_custom_min_quality(self, tmp_path: Path) -> None:
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "text": "moderate quality text content here yes", "ts_end": 1.0, "quality_score": 0.5},
            {"chunk_id": "s1-1", "text": "high quality text content here with more words", "ts_end": 2.0, "quality_score": 0.9},
        ])

        retriever = RecencyRetriever(min_quality=0.6)
        result = retriever.retrieve(storage)

        assert len(result) == 1
        assert result[0].chunk_id == "s1-1"
