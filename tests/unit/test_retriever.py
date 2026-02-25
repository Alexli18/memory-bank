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
            {"chunk_id": "s1-0", "session_id": "s1", "text": "The authentication module handles JWT token validation", "ts_start": 0.0, "ts_end": 1.0, "quality_score": 0.8},
            {"chunk_id": "s1-1", "session_id": "s1", "text": "Database migration scripts run automatically on deploy", "ts_start": 2.0, "ts_end": 3.0, "quality_score": 0.8},
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
        # Use hash-based unique content to avoid near-dedup
        import hashlib
        chunks = [
            {"chunk_id": f"s1-{i}", "session_id": "s1", "text": f"unique content {hashlib.sha256(str(i).encode()).hexdigest()} end marker", "ts_end": float(i), "quality_score": 0.8}
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
            {"chunk_id": "s1-0", "session_id": "s1", "text": "Refactored the authentication middleware to use async handlers", "ts_end": 1.0, "quality_score": 0.8},
        ])
        _write_chunks(root, "s2", [
            {"chunk_id": "s2-0", "session_id": "s2", "text": "Added PostgreSQL connection pooling with configurable limits", "ts_end": 5.0, "quality_score": 0.8},
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

    def test_dedup_removes_overlapping_chunks(self, tmp_path: Path) -> None:
        """RecencyRetriever.retrieve() deduplicates overlapping chunks from hook fires."""
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {"chunk_id": "s1-0", "session_id": "s1", "text": "unique first chunk with enough text to pass", "ts_end": 1.0, "quality_score": 0.8},
            {"chunk_id": "s1-1", "session_id": "s1", "text": "duplicate chunk content from hook fire one", "ts_end": 2.0, "quality_score": 0.7},
            {"chunk_id": "s1-2", "session_id": "s1", "text": "duplicate chunk content from hook fire one", "ts_end": 3.0, "quality_score": 0.9},
        ])

        retriever = RecencyRetriever()
        result = retriever.retrieve(storage)

        assert len(result) == 2
        chunk_ids = {c.chunk_id for c in result}
        assert "s1-0" in chunk_ids
        # s1-2 kept (higher quality), s1-1 removed
        assert "s1-2" in chunk_ids
        assert "s1-1" not in chunk_ids

    def test_decay_prioritizes_recent_chunks(self, tmp_path: Path) -> None:
        """With decay enabled, recent chunks win over old ones even with equal quality."""
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        now = 1_000_000.0
        _write_chunks(root, "s1", [
            {
                "chunk_id": "old", "session_id": "s1",
                "text": "Old session chunk with authentication flow details here",
                "ts_start": 0.0, "ts_end": now - 30 * 86400,  # 30 days old
                "quality_score": 0.8,
            },
            {
                "chunk_id": "new", "session_id": "s1",
                "text": "New session chunk with deployment pipeline configuration",
                "ts_start": 0.0, "ts_end": now - 1 * 86400,  # 1 day old
                "quality_score": 0.8,
            },
        ])

        retriever = RecencyRetriever(half_life_days=14.0, max_excerpts=1)
        # Monkey-patch time for deterministic results
        import mb.decay as _decay
        _orig = _decay.time.time
        _decay.time.time = lambda: now  # type: ignore[assignment]
        try:
            result = retriever.retrieve(storage)
        finally:
            _decay.time.time = _orig

        assert len(result) == 1
        assert result[0].chunk_id == "new"

    def test_decay_skips_artifact_chunks(self, tmp_path: Path) -> None:
        """Artifact chunks are not decayed — their original quality is used."""
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        now = 1_000_000.0
        _write_chunks(root, "s1", [
            {
                "chunk_id": "artifact", "session_id": "s1",
                "text": "Architecture plan artifact with long text for testing",
                "ts_start": 0.0, "ts_end": now - 60 * 86400,  # 60 days old
                "quality_score": 0.8, "artifact_type": "plan",
            },
            {
                "chunk_id": "conv", "session_id": "s1",
                "text": "Conversation chunk about debugging session failures",
                "ts_start": 0.0, "ts_end": now - 60 * 86400,  # 60 days old
                "quality_score": 0.8,
            },
        ])

        retriever = RecencyRetriever(half_life_days=14.0, min_quality=0.3)
        import mb.decay as _decay
        _orig = _decay.time.time
        _decay.time.time = lambda: now  # type: ignore[assignment]
        try:
            result = retriever.retrieve(storage)
        finally:
            _decay.time.time = _orig

        # Artifact survives (quality 0.8 unchanged), conversation decayed heavily
        chunk_ids = {c.chunk_id for c in result}
        assert "artifact" in chunk_ids

    def test_decay_disabled_preserves_original_behavior(self, tmp_path: Path) -> None:
        """half_life_days=0.0 means no decay — same as default."""
        root = tmp_path / ".mb"
        storage = _make_storage(root)
        _write_chunks(root, "s1", [
            {
                "chunk_id": "s1-0", "session_id": "s1",
                "text": "Some old chunk that has enough text for filter",
                "ts_start": 0.0, "ts_end": 1.0, "quality_score": 0.8,
            },
        ])

        r_no_decay = RecencyRetriever(half_life_days=0.0)
        r_default = RecencyRetriever()
        result_no_decay = r_no_decay.retrieve(storage)
        result_default = r_default.retrieve(storage)

        assert len(result_no_decay) == len(result_default)
        assert result_no_decay[0].chunk_id == result_default[0].chunk_id
