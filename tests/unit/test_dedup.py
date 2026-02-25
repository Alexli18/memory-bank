"""Tests for _deduplicate_chunks in mb.retriever."""

from __future__ import annotations

from mb.models import Chunk
from mb.retriever import _deduplicate_chunks


def _chunk(
    chunk_id: str = "c1",
    text: str = "some meaningful text content here",
    quality_score: float = 0.8,
    ts_end: float = 1.0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        session_id="s1",
        index=0,
        text=text,
        ts_start=0.0,
        ts_end=ts_end,
        token_estimate=len(text) // 4,
        quality_score=quality_score,
    )


class TestExactDedup:
    def test_exact_duplicates_removed(self) -> None:
        """Identical text (after normalization) keeps only one chunk."""
        a = _chunk(chunk_id="a", text="Hello world this is a test", quality_score=0.7)
        b = _chunk(chunk_id="b", text="Hello world this is a test", quality_score=0.9)

        result = _deduplicate_chunks([a, b])

        assert len(result) == 1
        assert result[0].chunk_id == "b"  # higher quality

    def test_exact_dedup_whitespace_normalized(self) -> None:
        """Whitespace differences are treated as exact duplicates."""
        a = _chunk(chunk_id="a", text="Hello   world\n\ttest", quality_score=0.5)
        b = _chunk(chunk_id="b", text="hello world test", quality_score=0.6)

        result = _deduplicate_chunks([a, b])

        assert len(result) == 1
        assert result[0].chunk_id == "b"

    def test_exact_dedup_tiebreak_by_ts_end(self) -> None:
        """Same quality_score: keep the chunk with later ts_end."""
        a = _chunk(chunk_id="a", text="same text content here", quality_score=0.8, ts_end=10.0)
        b = _chunk(chunk_id="b", text="same text content here", quality_score=0.8, ts_end=20.0)

        result = _deduplicate_chunks([a, b])

        assert len(result) == 1
        assert result[0].chunk_id == "b"  # later ts_end

    def test_different_texts_kept(self) -> None:
        """Non-duplicate chunks are all retained."""
        a = _chunk(chunk_id="a", text="Alpha content is unique here")
        b = _chunk(chunk_id="b", text="Beta content is different here")

        result = _deduplicate_chunks([a, b])

        assert len(result) == 2


class TestNearDedup:
    def test_near_duplicates_above_threshold_removed(self) -> None:
        """Chunks with SequenceMatcher ratio > 0.70 are deduped."""
        base = "This is a long chunk of text that contains important information about the project architecture and design patterns used in the codebase"
        # Change only a few characters to stay above 0.70
        variant = "This is a long chunk of text that contains important information about the project architecture and design patterns used in this codebase"

        a = _chunk(chunk_id="a", text=base, quality_score=0.7)
        b = _chunk(chunk_id="b", text=variant, quality_score=0.9)

        result = _deduplicate_chunks([a, b])

        assert len(result) == 1
        assert result[0].chunk_id == "b"  # higher quality kept

    def test_near_duplicates_below_threshold_kept(self) -> None:
        """Chunks with ratio <= 0.70 are both kept."""
        a = _chunk(chunk_id="a", text="Alpha bravo charlie delta echo foxtrot golf hotel")
        b = _chunk(chunk_id="b", text="Zulu yankee xray whiskey victor uniform tango sierra")

        result = _deduplicate_chunks([a, b])

        assert len(result) == 2

    def test_200_char_overlap_from_same_session(self) -> None:
        """200-char overlap in 2048-char chunks has ~10-30% ratio — kept."""
        # Simulate 2048-char chunks with 200-char overlap
        unique_a = "A" * 1848
        overlap = "X" * 200
        unique_b = "B" * 1848
        chunk_a_text = unique_a + overlap
        chunk_b_text = overlap + unique_b

        a = _chunk(chunk_id="a", text=chunk_a_text, quality_score=0.8)
        b = _chunk(chunk_id="b", text=chunk_b_text, quality_score=0.8)

        result = _deduplicate_chunks([a, b])

        # Overlap ratio ~10%, both should survive
        assert len(result) == 2


class TestEdgeCases:
    def test_empty_input(self) -> None:
        assert _deduplicate_chunks([]) == []

    def test_single_chunk_passthrough(self) -> None:
        c = _chunk(chunk_id="only")
        result = _deduplicate_chunks([c])
        assert result == [c]

    def test_order_preserved(self) -> None:
        """Surviving chunks keep their original order."""
        a = _chunk(chunk_id="a", text="The authentication system uses JWT tokens for stateless session management", ts_end=3.0)
        b = _chunk(chunk_id="b", text="Database migrations are applied automatically on startup via Alembic", ts_end=2.0)
        c = _chunk(chunk_id="c", text="Frontend components communicate through a Redux store and middleware", ts_end=1.0)

        result = _deduplicate_chunks([a, b, c])

        assert [c.chunk_id for c in result] == ["a", "b", "c"]

    def test_multiple_exact_duplicates(self) -> None:
        """Three copies of the same text keep only one."""
        a = _chunk(chunk_id="a", text="repeated content", quality_score=0.5, ts_end=1.0)
        b = _chunk(chunk_id="b", text="repeated content", quality_score=0.9, ts_end=2.0)
        c = _chunk(chunk_id="c", text="repeated content", quality_score=0.7, ts_end=3.0)

        result = _deduplicate_chunks([a, b, c])

        assert len(result) == 1
        # b has highest quality_score (0.9)
        assert result[0].chunk_id == "b"
