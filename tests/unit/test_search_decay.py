"""Tests for decay boost in VectorIndex.search()."""

from __future__ import annotations

import time

import numpy as np

from mb.search import VectorIndex, VECTOR_DIM


def _make_index(tmp_path, entries):
    """Create a VectorIndex with given entries.

    Each entry: {"vector": list[float], "metadata": dict}
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index = VectorIndex(index_dir)
    for e in entries:
        index.add(e["vector"], e["metadata"])
    return index


def _random_vector(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.randn(VECTOR_DIM).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


class TestSearchDecayBoost:
    def test_no_boost_when_disabled(self, tmp_path):
        """half_life_days=0 means no decay boost — raw cosine scores."""
        now = time.time()
        vec = _random_vector(42)
        entries = [
            {"vector": vec, "metadata": {"chunk_id": "c1", "session_id": "s1", "text": "old", "ts_start": 0, "ts_end": now - 30 * 86400, "quality_score": 0.8}},
            {"vector": vec, "metadata": {"chunk_id": "c2", "session_id": "s1", "text": "new", "ts_start": 0, "ts_end": now - 1 * 86400, "quality_score": 0.8}},
        ]
        index = _make_index(tmp_path, entries)
        results = index.search(vec, top_k=2, half_life_days=0.0)

        # Without boost, same vector → same cosine score
        assert abs(results[0].score - results[1].score) < 1e-5

    def test_boost_favors_recent(self, tmp_path):
        """With decay boost, newer chunk should score higher."""
        now = time.time()
        vec = _random_vector(42)
        entries = [
            {"vector": vec, "metadata": {"chunk_id": "old", "session_id": "s1", "text": "old", "ts_start": 0, "ts_end": now - 30 * 86400, "quality_score": 0.8}},
            {"vector": vec, "metadata": {"chunk_id": "new", "session_id": "s1", "text": "new", "ts_start": 0, "ts_end": now - 1 * 86400, "quality_score": 0.8}},
        ]
        index = _make_index(tmp_path, entries)
        results = index.search(vec, top_k=2, half_life_days=14.0)

        scores = {r.chunk_id: r.score for r in results}
        assert scores["new"] > scores["old"]

    def test_no_decay_flag_disables_boost(self, tmp_path):
        """no_decay=True returns raw cosine scores even with half_life_days > 0."""
        now = time.time()
        vec = _random_vector(42)
        entries = [
            {"vector": vec, "metadata": {"chunk_id": "c1", "session_id": "s1", "text": "a", "ts_start": 0, "ts_end": now - 30 * 86400}},
            {"vector": vec, "metadata": {"chunk_id": "c2", "session_id": "s1", "text": "b", "ts_start": 0, "ts_end": now - 1 * 86400}},
        ]
        index = _make_index(tmp_path, entries)
        results = index.search(vec, top_k=2, half_life_days=14.0, no_decay=True)

        assert abs(results[0].score - results[1].score) < 1e-5

    def test_artifact_chunks_not_boosted(self, tmp_path):
        """Artifact chunks should not receive decay boost."""
        now = time.time()
        vec = _random_vector(42)
        entries = [
            {"vector": vec, "metadata": {"chunk_id": "art", "session_id": "s1", "text": "plan", "ts_start": 0, "ts_end": now - 1 * 86400, "artifact_type": "plan"}},
            {"vector": vec, "metadata": {"chunk_id": "sess", "session_id": "s1", "text": "conv", "ts_start": 0, "ts_end": now - 1 * 86400}},
        ]
        index = _make_index(tmp_path, entries)
        results = index.search(vec, top_k=2, half_life_days=14.0)

        scores = {r.chunk_id: r.score for r in results}
        # Session chunk gets boosted, artifact does not → session scores higher
        assert scores["sess"] > scores["art"]

    def test_boost_magnitude_limited(self, tmp_path):
        """Max boost is 10% (alpha=0.1 * decay_factor=1.0 for fresh chunks)."""
        now = time.time()
        vec = _random_vector(42)
        entries = [
            {"vector": vec, "metadata": {"chunk_id": "c1", "session_id": "s1", "text": "a", "ts_start": 0, "ts_end": now}},
        ]
        index = _make_index(tmp_path, entries)

        results_no_boost = index.search(vec, top_k=1, half_life_days=0.0)
        results_with_boost = index.search(vec, top_k=1, half_life_days=14.0)

        raw = results_no_boost[0].score
        boosted = results_with_boost[0].score
        boost_pct = (boosted - raw) / raw
        assert abs(boost_pct - 0.1) < 0.01  # ~10% boost
