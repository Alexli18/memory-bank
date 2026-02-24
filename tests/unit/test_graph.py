"""Tests for session graph: episode classification, error detection, related sessions."""

from __future__ import annotations

import json

import pytest

from mb.graph import EpisodeType, SessionGraph, SessionNode
from mb.models import Chunk, SessionMeta
from mb.store import NdjsonStorage


# ---------------------------------------------------------------------------
# EpisodeType enum
# ---------------------------------------------------------------------------


class TestEpisodeType:
    def test_all_values_exist(self):
        assert set(EpisodeType) == {
            EpisodeType.BUILD,
            EpisodeType.TEST,
            EpisodeType.DEPLOY,
            EpisodeType.DEBUG,
            EpisodeType.REFACTOR,
            EpisodeType.EXPLORE,
            EpisodeType.CONFIG,
            EpisodeType.DOCS,
            EpisodeType.REVIEW,
        }

    def test_values_are_strings(self):
        for ep in EpisodeType:
            assert isinstance(ep.value, str)


# ---------------------------------------------------------------------------
# classify_episode
# ---------------------------------------------------------------------------


class TestClassifyEpisode:
    @pytest.fixture()
    def graph(self) -> SessionGraph:
        return SessionGraph()

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            (["pytest"], EpisodeType.TEST),
            (["pytest", "-v", "tests/"], EpisodeType.TEST),
            (["python", "-m", "pytest"], EpisodeType.TEST),
            (["jest"], EpisodeType.TEST),
            (["cargo", "test"], EpisodeType.TEST),
            (["go", "test", "./..."], EpisodeType.TEST),
            (["npm", "test"], EpisodeType.TEST),
            (["make", "test"], EpisodeType.TEST),
            (["make"], EpisodeType.BUILD),
            (["cmake", "--build", "."], EpisodeType.BUILD),
            (["cargo", "build"], EpisodeType.BUILD),
            (["go", "build", "."], EpisodeType.BUILD),
            (["npm", "run", "build"], EpisodeType.BUILD),
            (["docker", "build", "."], EpisodeType.BUILD),
            (["docker", "push", "img"], EpisodeType.DEPLOY),
            (["kubectl", "apply", "-f", "x.yaml"], EpisodeType.DEPLOY),
            (["terraform", "apply"], EpisodeType.DEPLOY),
            (["ansible-playbook", "x.yml"], EpisodeType.DEPLOY),
            (["gdb", "./a.out"], EpisodeType.DEBUG),
            (["lldb", "./a.out"], EpisodeType.DEBUG),
            (["python", "-m", "pdb", "script.py"], EpisodeType.DEBUG),
            (["claude"], EpisodeType.REFACTOR),
            (["claude", "code"], EpisodeType.REFACTOR),
        ],
    )
    def test_classify_common_commands(
        self,
        graph: SessionGraph,
        command: list[str],
        expected: EpisodeType,
    ):
        meta = SessionMeta(
            session_id="s1",
            command=command,
            cwd="/tmp",
            started_at=1.0,
        )
        assert graph.classify_episode(meta) == expected

    def test_unknown_command_defaults_to_build(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["unknown-tool"],
            cwd="/tmp",
            started_at=1.0,
        )
        assert graph.classify_episode(meta) == EpisodeType.BUILD


# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------


class TestErrorDetection:
    @pytest.fixture()
    def graph(self) -> SessionGraph:
        return SessionGraph()

    def test_nonzero_exit_code_is_error(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=1,
        )
        assert graph.detect_error(meta, []) is True

    def test_zero_exit_code_no_error(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=0,
        )
        assert graph.detect_error(meta, []) is False

    def test_none_exit_code_no_error(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=None,
        )
        assert graph.detect_error(meta, []) is False

    def test_error_keyword_in_chunks(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=0,
        )
        chunks = [
            Chunk(
                chunk_id="s1-0",
                session_id="s1",
                index=0,
                text="Traceback (most recent call last):\n  File 'x.py'\nError: failed",
                ts_start=1.0,
                ts_end=2.0,
                token_estimate=20,
                quality_score=0.5,
            ),
        ]
        assert graph.detect_error(meta, chunks) is True

    def test_error_summary_from_exit_code(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=1,
        )
        summary = graph.extract_error_summary(meta, [])
        assert summary is not None
        assert "exit code 1" in summary.lower()

    def test_error_summary_from_chunks(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=0,
        )
        chunks = [
            Chunk(
                chunk_id="s1-0",
                session_id="s1",
                index=0,
                text="Some output\nTraceback (most recent call last):\n  ValueError: bad value",
                ts_start=1.0,
                ts_end=2.0,
                token_estimate=20,
                quality_score=0.5,
            ),
        ]
        summary = graph.extract_error_summary(meta, chunks)
        assert summary is not None
        assert "Traceback" in summary or "ValueError" in summary

    def test_no_error_returns_none_summary(self, graph: SessionGraph):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
            exit_code=0,
        )
        summary = graph.extract_error_summary(meta, [])
        assert summary is None


# ---------------------------------------------------------------------------
# SessionNode
# ---------------------------------------------------------------------------


class TestSessionNode:
    def test_create_node(self):
        meta = SessionMeta(
            session_id="s1",
            command=["pytest"],
            cwd="/tmp",
            started_at=1.0,
        )
        node = SessionNode(
            meta=meta,
            episode_type=EpisodeType.TEST,
            has_error=False,
            error_summary=None,
            related_sessions=[],
        )
        assert node.meta is meta
        assert node.episode_type == EpisodeType.TEST
        assert node.has_error is False
        assert node.related_sessions == []


# ---------------------------------------------------------------------------
# find_related_sessions
# ---------------------------------------------------------------------------


class TestFindRelatedSessions:
    @pytest.fixture()
    def graph(self) -> SessionGraph:
        return SessionGraph()

    def test_temporal_neighbors_linked(self, graph: SessionGraph):
        metas = [
            SessionMeta(session_id="s1", command=["make"], cwd="/tmp", started_at=100.0, ended_at=110.0, exit_code=0),
            SessionMeta(session_id="s2", command=["pytest"], cwd="/tmp", started_at=111.0, ended_at=120.0, exit_code=1),
            SessionMeta(session_id="s3", command=["python", "-m", "pdb", "x.py"], cwd="/tmp", started_at=121.0, ended_at=130.0, exit_code=0),
        ]
        related = graph.find_related_sessions("s2", metas)
        assert "s1" in related
        assert "s3" in related

    def test_distant_sessions_not_linked(self, graph: SessionGraph):
        metas = [
            SessionMeta(session_id="s1", command=["make"], cwd="/tmp", started_at=100.0, ended_at=110.0, exit_code=0),
            SessionMeta(session_id="s2", command=["pytest"], cwd="/tmp", started_at=100000.0, ended_at=100010.0, exit_code=0),
        ]
        related = graph.find_related_sessions("s1", metas)
        assert "s2" not in related

    def test_self_not_in_related(self, graph: SessionGraph):
        metas = [
            SessionMeta(session_id="s1", command=["make"], cwd="/tmp", started_at=100.0, ended_at=110.0, exit_code=0),
        ]
        related = graph.find_related_sessions("s1", metas)
        assert "s1" not in related


# ---------------------------------------------------------------------------
# SessionGraph.build_graph (integration-level unit test)
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_build_graph_from_storage(self, storage: NdjsonStorage):
        # Create two sessions with chunks
        s1_dir = storage.root / "sessions" / "s1"
        s1_dir.mkdir(parents=True)
        (s1_dir / "meta.json").write_text(
            json.dumps({
                "session_id": "s1",
                "command": ["make"],
                "cwd": "/tmp",
                "started_at": 100.0,
                "ended_at": 110.0,
                "exit_code": 0,
            }) + "\n"
        )
        (s1_dir / "chunks.jsonl").write_text(
            json.dumps({
                "chunk_id": "s1-0", "session_id": "s1", "index": 0,
                "text": "Build succeeded", "ts_start": 100.0, "ts_end": 110.0,
                "token_estimate": 5, "quality_score": 0.8,
            }) + "\n"
        )

        s2_dir = storage.root / "sessions" / "s2"
        s2_dir.mkdir(parents=True)
        (s2_dir / "meta.json").write_text(
            json.dumps({
                "session_id": "s2",
                "command": ["pytest"],
                "cwd": "/tmp",
                "started_at": 111.0,
                "ended_at": 120.0,
                "exit_code": 1,
            }) + "\n"
        )
        (s2_dir / "chunks.jsonl").write_text(
            json.dumps({
                "chunk_id": "s2-0", "session_id": "s2", "index": 0,
                "text": "FAILED test_foo.py::test_bar", "ts_start": 111.0, "ts_end": 120.0,
                "token_estimate": 8, "quality_score": 0.7,
            }) + "\n"
        )

        graph = SessionGraph()
        nodes = graph.build_graph(storage)

        assert len(nodes) == 2
        node_map = {n.meta.session_id: n for n in nodes}

        assert node_map["s1"].episode_type == EpisodeType.BUILD
        assert node_map["s1"].has_error is False

        assert node_map["s2"].episode_type == EpisodeType.TEST
        assert node_map["s2"].has_error is True
        assert node_map["s2"].error_summary is not None

        # Temporal neighbors should be linked
        assert "s1" in node_map["s2"].related_sessions

    def test_build_graph_claude_session_uses_content(self, storage: NdjsonStorage):
        """Claude sessions should be classified by chunk content, not just command."""
        s1_dir = storage.root / "sessions" / "s1"
        s1_dir.mkdir(parents=True)
        (s1_dir / "meta.json").write_text(
            json.dumps({
                "session_id": "s1",
                "command": ["claude"],
                "cwd": "/tmp",
                "started_at": 100.0,
                "ended_at": 110.0,
                "exit_code": 0,
            }) + "\n"
        )
        (s1_dir / "chunks.jsonl").write_text(
            json.dumps({
                "chunk_id": "s1-0", "session_id": "s1", "index": 0,
                "text": "User: explain the architecture\nAssistant: The architecture uses...",
                "ts_start": 100.0, "ts_end": 110.0,
                "token_estimate": 20, "quality_score": 0.8,
            }) + "\n"
        )

        graph = SessionGraph()
        nodes = graph.build_graph(storage)
        assert len(nodes) == 1
        assert nodes[0].episode_type == EpisodeType.EXPLORE


# ---------------------------------------------------------------------------
# Content-based classification
# ---------------------------------------------------------------------------


def _make_claude_meta(session_id: str = "s1") -> SessionMeta:
    return SessionMeta(
        session_id=session_id, command=["claude"], cwd="/tmp", started_at=1.0
    )


def _make_chunk(text: str, session_id: str = "s1") -> Chunk:
    return Chunk(
        chunk_id=f"{session_id}-0",
        session_id=session_id,
        index=0,
        text=text,
        ts_start=1.0,
        ts_end=2.0,
        token_estimate=len(text) // 4,
        quality_score=0.8,
    )


class TestContentClassification:
    @pytest.fixture()
    def graph(self) -> SessionGraph:
        return SessionGraph()

    def test_claude_without_chunks_returns_refactor(self, graph: SessionGraph):
        meta = _make_claude_meta()
        assert graph.classify_episode(meta) == EpisodeType.REFACTOR

    def test_claude_with_empty_chunks_returns_refactor(self, graph: SessionGraph):
        meta = _make_claude_meta()
        assert graph.classify_episode(meta, []) == EpisodeType.REFACTOR

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Running pytest -v tests/\n5 PASSED, 0 FAILED", EpisodeType.TEST),
            ("cargo build --release\nCompiling project v0.1.0", EpisodeType.BUILD),
            ("kubectl apply -f deploy.yaml\ndeployment to staging", EpisodeType.DEPLOY),
            ("Traceback (most recent call last):\npdb> breakpoint", EpisodeType.DEBUG),
            ("Let's refactor this function and rename the variable", EpisodeType.REFACTOR),
            ("How does the authentication work? Explain the flow", EpisodeType.EXPLORE),
            ("Update pyproject.toml config and install dependency", EpisodeType.CONFIG),
            ("Update the README documentation and CHANGELOG", EpisodeType.DOCS),
            ("Review this PR and do a code review for the audit", EpisodeType.REVIEW),
        ],
    )
    def test_content_classification(
        self, graph: SessionGraph, text: str, expected: EpisodeType
    ):
        meta = _make_claude_meta()
        chunks = [_make_chunk(text)]
        assert graph.classify_episode(meta, chunks) == expected

    def test_scoring_picks_highest_count(self, graph: SessionGraph):
        """When multiple types match, the one with the most hits wins."""
        meta = _make_claude_meta()
        text = "pytest test_foo PASSED\npytest test_bar FAILED\ntest_baz assert\nsome refactor"
        chunks = [_make_chunk(text)]
        assert graph.classify_episode(meta, chunks) == EpisodeType.TEST

    def test_no_matching_content_returns_refactor(self, graph: SessionGraph):
        meta = _make_claude_meta()
        chunks = [_make_chunk("Hello world, nothing special here")]
        assert graph.classify_episode(meta, chunks) == EpisodeType.REFACTOR

    def test_non_claude_command_ignores_chunks(self, graph: SessionGraph):
        """Non-claude commands still use command-based classification."""
        meta = SessionMeta(
            session_id="s1", command=["pytest"], cwd="/tmp", started_at=1.0
        )
        chunks = [_make_chunk("deploy to production with kubectl")]
        assert graph.classify_episode(meta, chunks) == EpisodeType.TEST
