"""Session graph: episode classification, error detection, and related session linking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mb.models import Chunk, SessionMeta

if TYPE_CHECKING:
    from mb.store import NdjsonStorage


class EpisodeType(Enum):
    """Classification of session purpose."""

    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    DEBUG = "debug"
    REFACTOR = "refactor"
    EXPLORE = "explore"
    CONFIG = "config"
    DOCS = "docs"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class SessionNode:
    """A graph node representing a session with episode metadata."""

    meta: SessionMeta
    episode_type: EpisodeType
    has_error: bool
    error_summary: str | None
    related_sessions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristics tables (R10)
# ---------------------------------------------------------------------------

# command[0] → episode type
_CMD_MAP: dict[str, EpisodeType] = {
    "make": EpisodeType.BUILD,
    "cmake": EpisodeType.BUILD,
    "ninja": EpisodeType.BUILD,
    "pytest": EpisodeType.TEST,
    "jest": EpisodeType.TEST,
    "gdb": EpisodeType.DEBUG,
    "lldb": EpisodeType.DEBUG,
    "claude": EpisodeType.REFACTOR,
}

# (command[0], command[1]) → episode type (overrides _CMD_MAP)
_CMD_PAIR_MAP: dict[tuple[str, str], EpisodeType] = {
    ("cargo", "build"): EpisodeType.BUILD,
    ("cargo", "test"): EpisodeType.TEST,
    ("go", "build"): EpisodeType.BUILD,
    ("go", "test"): EpisodeType.TEST,
    ("npm", "test"): EpisodeType.TEST,
    ("npm", "run"): EpisodeType.BUILD,  # npm run build → BUILD
    ("docker", "build"): EpisodeType.BUILD,
    ("docker", "push"): EpisodeType.DEPLOY,
    ("make", "test"): EpisodeType.TEST,
    ("make", "install"): EpisodeType.BUILD,
}

# command[0] keywords that always mean DEPLOY
_DEPLOY_CMDS = {"kubectl", "terraform", "ansible", "ansible-playbook", "deploy"}

# Patterns in command arguments that override to TEST
_TEST_SUBCOMMAND_RE = re.compile(r"\btest\b", re.IGNORECASE)

# Patterns in command that mean DEBUG
_DEBUG_MODULE_RE = re.compile(r"\bpdb\b")

# Error keywords in chunk text
_ERROR_KEYWORDS_RE = re.compile(
    r"(Traceback \(most recent call last\)|FAILED |ERROR:|Exception:|"
    r"panic:|FATAL|segmentation fault|core dumped)",
    re.IGNORECASE,
)

# Content-based patterns for classifying "claude" sessions by chunk text
_CONTENT_PATTERNS: dict[EpisodeType, re.Pattern[str]] = {
    EpisodeType.TEST: re.compile(
        r"(pytest|unittest|test_|PASSED|FAILED|assert\b|coverage)", re.IGNORECASE
    ),
    EpisodeType.BUILD: re.compile(
        r"(compile|build|linking|cargo build|webpack|make\b)", re.IGNORECASE
    ),
    EpisodeType.DEPLOY: re.compile(
        r"(deploy|kubectl|terraform|docker push|production|staging)", re.IGNORECASE
    ),
    EpisodeType.DEBUG: re.compile(
        r"(Traceback|pdb|breakpoint|debugger|stack trace)", re.IGNORECASE
    ),
    EpisodeType.REFACTOR: re.compile(
        r"(refactor|rename|extract\b|restructure|simplify)", re.IGNORECASE
    ),
    EpisodeType.EXPLORE: re.compile(
        r"(how does|what is|explain|architecture|show me|understand)", re.IGNORECASE
    ),
    EpisodeType.CONFIG: re.compile(
        r"(config|\.env|settings|install\b|dependency|pyproject\.toml|yaml|setup\b)",
        re.IGNORECASE,
    ),
    EpisodeType.DOCS: re.compile(
        r"(README|documentation|docstring|CHANGELOG|markdown)", re.IGNORECASE
    ),
    EpisodeType.REVIEW: re.compile(
        r"(review|PR\b|pull request|code review|LGTM|audit)", re.IGNORECASE
    ),
}


def _classify_from_content(chunks: list[Chunk]) -> EpisodeType:
    """Classify episode type by counting regex matches across chunk text."""
    scores: dict[EpisodeType, int] = {ep: 0 for ep in _CONTENT_PATTERNS}
    for chunk in chunks:
        text = chunk.text
        for ep, pattern in _CONTENT_PATTERNS.items():
            scores[ep] += len(pattern.findall(text))

    best = max(scores, key=scores.__getitem__)
    if scores[best] == 0:
        return EpisodeType.REFACTOR
    return best


# Max temporal gap (seconds) for sessions to be considered neighbors
_MAX_NEIGHBOR_GAP = 600  # 10 minutes


class SessionGraph:
    """Builds a session graph with episode classification and relationship linking."""

    def classify_episode(
        self, meta: SessionMeta, chunks: list[Chunk] | None = None
    ) -> EpisodeType:
        """Classify session episode type from command heuristics.

        When the command is ``claude`` and *chunks* are provided, delegates
        to content-based classification via ``_classify_from_content``.
        """
        cmd = meta.command
        if not cmd:
            return EpisodeType.BUILD

        cmd0 = _basename(cmd[0])

        # For "claude" sessions, use content-based classification when chunks available
        if cmd0 == "claude" and chunks:
            return _classify_from_content(chunks)

        # Check two-word pairs first (e.g., "cargo test")
        if len(cmd) >= 2:
            pair = (cmd0, cmd[1])
            if pair in _CMD_PAIR_MAP:
                return _CMD_PAIR_MAP[pair]

        # Check for python -m <module> patterns
        if cmd0 == "python" and len(cmd) >= 3 and cmd[1] == "-m":
            module = cmd[2]
            if module in ("pdb",):
                return EpisodeType.DEBUG
            if module in ("pytest",):
                return EpisodeType.TEST

        # Check deploy commands
        if cmd0 in _DEPLOY_CMDS:
            return EpisodeType.DEPLOY

        # Check single-command map
        if cmd0 in _CMD_MAP:
            return _CMD_MAP[cmd0]

        # Fallback: check if "test" appears as a subcommand
        for arg in cmd[1:]:
            if _TEST_SUBCOMMAND_RE.search(arg):
                return EpisodeType.TEST

        return EpisodeType.BUILD

    def detect_error(self, meta: SessionMeta, chunks: list[Chunk]) -> bool:
        """Detect whether a session has errors."""
        if meta.exit_code is not None and meta.exit_code != 0:
            return True
        for chunk in chunks:
            if _ERROR_KEYWORDS_RE.search(chunk.text):
                return True
        return False

    def extract_error_summary(
        self, meta: SessionMeta, chunks: list[Chunk]
    ) -> str | None:
        """Extract a brief error summary, or None if no error detected."""
        parts: list[str] = []

        if meta.exit_code is not None and meta.exit_code != 0:
            parts.append(f"Exit code {meta.exit_code}")

        for chunk in chunks:
            match = _ERROR_KEYWORDS_RE.search(chunk.text)
            if match:
                # Extract the line containing the error keyword
                start = match.start()
                line_start = chunk.text.rfind("\n", 0, start) + 1
                line_end = chunk.text.find("\n", start)
                if line_end == -1:
                    line_end = len(chunk.text)
                error_line = chunk.text[line_start:line_end].strip()
                if error_line and error_line not in parts:
                    parts.append(error_line)

        if not parts:
            return None
        return "; ".join(parts[:3])  # At most 3 error fragments

    def find_related_sessions(
        self, session_id: str, all_metas: list[SessionMeta]
    ) -> list[str]:
        """Find sessions temporally adjacent to the given session."""
        target: SessionMeta | None = None
        for m in all_metas:
            if m.session_id == session_id:
                target = m
                break
        if target is None:
            return []

        related: list[str] = []
        t_start = target.started_at
        t_end = target.ended_at or target.started_at

        for m in all_metas:
            if m.session_id == session_id:
                continue
            m_start = m.started_at
            m_end = m.ended_at or m.started_at

            # Check if sessions are within _MAX_NEIGHBOR_GAP of each other
            gap = min(
                abs(t_start - m_end),
                abs(m_start - t_end),
                abs(t_start - m_start),
            )
            if gap <= _MAX_NEIGHBOR_GAP:
                related.append(m.session_id)

        return related

    def build_graph(self, storage: NdjsonStorage) -> list[SessionNode]:
        """Build a full session graph from storage."""
        all_metas = storage.list_sessions()
        nodes: list[SessionNode] = []

        for meta in all_metas:
            chunks = storage.read_chunks(meta.session_id)
            episode = self.classify_episode(meta, chunks)
            has_error = self.detect_error(meta, chunks)
            error_summary = self.extract_error_summary(meta, chunks) if has_error else None
            related = self.find_related_sessions(meta.session_id, all_metas)

            nodes.append(
                SessionNode(
                    meta=meta,
                    episode_type=episode,
                    has_error=has_error,
                    error_summary=error_summary,
                    related_sessions=related,
                )
            )

        return nodes


def _basename(cmd: str) -> str:
    """Extract the basename of a command (strip path)."""
    if "/" in cmd:
        return cmd.rsplit("/", 1)[-1]
    return cmd
