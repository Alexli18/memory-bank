"""Tests for mb.budgeter — token estimation, Section, budget allocation."""

from __future__ import annotations

from mb.budgeter import Section, apply_budget, estimate_tokens, truncate_elements


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_known_string(self) -> None:
        # 40 chars → 40/4 * 1.1 = 11
        text = "a" * 40
        assert estimate_tokens(text) == 11

    def test_returns_int(self) -> None:
        assert isinstance(estimate_tokens("hello world"), int)


class TestSection:
    def test_token_count_property(self) -> None:
        s = Section(name="A", content="hello world", priority=1, is_protected=False)
        assert s.token_count == estimate_tokens("hello world")

    def test_protected_flag(self) -> None:
        s = Section(name="A", content="x", priority=1, is_protected=True)
        assert s.is_protected is True


class TestTruncateElements:
    def test_fits_within_budget(self) -> None:
        content = "<a>one</a>\n<a>two</a>"
        result = truncate_elements(content, "</a>", budget=9999)
        assert result == content

    def test_removes_trailing_elements(self) -> None:
        content = "<a>one</a>\n<a>two</a>\n<a>three</a>"
        # Set budget small enough that last element is removed
        result = truncate_elements(content, "</a>", budget=6)
        assert "<a>three</a>" not in result
        assert "<a>one</a>" in result

    def test_returns_empty_when_nothing_fits(self) -> None:
        content = "<a>x</a>"
        result = truncate_elements(content, "</a>", budget=0)
        assert result == ""


class TestApplyBudget:
    def test_all_fit_within_budget(self) -> None:
        sections = [
            Section("A", "short", priority=1, is_protected=True),
            Section("B", "also short", priority=2, is_protected=False),
        ]
        result = apply_budget(sections, budget=9999)
        assert len(result) == 2
        assert result[0].content == "short"
        assert result[1].content == "also short"

    def test_protected_sections_never_truncated(self) -> None:
        sections = [
            Section("protected", "important content", priority=1, is_protected=True),
            Section("expendable", "x" * 1000, priority=2, is_protected=False),
        ]
        result = apply_budget(sections, budget=20)
        assert result[0].content == "important content"

    def test_truncation_removes_low_priority_first(self) -> None:
        sections = [
            Section("A", "hi", priority=1, is_protected=True),
            Section("B", "medium content", priority=2, is_protected=False),
            Section("C", "x" * 2000, priority=3, is_protected=False),
        ]
        result = apply_budget(sections, budget=10)
        # Protected "A" is intact
        assert result[0].content == "hi"
        # Higher priority "B" is fully preserved, lower priority "C" is truncated/empty
        b_section = next(s for s in result if s.name == "B")
        c_section = next(s for s in result if s.name == "C")
        assert b_section.content == "medium content"
        assert len(c_section.content) < 2000

    def test_preserves_section_order(self) -> None:
        sections = [
            Section("first", "a", priority=1, is_protected=True),
            Section("second", "b", priority=2, is_protected=False),
            Section("third", "c", priority=3, is_protected=False),
        ]
        result = apply_budget(sections, budget=9999)
        assert [s.name for s in result] == ["first", "second", "third"]

    def test_empty_content_when_no_budget(self) -> None:
        sections = [
            Section("A", "x" * 100, priority=1, is_protected=False),
        ]
        # Budget of 0 means nothing for non-protected
        result = apply_budget(sections, budget=0)
        assert result[0].content == ""
