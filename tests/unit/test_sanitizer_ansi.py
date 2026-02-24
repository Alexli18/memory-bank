"""Tests for AnsiStripper streaming ANSI escape sequence stripping."""

from __future__ import annotations

from mb.sanitizer import AnsiStripper


def test_ascii_passthrough() -> None:
    """Plain ASCII text passes through unchanged."""
    s = AnsiStripper()
    assert s.process(b"hello world") == "hello world"


def test_csi_color_stripped() -> None:
    """CSI color sequences like \\x1b[31m and \\x1b[0m are stripped."""
    s = AnsiStripper()
    assert s.process(b"hello\x1b[31mworld\x1b[0m") == "helloworld"


def test_osc_title_stripped() -> None:
    """OSC title sequence (\\x1b]0;...\\x07) is stripped."""
    s = AnsiStripper()
    assert s.process(b"\x1b]0;My Title\x07text") == "text"


def test_crlf_normalized() -> None:
    """CR+LF (\\r\\n) is normalized to a single \\n."""
    s = AnsiStripper()
    assert s.process(b"line1\r\nline2") == "line1\nline2"


def test_bare_cr_to_newline() -> None:
    """Bare \\r not followed by \\n is converted to \\n."""
    s = AnsiStripper()
    assert s.process(b"abc\rdef") == "abc\ndef"


def test_flush_pending_cr() -> None:
    """A trailing \\r deferred across chunks is emitted as \\n on flush."""
    s = AnsiStripper()
    text = s.process(b"text\r")
    remaining = s.flush()
    assert text == "text"
    assert remaining == "\n"


def test_tab_passthrough() -> None:
    """Tab characters pass through unchanged."""
    s = AnsiStripper()
    assert s.process(b"a\tb") == "a\tb"


def test_c0_control_stripped() -> None:
    """C0 control characters (except \\n, \\t, \\r) are stripped."""
    s = AnsiStripper()
    assert s.process(b"hello\x01\x02world") == "helloworld"


def test_utf8_incremental() -> None:
    """A multi-byte UTF-8 character split across two process() calls decodes correctly."""
    s = AnsiStripper()
    # U+00E9 (e-acute) is encoded as 0xC3 0xA9 in UTF-8
    part1 = s.process(b"caf\xc3")
    part2 = s.process(b"\xa9!")
    assert part1 + part2 == "caf\u00e9!"


def test_esc_two_char_stripped() -> None:
    """Two-character ESC sequences like \\x1bM (reverse index) are stripped."""
    s = AnsiStripper()
    assert s.process(b"\x1bMtext") == "text"
