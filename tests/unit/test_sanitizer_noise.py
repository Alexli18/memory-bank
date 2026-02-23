"""Tests for strip_terminal_noise() in mb.sanitizer."""

from __future__ import annotations

from mb.sanitizer import strip_terminal_noise


def test_strips_box_drawing() -> None:
    """Box-drawing characters (U+2500-U+257F) are removed."""
    assert strip_terminal_noise("hello─│├┤┬┴┼world") == "helloworld"


def test_strips_block_elements() -> None:
    """Block element characters (U+2580-U+259F) are removed."""
    assert strip_terminal_noise("foo█▓▒░bar") == "foobar"


def test_strips_braille_patterns() -> None:
    """Braille pattern characters (U+2800-U+28FF) are removed."""
    assert strip_terminal_noise("a⠁⠂⠃b") == "ab"


def test_strips_dingbats_spinners() -> None:
    """Dingbat spinner characters (U+2700-U+27BF) are removed."""
    assert strip_terminal_noise("loading✽✻✶✳✢done") == "loadingdone"


def test_strips_middle_dot() -> None:
    """Middle dot (U+00B7) is removed."""
    assert strip_terminal_noise("a·b·c") == "abc"


def test_strips_heavy_right_angle() -> None:
    """Heavy right-pointing angle quotation mark (U+276F) ❯ is removed."""
    assert strip_terminal_noise("❯ command") == " command"


def test_strips_arrows() -> None:
    """Arrow characters (U+2190-U+21FF) are removed."""
    assert strip_terminal_noise("←↑→↓text") == "text"


def test_strips_play_triangle() -> None:
    """Play triangle ⏵ is removed."""
    assert strip_terminal_noise("⏵⏵accepted") == "accepted"


def test_preserves_normal_text() -> None:
    """Normal ASCII text, numbers, and standard punctuation are preserved."""
    text = "Hello, World! 123 (foo) [bar] {baz} @#$%"
    assert strip_terminal_noise(text) == text


def test_preserves_cyrillic() -> None:
    """Cyrillic characters are preserved."""
    text = "Привет мир"
    assert strip_terminal_noise(text) == text


def test_collapses_blank_lines() -> None:
    """Three or more consecutive newlines collapse to two."""
    assert strip_terminal_noise("a\n\n\nb") == "a\n\nb"
    assert strip_terminal_noise("a\n\n\n\n\nb") == "a\n\nb"


def test_preserves_double_newline() -> None:
    """Exactly two newlines are preserved as-is."""
    assert strip_terminal_noise("a\n\nb") == "a\n\nb"


def test_combined_noise_and_blank_lines() -> None:
    """Stripping noise chars that leave blank lines also collapses them."""
    text = "hello\n───────\n\n\nworld"
    result = strip_terminal_noise(text)
    assert "───" not in result
    assert "\n\n\n" not in result
    assert "hello" in result
    assert "world" in result


# --- Terminal UI pattern tests ---


def test_strips_accept_edits_on() -> None:
    """Claude Code 'accept edits on' UI hint is stripped."""
    assert "accept edits on" not in strip_terminal_noise("foo accept edits on bar")


def test_strips_shift_tab_to_cycle() -> None:
    """Claude Code 'shift+tab to cycle' hint is stripped."""
    assert "shift+tab to cycle" not in strip_terminal_noise("shift+tab to cycle")


def test_strips_esc_to_cancel() -> None:
    """Claude Code 'Esc to cancel' hint is stripped."""
    assert "Esc to cancel" not in strip_terminal_noise("Esc to cancel")


def test_strips_running_indicator() -> None:
    """Claude Code 'Running…' indicator is stripped."""
    assert "Running…" not in strip_terminal_noise("⎿ Running…")


def test_strips_status_bar_with_tokens() -> None:
    """Claude Code status bar like 'Cascading… (1m 59s 7.0k tokens)' is stripped."""
    result = strip_terminal_noise("Cascading… (1m 59s 7.0k tokens)")
    assert "tokens" not in result


def test_strips_file_count() -> None:
    """Claude Code file count like '33 files +0 -0' is stripped."""
    assert "files" not in strip_terminal_noise("33 files +0 -0")


def test_strips_proceed_prompt() -> None:
    """Claude Code 'Do you want to proceed?' prompt is stripped."""
    assert "proceed" not in strip_terminal_noise("Do you want to proceed?")


def test_collapses_horizontal_whitespace() -> None:
    """Long runs of spaces/tabs are collapsed to single space."""
    assert strip_terminal_noise("hello      world") == "hello world"
    assert strip_terminal_noise("a\t\t\tb") == "a b"


def test_preserves_short_whitespace() -> None:
    """Single and double spaces are preserved."""
    assert strip_terminal_noise("a b") == "a b"
    assert strip_terminal_noise("a  b") == "a  b"


def test_strips_thought_for_timer() -> None:
    """Claude Code 'thought for Ns' timer is stripped."""
    assert "thought" not in strip_terminal_noise("thought for 28s")


def test_strips_token_counter() -> None:
    """Claude Code token counter like '5.7k tokens' is stripped."""
    assert "tokens" not in strip_terminal_noise("1m 5.7k tokens")


def test_strips_reading_files() -> None:
    """Claude Code 'Reading N files…' indicator is stripped."""
    result = strip_terminal_noise("Reading 1 file…")
    assert "Reading" not in result


def test_strips_bash_tab_indicator() -> None:
    """Terminal tab indicator '1 bash' is stripped."""
    assert "bash" not in strip_terminal_noise("1 bash")


def test_strips_shell_details() -> None:
    """Claude Code 'Shell details' panel header is stripped."""
    assert "Shell" not in strip_terminal_noise("Shell details")
