"""ANSI escape sequence stripping state machine for streaming PTY data."""

from __future__ import annotations

import codecs
import re
from enum import IntEnum, auto

# Box Drawing (U+2500–U+257F), Block Elements (U+2580–U+259F),
# Braille (U+2800–U+28FF), common spinner/prompt dingbats
_NOISE_CHARS_RE = re.compile(
    r"[\u2500-\u257F"  # box drawing ─│├┤┬┴┼
    r"\u2580-\u259F"  # block elements █▓▒░▟▙▝▜▛▐
    r"\u2800-\u28FF"  # braille patterns
    r"\u2700-\u27BF"  # dingbats ✽✻✶✳✢
    r"\u00B7"  # middle dot ·
    r"\u276F"  # heavy right angle ❯
    r"\u27E8\u27E9"  # angle brackets
    r"\u29C9"  # squared symbol ⧉
    r"\u23B0-\u23FF"  # misc technical brackets ⎿ and controls ⏸
    r"\u2190-\u21FF"  # arrows
    r"\u2B50-\u2B5F"  # stars
    r"⏵➜]+"
)

_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Common Claude Code / terminal UI text patterns that are never useful context
_TERMINAL_UI_RE = re.compile(
    r"(?:"
    # Claude Code UI hints
    r"accept\s*edits?\s*on"
    r"|shift\+tab\s*to\s*cycle"
    r"|Esc\s*to\s*cancel"
    r"|Tab\s*to\s*amend"
    r"|ctrl\+[a-z]\s*to\s*\w+"
    r"|Do\s*you\s*want\s*to\s*proceed\??"
    r"|Yes,?\s*and\s*always\s*allow"
    r"|for\s*bash\s*mode"
    r"|to\s*go\s*back"
    r"|Esc/Enter/Space\s*to\s*close"
    r"|k\s*to\s*kill"
    # Claude Code status/spinners/timers
    r"|\w+ing…\s*\([^)]*tokens?\)"
    r"|\w+ing…"
    r"|Running…"
    r"|Reading\s*\d+\s*files?…"
    r"|Loading\s*output…"
    r"|thought\s*for\s*\d+s"
    r"|\d+[ms]\s+[\d.]+[km]?\s*tokens?"
    # File counts and shell prompts
    r"|\d+\s*files?\s*\+\d+\s*-\d+"
    r"|\w+\s*git:\([^)]*\)"
    r"|\d+\s*bash\b"
    # Claude Code tool call headers
    r"|Shell\s*details"
    r"|Status:\s*running"
    r"|Runtime:\s*\d+[ms]\s*\d*[ms]?"
    r")",
    re.IGNORECASE,
)

# Collapse runs of whitespace (spaces/tabs) on a single line into one space
_HORIZ_WHITESPACE_RE = re.compile(r"[^\S\n]{3,}")


def strip_terminal_noise(text: str) -> str:
    """Remove terminal UI noise characters, UI patterns, and collapse blank lines."""
    text = _NOISE_CHARS_RE.sub("", text)
    text = _TERMINAL_UI_RE.sub("", text)
    text = _HORIZ_WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


class _State(IntEnum):
    """Parser states for ANSI escape sequence stripping."""

    GROUND = auto()
    ESC = auto()  # Received ESC (0x1B)
    ESC_INTER = auto()  # ESC followed by intermediate byte (0x20-0x2F)
    CSI_PARAM = auto()  # CSI sequence (ESC [ ...)
    OSC_STRING = auto()  # OSC sequence (ESC ] ...)
    DCS_STRING = auto()  # DCS sequence (ESC P ...)
    STRING_ESC = auto()  # ESC within string sequence (OSC/DCS/APC/PM/SOS)


class AnsiStripper:
    """Streaming ANSI escape sequence stripper.

    Processes raw bytes from PTY output, strips ANSI escape sequences,
    normalizes line endings, and decodes UTF-8 incrementally.

    Usage:
        stripper = AnsiStripper()
        text = stripper.process(raw_bytes)
        # ... more chunks ...
        remaining = stripper.flush()
    """

    def __init__(self) -> None:
        self._state = _State.GROUND
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending_cr = False  # Deferred \r for cross-chunk \r\n handling

    def process(self, data: bytes) -> str:
        """Process a chunk of raw bytes and return sanitized text."""
        decoded = self._decoder.decode(data, final=False)
        return self._strip(decoded)

    def flush(self) -> str:
        """Flush remaining state at end of stream."""
        remaining = self._decoder.decode(b"", final=True)
        text = self._strip(remaining)
        # Emit deferred \r as \n at stream end
        if self._pending_cr:
            self._pending_cr = False
            text = "\n" + text
        return text

    def _strip(self, text: str) -> str:
        """Strip ANSI sequences and normalize line endings from decoded text."""
        out: list[str] = []
        for ch in text:
            cp = ord(ch)

            if self._state == _State.GROUND:
                self._ground(ch, cp, out)

            elif self._state == _State.ESC:
                self._esc(ch, cp, out)

            elif self._state == _State.ESC_INTER:
                # Intermediate bytes 0x20-0x2F continue; final byte 0x30-0x7E ends
                if 0x30 <= cp <= 0x7E:
                    self._state = _State.GROUND
                elif not (0x20 <= cp <= 0x2F):
                    # Invalid — back to ground, reprocess
                    self._state = _State.GROUND
                    self._ground(ch, cp, out)

            elif self._state == _State.CSI_PARAM:
                # Parameter bytes (0x30-0x3F), intermediate (0x20-0x2F) continue
                # Final byte 0x40-0x7E ends the sequence
                if 0x40 <= cp <= 0x7E:
                    self._state = _State.GROUND
                # Ignore parameter and intermediate bytes (stay in CSI_PARAM)
                # C0 controls within CSI are executed (but we strip them)

            elif self._state == _State.OSC_STRING:
                self._string_seq(ch, cp)

            elif self._state == _State.DCS_STRING:
                self._string_seq(ch, cp)

            elif self._state == _State.STRING_ESC:
                # ESC within string: if followed by \, it's ST (String Terminator)
                if ch == "\\":
                    self._state = _State.GROUND
                else:
                    # False alarm — could be nested ESC sequence within string
                    # Most terminals treat this as ST, return to ground
                    self._state = _State.GROUND
                    # Reprocess current char from ground state
                    self._ground(ch, cp, out)

        return "".join(out)

    def _ground(self, ch: str, cp: int, out: list[str]) -> None:
        """Handle character in GROUND state."""
        # Handle deferred \r
        if self._pending_cr:
            self._pending_cr = False
            if ch == "\n":
                # \r\n -> \n
                out.append("\n")
                return
            else:
                # bare \r -> \n
                out.append("\n")
                # Fall through to process current character

        if ch == "\x1b":
            self._state = _State.ESC
        elif ch == "\r":
            # Defer \r — might be followed by \n
            self._pending_cr = True
        elif ch == "\n" or ch == "\t":
            out.append(ch)
        elif cp == 0x9B:
            # C1 CSI (8-bit)
            self._state = _State.CSI_PARAM
        elif cp == 0x9D:
            # C1 OSC (8-bit)
            self._state = _State.OSC_STRING
        elif cp == 0x90:
            # C1 DCS (8-bit)
            self._state = _State.DCS_STRING
        elif cp == 0x98 or cp == 0x9E or cp == 0x9F:
            # C1 SOS/PM/APC (8-bit)
            self._state = _State.OSC_STRING  # Treat same as OSC for stripping
        elif cp < 0x20:
            # Other C0 controls — strip (except \n, \t handled above)
            pass
        elif 0x80 <= cp <= 0x9F:
            # Other C1 controls — strip
            pass
        else:
            # Printable character
            out.append(ch)

    def _esc(self, ch: str, cp: int, out: list[str]) -> None:
        """Handle character after ESC."""
        if ch == "[":
            # CSI
            self._state = _State.CSI_PARAM
        elif ch == "]":
            # OSC
            self._state = _State.OSC_STRING
        elif ch == "P":
            # DCS
            self._state = _State.DCS_STRING
        elif ch == "X" or ch == "^" or ch == "_":
            # SOS / PM / APC — treat as string sequences
            self._state = _State.OSC_STRING
        elif ch == "N" or ch == "O":
            # SS2 / SS3 — next char is from alternate char set, skip it
            # Actually SS2/SS3 only affect the next character
            # For simplicity, just go back to ground (the next char will be processed normally)
            self._state = _State.GROUND
        elif 0x20 <= cp <= 0x2F:
            # Intermediate byte — ESC with intermediate
            self._state = _State.ESC_INTER
        elif 0x30 <= cp <= 0x7E:
            # Final byte — two-char ESC sequence complete
            self._state = _State.GROUND
        else:
            # Invalid after ESC — back to ground, reprocess
            self._state = _State.GROUND
            self._ground(ch, cp, out)

    def _string_seq(self, ch: str, cp: int) -> None:
        """Handle character within OSC/DCS/APC/PM/SOS string sequence."""
        if ch == "\x1b":
            # Might be ESC \ (ST)
            self._state = _State.STRING_ESC
        elif ch == "\x07":
            # BEL terminates OSC (common in xterm title sequences)
            self._state = _State.GROUND
        elif cp == 0x9C:
            # C1 ST (8-bit String Terminator)
            self._state = _State.GROUND
        # All other characters consumed (stripped) within string sequence
