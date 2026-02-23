"""PTY wrapper — transparent session capture with full interactivity."""

from __future__ import annotations

import atexit
import errno
import fcntl
import os
import pty
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

from mb.sanitizer import AnsiStripper
from mb.storage import create_session, finalize_session, write_event


def _get_winsize(fd: int) -> bytes:
    """Get terminal window size as packed struct."""
    return fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)


def _set_winsize(fd: int, winsize: bytes) -> None:
    """Set terminal window size from packed struct."""
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def run_session(child_cmd: list[str], storage_root: Path) -> int:
    """Run a command in a PTY with transparent capture.

    Launches child_cmd in a pseudo-terminal, forwards all I/O transparently
    while capturing sanitized events to storage.

    Args:
        child_cmd: Command and arguments to execute.
        storage_root: Path to .memory-bank/ directory.

    Returns:
        Child process exit code.
    """
    session_id = create_session(child_cmd, root=storage_root)
    t0 = time.monotonic()

    # Write system start event — silent on disk errors
    try:
        write_event(
            session_id, "system", "system", "session_start", ts=0.0, root=storage_root
        )
    except OSError:
        pass

    # Print session start banner to stderr
    sys.stderr.write(f"[mb] Session {session_id} started\n")
    sys.stderr.flush()

    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    is_tty = os.isatty(stdin_fd)

    # Save original terminal settings for restoration (only if stdin is a tty)
    old_attrs = termios.tcgetattr(stdin_fd) if is_tty else None
    master_fd: int | None = None
    child_pid: int = 0
    exit_code = 1  # Default if something goes wrong

    # Sanitizer for stripping ANSI from captured output
    output_stripper = AnsiStripper()
    input_stripper = AnsiStripper()

    def _restore_terminal() -> None:
        """Restore terminal to original settings."""
        if old_attrs is not None:
            try:
                termios.tcsetattr(stdin_fd, termios.TCSAFLUSH, old_attrs)
            except (termios.error, OSError):
                pass

    # Register atexit handler for safety
    atexit.register(_restore_terminal)

    try:
        # Fork with PTY
        child_pid, master_fd = pty.fork()

        if child_pid == 0:
            # Child process — exec the command
            try:
                os.execvp(child_cmd[0], child_cmd)
            except Exception:
                # Must use os._exit() to prevent child from running parent code
                os._exit(127)

        # Parent process
        # Sync initial window size to child PTY
        if is_tty:
            try:
                winsize = _get_winsize(stdin_fd)
                _set_winsize(master_fd, winsize)
            except OSError:
                pass

        # Set up SIGWINCH handler to sync window size changes
        def _sigwinch_handler(signum: int, frame: object) -> None:
            try:
                if master_fd is not None and is_tty:
                    winsize = _get_winsize(stdin_fd)
                    _set_winsize(master_fd, winsize)
            except OSError:
                pass

        old_sigwinch = signal.signal(signal.SIGWINCH, _sigwinch_handler)

        # Forward SIGINT to child instead of killing Memory Bank
        def _sigint_handler(signum: int, frame: object) -> None:
            if child_pid > 0:
                try:
                    os.kill(child_pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

        old_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        # Set terminal to raw mode (only if stdin is a tty)
        if is_tty:
            tty.setraw(stdin_fd)

        # Main I/O loop
        stdin_open = True
        try:
            while True:
                watch_fds = [master_fd]
                if stdin_open:
                    watch_fds.append(stdin_fd)

                try:
                    rfds, _, _ = select.select(watch_fds, [], [], 0.1)
                except (select.error, OSError) as e:
                    if getattr(e, "errno", None) == errno.EINTR:
                        continue
                    raise

                if master_fd in rfds:
                    try:
                        data = os.read(master_fd, 16384)
                    except OSError as e:
                        if e.errno == errno.EIO:
                            # Child exited — normal PTY behavior
                            break
                        raise

                    if not data:
                        break

                    # Forward to user FIRST (FR-014)
                    os.write(stdout_fd, data)

                    # Then capture (sanitized) — silent on disk errors
                    ts = time.monotonic() - t0
                    sanitized = output_stripper.process(data)
                    if sanitized:
                        try:
                            write_event(
                                session_id,
                                "stdout",
                                "terminal",
                                sanitized,
                                ts=ts,
                                root=storage_root,
                            )
                        except OSError:
                            pass

                if stdin_open and stdin_fd in rfds:
                    try:
                        data = os.read(stdin_fd, 16384)
                    except OSError as e:
                        if e.errno == errno.EIO:
                            stdin_open = False
                            continue
                        raise

                    if not data:
                        # EOF on stdin — stop monitoring but keep reading child output
                        stdin_open = False
                        # Send EOF to child via PTY master (Ctrl+D)
                        try:
                            os.write(master_fd, b"\x04")
                        except OSError:
                            pass
                        continue

                    # Forward to child
                    os.write(master_fd, data)

                    # Capture input (sanitized) — silent on disk errors
                    ts = time.monotonic() - t0
                    sanitized = input_stripper.process(data)
                    if sanitized:
                        try:
                            write_event(
                                session_id,
                                "stdin",
                                "user",
                                sanitized,
                                ts=ts,
                                root=storage_root,
                            )
                        except OSError:
                            pass

        finally:
            # Restore signal handlers
            signal.signal(signal.SIGWINCH, old_sigwinch)
            signal.signal(signal.SIGINT, old_sigint)

        # Flush remaining sanitizer state — silent on disk errors
        remaining_out = output_stripper.flush()
        if remaining_out:
            ts = time.monotonic() - t0
            try:
                write_event(
                    session_id,
                    "stdout",
                    "terminal",
                    remaining_out,
                    ts=ts,
                    root=storage_root,
                )
            except OSError:
                pass
        remaining_in = input_stripper.flush()
        if remaining_in:
            ts = time.monotonic() - t0
            try:
                write_event(
                    session_id,
                    "stdin",
                    "user",
                    remaining_in,
                    ts=ts,
                    root=storage_root,
                )
            except OSError:
                pass

        # Wait for child process to finish
        try:
            _, status = os.waitpid(child_pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            # Child already reaped (e.g., by signal handler) — keep default exit_code
            pass

    except Exception:
        # If child was started, try to get its exit status
        if child_pid > 0:
            try:
                _, status = os.waitpid(child_pid, os.WNOHANG)
                if status != 0:
                    exit_code = os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                pass
        raise
    finally:
        # Close master fd
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass

        # Restore terminal
        _restore_terminal()
        atexit.unregister(_restore_terminal)

        # Write system end event and finalize — silent on disk errors
        ts_end = time.monotonic() - t0
        try:
            write_event(
                session_id,
                "system",
                "system",
                "session_end",
                ts=ts_end,
                root=storage_root,
            )
            finalize_session(session_id, exit_code, root=storage_root)
        except OSError:
            pass

        # Print session end banner to stderr
        sys.stderr.write(f"[mb] Session {session_id} ended (exit code: {exit_code})\n")
        sys.stderr.flush()

    return exit_code
