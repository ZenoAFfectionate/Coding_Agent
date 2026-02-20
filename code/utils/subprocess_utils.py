"""Safe subprocess execution with proper process group cleanup.

When ``subprocess.run(..., timeout=N)`` times out, it sends SIGKILL to the
direct child process only.  If the child has spawned its own children (or is
a shell wrapper), those grandchild processes survive and become orphans —
exactly the root cause of runaway ``python -c`` processes eating 100 % CPU
for days.

This module provides ``safe_run()``, a drop-in wrapper around
``subprocess.run`` that:

1. Puts the subprocess in its own **process group** (``start_new_session=True``).
2. On timeout, kills the **entire process group** via ``os.killpg()``.
3. On any other exception, still attempts to clean up the process group.

All tool classes that spawn subprocesses should use ``safe_run()`` instead of
calling ``subprocess.run()`` directly.
"""

import os
import signal
import subprocess
import sys
from typing import Any


def safe_run(
    *args: Any,
    timeout: int | float | None = 30,
    input: str | bytes | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run a subprocess with process-group-level timeout enforcement.

    Accepts the same arguments as ``subprocess.run()``.  Key differences:

    - ``start_new_session=True`` is forced so the child gets its own
      process group (POSIX) / job object (Windows).
    - On timeout, the **entire process group** is killed, not just the
      leader process.  This prevents orphaned grandchild processes.

    Args:
        *args: Positional arguments forwarded to ``subprocess.Popen``.
        timeout: Seconds before the process group is killed.
            ``None`` means wait indefinitely (not recommended).
        input: Optional data to send to the process's stdin.  When
            provided, ``stdin`` is automatically set to ``PIPE``.
        **kwargs: Keyword arguments forwarded to ``subprocess.Popen``.
            ``start_new_session`` is always overridden to ``True``.

    Returns:
        ``subprocess.CompletedProcess`` — same as ``subprocess.run``.

    Raises:
        subprocess.TimeoutExpired: If the process exceeds *timeout*.
        All other exceptions from ``subprocess.Popen`` propagate as-is.
    """
    # Force a new session / process group so we can kill the whole tree.
    kwargs["start_new_session"] = True

    # Translate capture_output=True into Popen-compatible args,
    # since Popen doesn't accept capture_output directly.
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    # When input data is provided, wire up stdin automatically.
    if input is not None:
        kwargs.setdefault("stdin", subprocess.PIPE)

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(*args, **kwargs)
        stdout, stderr = proc.communicate(input=input, timeout=timeout)

        return subprocess.CompletedProcess(
            args=proc.args,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        # Collect any remaining output so file descriptors are closed.
        stdout, stderr = b"", b""
        if proc is not None:
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                pass
        raise subprocess.TimeoutExpired(
            cmd=proc.args if proc else args,
            timeout=timeout,
            output=stdout,
            stderr=stderr,
        )

    except Exception:
        # On unexpected errors, still clean up the process tree.
        _kill_process_group(proc)
        raise


def _kill_process_group(proc: subprocess.Popen | None) -> None:
    """Send SIGKILL to the entire process group of *proc*."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # Process already exited or we lack permissions — that's fine.
        pass
    # Belt-and-suspenders: make sure the direct child is dead too.
    try:
        proc.kill()
    except OSError:
        pass
