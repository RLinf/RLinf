# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_logger():
    """Get the logger instance of the current worker."""
    from rlinf.scheduler.worker import Worker

    return Worker.logger


_LIBAV_LOGS_SILENCED = False


def silence_libav_logs() -> None:
    """Suppress the ``[libdav1d @ 0x..] libdav1d 0.9.2`` chatter that
    torchcodec (LeRobot's default video backend) emits every frame.

    The messages are written by libav* **directly to file-descriptor 2**
    — pyav's log level does not intercept them because torchcodec
    bypasses pyav. We splice our own ``fd=2`` through a long-lived
    ``grep -v '\\[libdav1d'`` subprocess so every write to stderr is
    filtered line-by-line; all other stderr output (our own logs,
    tracebacks, etc.) still reaches the terminal.

    Idempotent: once the fd=2 redirect is installed, repeated calls return
    immediately so we never stack multiple ``grep`` filter subprocesses or
    re-duplicate the file descriptor.

    Call this once, as early as possible, before the heavy ``torch`` /
    ``torchcodec`` imports so the redirect is installed before libav loads.
    """
    global _LIBAV_LOGS_SILENCED
    if _LIBAV_LOGS_SILENCED:
        return

    import atexit
    import os
    import shutil
    import subprocess
    import sys

    # pyav's log level still helps when pyav IS the backend (older lerobot
    # fallback); cheap to do alongside the fd-level filter.
    try:
        import av

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass

    if not shutil.which("grep"):
        return
    # Save the original fd=2 BEFORE we redirect — we need it back at exit
    # so grep can see EOF on its stdin (otherwise grep blocks forever and
    # ``atexit`` deadlocks because our fd=2 is still a write-end of the pipe).
    saved_stderr_fd = os.dup(2)
    try:
        # grep's stdout points at the ORIGINAL stderr so filtered lines
        # keep showing. grep's stdin becomes our new fd=2.
        grep = subprocess.Popen(
            # Match any libdav1d / libav* chatter — both the `[libdav1d @
            # 0x..] libdav1d 0.9.2` form AND any continuation line that
            # contains just `libdav1d`. Cast a wider net so child workers
            # whose output formatting differs don't slip through.
            ["grep", "-v", "-E", "--line-buffered", r"libdav1d|libdav1d 0\.9"],
            stdin=subprocess.PIPE,
            stdout=saved_stderr_fd,
        )
    except Exception:
        os.close(saved_stderr_fd)
        return

    sys.stderr.flush()
    os.dup2(grep.stdin.fileno(), 2)

    def _restore_and_drain() -> None:
        # Restore fd=2 first so subsequent stderr writes (e.g. tracebacks)
        # still reach the terminal — and crucially so grep sees EOF on its
        # stdin instead of waiting on us forever.
        try:
            os.dup2(saved_stderr_fd, 2)
        except OSError:
            pass
        try:
            os.close(saved_stderr_fd)
        except OSError:
            pass
        try:
            grep.stdin.close()
        except Exception:
            pass
        try:
            grep.wait(timeout=3)
        except Exception:
            grep.kill()

    atexit.register(_restore_and_drain)
    _LIBAV_LOGS_SILENCED = True
