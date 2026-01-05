import os
import signal


def chunkify(iterable, size):
    """Yield successive chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def nothrow_killpg(pid: int = None, pgid:int = None, sig: int = signal.SIGKILL):
    try:
        pgid = os.getpgid(pid) if pgid is None else pgid
        os.killpg(pgid, sig)
    except OSError:
        pass  # ignore errors, process may have already exited
