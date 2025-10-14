from contextlib import contextmanager
import tempfile
import io
import shlex

from .executor import ScriptExecutor, ProcessExecuteResult, TIMEOUT_EXIT_CODE


SCRIPT_ENDING_MARK = "@@E"
DURATION_MARK = "@@D"


PRE_TEMPLATE = f"""
def _exec_prepare():
    import signal
    import resource
    import os
    import time

    # preventing multi-threading for numpy
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    def _exec_set_alarm_timeout(timeout):
        signal.signal(signal.SIGALRM, _exec_time_exceeded)
        signal.alarm(timeout)

    # checking time limit exceed
    def _exec_time_exceeded(*_):
        print('Suicide from timeout.', flush=True)
        try:
            os.killpg(0, signal.SIGKILL)  # sometime this can still fail
        except Exception:
            pass
        try:
            os.kill(0, signal.SIGKILL)  # sometime this can still fail
        except Exception:
            pass
        os._exit({TIMEOUT_EXIT_CODE})  # may not run here.

    def _exec_set_max_runtime(seconds):
        # setting up the resource limit
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))
        # Just use its default behavior to terminate the process.
        # signal.signal(signal.SIGXCPU, _exec_time_exceeded)

    def _exec_limit_memory(maxsize):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if {{timeout}}:
        _exec_set_alarm_timeout({{timeout}})
        _exec_set_max_runtime({{timeout}})

    if {{memory_limit}}:
        _exec_limit_memory({{memory_limit}})

    return time.perf_counter()

_exec_time_start = _exec_prepare()

""".strip()

POST_TEMPLATE = f"""

def _exec_end():
    import time
    _exec_time_end = time.perf_counter()
    _exec_duration = _exec_time_end - _exec_time_start
    print("{SCRIPT_ENDING_MARK}")
    print(f"{DURATION_MARK}{{_exec_duration}}", flush=True)

_exec_end()

""".strip()

class PythonExecutor(ScriptExecutor):
    def __init__(self, run_cl: str, timeout: int = None, memory_limit: int = None):
        self.timeout = timeout
        self.memory_limit = (
            memory_limit + 128 * 1024 * 1024  # extra 128MB for python overhead
            if memory_limit
            else None
        )
        self.run_cl = run_cl

    def setup_command(self, tmp_path: str, script: str):
        source_path = f"{tmp_path}/source.py"
        with open(source_path, mode='w') as f:
            f.write(PRE_TEMPLATE.format(timeout=self.timeout, memory_limit=self.memory_limit))
            f.write("\n")
            f.write(script)
            f.write("\n")
            f.write(POST_TEMPLATE)
            f.flush()
        yield shlex.split(self.run_cl.format(
            source=shlex.quote(source_path),
            workdir=shlex.quote(str(tmp_path))
        ))

    def process_result(self, result):
        if SCRIPT_ENDING_MARK in result.stdout:
            result.stdout, meta_info = result.stdout.split(SCRIPT_ENDING_MARK, 2)
            for line in io.StringIO(meta_info):
                if line.startswith(DURATION_MARK):
                    result.cost = float(line[len(DURATION_MARK):])
                    break
        return result
