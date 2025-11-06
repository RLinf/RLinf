import subprocess
from dataclasses import dataclass, field
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Generator, Protocol

from ..utils import nothrow_killpg


class ExecuteResult(Protocol):
    success: bool
    cost: float # in seconds


@dataclass
class ProcessExecuteResult:
    stdout: str
    stderr: str
    exit_code: int
    cost: float # in seconds
    success: bool = field(init=False)

    def __post_init__(self):
        self.success = self.exit_code == 0


class CompileError(Exception):
    pass


def _run_as_pg(args: list[str],
        input=None, capture_output=False, timeout=None, check=False, **kwargs):
    # copied from subprocess.run
    # For most cases, this is enough to make sure all subprocesses are
    # killed when the parent process is killed.
    # But if the subprocess creates a child process with new session,
    # this will not work.
    from subprocess import Popen, PIPE, TimeoutExpired, CalledProcessError, CompletedProcess

    kwargs['start_new_session'] = True
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = PIPE

    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used '
                             'with capture_output.')
        kwargs['stdout'] = PIPE
        kwargs['stderr'] = PIPE

    with Popen(args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired as exc:
            # as we set start_new_session=True, pid is the process group id
            nothrow_killpg(pgid=process.pid)
            process.wait()
            raise
        except:  # Including KeyboardInterrupt, communicate handled that.
            nothrow_killpg(pgid=process.pid)
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        # in case some orphaned child process is still running
        nothrow_killpg(pgid=process.pid)
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args,
                                     output=stdout, stderr=stderr)
    return CompletedProcess(process.args, retcode, stdout, stderr)


TIMEOUT_EXIT_CODE = -101
COMPILE_ERROR_EXIT_CODE = -102


class ProcessExecutor:
    def execute(self, command_args: list[str], cwd=None, stdin: str | None = None, timeout: float | None = None) -> ProcessExecuteResult:
        time_start = time.perf_counter()
        try:
            std_input = stdin.encode() if stdin else None
            result = _run_as_pg(command_args, cwd=cwd, shell=False, check=False, capture_output=True, timeout=timeout, input=std_input)
            stdout = result.stdout.decode()
            stderr = result.stderr.decode()
            exit_code = result.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode()
            stderr = e.stderr.decode()
            exit_code = TIMEOUT_EXIT_CODE

        time_end = time.perf_counter()

        return ProcessExecuteResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            cost=time_end - time_start
        )


class ScriptExecutor(ProcessExecutor):
    def setup_command(self, tmp_path: str, script: str) -> Generator[list[str], ProcessExecuteResult, None]:
        """
        Prepare the command to execute the script
        """
        raise NotImplementedError

    def process_result(self, result: ProcessExecuteResult) -> ProcessExecuteResult:
        return result

    def execute_script(self, script: str, stdin: str | None = None, timeout: float | None = None) -> ProcessExecuteResult:
        # add 1 second to timeout as the overhead of the pre/post processing
        timeout = timeout + 1 if timeout else None

        with tempfile.TemporaryDirectory() as tmp_path:
            gen_command = self.setup_command(tmp_path, script)
            command = next(gen_command)
            while True:
                try:
                    result = self.execute(command, cwd=tmp_path, stdin=stdin, timeout=timeout)
                    command = gen_command.send(result)
                except StopIteration:
                    break
            # return the last result
            return self.process_result(result)
