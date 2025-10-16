from multiprocessing import Process
import logging
import threading
from time import sleep, time
from pathlib import Path
import json
from dataclasses import asdict
import traceback
import uuid
import json

import psutil
from pydantic import ValidationError

from app.libs.executors.executor import ProcessExecuteResult
from app.model import Submission, SubmissionResult, WorkPayload, ResultReason
from app.libs.executors.python_executor import PythonExecutor, ScriptExecutor
from app.libs.executors.cpp_executor import CppExecutor
from app.libs.executors.executor import TIMEOUT_EXIT_CODE
import app.config as app_config
from app.work_queue import connect_queue

from app.libs.utils import nothrow_killpg


logger = logging.getLogger(__name__)


def save_error_case(sub: Submission, result: ProcessExecuteResult | None = None, exception: Exception | None = None):
    if not app_config.ERROR_CASE_SAVE_PATH:
        return

    try:
        save_path = Path(app_config.ERROR_CASE_SAVE_PATH) / sub.sub_id
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'submission.json', 'w') as f:
            f.write(sub.model_dump_json(indent=True))
        with open(save_path / 'solution.txt', 'w') as f:
            f.write(sub.solution)
        if result:
            with open(save_path / 'result.json', 'w') as f:
                json.dump(asdict(result), f, indent=True)
        if exception:
            with open(save_path / 'exception.txt', 'w') as f:
                for line in traceback.format_exception(exception):
                    f.write(line)
    except Exception:
        logger.exception(f'Failed to save error case for submission {sub.sub_id}')


def executor_factory(type: str) -> ScriptExecutor:
    if type == 'python':
        return PythonExecutor(
            run_cl=app_config.PYTHON_EXECUTE_COMMAND,
            timeout=app_config.MAX_EXECUTION_TIME,
            memory_limit=app_config.MAX_MEMORY * 1024 * 1024,
        )
    elif type == 'cpp':
        return CppExecutor(
            compiler_cl=app_config.CPP_COMPILE_COMMAND,
            run_cl=app_config.CPP_EXECUTE_COMMAND,
            timeout=app_config.MAX_EXECUTION_TIME,
            memory_limit=app_config.MAX_MEMORY * 1024 * 1024,
        )
    else:
        raise ValueError(f'Unsupported type: {type}')


def judge(sub: Submission):
    try:
        executor = executor_factory(sub.type)
        result = executor.execute_script(sub.solution, sub.input)

        success = result.success
        run_success = result.success
        if sub.expected_output is not None:
            success = success and result.stdout.strip() == sub.expected_output.strip()
        if not success:
            save_error_case(sub, result)
        sub_result = SubmissionResult(
            sub_id=sub.sub_id, success=success, cost=result.cost,
            run_success=run_success,
            # only save stdout and stderr if expected_output is None
            stdout=result.stdout[:app_config.MAX_STDOUT_ERROR_LENGTH]
                if result.stdout is not None else None,
            stderr=result.stderr[:app_config.MAX_STDOUT_ERROR_LENGTH]
                if result.stderr is not None else None,
            reason=ResultReason.WORKER_TIMEOUT
                if result.exit_code == TIMEOUT_EXIT_CODE
                else ResultReason.UNSPECIFIED
        )
    except Exception as e:
        logger.exception(f'Worker failed to judge submission {sub.sub_id}')
        save_error_case(sub, None, e)
        sub_result = SubmissionResult(
            sub_id=sub.sub_id, run_success=False, success=False, cost=0, reason=ResultReason.INTERNAL_ERROR
        )
    return sub_result


class Worker(Process):
    def _run_loop(self):
        worker_id = str(uuid.uuid4())
        redis_queue = connect_queue(False)
        # warm up the connection
        for _ in range(10):
            time_offset = redis_queue.time() - time()
        if abs(time_offset) > 1:
            logger.warning(f'Clock skew detected: {time_offset:.2f} seconds. '
                           f'This may cause issues with timeouts.'
                           f'Please make sure MAX_QUEUE_WORK_LIFE_TIME{app_config.MAX_QUEUE_WORK_LIFE_TIME} is large enough.')
        while True:
            # register worker id
            redis_queue.set(
                f'{app_config.REDIS_WORKER_ID_PREFIX}{worker_id}',
                1,
                app_config.REDIS_WORKER_REGISTER_EXPIRE
            )
            work_item = redis_queue.pqueue.block_pop(app_config.REDIS_WORK_QUEUE_NAME, timeout=app_config.REDIS_WORK_QUEUE_BLOCK_TIMEOUT)
            if not work_item:
                continue
            _, payload_json, _ = work_item

            payload = None
            result = None
            result_queue_name = None
            long_running = False
            try:
                payload = WorkPayload.model_validate_json(payload_json)
                long_running = payload.long_running
                result_queue_name = f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}'
                if not long_running and (lifetime := time() - payload.timestamp) >= app_config.MAX_QUEUE_WORK_LIFE_TIME:
                    logger.warning(f'Work {payload.work_id} lifetime ({lifetime:.2f}>{app_config.MAX_QUEUE_WORK_LIFE_TIME}) timed out. '
                                f'Ignored. Concurrency is too hight?')
                    continue
                result = judge(payload.submission)
            except ValidationError:
                logger.exception(f'Failed to parse payload {payload_json}')
                try:
                    payload_dict = json.load(payload_json)
                    work_id = payload_dict.get('work_id')
                    sub_id = payload_dict.get('submission', {}).get('sub_id')
                    long_running = payload_dict.get('long_running', False)
                except Exception:
                    work_id = None
                    sub_id = None
                    long_running = False
                if work_id and sub_id:
                    result_queue_name = f'{app_config.REDIS_RESULT_PREFIX}{work_id}'
                    result = SubmissionResult(
                        sub_id=sub_id,
                        run_success=False,
                        success=False,
                        cost=0,
                        reason=ResultReason.INVALID_INPUT
                    )
                else:
                    logger.error(f'Failed to parse payload {payload_json}')
                    continue
            except Exception:
                logger.exception(f'Worker failed to process work item {payload_json}')
                if payload is not None and result_queue_name is not None:
                    long_running = payload.long_running
                    result = SubmissionResult(
                        sub_id=payload.submission.sub_id,
                        run_success=False,
                        success=False,
                        cost=0,
                        reason=ResultReason.INTERNAL_ERROR
                    )
                else:
                    logger.error(f'Failed to process work item {payload_json}')
                    continue

            redis_queue.queue.push(result_queue_name, result.model_dump_json())
            redis_queue.expire(
                result_queue_name,
                app_config.REDIS_RESULT_EXPIRE
                    if not long_running
                    else app_config.REDIS_RESULT_LONG_BATCH_EXPIRE
            )

    def run(self):
        while True:
            try:
                self._run_loop()
            except Exception:
                logger.exception(f'Worker failed. Will retry in 60 seconds...')
                sleep(60)


class WorkerManager:
    def __init__(self):
        max_workers = app_config.MAX_WORKERS
        self.workers: list[Worker] = []
        logger.info(f'Starting {max_workers} workers...')
        for _ in range(max_workers):
            worker = Worker()
            worker.start()
            self.workers.append(worker)
        logger.info(f'Started {max_workers} workers')

    def run(self):
        while True:
            try:
                logger.info('Checking workers...')
                self._check_workers()
            except Exception as e:
                logger.exception(f'Check worker failed. Will retry in 60 seconds...')
            sleep(30)

    def run_background(self):
        self._check_thread = threading.Thread(target=self.run, name='worker-checker')
        self._check_thread.daemon = True
        self._check_thread.start()

    def _check_workers(self):
        failed_workers = 0
        busy_workers = 0
        hanged_workers = 0
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                logger.error('Worker dead. Restarting...')
                worker = Worker()
                worker.start()
                self.workers[i] = worker
                failed_workers += 1
            else:
                try:
                    worker_p = psutil.Process(worker.pid)
                    is_busy = 0
                    is_hanged = 0
                    for subp in worker_p.children(recursive=True):
                        is_busy = 1
                        if subp.is_running() and time() - subp.create_time() > app_config.MAX_PROCESS_TIME:
                            is_hanged = 1
                            logger.info(f'Worker {subp.pid} is running for {time() - subp.create_time()} seconds. Terminating...')
                            nothrow_killpg(pid=subp.pid)
                        try:
                            # it should have been terminated by now
                            # let's wait to avoid zombie process
                            subp.wait(timeout=0.1)
                        except Exception:
                            # will try in the next loop
                            pass
                    busy_workers += is_busy
                    hanged_workers += is_hanged
                except Exception:
                    logger.exception(f'Failed to check worker {worker.pid}')

        logger.info(f'Total: {len(self.workers)}, free: {len(self.workers) - busy_workers} failed: {failed_workers}, busy: {busy_workers}, hanged: {hanged_workers}')
