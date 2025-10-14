import logging
from time import time
import asyncio
import uuid

import app.config as app_config
from app.libs.redis_queue import RedisQueue
from app.libs.utils import chunkify
from app.model import (
    Submission,
    SubmissionResult,
    WorkPayload,
    BatchSubmission,
    BatchSubmissionResult,
    ResultReason,
)


logger = logging.getLogger(__name__)


def _to_result(submission: Submission, start_time: float, result_json: tuple[str, bytes] | None):
    if result_json is None: # timeout
        return SubmissionResult(sub_id=submission.sub_id, run_success=False, success=False, cost=time() - start_time, reason=ResultReason.QUEUE_TIMEOUT)
    else:
        result = SubmissionResult.model_validate_json(result_json[1])
        if not result.run_success and result.cost >= app_config.MAX_EXECUTION_TIME:
            result.reason = ResultReason.WORKER_TIMEOUT
        return result


async def judge(redis_queue: RedisQueue, submission: Submission):
    start_time = time()
    try:
        payload = WorkPayload(submission=submission)
        payload_json = payload.model_dump_json()
        await redis_queue.pqueue.push(app_config.REDIS_WORK_QUEUE_NAME, {payload_json: time()})
        result_queue_name = f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}'
        result_json = await redis_queue.queue.block_pop(result_queue_name, timeout=app_config.MAX_QUEUE_WAIT_TIME)
        await redis_queue.delete(result_queue_name)
        return _to_result(submission, start_time, result_json)
    except Exception:
        logger.exception(f'Failed to judge submission {submission.sub_id}')
        return SubmissionResult(sub_id=submission.sub_id, run_success=False, success=False, cost=time() - start_time, reason=ResultReason.INTERNAL_ERROR)


async def _judge_batch_impl(redis_queue: RedisQueue, subs: list[Submission], long_batch=False):
    start_time = time()
    max_wait_time = app_config.LONG_BATCH_MAX_QUEUE_WAIT_TIME \
        if long_batch else app_config.MAX_QUEUE_WAIT_TIME
    batch_chunk_size = app_config.MAX_LONG_BATCH_CHUNK_SIZE \
        if long_batch else app_config.MAX_BATCH_CHUNK_SIZE
    # use a hash tag to make sure all payloads are in the same slot in redis cluster
    hash_tag = '{' + str(uuid.uuid4()) + '}'
    sub_chunks = chunkify(subs, batch_chunk_size)

    async def _submit(payloads: list[WorkPayload]):
        # payload.work_id is different, so we can safely use dict
        payload_jsons = {payload.model_dump_json(): payload.timestamp for payload in payloads}
        await redis_queue.pqueue.push(app_config.REDIS_WORK_QUEUE_NAME, payload_jsons)

    async def _sync_pop(queue_names: list[str]):
        step_results = await redis_queue.queue.pop_multi(*queue_names)
        name_results = [(k, v) for k, v in zip(queue_names, step_results) if v is not None]
        return name_results

    async def _async_pop(queue_names: list[str], timeout: int):
        name_result = await redis_queue.queue.block_pop(*queue_names, timeout=timeout)
        if name_result is not None:
            return [(name_result[0].decode(), name_result[1])]
        return []

    async def _pop_results(queue_names: list[str], timeout: int):
        name_results = await _sync_pop(queue_names)
        if not name_results and timeout > 0:
            name_results = await _async_pop(queue_names, min(timeout, app_config.MAX_PROCESS_TIME))
        return name_results

    async def _get_result(payloads: list[WorkPayload], max_chunk_wait_time):
        """max_chunk_wait_time <= 0 means no wait (which is different from block_pop)"""
        result_queue_names = {
            f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}': payload
            for payload in payloads
        }
        results = {}
        result_start_time = time()
        left_time = max_chunk_wait_time
        start_working_time = 0
        left_result_queue_names = list(result_queue_names.keys())

        while left_result_queue_names:
            max_timestamp = max(
                result_queue_names[result_queue_name].timestamp
                for result_queue_name in left_result_queue_names
            )
            name_results = await _pop_results(left_result_queue_names, left_time)
            if not name_results: # if no result, check if timeout
                if start_working_time == 0:
                    # the queue is ordered by timestamp
                    next_payload_info = await redis_queue.pqueue.peak(app_config.REDIS_WORK_QUEUE_NAME)
                    if not next_payload_info:
                        start_working_time = time()
                    else:
                        # before next_work_timestamp, all work is done or processing.
                        # so if it is bigger than max_timestamp, we can assume all work is done or in progress.
                        next_work_timestamp = next_payload_info[1]
                        if next_work_timestamp > max_timestamp:
                            start_working_time = time()
                else:
                    # if start_working_time is set, it means all work is done or in progress.
                    # so we only wait for app_config.MAX_PROCESS_TIME for them to finish.
                    # if it is still not finished, we assume some error happened.
                    # and we can break the loop.
                    if time() - start_working_time > app_config.MAX_PROCESS_TIME:
                        logger.warning(f'No result for {len(left_result_queue_names)} submissions. '
                                       f'Assuming all submissions are timed out.')
                        logger.warning('This is mostly caused by redis OOM or workers killed or potential bug. ')
                        break
            else:
                start_working_time = 0

            for name_result in name_results:
                result_queue_name, _ = name_result
                payload = result_queue_names[result_queue_name]
                results[result_queue_name] = _to_result(payload.submission, start_time, name_result)
                left_result_queue_names.remove(result_queue_name)

            left_time = max_chunk_wait_time - int(time() - result_start_time)
            if left_time <= 0:
                break

        # fill non-ready work as timeout
        for result_queue_name in left_result_queue_names:
            results[result_queue_name] = _to_result(result_queue_names[result_queue_name].submission, start_time, None)

        await redis_queue.delete(*result_queue_names)
        return [results[result_queue_name] for result_queue_name in result_queue_names]

    # submit all submissions to the queue
    payload_chunks = []
    for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
        payload_chunk = [
            WorkPayload(work_id=f'{hash_tag}:{sub_chunk_id}-{idx}', submission=sub, long_running=long_batch)
            for idx, sub in enumerate(sub_chunk)
        ]
        payload_chunks.append(payload_chunk)
        await _submit(payload_chunk)

    results = []
    wait_start_time = time()
    for chunk in payload_chunks:
        # get all results from the queue
        left_time = max_wait_time - int(time() - wait_start_time)
        chunk_results = await _get_result(chunk, left_time)
        results.extend(chunk_results)
    return results


async def judge_batch(redis_queue: RedisQueue, batch_sub: BatchSubmission, long_batch=False):
    try:
        results = await _judge_batch_impl(redis_queue, batch_sub.submissions, long_batch)
    except Exception:
        logger.exception(f'Failed to judge batch submission {batch_sub.sub_id}')
        results=[
            SubmissionResult(
                sub_id=sub.sub_id,
                run_success=False,
                success=False,
                cost=0,
                reason=ResultReason.INTERNAL_ERROR
            ) for sub in batch_sub.submissions
        ]
    return BatchSubmissionResult(
        sub_id=batch_sub.sub_id,
        results=results
    )
