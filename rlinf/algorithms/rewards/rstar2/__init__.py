# Copyright 2026 The RLinf Authors.
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

import multiprocessing
import time
from multiprocessing import Process, Queue

from omegaconf import DictConfig

from toolkits.rstar2.fused_compute_score.compute_score import compute_score


def _compute_score_wrapper(
    response: str, reference: str, index: int, result_queue: Queue
):
    """
    Wrapper function to run compute_score in a separate process.

    Args:
        response: The response string to evaluate
        reference: The reference string to compare against
        index: Index to track which response this is for
        result_queue: Queue to store the result
    """
    try:
        score = compute_score(response, reference)
        result_queue.put((index, score))
    except Exception as e:
        result_queue.put((index, e))


class Rstar2Reward:
    def __init__(self, config: DictConfig):
        self.scale = config.get("reward_scale", 1.0)
        self.timeout = config.get("compute_score_timeout", 6.0)
        self.default_score = config.get("default_score_on_timeout", 0.0)
        self.max_workers = config.get("max_workers", None)

    def get_reward(
        self, response: list[str], reference: list[list[str]]
    ) -> list[float]:
        # Calculate rewards in parallel processes
        n = len(response)
        result_queue = multiprocessing.Queue()
        processes = []
        process_start_times = {}

        try:
            # start processes
            for i, (resp, ref) in enumerate(zip(response, reference, strict=False)):
                process = Process(
                    target=_compute_score_wrapper,
                    args=(str(resp), str(ref[0]), i, result_queue),
                )
                process.start()
                processes.append((i, process))
                process_start_times[i] = time.time()

            # collect results
            results = {}

            while len(results) < n:
                current_time = time.time()

                self._collect_results(result_queue, results, process_start_times)

                for i, process in processes:
                    if i in results:
                        continue

                    elapsed = current_time - process_start_times[i]

                    if elapsed > self.timeout:
                        print(f"Process {i}: Timeout after {elapsed:.2f}s")
                        results[i] = self.default_score
                        self._terminate_process(process, i)

                    elif not process.is_alive():
                        print(f"Process {i}: Died without result")
                        results[i] = self.default_score

                if len(results) < n:
                    time.sleep(0.01)

            self._collect_results(result_queue, results, process_start_times)

            rewards = [results.get(i, self.default_score) for i in range(n)]

            return [float(reward) * self.scale for reward in rewards]

        finally:
            # terminate all processes
            for i, process in processes:
                if process.is_alive():
                    try:
                        process.terminate()
                    except Exception as e:
                        print(f"Error terminating {i}: {e}")

            time.sleep(0.3)

            for i, process in processes:
                if process.is_alive():
                    try:
                        process.kill()
                    except Exception as e:
                        print(f"Error killing {i}: {e}")

            for i, process in processes:
                try:
                    process.join(timeout=0.05)
                except Exception:
                    pass

            self._close_queue(result_queue)

    def _collect_results(
        self,
        result_queue: Queue,
        results: dict[int, float],
        process_start_times: dict[int, float],
    ) -> None:
        # Collect results from the result queue
        while not result_queue.empty():
            try:
                index, result = result_queue.get_nowait()
                if index not in results:
                    if isinstance(result, Exception):
                        print(f"Process {index}: Exception - {result}")
                        results[index] = self.default_score
                    else:
                        results[index] = result
            except Exception as e:
                print(f"Error collecting result: {e}")
                break

    def _terminate_process(
        self, process: Process, index: int, force: bool = False
    ) -> None:
        # Terminate or kill a process
        if not process.is_alive():
            return

        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except Exception as e:
            print(f"Error terminating process {index}: {e}")

    def _close_queue(self, result_queue: Queue) -> None:
        # Close the result queue properly
        try:
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except Exception:
                    break

            result_queue.close()

            result_queue.join_thread()
        except Exception as e:
            print(f"Error closing queue: {e}")
