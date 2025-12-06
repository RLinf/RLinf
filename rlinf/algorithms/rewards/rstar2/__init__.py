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

from typing import List
import torch
import multiprocessing
from multiprocessing import Process, Queue

from omegaconf import DictConfig

from toolkits.rstar2.fused_compute_score.compute_score import compute_score


def _compute_score_wrapper(response: str, reference: str, index: int, result_queue: Queue):
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
        self.timeout = config.get("compute_score_timeout", 30.0)  
        self.default_score = config.get("default_score_on_timeout", 0.0) 
        self.max_workers = config.get("max_workers", None) 

    def get_reward(
        self, response: List[str], reference: List[List[str]]
    ) -> List[float]:
        """
        Calculates reward scores for a list of responses compared to corresponding lists of reference answers.
        All compute_score calls are executed in parallel with individual timeout protection.
        
        Args:
            response (List[str]): A list of response strings to be evaluated.
            reference (List[List[str]]): A list where each element is a list of reference strings corresponding to each response.
        
        Returns:
            List[float]: A list of reward scores, one for each response.
        """
        n = len(response)
        result_queue = multiprocessing.Manager().Queue()
        processes = []
        
        # Start all processes
        for i, (resp, ref) in enumerate(zip(response, reference, strict=False)):
            process = Process(
                target=_compute_score_wrapper,
                args=(str(resp), str(ref[0]), i, result_queue)
            )
            process.start()
            processes.append((i, process))
        
        # Wait for all processes to complete or time out
        results = {}
        for i, process in processes:
            process.join(timeout=self.timeout)
            
            if process.is_alive():
                process.terminate()
                process.join()
                print(f"Warning: compute_score timed out after {self.timeout}s for response {i}: {response[i][:50]}...")
                results[i] = self.default_score
        
        # Collect all the results
        while not result_queue.empty():
            try:
                index, result = result_queue.get_nowait()
                if isinstance(result, Exception):
                    print(f"Warning: compute_score raised exception for response {index}: {result}")
                    results[index] = self.default_score
                else:
                    results[index] = result
            except:
                break
        
        # Make sure that all indexes have results
        for i in range(n):
            if i not in results:
                print(f"Warning: No result for response {i}")
                results[i] = self.default_score
        # Return the results in sequence
        rewards = [results[i] for i in range(n)]
        return [float(reward) * self.scale for reward in rewards]
