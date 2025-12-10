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
        n = len(response)
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
        processes = []
        
        try:
            # Start all processes
            for i, (resp, ref) in enumerate(zip(response, reference, strict=False)):
                process = Process(
                    target=_compute_score_wrapper,
                    args=(str(resp), str(ref[0]), i, result_queue)
                )
                process.start()
                processes.append((i, process))
            
            # Wait with timeout and force kill if needed
            results = {}
            for i, process in processes:
                process.join(timeout=self.timeout)
                
                if process.is_alive():
                    print(f"Warning: compute_score timed out for response {i}")
                    process.terminate()
                    process.join(timeout=1.0)  # 给 terminate 1秒时间
                    
                    if process.is_alive():
                        process.kill()  # 强制杀死
                        process.join()
                    
                    results[i] = self.default_score
            
            # Collect results with timeout protection
            collected = 0
            max_attempts = n * 2  # 防止无限循环
            attempts = 0
            
            while collected < n and attempts < max_attempts:
                try:
                    index, result = result_queue.get(timeout=0.1)  # 使用 get(timeout) 而非 get_nowait
                    if isinstance(result, Exception):
                        print(f"Warning: Exception for response {index}: {result}")
                        results[index] = self.default_score
                    else:
                        results[index] = result
                    collected += 1
                except:
                    attempts += 1
            
            # Fill missing results
            for i in range(n):
                if i not in results:
                    print(f"Warning: No result for response {i}")
                    results[i] = self.default_score
            
            rewards = [results[i] for i in range(n)]
            return [float(reward) * self.scale for reward in rewards]
        
        finally:
            # Cleanup
            for _, process in processes:
                if process.is_alive():
                    process.kill()
            manager.shutdown()
