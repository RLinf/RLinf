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

from typing import List, Dict
import torch
import time
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
        self.timeout = config.get("compute_score_timeout", 6.0)  
        self.default_score = config.get("default_score_on_timeout", 0.0) 
        self.max_workers = config.get("max_workers", None) 


    def get_reward(
        self, response: List[str], reference: List[List[str]]
    ) -> List[float]:
        """并行计算奖励,每个进程单独超时"""
        n = len(response)
        result_queue = multiprocessing.Queue()
        processes = []
        process_start_times = {}
        start_time = time.time()
        
        try:
            # 1. 启动所有进程
            for i, (resp, ref) in enumerate(zip(response, reference, strict=False)):
                process = Process(
                    target=_compute_score_wrapper,
                    args=(str(resp), str(ref[0]), i, result_queue)
                )
                process.start()
                processes.append((i, process))
                process_start_times[i] = time.time()
                # print(f"Started process {i}")
            
            # 2. 并行等待 + 收集结果
            results = {}
            
            while len(results) < n:
                current_time = time.time()
                
                # 2.1 收集已完成的结果
                self._collect_results(result_queue, results, process_start_times)
                
                # 2.2 检查超时并终止
                for i, process in processes:
                    if i in results:
                        continue
                    
                    elapsed = current_time - process_start_times[i]
                    
                    if elapsed > self.timeout:
                        print(f"⏰ Process {i}: Timeout after {elapsed:.2f}s")
                        results[i] = self.default_score
                        self._terminate_process(process, i)
                    
                    elif not process.is_alive():
                        print(f"⚠️  Process {i}: Died without result")
                        results[i] = self.default_score
                
                if len(results) < n:
                    time.sleep(0.01)
            
            # 3. 最后一次收集结果
            self._collect_results(result_queue, results, process_start_times)
            
            # 4. 返回结果
            rewards = [results.get(i, self.default_score) for i in range(n)]
            
            # total_elapsed = time.time() - start_time
            # success_count = sum(1 for i in range(n) if results.get(i) != self.default_score)
            # print(f"\n{'='*70}")
            # print(f"Completed in {total_elapsed:.2f}s | Success: {success_count}/{n}")
            # print(f"{'='*70}")
            
            return [float(reward) * self.scale for reward in rewards]
        
        finally:
            # 5. 批量终止所有进程
            for i, process in processes:
                if process.is_alive():
                    try:
                        process.terminate()
                    except Exception as e:
                        print(f"Error terminating {i}: {e}")
            
            # 6. 短暂等待优雅退出
            time.sleep(0.3)
            
            # 7. 强制 kill 残留进程
            for i, process in processes:
                if process.is_alive():
                    try:
                        process.kill()
                    except Exception as e:
                        print(f"Error killing {i}: {e}")
            
            # 8. 统一 join 一次(短超时)
            for i, process in processes:
                try:
                    process.join(timeout=0.05)
                except:
                    pass
            
            # 9. 清理 queue
            self._close_queue(result_queue)


    def _collect_results(
        self, 
        result_queue: Queue, 
        results: Dict[int, float],
        process_start_times: Dict[int, float]
    ) -> None:
        """从 queue 中收集结果"""
        while not result_queue.empty():
            try:
                index, result = result_queue.get_nowait()
                if index not in results:
                    if isinstance(result, Exception):
                        print(f"⚠️  Process {index}: Exception - {result}")
                        results[index] = self.default_score
                    else:
                        elapsed = time.time() - process_start_times[index]
                        # print(f"✅ Process {index}: Completed in {elapsed:.2f}s")
                        results[index] = result
            except Exception as e:
                print(f"Error collecting result: {e}")
                break


    def _terminate_process(
        self, 
        process: Process, 
        index: int, 
        force: bool = False
    ) -> None:
        """终止进程 (不等待 join)"""
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
        """安全关闭 queue"""
        try:
            # 清空队列中的剩余数据
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except:
                    break
            
            # 关闭队列
            result_queue.close()
            
            # 等待后台线程结束
            result_queue.join_thread()
        except Exception as e:
            print(f"Error closing queue: {e}")
