from vllm.v1.worker.gpu_worker import Worker as _Worker
from vllm.config import VllmConfig

class VLLMWorker(_Worker):
    def __init__(self, 
                 vllm_config: VllmConfig, 
                 local_rank: int,
                 rank: int, 
                 distrubuted_init_method: str,
                 is_driver_worker: bool = False):
        super().__init__(vllm_config, local_rank, rank, distrubuted_init_method, is_driver_worker)
    
    