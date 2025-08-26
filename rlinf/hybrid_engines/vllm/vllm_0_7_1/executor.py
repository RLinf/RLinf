from vllm.executor.executor_base import ExecutorBase
from vllm.config import VllmConfig
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class VLLMExecutor(ExecutorBase):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        
        
    def _init_executor(self):
        raise NotImplementedError("VLLMExecutor._init_executor is not implemented yet.")

    def collective_rpc(self, method: Union[str,Callable], timeout: Optional[float] = None, args: Tuple = (), kwargs: Optional[Dict]=None) -> List[Any]:
        raise NotImplementedError("VLLMExecutor.collective_rpc is not implemented yet.")
    
    def check_health(self):
        raise NotImplementedError("VLLMExecutor.check_health is not implemented yet.")
