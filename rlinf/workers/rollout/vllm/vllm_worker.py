from omegaconf import DictConfig
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.placement import ComponentPlacement
from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult
)


class VLLMWorker(Worker):

    def __init__(self, config: DictConfig, placement: ComponentPlacement):
        Worker.__init__(self)
        self._cfg = config
        self._placement = placement

        self._return_logprobs = self._cfg.rollout.return_logprobs
        self._sampling_params = self._get_sampling_params_from_config()
        if self._cfg.algorithm.rollout_batch_size_per_gpu is None:
            self._rollout_batch_size = None
        else:
            self._rollout_batch_size = (
                self._cfg.algorithm.rollout_batch_size_per_gpu
                * self._cfg.rollout.tensor_parallel_size
                * self._cfg.rollout.pipeline_parallel_size
            )
        
    def _get_sampling_params_from_config(self) -> dict:
        raise NotImplementedError("VLLMWorker._get_sampling_params_from_config is not implemented yet.")

    def sync_model_from_actor(self) -> None:
        raise NotImplementedError("VLLMWorker.sync_model_from_actor is not implemented yet.")
    
    def init_worker(self) -> None:
        raise NotImplementedError("VLLMWorker.init_worker is not implemented yet.")

    def _stop(self) -> None:
        raise NotImplementedError("VLLMWorker._stop is not implemented yet."):

    def rollout(self, input_channel: Channel, output_channel: Channel) -> None:
        while True:
            request: RolloutRequest = input_channel.get()
            if request is None:
                self._stop()
                