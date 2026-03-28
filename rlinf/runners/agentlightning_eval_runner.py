import logging
import os
import time
from typing import Any, Optional, Union

import requests

if "CUDA_LAUNCH_BLOCKING" not in os.environ:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from agentlightning.adapter.triplet import TraceToTripletBase
from agentlightning.store.base import LightningStore
from rlinf.workers.agent.agentlightning_rollout_worker import AgentLightningRolloutWorker
import typing

if typing.TYPE_CHECKING:
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.rollout.sglang.sglang_worker_server import SGLangWorkerWithHTTPServer

class AgentLightningEvalRunner:

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        val_dataset: Dataset,
        rollout: SGLangWorkerWithHTTPServer,
        actor: MAMegatronActor,
        store: LightningStore,
        adapter: TraceToTripletBase,
        agentlightning_rollout_worker: AgentLightningRolloutWorker,
    ):
        self.cfg = cfg
        self.placement = placement
        self.val_dataset = val_dataset
        self.rollout = rollout
        self.actor = actor
        self.store = store
        self.adapter = adapter
        self.agentlightning_rollout_worker = agentlightning_rollout_worker

        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")
        self._build_dataloader()

    def _build_dataloader(self):
        def agl_collate_fn(data_list: list[dict]) -> dict[str, Any]:
            batch = {}
            keys = list(data_list[0].keys())
            for key in keys:
                batch[key] = [item[key] for item in data_list]
            return batch

        val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=self.cfg.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=agl_collate_fn,
        )

    def init_rollout_workers(self):
        logging.info("[AgentLightningEvalRunner] init_rollout_workers: calling rollout.init_worker()")
        rollout_handle = self.rollout.init_worker()
        rollout_handle.wait()
        logging.info("[AgentLightningEvalRunner] rollout.init_worker finished")

        use_pre_process_policy = getattr(self.cfg.cluster, "use_pre_process_policy", False)
        if use_pre_process_policy:
            self.rollout.offload_engine().wait()

        agl_server_addresses = self.rollout.get_server_address().wait()

        logging.info(
            "[AgentLightningEvalRunner] initializing AgentLightningRolloutWorker with server_addresses=%s",
            agl_server_addresses,
        )
        self.agentlightning_rollout_worker.init_worker(
            store=self.store,
            adapter=self.adapter,
            server_addresses=agl_server_addresses,
            group_size=self.cfg.algorithm.group_size,
            model=self.cfg.rollout.model.model_path,
            reward_fillna_value=self.cfg.algorithm.get("reward_fillna_value", 0.0),
            is_eval_mode=True,
        ).wait()
        logging.info("[AgentLightningEvalRunner] AgentLightningRolloutWorker.init_worker finished")

    def init_workers(self):
        self.init_rollout_workers()

    def _put_batch(self, batch: dict):
        self.dataloader_channel.put(batch, async_op=True)

    def _run_eval_loop(self) -> float:
        logging.info("[AgentLightningEvalRunner] _run_eval_loop: fetching first batch from val_dataloader")
        batch = next(iter(self.val_dataloader))
        logging.info(
            "[AgentLightningEvalRunner] _run_eval_loop: got batch with keys=%s size=%d",
            list(batch.keys()) if isinstance(batch, dict) else type(batch),
            len(next(iter(batch.values()))) if isinstance(batch, dict) and batch else 0,
        )

        self._put_batch(batch)
        logging.info("[AgentLightningEvalRunner] _run_eval_loop: submitted batch to dataloader_channel, calling process_eval_batch")

        rollout_handle: Handle = self.agentlightning_rollout_worker.process_eval_batch(
            input_channel=self.dataloader_channel
        )

        logging.info("[AgentLightningEvalRunner] _run_eval_loop: waiting for rollout_handle")
        results = rollout_handle.wait()
        logging.info("[AgentLightningEvalRunner] _run_eval_loop: rollout_handle returned results=%r", results)
        avg_reward = results[0] if results and len(results) > 0 else 0.0
        return avg_reward

    def eval(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir not in (None, "", "original"):
            logging.warning(
                "checkpoint_dir is ignored in HF eval mode. "
                "Current eval path always uses rollout HF model directly."
            )

        if (
            not self.cfg.rollout.validate_weight
            and not self.cfg.rollout.get("validate_weight_first_sync", False)
        ):
            logging.warning(
                "rollout.validate_weight and rollout.validate_weight_first_sync are both false; "
                "set validate_weight_first_sync=true for HF eval."
            )
            self.cfg.rollout.validate_weight_first_sync = True

        self.init_workers()
        avg_reward = self._run_eval_loop()
        logging.info(f"Evaluation Results:")
        logging.info(f"  Model: HF rollout model ({self.cfg.rollout.model.model_path})")
        logging.info(f"  Batches: {len(self.val_dataloader)}")
        logging.info(f"  Average Reward: {avg_reward:.6f}")
