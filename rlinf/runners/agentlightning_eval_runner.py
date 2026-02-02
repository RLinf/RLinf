import logging
import os
import glob
import time
from typing import Optional, Any

if "CUDA_LAUNCH_BLOCKING" not in os.environ:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from omegaconf.dictconfig import DictConfig
import ray
import ray.util
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from agentlightning.adapter.triplet import TraceToTripletBase
from agentlightning.store.base import LightningStore
from rlinf.workers.agent.agentlightning_rollout_worker import AgentLightningRolloutWorker
import typing

if typing.TYPE_CHECKING:
    from rlinf.scheduler import Channel
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor
    from rlinf.workers.rollout.sglang.sglang_worker_server import SGLangWorkerWithHTTPServer


class AgentLightningEvalRunner:

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        val_dataset: Dataset,
        rollout: "SGLangWorkerWithHTTPServer",
        actor: Optional["MegatronActor"],
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

        from rlinf.scheduler import Channel
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

    def init_rollout_workers(self, use_original_model: bool = False):
        rollout_handle = self.rollout.init_worker()

        if use_original_model:
            if (
                self.actor is not None
                and self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from toolkits.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                    convert_hf_to_mg,
                )
                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )

        rollout_handle.wait()
        use_pre_process_policy = getattr(self.cfg.cluster, "use_pre_process_policy", False)
        if use_pre_process_policy:
            self.rollout.offload_engine().wait()

        server_addresses = []
        if hasattr(self.rollout, 'http_server_start') and ray is not None:
            configured_host = self.cfg.server.sglang_http.get('host', '0.0.0.0')
            if configured_host == '0.0.0.0':
                node_ip = ray.util.get_node_ip_address()
            else:
                node_ip = configured_host
            
            base_port = self.cfg.server.sglang_http.get('port', 8020)
            num_workers = len(self.rollout.worker_info_list)
            for rank in range(num_workers):
                port = base_port + rank
                server_addresses.append(f"{node_ip}:{port}")
            
            for rank in range(num_workers):
                self.rollout.execute_on(rank).http_server_start().wait()

        self.agentlightning_rollout_worker.init_worker(
            store=self.store,
            adapter=self.adapter,
            server_addresses=server_addresses,
            group_size=self.cfg.algorithm.group_size,
            model=self.cfg.rollout.model.model_path,
            reward_fillna_value=self.cfg.algorithm.get("reward_fillna_value", 0.0),
            is_eval_mode=True,
        ).wait()

    def init_actor_workers(self):
        if self.actor is not None:
            actor_handle = self.actor.init_worker()
            actor_handle.wait()
            
            megatron_checkpoint = self.cfg.actor.model.get("megatron_checkpoint")
            if megatron_checkpoint is not None and megatron_checkpoint != "null" and os.path.exists(megatron_checkpoint):
                logging.info(f"Loading checkpoint from config: {megatron_checkpoint}")
                self.actor.load_checkpoint(megatron_checkpoint).wait()

    def init_workers(self, use_original_model: bool = False):
        self.init_rollout_workers(use_original_model=use_original_model)
        self.init_actor_workers()

    def _put_batch(self, batch: dict):
        self.dataloader_channel.put(batch, async_op=True)

    def _sync_weights(self):
        if self.actor is not None:
            self.actor.sync_model_to_rollout()
            self.rollout.sync_model_from_actor().wait()
            self.actor.del_reshard_state_dict().wait()

    def _run_eval_loop(self) -> float:
        batch = next(iter(self.val_dataloader))
        
        self._put_batch(batch)
        
        rollout_handle: Handle = self.agentlightning_rollout_worker.process_eval_batch(
            input_channel=self.dataloader_channel
        )
        
        results = rollout_handle.wait()
        avg_reward = results[0] if results and len(results) > 0 else 0.0
        return avg_reward

    def eval(self, checkpoint_dir: Optional[str] = None):
        use_original_model = False
        if checkpoint_dir is None:
            megatron_checkpoint = self.cfg.actor.model.get("megatron_checkpoint")
            if megatron_checkpoint is None or megatron_checkpoint == "null":
                checkpoint_dir = self.cfg.runner.get("resume_dir")
                if checkpoint_dir is None or checkpoint_dir.lower() == "original":
                    use_original_model = True
        elif checkpoint_dir.lower() == "original":
            use_original_model = True
        
        if use_original_model:
            logging.info("Using original HuggingFace model")
            self.init_workers(use_original_model=True)
            self._sync_weights()
            avg_reward = self._run_eval_loop()
            print(f"Evaluation Results:")
            print(f"  Model: Original HuggingFace model ({self.cfg.actor.megatron.ckpt_convertor.hf_model_path})")
            print(f"  Batches: {len(self.val_dataloader)}")
            print(f"  Average Reward: {avg_reward:.6f}")
            return
        
        if checkpoint_dir is None:
            actor_checkpoint_path = self.cfg.actor.model.get("megatron_checkpoint")
            if actor_checkpoint_path is None or actor_checkpoint_path == "null":
                raise ValueError("No checkpoint path provided in config or parameter")
        else:
            if os.path.basename(checkpoint_dir).startswith("global_step_"):
                ckpt_path = checkpoint_dir
            else:
                pattern = os.path.join(checkpoint_dir, "global_step_*")
                checkpoint_dirs = sorted(glob.glob(pattern), key=lambda x: int(x.split("_")[-1]))
                if not checkpoint_dirs:
                    raise ValueError(f"No checkpoint found in {checkpoint_dir}")
                ckpt_path = checkpoint_dirs[0]
                logging.info(f"Found {len(checkpoint_dirs)} checkpoints, evaluating the first one: {ckpt_path}")
            
            actor_checkpoint_path = os.path.join(ckpt_path, "actor")
            self.cfg.actor.model.megatron_checkpoint = actor_checkpoint_path
        
        if self.actor is None or not os.path.exists(actor_checkpoint_path):
            raise ValueError(f"Actor checkpoint not found at {actor_checkpoint_path}")
        
        self.init_workers()
        self._sync_weights()
        
        step = 0
        ckpt_dir = os.path.dirname(actor_checkpoint_path)
        if os.path.basename(ckpt_dir).startswith("global_step_"):
            step = int(os.path.basename(ckpt_dir).split("_")[-1])
        
        avg_reward = self._run_eval_loop()
        print(f"Evaluation Results:")
        print(f"  Checkpoint: {actor_checkpoint_path}")
        print(f"  Step: {step}")
        print(f"  Batches: {len(self.val_dataloader)}")
        print(f"  Average Reward: {avg_reward:.6f}")
