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

import asyncio
import logging
import os
import typing
from typing import Any, Dict, List, Optional, Union

from megatron.core.transformer import attention

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference

if typing.TYPE_CHECKING:
    from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop

logging.getLogger().setLevel(logging.INFO)


class Rstar2Runner:
    """Runner for math model training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        rollout: Optional[Worker] = None,  # 添加rollout_worker参数
        reward: Optional[Worker] = None,
        down_sample: Optional[Worker] = None,
        cluster=None,
    ):
        """"""
        self.cfg = cfg
        self.component_placement = placement
        self.is_pipeline = self.component_placement.is_disaggregated
        self.has_dedicated_inference = inference is not None
        self.has_dedicated_reward = reward is not None
        self.has_dedicated_down_sample = down_sample is not None

        # Workers
        self.actor = actor
        self.rollout = rollout  # 存储rollout_worker引用
        # Collocated mode uses actor as inference
        self.inference = inference if self.has_dedicated_inference else self.actor
        self.reward = reward if self.has_dedicated_reward else self.actor
        self.down_sample = down_sample if self.has_dedicated_down_sample else self.actor
        # AgentLoop components (替代传统rollout)
        # self._tool_agent_loop = None
        self._cluster = cluster  # 存储cluster对象，用于后续初始化AgentLoop组件

        # Data channels
        self.dataloader_channel = Channel.create("DataLoader")
        self.rollout_channel = Channel.create("Rollout")
        # Create a local channel (i.e., a channel that is different in every process)
        # if inference is not a dedicated worker
        self.inference_channel = Channel.create(
            "Inference", local=not self.has_dedicated_inference
        )
        self.reward_channel = Channel.create(
            "Reward", local=not self.has_dedicated_reward
        )
        self.down_sample_channel = Channel.create(
            "DownSample", local=not self.has_dedicated_down_sample
        )
        self.actor_channel = Channel.create("Actor", local=True)
        
        # 不再使用AgentLoop通道，ToolAgentLoop为本地类直接调用

        # Configurations
        self.compute_ref_logprobs = self.cfg.algorithm.kl_beta > 0
        self.recompute_logprobs = self.cfg.algorithm.recompute_logprobs
        self.consumed_samples = 0
        self.global_steps = 0

        # Build dataloader and compute `max_steps`
        self._build_dataloader(train_dataset, val_dataset)
        self._set_max_steps()

        # Wandb table
        self.train_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])
        self.val_df = pd.DataFrame(columns=["step", "prompt", "response", "reward"])

        # Timers
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.metric_logger = MetricLogger(cfg)
        
        # Initialize AgentLoop components if needed
        # 注意：AgentLoop组件将在init_workers()中初始化，避免与权重转换冲突

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        """
        Creates the train and validation dataloaders.
        """
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if collate_fn is None:
            from rlinf.data.datasets import collate_fn

        # Use a sampler to facilitate checkpoint resumption.
        # If shuffling is enabled in the data configuration, create a random sampler.
        if self.cfg.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.cfg.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
            sampler = SequentialSampler(data_source=self.train_dataset)

        num_workers = self.cfg.data.num_workers

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("max_num_gen_batches", 1),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        val_batch_size = (
            self.cfg.data.val_rollout_batch_size
        )  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.cfg.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        logging.info(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

    # def _init_agentloop_components(self, cluster):
    #     """初始化AgentLoop组件（本地实例）"""
    #     from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop
    #     self._tool_agent_loop = ToolAgentLoop(
    #         self.cfg,
    #         self.component_placement,
    #         tools=self._get_agentloop_tools(),
    #         rollout=self.rollout,
    #     )
    #     logging.info("ToolAgentLoop instance created")

    def _pre_process_rollout_request(self, request: RolloutRequest):
        """Split rollout request into smaller groups (reference: SGLangWorker._pre_process_rollout_request).

        Returns: List[List[RolloutRequest]]
        - Outer list: batches
        - Inner list: requests aligned to group_size
        """
        group_size = request.n
        repeated_request = request.repeat()

        # Derive rollout batch size similar to AsyncSGLangWorker
        per_gpu = self.cfg.algorithm.get("rollout_batch_size_per_gpu", None)
        if per_gpu is None:
            rollout_batch_size = None
        else:
            tp = getattr(self.component_placement, "rollout_tp_size", 1)
            pp = getattr(self.component_placement, "rollout_pipeline_parallel_size", 1)
            rollout_batch_size = per_gpu * tp * pp

        if rollout_batch_size is not None:
            assert len(repeated_request.input_ids) % rollout_batch_size == 0, (
                f"rollout_batch_size {rollout_batch_size} must divide the total number of requests {len(repeated_request)}"
            )
            num_batch = len(repeated_request.input_ids) // rollout_batch_size
        else:
            num_batch = 1

        split_requests = repeated_request.split(num_batch)
        if self.component_placement.is_disaggregated:
            num_prompts_per_request = len(split_requests[0].input_ids) // group_size
            return [r.split(num_prompts_per_request) for r in split_requests]
        else:
            return [r.split(1) for r in split_requests]

    def _get_agentloop_tools(self):
        """获取AgentLoop工具配置"""
        tools = {}
        
        # 添加CodeJudge工具（如果配置了）
        if self.cfg.get("tools", {}).get("code_judge"):
            from toolkits.tools.code_judge_tool import CodeJudgeTool
            
            code_judge_config = self.cfg.tools.code_judge
            tools["python_code_with_standard_io"] = CodeJudgeTool(
                name="python_code_with_standard_io",
                host_addr=code_judge_config.get("host_addr", "localhost"),
                host_port=code_judge_config.get("host_port", 8088),
                batch_size=code_judge_config.get("batch_size", 4),
                concurrency=code_judge_config.get("concurrency", 2),
                batch_timeout_seconds=code_judge_config.get("batch_timeout_seconds", 30.0),
            )
        
        return tools

    # 已移除通道处理器

    # 已移除旧的兼容生成函数

    def _run_agentloop_rollout(self):
        """运行AgentLoop rollout - 参考SGLangWorker的rollout方法"""
        import asyncio
        import time
        
        async def _async_rollout():
            try:
                # 从输入通道获取请求 - 参考SGLangWorker
                rollout_request: RolloutRequest = await self.dataloader_channel.get(
                    async_op=True
                ).async_wait()

                # 使用 SGLangWorker 的默认采样参数，不需要传入
                
                # 预处理：拆分请求为小批次和对齐group的子请求
                request_groups_list = self._pre_process_rollout_request(rollout_request)

                last_results = []
                from rlinf.data.io_struct import RolloutResult
                for request_groups in request_groups_list:
                    rollout_tasks = []
                    for group in request_groups:
                        # 为当前group创建任务（直接调用本地 ToolAgentLoop）
                        
                        for raw_id, input_ids in enumerate(group.input_ids):
                            task = asyncio.create_task(
                                self._async_agentloop_generate(raw_id, input_ids)
                            )
                            rollout_tasks.append(task)

                    # 等待所有任务完成，保持顺序
                    task_results = await asyncio.gather(*rollout_tasks)
                    results = []
                    for raw_id, input_ids, result in task_results:
                        results.append(result)

                    # 汇总为 RolloutResult 对象，供后续 actor 使用
                    # Clip to model limits to avoid mask/position size mismatch
                    max_prompt_len = int(self.cfg.data.max_prompt_length)
                    max_total_len = int(self.cfg.actor.model.encoder_seq_length)
                    max_resp_len = max(1, max_total_len - max_prompt_len)

                    prompt_ids = [r["input_ids"][:max_prompt_len] for r in results]
                    response_ids = [r["output_ids"][:max_resp_len] for r in results]
                    prompt_lengths = [len(p) for p in prompt_ids]
                    response_lengths = [len(o) for o in response_ids]
                    response_mask = [r["response_mask"][:max_resp_len] for r in results]
                    is_end = [True for _ in results]
                    print(f"len(results): {len(results)}")
                    rollout_obj = RolloutResult(
                        num_sequence=len(results),
                        group_size=group.n,
                        prompt_lengths=prompt_lengths,
                        prompt_ids=prompt_ids,
                        response_lengths=response_lengths,
                        response_ids=response_ids,
                        is_end=is_end,
                        answers=group.answers,
                        response_mask=response_mask,
                    )

                    # 将结果发送到输出通道（回退为直接发送）
                    await self.rollout_channel.put(rollout_obj, async_op=True).async_wait()
                    last_results = results

                return last_results
                
            except Exception as e:
                logging.error(f"Error in AgentLoop rollout: {e}")
                raise
        
        # 运行异步rollout（本地），并返回本地句柄
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        t0 = time.time()
        try:
            results = loop.run_until_complete(_async_rollout())
            duration = time.time() - t0

            class _LocalHandle:
                def __init__(self, results_obj, duration_sec: float):
                    self._results = results_obj
                    self._duration = duration_sec

                def wait(self):
                    return self._results

                def consume_duration(self, reduction_type: str | None = None):
                    return self._duration
            self.rollout.offload_engine().wait()
            return _LocalHandle(results, duration)
        finally:
            loop.close()
        print(f"run 3")
    
    async def _async_agentloop_generate(self, raw_id: int, input_ids: List[int]):
        """异步AgentLoop生成 - 为每个任务创建独立的ToolAgentLoop避免并发冲突"""
        try:
            # 为避免并发冲突，每个任务使用独立的ToolAgentLoop实例
            from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoop
            task_agent_loop = ToolAgentLoop(
                self.cfg,
                self.component_placement,
                tools=self._get_agentloop_tools(),
                rollout=self.rollout,
            )
            
            # 使用默认采样参数，ToolAgentLoop内部会调用SGLangWorker的agenerate
            agent_output = await task_agent_loop.run(input_ids, {})

            # 构建结果 - 参考SGLangWorker的结果格式（保持 output_ids 为 ids）
            result = {
                "input_ids": input_ids,
                "output_ids": agent_output.response_ids,
                "response_lengths": len(agent_output.response_ids),
                "finished": True,  # AgentLoop总是完成
                "num_turns": agent_output.num_turns,
                "prompt_text": agent_output.prompt_text,
                "response_text": agent_output.response_text,
                "response_mask": agent_output.response_mask,
            }
            
            return raw_id, input_ids, result
            
        except Exception as e:
            logging.error(f"Error in AgentLoop generate (raw_id={raw_id}): {e}")
            raise

    def init_workers(self):
        # Must be done before actor init
        if self.cfg.runner.resume_dir is None:
            logging.info("Training from scratch")
            if (
                self.cfg.actor.training_backend == "megatron"
                and self.cfg.actor.megatron.use_hf_ckpt
            ):
                from toolkits.ckpt_convertor.convert_hf_to_mg import convert_hf_to_mg

                convert_hf_to_mg(
                    self.cfg.actor.megatron.ckpt_convertor.hf_model_path,
                    self.cfg.actor.megatron.ckpt_convertor,
                )

        # Init workers
        self.actor.init_worker().wait()
        if self.has_dedicated_inference:
            self.inference.init_worker().wait()
        if self.has_dedicated_reward:
            self.reward.init_worker().wait()
        
        # Init rollout worker
        if self.rollout is not None:
            self.rollout.init_worker().wait()

        # Init AgentLoop components（本地实例，无需集群）
        # self._init_agentloop_components(self._cluster)

        if self.cfg.runner.resume_dir is None:
            return

        # Resume from checkpoint
        logging.info(f"Load from checkpoint folder: {self.cfg.runner.resume_dir}")
        # set global step
        self.global_steps = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
        logging.info(f"Setting global step to {self.global_steps}")

        actor_checkpoint_path = os.path.join(self.cfg.runner.resume_dir, "actor")
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        # load data
        dataloader_local_path = os.path.join(self.cfg.runner.resume_dir, "data/data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            logging.warning(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _compute_flops_metrics(self, time_metrics, act_rollout_metrics) -> dict:
        rollout_time = time_metrics.get("rollout")
        inference_time = time_metrics.get("inference", -1)
        training_time = time_metrics.get("training")

        num_gpus_actor = self.component_placement.actor_world_size
        num_gpus_rollout = self.component_placement.rollout_world_size

        rollout_tflops = act_rollout_metrics["rollout_tflops"]
        inference_tflops = act_rollout_metrics["inference_tflops"]
        training_tflops = act_rollout_metrics["training_tflops"]

        flops_metrics = {
            "rollout_tflops_per_gpu": 0.0,
            "inference_tflops_per_gpu": 0.0,
            "training_tflops_per_gpu": 0.0,
        }
        if rollout_time > 0 and rollout_tflops > 0:
            flops_metrics["rollout_tflops_per_gpu"] = (
                rollout_tflops / rollout_time / num_gpus_rollout
            )

        if inference_time > 0 and inference_tflops > 0:
            num_gpus_inference = self.component_placement.inference_world_size
            if num_gpus_inference == 0:
                num_gpus_inference = self.component_placement.actor_world_size
            flops_metrics["inference_tflops_per_gpu"] = (
                inference_tflops / inference_time / num_gpus_inference
            )

        if training_time > 0 and training_tflops > 0:
            flops_metrics["training_tflops_per_gpu"] = (
                training_tflops / training_time / num_gpus_actor
            )

        return flops_metrics

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.output_dir,
            self.cfg.runner.experiment_name,
            f"checkpoints/global_step_{self.global_steps}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        data_save_path = os.path.join(base_output_dir, "data")

        # actor
        self.actor.save_checkpoint(actor_save_path, self.global_steps).wait()

        # data
        local_mkdir_safe(data_save_path)
        dataloader_local_path = os.path.join(data_save_path, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

    def _set_max_steps(self):
        self.num_steps_per_epoch = len(self.train_dataloader)
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_steps // self.num_steps_per_epoch

    def _put_batch(self, batch: Dict[str, torch.Tensor]):
        prompt_ids = batch["prompt"].tolist()
        lengths = batch["length"].tolist()
        answers = batch["answer"].tolist()
        prompts = [ids[-pmp_len:] for ids, pmp_len in zip(prompt_ids, lengths)]
        rollout_dp_size = self.component_placement.rollout_dp_size

        for input_ids, answers in zip(
            split_list(prompts, rollout_dp_size, enforce_divisible_batch=False),
            split_list(answers, rollout_dp_size, enforce_divisible_batch=False),
        ):
            request = RolloutRequest(
                n=self.cfg.algorithm.group_size,
                input_ids=input_ids,
                answers=answers,
            )
            self.dataloader_channel.put(request, async_op=True)

    def _sync_weights(self):
        # 完全对齐 math_runner 的权重同步顺序：
        # 1) actor -> rollout 权重同步
        # 2) rollout 侧从 actor 拉取
        # 3) actor 清理 reshard
        # 4) （可选）inference 同步
        # 1) 将权重发送给 rollout
        self.actor.sync_model_to_rollout()
        # 2) rollout 从 actor 拉取权重（异步接口，需等待）
        self.rollout.sync_model_from_actor().wait()

        # 3) 清理 actor 的 reshard 状态
        self.actor.del_reshard_state_dict().wait()

        # 4) 如有专用 inference，再进行推理侧同步
        if self.has_dedicated_inference:
            self.actor.sync_model_to_inference()
            self.inference.sync_model_from_actor().wait()

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.run_timer.start_time()
        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    with self.timer("sync_weights"):
                        self._sync_weights()

                    # Rollout - 使用AgentLoop
                    rollout_handle: Handle = self._run_agentloop_rollout()

                    if self.recompute_logprobs:
                        # Inference prev/ref logprobs
                        infer_handle: Handle = self.inference.run_inference(
                            input_channel=self.rollout_channel,
                            output_channel=self.inference_channel,
                            compute_ref_logprobs=self.compute_ref_logprobs,
                        )
                        inference_channel = self.inference_channel
                    else:
                        infer_handle = None
                        inference_channel = self.rollout_channel

                    # Rewards
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=inference_channel,
                        output_channel=self.reward_channel,
                    )
                    
                    down_sample_handle: Handle = self.down_sample.down_sample_batch(
                        input_channel=self.reward_channel,
                        output_channel=self.down_sample_channel,
                    )

                    # Advantages and returns
                    adv_handle: Handle = self.actor.compute_advantages_and_returns(
                        input_channel=self.down_sample_channel,
                        output_channel=self.actor_channel,
                    )

                    # Actor training
                    actor_input_channel = self.actor_channel
                    if self.is_pipeline:
                        # In pipeline mode, the rollout already contains the advantages and returns
                        # So the above two steps are in fact no-ops, and we should directly use the inference channel as the input
                        actor_input_channel = inference_channel
                    actor_handle: Handle = self.actor.run_training(
                        input_channel=actor_input_channel,
                    )

                    metrics = actor_handle.wait()
                    self.global_steps += 1

                    run_time_exceeded = self.run_timer.is_finished()
                    _, save_model, is_train_end = check_progress(
                        self.global_steps,
                        self.max_steps,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=run_time_exceeded,
                    )

                    if save_model:
                        self._save_checkpoint()

                    if is_train_end:
                        logging.info(
                            f"Step limit given by max_steps={self.max_steps} reached. Stopping run"
                        )
                        return

                    if run_time_exceeded:
                        logging.info(
                            f"Time limit given by run_timer={self.run_timer} reached. Stopping run"
                        )
                        return

                time_metrics = self.timer.consume_durations()
                time_metrics["training"] = actor_handle.consume_duration()
                time_metrics["rollout"] = rollout_handle.consume_duration()
                time_metrics["reward"] = reward_handle.consume_duration()
                time_metrics["advantage"] = adv_handle.consume_duration()
                if infer_handle is not None:
                    # Inference time should be the min time across ranks, because different DP receive the rollout results differently
                    # But at the begin of the pp schedule, there is a timer barrier
                    # This makes all DP end at the same time, while they start at differnt times, and thus only the min time is correct
                    time_metrics["inference"] = infer_handle.consume_duration(
                        reduction_type="min"
                    )
                num_minibatches = len(metrics[0][1])
                print("num_minibatches:", num_minibatches)
                logging_steps = (
                    self.global_steps - 1
                ) * num_minibatches
                # add prefix to the metrics
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                rollout_metrics = {f"rollout/{k}": v for k, v in metrics[0][0].items()}

                self.metric_logger.log(log_time_metrics, logging_steps)
                self.metric_logger.log(rollout_metrics, logging_steps)
                for i in range(num_minibatches):
                    training_metrics = {
                        f"train/{k}": v for k, v in metrics[0][1][i].items()
                    }
                    self.metric_logger.log(training_metrics, logging_steps + i)

                logging_metrics = time_metrics

                if self.cfg.actor.get("calculate_flops", False):
                    flops_metrics = self._compute_flops_metrics(
                        time_metrics, metrics[0][0]
                    )
                    flops_metrics = {f"flops/{k}": v for k, v in flops_metrics.items()}
                    self.metric_logger.log(flops_metrics, logging_steps)
                    logging_metrics.update(flops_metrics)

                logging_metrics.update(metrics[0][0])
                if num_minibatches > 0:
                    logging_metrics.update(metrics[0][1][-1])

                global_pbar.set_postfix(logging_metrics)
                global_pbar.update(1)

        self.metric_logger.finish()
