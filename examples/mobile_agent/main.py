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

import json
import time

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import (
    Channel,
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)
from rlinf.scheduler.dynamic_scheduler.scheduler_worker import SchedulerWorker
from rlinf.utils.placement import (
    ComponentPlacement,
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor import get_actor_worker
from rlinf.agents.mobile.mobile_agent_loop import MobileAgentLoopWorker
from rlinf.workers.env.phone_worker import PhoneWorker
from rlinf.workers.inference.utils import get_inference_backend_worker
from rlinf.workers.reward.reward_worker import RewardWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker

from rlinf.data.io_struct import RolloutRequest

"""Script to start GRPO training"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(cluster_cfg=cfg.cluster)

    component_placement = ComponentPlacement(cfg, cluster)
    phone_placement_strategy = component_placement.get_strategy("phone")
    phone_wg = PhoneWorker.create_group(cfg).launch(
        cluster, placement_strategy=phone_placement_strategy, name=cfg.phone.group_name
    )
    phone_wg.init_worker().wait()

    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(cfg, component_placement).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    rollout_group.init_worker().wait()

    channel_a2p = Channel.create("a2p")
    channel_p2a = Channel.create("p2a")
    channel_a2l = Channel.create("a2l")
    channel_l2a = Channel.create("l2a")

    channel_agent_input = Channel.create("agent_input")
    channel_agent_output = Channel.create("agent_output")

    mobile_agent_loop_worker = MobileAgentLoopWorker.create_group(
        cfg, component_placement
    ).launch(cluster, placement_strategy=NodePlacementStrategy(node_ranks=[0]))

    num_devices = phone_wg.get_num_devices().wait()

    print(f"{num_devices=}")

    mobile_agent_loop_worker.init_mobile_worker(
        channel_a2l, channel_l2a, channel_a2p, channel_p2a, num_devices
    )

    input_ids = [
        # "Return to desktop first, then open camera and take a photo",
        # "Return to desktop first, then open Photos App and select the last photo",
        "Set a alarm clock with the time 21:00 today.",
        # "Return to desktop first, Delete all alarm clock earlier than 23:00."
    ]

    n = 1
    rollout_request = RolloutRequest(
        n=n,
        input_ids=input_ids,
        answers=[None] * len(input_ids),
        image_data=[None] * len(input_ids),
        multi_modal_inputs=[None] * len(input_ids),
    )

    channel_agent_input.put(rollout_request)


    agent_handle = mobile_agent_loop_worker.run_agentloop_rollout(channel_agent_input, channel_agent_output)
    phone_handle = phone_wg.interact_multi_device(channel_a2p, channel_p2a)
    llm_handle = rollout_group.vl_generate_serverless(channel_a2l, channel_l2a)


    agent_handle.wait()


    for i in range(len(input_ids)):
        print("=" * 60)
        output = channel_agent_output.get()
        for k in range(n):
            print(f"Input {i} Rollout {k}: {output[k].prompt_text}")
            for j in range(len(output[k].trace_prints)):
                print(f"\tTrace print {j}: {output[k].trace_prints[j]}")
            print("=" * 60)
            print()


    # agent_handle = mobile_agent_loop_worker.run_one_test_query(channel_a2p, channel_p2a, channel_a2l, channel_l2a)
    # phone_handle = phone_wg.interact(channel_a2p, channel_p2a)
    # llm_handle = rollout_group.vl_generate(channel_a2l, channel_l2a)

    # agent_handle.wait()
    # phone_handle.wait()
    # llm_handle.wait()


if __name__ == "__main__":
    main()
