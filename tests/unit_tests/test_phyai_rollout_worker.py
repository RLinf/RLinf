# Copyright 2026 The RLinf Authors.
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
import importlib.util
import inspect
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.scheduler import Worker

_HAS_PHYAI = importlib.util.find_spec("phyai") is not None
pytestmark = pytest.mark.skipif(not _HAS_PHYAI, reason="PhyAI is not installed")

if _HAS_PHYAI:
    from phyai.models.pi05.scheduler_pi05 import PI05RolloutRequest

    from rlinf.models.embodiment.openpi.modules.model import IMAGE_KEYS
    from rlinf.workers.rollout.phyai import phyai_rollout_worker
    from rlinf.workers.rollout.phyai.phyai_rollout_worker import (
        PhyAIRolloutWorker,
        _PhyAIWeightTarget,
    )


class _FakeEngine:
    def __init__(self) -> None:
        self.events = []

    def begin_weight_update(self) -> None:
        self.events.append("begin")

    def update_weights(self, state_dict) -> None:
        self.events.append(("update", state_dict))

    def finish_weight_update(self, version=None):
        self.events.append(("finish", version))
        return SimpleNamespace(loaded=["model.weight"])

    def abort_weight_update(self) -> None:
        self.events.append("abort")


class _FakeBucketWeightSyncer(BucketWeightSyncer):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(
            bucket_size=1024,
            bucket_dtype=None,
            bucket_device="cpu",
        )
        self.fail = fail

    async def init_receiver(self, state_dict, recv, send=None) -> None:
        del state_dict, recv, send
        self._receiver_initialized = True

    async def apply(self, model, recv) -> int:
        del recv
        model.load_state_dict({"model.weight": torch.ones(2, 2)})
        if self.fail:
            raise RuntimeError("sync failed")
        return 7


def _make_rollout_worker(*, fail: bool = False):
    worker = object.__new__(PhyAIRolloutWorker)
    engine = _FakeEngine()
    worker._accelerator_type = "cpu"
    worker._timer_metrics = {}
    worker._rank = 0
    worker._engine = engine
    worker._weight_target = _PhyAIWeightTarget(engine)
    worker.weight_syncer = _FakeBucketWeightSyncer(fail=fail)
    worker.actor_group_name = "actor"
    worker.actor_weight_src_rank = 0
    worker._group_name = "rollout"
    worker._weight_sync_rollout_ranks = [0]
    worker._weight_sync_is_sender = False
    worker._sync_weight_comm_options = None
    worker.finished_episodes = None
    worker.total_num_train_envs = 4
    worker.rollout_epoch = 2
    worker.version = 0
    worker.torch_platform = SimpleNamespace(empty_cache=lambda: None)
    worker.log_info = lambda _message: None
    return worker, engine


def _openpi_input_transform(sample):
    base = np.asarray(sample["observation/image"], dtype=np.uint8)
    wrist = np.asarray(sample["observation/wrist_image"], dtype=np.uint8)
    state = np.asarray(sample["observation/state"], dtype=np.float64) / 10
    prompt_length = min(len(sample["prompt"]), 8)
    return {
        "image": {
            IMAGE_KEYS[0]: base,
            IMAGE_KEYS[1]: wrist,
            IMAGE_KEYS[2]: np.zeros_like(base),
        },
        "image_mask": {
            IMAGE_KEYS[0]: np.bool_(True),
            IMAGE_KEYS[1]: np.bool_(True),
            IMAGE_KEYS[2]: np.bool_(False),
        },
        "state": state,
        "tokenized_prompt": np.arange(8, dtype=np.int64),
        "tokenized_prompt_mask": np.arange(8) < prompt_length,
    }


def _openpi_output_transform(sample):
    return {"actions": sample["actions"][..., :7] + sample["state"][0]}


def _make_env_obs(batch_size: int = 2):
    image_size = batch_size * 224 * 224 * 3
    main_images = torch.arange(image_size, dtype=torch.int64)
    wrist_images = torch.arange(image_size, dtype=torch.int64).flip(0)
    return {
        "main_images": main_images.remainder(251)
        .to(torch.uint8)
        .view(batch_size, 224, 224, 3),
        "wrist_images": wrist_images.remainder(251)
        .to(torch.uint8)
        .view(batch_size, 224, 224, 3),
        "states": torch.arange(batch_size * 8, dtype=torch.float32).view(batch_size, 8),
        "task_descriptions": ["pick", "place"][:batch_size],
    }


def _configure_preprocessing(worker):
    worker._engine_device = torch.device("cpu")
    worker._num_images = 2
    worker._input_transform_fn = _openpi_input_transform
    worker._output_transform_fn = _openpi_output_transform
    worker._openpi_config_name = "pi05_libero"
    worker._state_indices = None


def test_sync_model_from_actor_commits_only_complete_updates():
    worker, engine = _make_rollout_worker()
    asyncio.run(inspect.unwrap(PhyAIRolloutWorker.sync_model_from_actor)(worker))

    event_names = [
        event[0] if isinstance(event, tuple) else event for event in engine.events
    ]
    assert event_names == ["begin", "update", "finish"]
    assert engine.events[-1] == ("finish", 7)
    assert worker.version == 7
    assert worker.finished_episodes == 56

    worker, engine = _make_rollout_worker(fail=True)
    with pytest.raises(RuntimeError, match="sync failed"):
        asyncio.run(inspect.unwrap(PhyAIRolloutWorker.sync_model_from_actor)(worker))
    assert engine.events[-1] == "abort"
    assert worker.version == 0


def test_evaluate_preserves_channel_contract():
    class _AsyncResult:
        def __init__(self, value):
            self.value = value

        async def async_wait(self):
            return self.value

    worker = object.__new__(PhyAIRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"group_name": "env"}})
    worker.eval_rollout_epoch = 1
    worker.n_eval_chunk_steps = 1
    worker.num_pipeline_stages = 2
    worker.eval_batch_size = 2
    worker._rank = 0

    recv_calls = []
    send_calls = []
    predict_calls = []

    def recv_from(**kwargs):
        recv_calls.append(kwargs)
        stage_id = kwargs["route_key"]
        chunks = [
            {
                "obs": {
                    "states": torch.tensor([[stage_id * 2 + offset]]),
                    "task_descriptions": [f"stage-{stage_id}-{offset}"],
                },
                "final_obs": {"states": torch.tensor([[99]])},
                "rlt_switch_flags": torch.ones(1, dtype=torch.bool),
                "intervene_flags": torch.ones(1, dtype=torch.bool),
            }
            for offset in range(2)
        ]
        assert all(kwargs["infer_batch_size_fn"](chunk) == 1 for chunk in chunks)
        merged = kwargs["merge_fn"](chunks)
        assert set(merged) == {"obs"}
        return _AsyncResult(merged)

    def predict(obs, mode):
        predict_calls.append((obs, mode))
        return torch.zeros(2, 1, 1), {}

    worker.recv_from = recv_from
    worker.predict = predict
    worker.send_to = lambda **kwargs: send_calls.append(kwargs)

    asyncio.run(inspect.unwrap(PhyAIRolloutWorker.evaluate)(worker, "input", "output"))

    assert [call["route_key"] for call in recv_calls] == [0, 1]
    assert all(call["group_name"] == "env" for call in recv_calls)
    assert all(call["channel"] == "input" for call in recv_calls)
    assert all(call["tag"] == "eval_rollout_results" for call in recv_calls)
    assert all(call["batch_size"] == 2 for call in recv_calls)
    assert len(predict_calls) == 2
    for stage_id, (obs, mode) in enumerate(predict_calls):
        assert mode == "eval"
        assert torch.equal(
            obs["states"], torch.tensor([[stage_id * 2], [stage_id * 2 + 1]])
        )
        assert obs["task_descriptions"] == [
            f"stage-{stage_id}-0",
            f"stage-{stage_id}-1",
        ]
    assert [call["route_key"] for call in send_calls] == [0, 1]
    assert all(call["group_name"] == "env" for call in send_calls)
    assert all(call["channel"] == "output" for call in send_calls)
    assert all(call["tag"] == "eval_rollout_results" for call in send_calls)
    assert all(call["batch_size"] == 2 for call in send_calls)
    assert all(call["data"].device.type == "cpu" for call in send_calls)
    assert all(call["data"].is_contiguous() for call in send_calls)


def test_phyai_weight_target_converts_actor_layout():
    engine = _FakeEngine()
    target = _PhyAIWeightTarget(engine)
    qkv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    gating = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    value_bias = torch.ones(4)

    target.load_state_dict(
        {
            "model.img.encoder.layers.0.attn.in_proj_weight": qkv,
            "model.llm.layers.0.mlps.1.w_gating": gating,
            "value_head.mlp.0.bias": value_bias,
        }
    )

    weights = engine.events[0][1]
    vision_prefix = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "encoder.layers.0.self_attn"
    )
    assert torch.equal(weights[f"{vision_prefix}.q_proj.weight"], qkv[:2])
    assert torch.equal(weights[f"{vision_prefix}.k_proj.weight"], qkv[2:4])
    assert torch.equal(weights[f"{vision_prefix}.v_proj.weight"], qkv[4:])
    expert_prefix = "paligemma_with_expert.gemma_expert.model.layers.0.mlp"
    assert torch.equal(weights[f"{expert_prefix}.gate_proj.weight"], gating[0].T)
    assert torch.equal(weights[f"{expert_prefix}.up_proj.weight"], gating[1].T)
    assert weights["value_head.mlp.0.bias"] is value_bias


def test_phyai_rollout_worker_rejects_unknown_plugin(monkeypatch):
    monkeypatch.setattr(Worker, "__init__", lambda _worker: None)
    cfg = OmegaConf.create({"rollout": {"phyai": {"plugin": "unknown"}}})

    with pytest.raises(NotImplementedError, match="pi05"):
        PhyAIRolloutWorker(cfg)


@pytest.mark.parametrize("only_eval", [False, True])
@pytest.mark.parametrize(
    "kernel_config",
    [
        None,
        "~/phyai-kernel.yaml",
        {
            "schema": "phyai.kernel/v1",
            "profile": "static",
            "defaults": {"fallback": "error"},
            "overrides": [],
        },
    ],
)
def test_init_worker_loads_runner_checkpoint_and_passes_engine_config(
    monkeypatch, only_eval, kernel_config
):
    captured = {}
    trained_weight = torch.ones(4, 4)
    monkeypatch.setenv("PHYAI_KERNEL_CONFIG", "/environment/kernel.yaml")

    class _InitializedEngine:
        def __init__(self, args):
            expected_env = (
                None if isinstance(kernel_config, dict) else "/environment/kernel.yaml"
            )
            assert os.environ.get("PHYAI_KERNEL_CONFIG") == expected_env
            captured["engine_args"] = args

        def begin_weight_update(self):
            captured.setdefault("events", []).append("begin")

        def update_weights(self, state_dict):
            captured.setdefault("events", []).append(("update", state_dict))

        def finish_weight_update(self):
            captured.setdefault("events", []).append("finish")

        def abort_weight_update(self):
            captured.setdefault("events", []).append("abort")

    def load_checkpoint(path, *, map_location, weights_only):
        assert path == "/trained.pt"
        assert map_location == "cpu"
        assert weights_only
        return {"model.state_proj.weight": trained_weight}

    monkeypatch.setattr(phyai_rollout_worker, "RuntimeConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_rollout_worker, "DeviceConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_rollout_worker, "EngineConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_rollout_worker, "PI05Args", SimpleNamespace)
    monkeypatch.setattr(phyai_rollout_worker, "EngineArgs", SimpleNamespace)
    monkeypatch.setattr(phyai_rollout_worker, "Engine", _InitializedEngine)
    monkeypatch.setattr(torch, "load", load_checkpoint)

    def input_transform(sample):
        return sample

    def output_transform(sample):
        return sample

    def _build_transforms(model_path, config_name, data_kwargs=None):
        captured["transforms"] = (model_path, config_name, data_kwargs)
        return [input_transform], [output_transform]

    openpi = ModuleType("openpi")
    transforms = ModuleType("openpi.transforms")
    transforms.compose = lambda functions: functions[0]
    openpi.transforms = transforms
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms)
    monkeypatch.setattr(
        phyai_rollout_worker, "build_openpi_transforms", _build_transforms
    )

    worker = object.__new__(PhyAIRolloutWorker)
    worker.cfg = OmegaConf.create({"runner": {"ckpt_path": "/trained.pt"}})
    worker._engine = None
    worker._phyai_plugin = "pi05"
    worker._phyai_cfg = OmegaConf.create(
        {"params_dtype": "bf16", "kernel_config": kernel_config}
    )
    worker.model_cfg = OmegaConf.create(
        {
            "model_path": "/checkpoint",
            "model_type": "openpi",
            "precision": "bf16",
            "num_action_chunks": 5,
            "action_dim": 7,
            "num_steps": 3,
            "add_value_head": True,
            "openpi": {
                "config_name": "pi05_libero",
                "action_horizon": 10,
                "model_action_dim": 32,
                "max_token_len": 200,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "num_images_in_input": 2,
                "detach_critic_input": True,
                "value_after_vlm": True,
            },
        }
    )
    worker.only_eval = only_eval
    worker.torch_device_type = "cpu"
    worker.per_node_train_batch_size = 2
    worker.per_node_eval_batch_size = 2
    worker.log_info = lambda _message: None
    worker._rank = 0

    worker.init_worker()

    assert os.environ["PHYAI_KERNEL_CONFIG"] == "/environment/kernel.yaml"
    engine_args = captured["engine_args"]
    if isinstance(kernel_config, dict):
        assert engine_args.config.kernel.policy_config == kernel_config
        assert engine_args.config.kernel.config_path is None
    else:
        expected_path = str(Path(kernel_config).expanduser()) if kernel_config else None
        assert engine_args.config.kernel.config_path == expected_path
        assert engine_args.config.kernel.policy_config is None
    plugin_args = engine_args.plugin_args
    plugin_cfg = plugin_args.config
    assert engine_args.config.device.params_dtype is torch.bfloat16
    assert plugin_args.checkpoint_dir == "/checkpoint"
    assert plugin_cfg.chunk_size == 10
    assert plugin_cfg.max_action_dim == 32
    assert plugin_cfg.num_inference_steps == 3
    assert plugin_cfg.critic_action_chunk is None
    assert plugin_cfg.add_value_head
    assert plugin_cfg.value_after_vlm
    assert captured["transforms"] == ("/checkpoint", "pi05_libero", None)
    assert worker._input_transform_fn is input_transform
    assert worker._output_transform_fn is output_transform
    assert captured["events"][0] == "begin"
    assert captured["events"][1][0] == "update"
    assert captured["events"][1][1]["state_proj.weight"] is trained_weight
    assert captured["events"][2] == "finish"


def test_phyai_preprocessing_builds_canonical_request():
    env_obs = _make_env_obs()
    worker = object.__new__(PhyAIRolloutWorker)
    _configure_preprocessing(worker)

    request, replay = worker._build_request(env_obs)
    assert torch.equal(request.input_ids, torch.arange(8).expand(2, -1))
    assert torch.equal(request.lang_lens, torch.tensor([4, 5]))
    assert torch.equal(replay.state, env_obs["states"] / 10)
    assert torch.equal(replay.tokenized_prompt, torch.arange(8).expand(2, -1))
    expected_token_mask = torch.arange(8)[None] < torch.tensor([[4], [5]])
    assert torch.equal(replay.tokenized_prompt_mask, expected_token_mask)
    assert torch.equal(
        replay.images[IMAGE_KEYS[0]],
        env_obs["main_images"].to(torch.float32) / 255 * 2 - 1,
    )
    assert torch.equal(
        replay.images[IMAGE_KEYS[1]],
        env_obs["wrist_images"].to(torch.float32) / 255 * 2 - 1,
    )
    assert replay.images[IMAGE_KEYS[2]].eq(-1).all()
    for key, valid in zip(IMAGE_KEYS, (True, True, False), strict=True):
        assert torch.equal(replay.image_masks[key], torch.full((2,), valid))
    assert request.pixel_values.shape[1] == 2
    for index, key in enumerate(IMAGE_KEYS[:2]):
        assert torch.equal(
            request.pixel_values[:, index],
            replay.images[key].permute(0, 3, 1, 2),
        )
    assert request.pixel_values.dtype is torch.float32
    assert request.pixel_values.is_contiguous()


def test_training_predict_returns_actor_replay_contract(monkeypatch):
    batch_size = 2
    env_obs = _make_env_obs(batch_size)
    model_actions = torch.arange(batch_size * 10 * 32, dtype=torch.float32).view(
        batch_size, 10, 32
    )
    chains = torch.arange(batch_size * 4 * 10 * 32, dtype=torch.float32).view(
        batch_size, 4, 10, 32
    )

    transition_logprobs = torch.arange(
        batch_size * 3 * 10 * 32, dtype=torch.float32
    ).view(batch_size, 3, 10, 32)

    class _Engine:
        def rollout_step(self, request):
            self.request = request
            return SimpleNamespace(
                actions=model_actions,
                chains=chains,
                transition_logprobs=transition_logprobs,
                raw_values=torch.tensor([[1.0], [2.0]]),
            )

    worker = object.__new__(PhyAIRolloutWorker)
    worker._accelerator_type = "cpu"
    worker._timer_metrics = {}
    worker._engine = _Engine()
    _configure_preprocessing(worker)
    worker._model_action_horizon = 10
    worker._model_action_dim = 32
    worker._num_steps = 3
    worker.only_eval = False
    worker.model_cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "action_dim": 7,
            "add_value_head": True,
            "openpi": {
                "model_action_dim": 32,
                "noise_method": "flow_sde",
                "noise_level": 0.5,
                "joint_logprob": False,
                "ignore_last": True,
            },
        }
    )

    def select_denoise_step(low, high):
        assert (low, high) == (0, 1)
        return 1

    monkeypatch.setattr(phyai_rollout_worker.random, "randint", select_denoise_step)

    actions, result = inspect.unwrap(PhyAIRolloutWorker.predict)(
        worker, env_obs, mode="train"
    )

    normalized_state = env_obs["states"] / 10
    expected_actions = model_actions[:, :5, :7] + normalized_state[:, :1, None]

    request = worker._engine.request
    assert isinstance(request, PI05RolloutRequest)
    assert request.pixel_values.dtype is torch.float32
    assert request.noise is None
    assert request.step_noise is None
    torch.testing.assert_close(
        request.sigmas, torch.tensor([[0.0, math.sqrt(0.5), 0.0]]).expand(2, -1)
    )
    assert actions.shape == (batch_size, 5, 7)
    assert torch.equal(actions, expected_actions)
    assert torch.equal(result["prev_logprobs"], transition_logprobs[:, 1, :5, :7])
    assert torch.equal(result["prev_values"], torch.tensor([[1.0], [2.0]]))
    forward_inputs = result["forward_inputs"]
    assert torch.equal(forward_inputs["chains"], chains)
    assert forward_inputs["denoise_inds"].eq(1).all()
    assert torch.equal(
        forward_inputs["obs_state"],
        torch.nn.functional.pad(normalized_state, (0, 24)),
    )
    assert torch.equal(
        forward_inputs["tokenized_prompt"], torch.arange(8).expand(2, -1)
    )
    assert torch.equal(
        forward_inputs["tokenized_prompt_mask"],
        torch.arange(8)[None] < torch.tensor([[4], [5]]),
    )
    assert torch.equal(forward_inputs["action"], actions.reshape(batch_size, -1))
    assert torch.equal(
        forward_inputs["model_action"], model_actions.reshape(batch_size, -1)
    )
    for key, image in zip(
        IMAGE_KEYS,
        (
            env_obs["main_images"],
            env_obs["wrist_images"],
            torch.zeros_like(env_obs["main_images"]),
        ),
        strict=True,
    ):
        assert torch.equal(
            forward_inputs[f"obs_image__{key}"], image.to(torch.float32) / 255 * 2 - 1
        )
        assert torch.equal(
            forward_inputs[f"obs_image_mask__{key}"],
            torch.full((batch_size,), key != IMAGE_KEYS[2]),
        )


def test_rollout_plan_uses_actor_sampling_defaults():
    worker = object.__new__(PhyAIRolloutWorker)
    worker._engine_device = torch.device("cpu")
    worker._model_action_horizon = 10
    worker._model_action_dim = 32
    worker._num_steps = 3
    worker.model_cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "action_dim": 7,
            "add_value_head": False,
            "openpi": {},
        }
    )
    request = SimpleNamespace(
        pixel_values=torch.empty(2, 2, 3, 4, 4),
        input_ids=torch.zeros(2, 200, dtype=torch.int64),
        lang_lens=torch.ones(2, dtype=torch.int64),
    )

    plan = worker._build_rollout_plan(request)

    assert plan.request.sigmas.eq(0).all()


def test_joint_initial_logprob_uses_realized_chain():
    request = PI05RolloutRequest(
        pixel_values=torch.empty(1, 2, 3, 4, 4),
        input_ids=torch.empty(1, 2, dtype=torch.int64),
        lang_lens=torch.ones(1, dtype=torch.int64),
        sigmas=torch.zeros(1, 2),
        compute_values=False,
    )
    plan = phyai_rollout_worker._PI05SamplingPlan(
        request=request,
        denoise_inds=torch.tensor([[0, 1]]),
        action_chunk=1,
        action_dim=1,
        joint_logprob=True,
    )
    rollout_result = SimpleNamespace(
        chains=torch.tensor([[[[2.0]], [[0.0]], [[0.0]]]]),
        transition_logprobs=torch.zeros(1, 2, 1, 1),
        raw_values=None,
    )

    actual, values = PhyAIRolloutWorker._reduce_rollout_outputs(plan, rollout_result)

    expected = (-0.5 * math.log(2 * math.pi) - 2.0) / 3
    torch.testing.assert_close(actual, torch.full((1, 1, 1), expected))
    assert values is None
