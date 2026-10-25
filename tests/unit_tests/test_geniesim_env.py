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

"""GenieSim environment contract without a running simulator."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.geniesim import geniesim_env
from rlinf.envs.sim.geniesim.adapter import GenieSimAdapter
from rlinf.envs.sim.geniesim.geniesim_env import GenieSimEnv


@pytest.mark.parametrize(
    "recipe,devices",
    [
        ("examples/embodiment/config/geniesim_pick_block_color_ppo_openpi_pi05", 8),
        ("examples/embodiment/config/geniesim_pick_block_color_grpo_openpi_pi05", 8),
        ("examples/embodiment/config/geniesim_multitask4_ppo_openpi_pi05", 8),
        ("examples/embodiment/config/geniesim_multitask6_ppo_openpi_pi05", 8),
        ("examples/embodiment/config/geniesim_multitask10_ppo_openpi_pi05", 8),
        ("evaluations/geniesim/geniesim_pick_block_color_eval", 1),
        ("evaluations/geniesim/geniesim_pick_block_color_eval", 8),
        ("tests/e2e_tests/embodied/geniesim_ppo_openpi_pi05", 8),
    ],
)
def test_example_endpoints_match_placement(recipe, devices, monkeypatch):
    from test_placement import create_fake_cluster

    from rlinf.config import validate_only_eval_rollout_model
    from rlinf.utils.placement import HybridComponentPlacement

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setenv("REPO_PATH", str(root))
    monkeypatch.setenv("GENIESIM_SERVER_ADDR", "tcp://sim.example:5601")
    monkeypatch.setenv("GENIESIM_HOST_0", "sim.example")
    monkeypatch.setenv("GENIESIM_EVAL_HOST", "eval.example")
    monkeypatch.setenv("GENIESIM_CHECKPOINT_PATH", "/tmp/geniesim-checkpoint")
    path = root / recipe
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = compose(config_name=path.name)
    OmegaConf.resolve(cfg)
    placement = HybridComponentPlacement(cfg, create_fake_cluster(1, devices))
    slots = placement.get_world_size("env") * cfg.rollout.pipeline_stage_num
    endpoints = []
    for mode in ("train", "eval"):
        if mode not in cfg.env:
            continue
        env_cfg = cfg.env[mode]
        assert env_cfg.total_num_envs % slots == 0
        assert (env_cfg.total_num_envs // slots) % env_cfg.group_size == 0
        override = OmegaConf.to_container(env_cfg.override_cfg, resolve=True)
        endpoints.extend(
            geniesim_env.resolve_geniesim_address(override, rank=i, world_size=slots)
            for i in range(slots)
        )
    assert len(endpoints) == len(set(endpoints))
    if "train" not in cfg.env:
        assert slots == 1
        validate_only_eval_rollout_model(cfg.rollout.model)
    else:
        actor_world_size = placement.get_world_size("actor")
        assert (
            cfg.actor.global_batch_size
            % (cfg.actor.micro_batch_size * actor_world_size)
            == 0
        )
        rollout_samples = (
            cfg.env.train.total_num_envs
            * cfg.env.train.rollout_epoch
            * (
                cfg.env.train.max_steps_per_rollout_epoch
                // cfg.actor.model.num_action_chunks
            )
        )
        assert rollout_samples % cfg.actor.global_batch_size == 0


@pytest.mark.parametrize("dimension", ["noise", "occlusion", "drop_frame"])
def test_sensor_randomization_is_rejected(dimension):
    with pytest.raises(ValueError, match="sensor"):
        geniesim_env.resolve_geniesim_randomization(
            {"randomization": {"sensor": {dimension: {"enable": True}}}}
        )


class _SimStateNonFinite(RuntimeError):
    pass


class _Client:
    def __init__(self):
        self.closed = False
        self.metadata_requests = []

    def set_backends(self, backend_cfg):
        self.backend_cfg = backend_cfg

    def task_metadata(self, *, sub_task_name):
        self.metadata_requests.append(sub_task_name)
        return {
            "sub_tasks": {
                sub_task_name: {
                    "task_names": [{"task_name": "table_task"}],
                    "instance_ids": [0, 1, 2, 3],
                    "switchable_instance_ids": [0, 1, 2, 3],
                    "instructions_by_instance": {
                        str(index): [{"id": 10 + index, "text": f"pick block {index}"}]
                        for index in range(4)
                    },
                }
            }
        }

    def close(self):
        self.closed = True


class _VecEnv:
    def __init__(self, client):
        self.client = client
        self.action_dim = 16
        self.closed = False
        self.reset_calls = []
        self.step_actions = []
        self.fail_next_step = False
        self.reset_failures = 0
        self.success_at_step = None
        self.steps = 0

    def init(self, task_cfg, config=None):
        self.num_envs = task_cfg["num_envs"]
        self.task_cfg = task_cfg
        self.progress = np.zeros(self.num_envs, dtype=np.float32)

    def _observation(self):
        frame = np.zeros((self.num_envs, 8, 8, 3), dtype=np.uint8)
        return {
            "states": np.zeros((self.num_envs, 24), dtype=np.float32),
            "main_images": frame,
            "wrist_images": frame,
            "wrist_images_right": frame,
            "task_descriptions": ["pick a block"] * self.num_envs,
        }

    def reset(
        self,
        *,
        env_ids,
        seeds,
        instance_ids,
        instruction_ids,
        block,
    ):
        self.reset_calls.append(
            {
                "env_ids": list(env_ids),
                "seeds": np.asarray(seeds).copy(),
                "instance_ids": np.asarray(instance_ids).copy(),
                "instruction_ids": np.asarray(instruction_ids).copy(),
            }
        )
        if self.reset_failures:
            self.reset_failures -= 1
            raise _SimStateNonFinite("non-finite joint state: env [1]")
        self.progress[env_ids] = 0
        return self._observation(), {}

    def step(self, actions, *, auto_reset, need_images, return_per_step_obs):
        """Mirrors GenieSimVecEnv.step: an ``(N, chunk, A)`` chunk in,
        ``(N, chunk)`` scalars out, with per-substep success in the info."""
        if self.fail_next_step:
            self.fail_next_step = False
            raise _SimStateNonFinite("non-finite joint state: env [1]")
        actions = np.asarray(actions)
        self.step_actions.append(actions.copy())
        chunk = actions.shape[1]
        progress = np.stack(
            [self.progress + 0.5 * (i + 1) for i in range(chunk)], axis=1
        )
        self.progress[:] = progress[:, -1]
        shape = (self.num_envs, chunk)
        successes = np.zeros(shape, dtype=bool)
        if self.success_at_step is not None:
            successes[0] = self.steps + np.arange(1, chunk + 1) == self.success_at_step
        self.steps += chunk
        return (
            self._observation(),
            progress.astype(np.float32),
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=bool),
            {
                "is_success": successes[:, -1],
                "chunk_success": successes,
            },
        )

    def close(self):
        self.closed = True


def _config(*, addresses=None, recovery=False, chunk_end_only=False, auto_reset=False):
    override = {
        "comm_backend": "zmq",
        "task_name": "table_task",
        "sub_task_name": "pick_block_color",
        "reset_blocking": True,
        "geniesim_env_seed": 7,
        "venv_device": "cpu",
        "nonfinite_state_recovery": recovery,
        "images_at_chunk_end_only": chunk_end_only,
    }
    if addresses is None:
        override["address"] = "tcp://localhost:5601"
    else:
        override["addresses"] = addresses
    return OmegaConf.create(
        {
            "seed": 7,
            "group_size": 1,
            "auto_reset": auto_reset,
            "max_episode_steps": 2,
            "main_image_key": "main",
            "extra_image_keys": ["wrist_left", "wrist_right"],
            "override_cfg": override,
        }
    )


@pytest.fixture
def simulator(monkeypatch):
    clients = []
    vec_envs = []

    def make_client(*args, **kwargs):
        client = _Client()
        clients.append(client)
        return client

    def make_vec_env(client, **kwargs):
        vec_env = _VecEnv(client)
        vec_envs.append(vec_env)
        return vec_env

    monkeypatch.setattr(geniesim_env, "_make_geniesim_client", make_client)
    monkeypatch.setattr(geniesim_env, "_make_geniesim_venv", make_vec_env)
    return clients, vec_envs


def test_registration_and_observation_action_contract():
    assert get_env_cls("geniesim") is GenieSimEnv
    raw = _VecEnv(_Client())
    raw.init({"num_envs": 2})
    obs = GenieSimAdapter().wrap_batch(raw._observation())
    assert tuple(obs["main_images"].shape) == (2, 8, 8, 3)
    assert tuple(obs["extra_view_images"].shape) == (2, 2, 8, 8, 3)
    assert tuple(obs["states"].shape) == (2, 24)
    assert obs["task_descriptions"] == ["pick a block", "pick a block"]

    actions = np.zeros((2, 3, 16), dtype=np.float64)
    assert prepare_actions(actions, "geniesim", "openpi", 3, 16) is actions


def test_reset_and_action_chunk_match_env_worker_contract(simulator):
    clients, vec_envs = simulator
    env = GenieSimEnv(_config(), num_envs=2)
    obs, _ = env.reset()
    assert tuple(obs["states"].shape) == (2, 24)
    assert vec_envs[0].reset_calls[0]["env_ids"] == [0, 1]

    observations, rewards, terminated, truncated, infos = env.chunk_step(
        np.zeros((2, 2, 16), dtype=np.float64)
    )
    assert len(observations) == len(infos) == 2
    assert tuple(rewards.shape) == tuple(terminated.shape) == (2, 2)
    assert rewards.tolist() == [[0.5, 0.5], [0.5, 0.5]]
    assert not terminated.any()
    assert truncated[:, -1].tolist() == [True, True]
    assert infos[-1]["episode"]["return"].tolist() == [1.0, 1.0]
    assert all(call.shape == (2, 1, 16) for call in vec_envs[0].step_actions)
    assert all(call.dtype == np.float32 for call in vec_envs[0].step_actions)
    with pytest.raises(ValueError, match="action chunks must have shape"):
        env.chunk_step(np.zeros((2, 2, 14), dtype=np.float32))

    env.close()
    assert clients[0].closed and vec_envs[0].closed


def test_rank_endpoints_and_work_items_are_distinct(simulator):
    addresses = ["tcp://sim:5601", "tcp://sim:5602"]
    envs = [
        GenieSimEnv(
            _config(addresses=addresses),
            num_envs=1,
            seed_offset=rank,
            total_num_processes=2,
        )
        for rank in range(2)
    ]
    assert [env._address for env in envs] == addresses
    for env in envs:
        env.reset()
    assert envs[0]._current_work_items()[0] != envs[1]._current_work_items()[0]
    for env in envs:
        env.close()

    with pytest.raises(ValueError, match="unique endpoints"):
        GenieSimEnv(
            _config(addresses=[addresses[0], addresses[0]]),
            num_envs=1,
            total_num_processes=2,
        )


def test_chunk_goes_to_the_simulator_in_one_step_call(simulator):
    _, vec_envs = simulator
    env = GenieSimEnv(_config(chunk_end_only=True), num_envs=2)
    env.reset()
    observations, rewards, terminated, truncated, infos = env.chunk_step(
        np.zeros((2, 3, 16), dtype=np.float32)
    )
    # One call carrying the whole chunk, not one call per substep.
    assert len(vec_envs[0].step_actions) == 1
    assert vec_envs[0].step_actions[0].shape == (2, 3, 16)
    assert len(observations) == len(infos) == 3
    assert tuple(rewards.shape) == tuple(terminated.shape) == (2, 3)
    # Progress rises 0.5 per substep, so each substep pays that delta until the
    # episode ends. ``max_episode_steps`` is 2 here, so the third substep falls
    # outside the episode and pays nothing.
    assert rewards.tolist() == [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    assert not terminated.any()
    # Truncation is reported where it happened and latched into the last column,
    # which is the only one env_worker reads.
    assert truncated[:, 1].tolist() == [True, True]
    assert truncated[:, -1].tolist() == [True, True]
    env.close()


@pytest.mark.parametrize("recovery", [False, True])
def test_nonfinite_step_does_not_return_cached_observations(
    simulator, monkeypatch, recovery
):
    monkeypatch.setattr(
        geniesim_env, "_nonfinite_state_error_types", lambda: (_SimStateNonFinite,)
    )
    _, vec_envs = simulator
    env = GenieSimEnv(_config(recovery=recovery, auto_reset=True), num_envs=2)
    env.reset()
    vec_envs[0].fail_next_step = True
    with pytest.raises(_SimStateNonFinite, match="non-finite joint state"):
        env.step(np.zeros((2, 16), dtype=np.float32))
    assert len(vec_envs[0].reset_calls) == 1
    env.close()


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("failures", [1, 3])
def test_nonfinite_reset_retries_are_bounded(
    simulator, monkeypatch, recovery, failures
):
    monkeypatch.setattr(
        geniesim_env, "_nonfinite_state_error_types", lambda: (_SimStateNonFinite,)
    )
    _, vec_envs = simulator
    env = GenieSimEnv(_config(recovery=recovery), num_envs=2)
    vec_envs[0].reset_failures = failures
    if not recovery:
        with pytest.raises(_SimStateNonFinite, match="non-finite joint state"):
            env.reset()
        assert len(vec_envs[0].reset_calls) == 1
    elif failures == 3:
        with pytest.raises(RuntimeError, match="all 3 attempts"):
            env.reset()
        assert len(vec_envs[0].reset_calls) == 3
    else:
        obs, _ = env.reset()
        assert obs["task_descriptions"] == ["pick a block"] * 2
        assert len(vec_envs[0].reset_calls) == 2
        for reset in vec_envs[0].reset_calls:
            assert reset["instance_ids"].tolist() == [0, 1]
    env.close()


@pytest.mark.parametrize("batched", [False, True])
def test_partial_done_preserves_other_episode_metrics(simulator, batched):
    _, vec_envs = simulator
    cfg = _config(chunk_end_only=batched)
    cfg.max_episode_steps = 400
    env = GenieSimEnv(cfg, num_envs=2)
    env.reset()
    vec_envs[0].success_at_step = 1
    actions = np.zeros((2, 10, 16), dtype=np.float32)
    _, rewards, terminated, _, infos = env.chunk_step(actions)
    assert rewards.sum(dim=1).tolist() == [0.5, 5.0]
    assert terminated[:, -1].tolist() == [True, False]
    assert infos[-1]["episode"]["return"].tolist() == [0.5, 5.0]
    assert infos[-1]["episode"]["length"].tolist() == [1.0, 10.0]
    assert infos[-1]["episode"]["success_once"].tolist() == [1.0, 0.0]
    _, rewards, _, _, infos = env.chunk_step(actions)
    assert rewards.sum(dim=1).tolist() == [0.0, 5.0]
    assert infos[-1]["episode"]["return"].tolist() == [0.5, 10.0]
    assert infos[-1]["episode"]["length"].tolist() == [1.0, 20.0]
    env.close()


def test_task_rotation_reuses_metadata_and_resumes_group_cursors(simulator):
    clients, vec_envs = simulator
    cfg = _config()
    cfg.group_size = 2
    cfg.override_cfg.sub_task_names = ["pick_block_color", "place_block"]
    cfg.override_cfg.sub_task_rotate_every = 1
    env = GenieSimEnv(cfg, num_envs=2)
    for expected_task, expected_instance in [
        ("pick_block_color", 0),
        ("place_block", 0),
        ("pick_block_color", 1),
    ]:
        env.reset()
        reset = vec_envs[0].reset_calls[-1]
        assert env._sub_task_name == expected_task
        assert reset["instance_ids"].tolist() == [expected_instance] * 2
        assert reset["instruction_ids"][0] == reset["instruction_ids"][1]
        assert reset["seeds"][0] == reset["seeds"][1]
    assert clients[0].metadata_requests == ["pick_block_color", "place_block"]
    env.close()


def _worker_for_env(env, cfg):
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker._accelerator_type = EnvWorker.accelerator_type
    worker._timer_metrics = {}
    cfg.env_type = "geniesim"
    cfg.ignore_terminations = False
    worker.cfg = OmegaConf.create({"env": {"train": cfg, "eval": cfg}})
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "num_action_chunks": 2,
            "action_dim": 16,
        }
    )
    worker.env_list = worker.eval_env_list = [env]
    worker.eval_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker.enable_online_lerobot = False
    worker.use_external_reward_model = False
    return worker


@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_worker_propagates_geniesim_divergence(
    simulator, monkeypatch, auto_reset, batched, recovery, mode
):
    monkeypatch.setattr(
        geniesim_env, "_nonfinite_state_error_types", lambda: (_SimStateNonFinite,)
    )
    _, vec_envs = simulator
    cfg = _config(recovery=recovery, auto_reset=auto_reset, chunk_end_only=batched)
    env = GenieSimEnv(cfg, num_envs=2)
    env.reset()
    worker = _worker_for_env(env, cfg)
    vec_envs[0].fail_next_step = True
    step = worker.env_interact_step if mode == "train" else worker.env_evaluate_step
    with pytest.raises(_SimStateNonFinite, match="non-finite joint state"):
        step(torch.zeros(2, 2, 16), 0)
    assert len(vec_envs[0].reset_calls) == 1
    env.close()


@pytest.fixture
def transport(monkeypatch):
    client_module = pytest.importorskip("rlinf_geniesim.envs.vec_env")
    from rlinf_geniesim.comm.base import ResultBuffer, SimDims

    layout = client_module.GenieSimVecEnvConfig()
    joint_names = (
        layout.left_arm_joints
        + layout.right_arm_joints
        + layout.gripper_joints
        + layout.waist_joints
        + layout.head_joints
    )

    class Transport(_Client):
        """Scripted simulator boundary; the real VecEnv owns client-side behavior."""

        def __init__(self):
            super().__init__()
            self.steps = np.zeros(2, dtype=np.int32)
            self.success_at_step = None
            self.termination_at_step = None
            self.truncation_at_step = None
            self.chunk_calls = 0
            self.corrupt_next = False
            self.sub_task_name = "pick_block_color"

        def handshake(self):
            return SimDims(
                num_envs=2,
                action_dim=16,
                state_dim=48,
                num_joints=24,
                num_bodies=0,
                num_objects=0,
                num_cameras=3,
                image_h=8,
                image_w=8,
            ), {"joint_names": joint_names}

        def backends_declared(self):
            return False

        def init(self, payload, timeout_s):
            self.sub_task_name = payload["task"]["sub_task_name"]
            return self.handshake()

        def task_metadata(self, sub_task_name=None):
            return super().task_metadata(
                sub_task_name=sub_task_name or self.sub_task_name
            )

        def _result(self, need_images=True):
            success = np.zeros(2, dtype=bool)
            if self.success_at_step is not None:
                success[0] = self.steps[0] == self.success_at_step
            terminations = success.copy()
            truncations = np.zeros(2, dtype=bool)
            if self.termination_at_step is not None:
                terminations[0] |= self.steps[0] == self.termination_at_step
            if self.truncation_at_step is not None:
                truncations[0] = self.steps[0] == self.truncation_at_step
            joints = np.zeros((2, 24, 2), dtype=np.float32)
            if self.corrupt_next:
                joints[1, 0, 0] = np.nan
                self.corrupt_next = False
            return ResultBuffer(
                states=np.zeros((2, 48), dtype=np.float32),
                joint_states=joints,
                body_poses=np.zeros((2, 0, 7)),
                object_poses=np.zeros((2, 0, 7)),
                images=np.zeros((2, 3, 8, 8, 3), dtype=np.uint8)
                if need_images
                else None,
                rewards=self.steps.astype(np.float32) * 0.5,
                terminations=terminations,
                truncations=truncations,
                sim_time=self.steps.astype(np.float64),
                success=success,
            )

        def reset(self, mask, seeds=None, instances=None, instructions=None):
            self.steps[np.asarray(mask, dtype=bool)] = 0
            return self._result()

        def step(self, actions, need_images=True):
            self.steps += 1
            return self._result(need_images)

        def step_chunk(self, actions, need_images=True):
            self.chunk_calls += 1
            count = len(actions)
            elapsed = self.steps[None, :] + np.arange(1, count + 1)[:, None]
            success = np.zeros((count, 2), dtype=bool)
            if self.success_at_step is not None:
                success[:, 0] = elapsed[:, 0] == self.success_at_step
            terminations = success.copy()
            truncations = np.zeros_like(success)
            if self.termination_at_step is not None:
                terminations[:, 0] |= elapsed[:, 0] == self.termination_at_step
            if self.truncation_at_step is not None:
                truncations[:, 0] = elapsed[:, 0] == self.truncation_at_step
            self.steps += count
            result = self._result(need_images)
            scalars = np.stack(
                [
                    elapsed * 0.5,
                    terminations,
                    truncations,
                    success,
                ]
            ).astype(np.float32)
            return SimpleNamespace(**vars(result), chunk_scalars=scalars)

    client = Transport()
    monkeypatch.setattr(geniesim_env, "_make_geniesim_client", lambda *a, **k: client)
    return client


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("boundary", ["success", "termination", "truncation"])
def test_real_client_partial_done_metrics(transport, batched, auto_reset, boundary):
    cfg = _config(chunk_end_only=batched, auto_reset=auto_reset)
    cfg.max_episode_steps = 400
    env = GenieSimEnv(cfg, num_envs=2)
    obs, _ = env.reset()
    assert obs["task_descriptions"] == ["pick block 0", "pick block 1"]
    setattr(transport, f"{boundary}_at_step", 1)
    observations, rewards, terminated, truncated, infos = env.chunk_step(
        np.zeros((2, 10, 16))
    )
    assert transport.chunk_calls == int(batched)
    assert rewards.sum(dim=1).tolist() == [0.5, 5.0]
    assert terminated[:, -1].tolist() == [boundary != "truncation", False]
    assert truncated[:, -1].tolist() == [boundary == "truncation", False]
    info = infos[-1].get("final_info", infos[-1])
    assert info["episode"]["return"].tolist() == [0.5, 5.0]
    assert info["episode"]["length"].tolist() == [1.0, 10.0]
    assert info["episode"]["success_once"].tolist() == [
        float(boundary == "success"),
        0.0,
    ]
    assert info["episode"]["success_at_end"].tolist() == [boundary == "success", False]
    assert info["episode"]["success_at_end"].dtype == torch.bool
    if auto_reset:
        assert observations[-1]["task_descriptions"] == ["pick block 2", "pick block 1"]
        assert infos[-1]["final_observation"]["task_descriptions"] == [
            "pick block 0",
            "pick block 1",
        ]
    env.close()


def test_real_client_respects_episode_limit(transport):
    cfg = _config(chunk_end_only=True)
    cfg.max_episode_steps = 400
    env = GenieSimEnv(cfg, num_envs=2)
    env.reset()
    for index in range(40):
        _, _, _, truncated, infos = env.chunk_step(np.zeros((2, 10, 16)))
        assert truncated[:, -1].tolist() == [index == 39] * 2
    assert infos[-1]["episode"]["length"].tolist() == [400.0, 400.0]
    assert infos[-1]["episode"]["return"].tolist() == [200.0, 200.0]
    env.close()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_real_client_propagates_nonfinite_state(transport, batched, mode):
    from rlinf_geniesim.envs.vec_env import SimStateNonFiniteError

    cfg = _config(recovery=True, chunk_end_only=batched, auto_reset=True)
    env = GenieSimEnv(cfg, num_envs=2)
    env.reset()
    transport.corrupt_next = True
    worker = _worker_for_env(env, cfg)
    step = worker.env_interact_step if mode == "train" else worker.env_evaluate_step
    with pytest.raises(SimStateNonFiniteError):
        step(torch.zeros(2, 2, 16), 0)
    env.close()
