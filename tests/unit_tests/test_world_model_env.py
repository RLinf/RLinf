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

"""World-model env episode handling and the BWM backend."""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.env.env_worker as env_worker_module
from rlinf.envs.sim.world_model import registry
from rlinf.envs.sim.world_model.backend.remote import RemoteWorldModelBackend
from rlinf.workers.env.env_worker import EnvWorker, push_world_model_server_urls


class _ConstantRewardModel(torch.nn.Module):
    def __init__(self, reward: float):
        super().__init__()
        self.reward = reward

    def predict_rew(self, frames):
        return torch.full((frames.shape[0],), self.reward, device=frames.device)


class _BlankFrameBackend:
    """World-model backend that draws blank frames; the reward model scores a constant."""

    supports_kir = False
    condition_frame_length = 2
    image_size = (8, 8)

    def __init__(self, cfg, device):
        self.chunk = cfg.chunk
        self.last_seeds = None

    @staticmethod
    def load_reward_model(cfg):
        return _ConstantRewardModel(cfg.unit_test_reward)

    def reward_instructions(self, env):
        return None

    def open_session(self, env_ids, init_frames, init_actions, seeds):
        self.last_seeds = list(seeds)

    def generate(self, env_ids, actions):
        return torch.zeros(len(env_ids), 3, self.chunk, *self.image_size)

    def close_session(self, env_ids):
        pass

    def offload(self):
        pass

    def onload(self):
        pass


def _blank_world_model_env(tmp_path, monkeypatch, *, reward, **overrides):
    from rlinf.envs.sim.world_model.env import WorldModelEnv

    monkeypatch.setitem(registry._REGISTRY, "unit_test_blank", _BlankFrameBackend)
    frame = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "task": "adjust the bottle",
        "delta_action": np.zeros(14, dtype=np.float32),
    }
    for episode in range(2):
        np.save(tmp_path / f"episode_{episode}.npy", np.array([frame, frame]))
    cfg = OmegaConf.create(
        {
            "backend": "unit_test_blank",
            "wm_env_type": "robotwin",
            "seed": 0,
            "auto_reset": True,
            "ignore_terminations": True,
            "max_episode_steps": 4,
            "chunk": 2,
            "action_dim": 14,
            "reward_coef": 1.0,
            "unit_test_reward": reward,
            "initial_image_path": str(tmp_path),
            "enable_kir": False,
            "use_fixed_reset_state_ids": True,
            "group_size": 1,
            "video_cfg": {"save_video": False},
            **overrides,
        }
    )
    env = WorldModelEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1)
    env.reset()
    return env


class _ScriptedRewardModel(torch.nn.Module):
    """Scores every frame of the n-th scored chunk with ``scores[n]``."""

    def __init__(self, scores):
        super().__init__()
        self.scores = list(scores)
        self.calls = 0

    def predict_rew(self, frames):
        self.calls += 1
        return torch.full((frames.shape[0],), self.scores.pop(0), device=frames.device)


class _ScriptedRewardBackend(_BlankFrameBackend):
    @staticmethod
    def load_reward_model(cfg):
        return _ScriptedRewardModel(cfg.unit_test_scores)


def test_world_model_env_episode_reward_pays_the_window_peak_at_the_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        registry._REGISTRY, "unit_test_scripted", _ScriptedRewardBackend
    )
    # Episodes are two chunks of two frames; frames 1-2 precede reward_first_frame,
    # so the first chunk of each episode is not scored at all.
    env = _blank_world_model_env(
        tmp_path,
        monkeypatch,
        reward=0.0,
        backend="unit_test_scripted",
        unit_test_scores=[1.0, 0.5],
        reward_mode="episode_window_max",
        reward_first_frame=3,
        terminate_at_max_episode_steps=True,
    )
    actions = torch.zeros(2, 2, 14)

    rewards = [env.chunk_step(actions)[1] for _ in range(4)]

    assert env.reward_model.calls == 2
    assert not rewards[0].any()
    assert rewards[1][:, -1].eq(1).all() and not rewards[1][:, :-1].any()
    # The previous episode's peak does not carry over.
    assert not rewards[2].any() and not rewards[3].any()


def test_world_model_env_echoes_the_last_command_as_state(tmp_path, monkeypatch):
    rest = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
    env = _blank_world_model_env(
        tmp_path,
        monkeypatch,
        reward=0.0,
        seed=11,
        state_from_last_action=True,
        rest_action=rest,
    )
    assert env.backend.last_seeds == [11, 11]
    rest_states = torch.tensor([rest, rest])
    actions = torch.arange(2 * 2 * 14, dtype=torch.float32).reshape(2, 2, 14)

    obs, _ = env.reset()
    assert torch.equal(obs["states"].cpu(), rest_states)

    obs_list, *_ = env.chunk_step(actions)
    assert torch.equal(obs_list[0]["states"].cpu(), actions[:, -1])

    # The episode limit restarts both slots at the rest pose.
    obs_list, _, _, _, infos = env.chunk_step(actions + 1)
    assert torch.equal(obs_list[0]["states"].cpu(), rest_states)
    assert torch.equal(
        infos[0]["final_observation"]["states"].cpu(), actions[:, -1] + 1
    )


class _RecordingBwmPipeline:
    torch_dtype = torch.float32

    def __init__(self, image_size):
        self.image_size = image_size
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full((1, 3, 61, *self.image_size), float(len(self.calls)))


class _HalfActionBridge:
    def __init__(self, **kwargs):
        pass

    def __call__(self, joint14):
        return np.asarray(joint14, dtype=np.float32) * 0.5


def test_bwm_history_condition_noises_all_but_the_reset_frame():
    from rlinf.envs.sim.world_model.backend import bwm

    first = torch.ones(1, 2, 3, 2, 2)

    def condition(noise, seed=5):
        latents = torch.zeros(1, 2, 5, 2, 2)
        inputs = {"first_frame_latents": first, "latents": latents, "seed": seed}
        cond, history_t = bwm._history_condition_latents(
            None, inputs, use_history_condition_noise_in_inference=noise
        )
        assert history_t == 3
        assert torch.equal(latents[:, :, :3], cond)
        assert latents[:, :, 3:].eq(0).all()
        return cond

    assert torch.equal(condition(noise=False), first)
    noised = condition(noise=True)
    assert torch.equal(noised[:, :, 0], first[:, :, 0])
    assert not torch.equal(noised[:, :, 1:], first[:, :, 1:])
    assert torch.equal(noised, condition(noise=True))
    assert not torch.equal(noised, condition(noise=True, seed=6))


def test_bwm_backend_rolls_its_condition_window(monkeypatch):
    from rlinf.envs.sim.world_model.backend import bwm

    pipe = _RecordingBwmPipeline((4, 6))
    monkeypatch.setattr(bwm, "RobotwinActionBridge", _HalfActionBridge)
    monkeypatch.setattr(bwm.BwmBackend, "_build_pipeline", lambda self: pipe)
    cfg = OmegaConf.create(
        {
            "chunk": 50,
            "condition_frame_length": 9,
            "image_size": [4, 6],
            "num_inference_steps": 2,
            "action_dim": 14,
            "rest_action": [0.0] * 14,
            "action_bridge": {
                "urdf_path": "",
                "srdf_path": "",
                "stat_path": "",
                "robot_pose": [0.0] * 7,
            },
        }
    )
    backend = bwm.BwmBackend(cfg, torch.device("cpu"))
    reset_window = [torch.full((3, 1, 4, 6), -1.0)] * 9
    backend.open_session([0], [reset_window], torch.zeros(1, 9, 14), seeds=[7])

    frames = backend.generate([0], torch.ones(1, 50, 14))
    backend.generate([0], torch.full((1, 50, 14), 2.0))

    assert frames.shape == (1, 3, 50, 4, 6)
    first, second = pipe.calls
    assert (first["seed"], second["seed"]) == (7, 8)
    # Bridged history (rest pose), the chunk, then its last command held twice.
    assert first["action"][0, :9].eq(0.0).all()
    assert first["action"][0, 9:].eq(0.5).all()
    # The reset frame stays in slot 0; the last 8 generated frames and poses roll in.
    assert second["input_video"][0, :, 0].eq(-1.0).all()
    assert second["input_video"][0, :, 1:].eq(1.0).all()
    assert second["action"][0, 0].eq(0.0).all()
    assert second["action"][0, 1:9].eq(0.5).all()
    assert second["action"][0, 9:].eq(1.0).all()


# mp4 is lossy, so uniform gray levels are compared within this tolerance.
_GRAY_ATOL = 4.0


def _remote_cfg(chunk=8, max_retries=1, adapter="videos_sync"):
    return OmegaConf.create(
        {
            "chunk": chunk,
            "condition_frame_length": 5,
            "num_frames": 5 + chunk,
            "image_size": [64, 64],
            "retain_action": True,
            "remote": {
                "adapter": adapter,
                "model": "bwm",
                "prompt": "",
                "num_inference_steps": 4,
                "fps": 24,
                "request_timeout_s": 30,
                "warmup_timeout_s": 30,
                "max_retries": max_retries,
                "retry_backoff_s": 0.0,
            },
        }
    )


def _gray_mp4(levels):
    import io

    import imageio.v2 as imageio

    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=24, quality=8)
    for level in levels:
        writer.append_data(np.full((64, 64, 3), level, dtype=np.uint8))
    writer.close()
    return buf.getvalue()


def _gray_levels(mp4: bytes) -> np.ndarray:
    import io

    import imageio.v3 as iio

    frames = iio.imread(io.BytesIO(mp4), extension=".mp4", plugin="FFMPEG")
    return np.asarray(frames, dtype=np.float32).mean(axis=(1, 2, 3))


_SERVED = [15 * t + 10 for t in range(13)]


def _open_sample_session(backend):
    window = [torch.full((3, 1, 64, 64), 40 * k / 255 * 2 - 1) for k in range(5)]
    history = torch.stack([torch.full((7,), 0.1 * i) for i in range(5)])
    backend.open_session([0], [window], history[None], seeds=[7])


def _chunk(length):
    return torch.stack([torch.full((7,), 1.0 + j) for j in range(length)])[None]


def test_remote_backend_sends_the_sample_request_and_rolls_forward():
    import json

    from _fake_http_server import FakeHTTPServer, parse_multipart

    with FakeHTTPServer([(200, "video/mp4", _gray_mp4(_SERVED))]) as server:
        backend = RemoteWorldModelBackend(
            _remote_cfg(), torch.device("cpu"), server.url
        )
        _open_sample_session(backend)
        frames = backend.generate([0], _chunk(8))

    (request,) = server.requests
    assert request["path"] == "/v1/videos/sync"
    fields, files = parse_multipart(request["content_type"], request["body"])
    assert {k: fields[k] for k in ("num_frames", "seed", "size", "fps", "model")} == {
        "num_frames": "13",
        "seed": "7",
        "size": "64x64",
        "fps": "24",
        "model": "bwm",
    }
    assert fields["guidance_scale"] == "1.0"
    action = np.array(json.loads(fields["extra_params"])["action"])
    assert action.shape == (13, 7)
    np.testing.assert_allclose(action[:5, 0], [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(action[5:, 0], np.arange(1, 9))
    filename, content, content_type = files["input_reference"]
    assert (filename, content_type) == ("history.mp4", "video/mp4")
    np.testing.assert_allclose(
        _gray_levels(content), [0, 40, 80, 120, 160], atol=_GRAY_ATOL
    )

    assert frames.shape == (1, 3, 8, 64, 64)
    got = ((frames[0].mean(dim=(0, 2, 3)) + 1) / 2 * 255).numpy()
    np.testing.assert_allclose(got, _SERVED[5:], atol=_GRAY_ATOL)
    session = backend._sessions[0]
    window = [frame.astype(np.float32).mean() for frame in session["frames"]]
    np.testing.assert_allclose(window, [0] + _SERVED[9:], atol=_GRAY_ATOL)
    assert session["actions"][0].eq(0.0).all()
    np.testing.assert_allclose(session["actions"][1:, 0], [5, 6, 7, 8])
    assert session["step_id"] == 1


def test_remote_backend_pads_actions_to_the_latent_grid():
    import json

    from _fake_http_server import FakeHTTPServer, parse_multipart

    with FakeHTTPServer([(200, "video/mp4", _gray_mp4(_SERVED))]) as server:
        backend = RemoteWorldModelBackend(
            _remote_cfg(chunk=6), torch.device("cpu"), server.url
        )
        _open_sample_session(backend)
        frames = backend.generate([0], _chunk(6))

    (request,) = server.requests
    fields, _ = parse_multipart(request["content_type"], request["body"])
    assert fields["num_frames"] == "13"
    action = np.array(json.loads(fields["extra_params"])["action"])
    assert action.shape == (13, 7)
    np.testing.assert_allclose(action[11:], np.full((2, 7), 6.0))
    # The frames of the two padded actions are not part of the chunk.
    got = ((frames[0].mean(dim=(0, 2, 3)) + 1) / 2 * 255).numpy()
    np.testing.assert_allclose(got, _SERVED[5:11], atol=_GRAY_ATOL)


def test_remote_backend_retry_resends_the_same_step_once():
    from _fake_http_server import FakeHTTPServer, parse_multipart

    responses = [(503, "text/plain", b"busy"), (200, "video/mp4", _gray_mp4(_SERVED))]
    with FakeHTTPServer(responses) as server:
        backend = RemoteWorldModelBackend(
            _remote_cfg(), torch.device("cpu"), server.url
        )
        _open_sample_session(backend)
        backend.generate([0], _chunk(8))

    first, second = (
        parse_multipart(r["content_type"], r["body"]) for r in server.requests
    )
    assert first == second
    assert backend._sessions[0]["step_id"] == 1


def test_remote_backend_exhausted_retries_leave_the_session_unchanged():
    from _fake_http_server import FakeHTTPServer

    with FakeHTTPServer([(503, "text/plain", b"busy")]) as server:
        backend = RemoteWorldModelBackend(
            _remote_cfg(), torch.device("cpu"), server.url
        )
        _open_sample_session(backend)
        before = backend._sessions[0]
        frames = [f.copy() for f in before["frames"]]
        actions = before["actions"].clone()
        with pytest.raises(RuntimeError, match="503"):
            backend.generate([0], _chunk(8))

    assert len(server.requests) == 2
    after = backend._sessions[0]
    assert all(np.array_equal(a, b) for a, b in zip(after["frames"], frames))
    assert torch.equal(after["actions"], actions)
    assert after["step_id"] == 0


def test_remote_backend_rejects_an_unregistered_adapter():
    cfg = _remote_cfg(adapter="nope")
    with pytest.raises(ValueError, match=r"available: \['videos_sync'\]"):
        RemoteWorldModelBackend(cfg, torch.device("cpu"), "http://127.0.0.1:1")


def _bridged_remote_backend(monkeypatch, url):
    from rlinf.envs.sim.world_model.backend import remote

    monkeypatch.setattr(remote, "RobotwinActionBridge", _HalfActionBridge)
    cfg = _remote_cfg()
    cfg.rest_action = [2.0] * 7
    cfg.action_bridge = {
        "urdf_path": "",
        "srdf_path": "",
        "stat_path": "",
        "robot_pose": [0.0] * 7,
    }
    return RemoteWorldModelBackend(cfg, torch.device("cpu"), url)


def _sent_actions(request) -> np.ndarray:
    import json

    from _fake_http_server import parse_multipart

    fields, _ = parse_multipart(request["content_type"], request["body"])
    return np.array(json.loads(fields["extra_params"])["action"])


def test_remote_backend_bridges_actions_and_fills_history_with_the_rest_pose(
    monkeypatch,
):
    from _fake_http_server import FakeHTTPServer

    with FakeHTTPServer([(200, "video/mp4", _gray_mp4(_SERVED))]) as server:
        backend = _bridged_remote_backend(monkeypatch, server.url)
        _open_sample_session(backend)
        backend.generate([0], _chunk(8))

    (request,) = server.requests
    action = _sent_actions(request)
    assert action.shape == (13, 7)
    # The joint-space init_actions are ignored; the history is the bridged rest pose.
    np.testing.assert_allclose(action[:5], np.full((5, 7), 1.0))
    np.testing.assert_allclose(action[5:, 0], 0.5 * np.arange(1, 9))


def test_remote_backend_keeps_the_bridged_history_across_steps(monkeypatch):
    from _fake_http_server import FakeHTTPServer

    responses = [(200, "video/mp4", _gray_mp4(_SERVED))] * 2
    with FakeHTTPServer(responses) as server:
        backend = _bridged_remote_backend(monkeypatch, server.url)
        _open_sample_session(backend)
        backend.generate([0], _chunk(8))
        backend.generate([0], _chunk(8) + 10)

    second = _sent_actions(server.requests[1])
    np.testing.assert_allclose(second[0], np.full(7, 1.0))
    np.testing.assert_allclose(second[1:5, 0], 0.5 * np.array([5, 6, 7, 8]))
    np.testing.assert_allclose(second[5:, 0], 0.5 * np.arange(11, 19))


def test_remote_backend_borrows_the_scorer_of_the_named_backend(monkeypatch):
    from rlinf.envs.sim.world_model.backend import bwm

    scorer = _ConstantRewardModel(0.0)
    monkeypatch.setattr(
        bwm.BwmBackend, "load_reward_model", staticmethod(lambda cfg: scorer)
    )
    cfg = _remote_cfg()
    cfg.remote.reward_backend = "bwm"
    assert RemoteWorldModelBackend.load_reward_model(cfg) is scorer

    backend = RemoteWorldModelBackend(cfg, torch.device("cpu"), "http://127.0.0.1:1")
    env = type("Env", (), {"num_envs": 2, "task_descriptions": ["a", "b"]})()
    assert backend.reward_instructions(env) == ["a"] * 8 + ["b"] * 8


def test_remote_backend_rejects_an_unregistered_reward_backend():
    cfg = _remote_cfg()
    cfg.remote.reward_backend = "nope"
    with pytest.raises(KeyError, match="'nope' is not registered"):
        RemoteWorldModelBackend.load_reward_model(cfg)


class _ServedBackend(RemoteWorldModelBackend):
    """The remote backend with a constant scorer, so no reward checkpoint is needed."""

    @staticmethod
    def load_reward_model(cfg):
        return _ConstantRewardModel(0.0)

    def reward_instructions(self, env):
        return None


def _served_env_cfg(data_dir) -> dict:
    """A world-model env on the remote backend: chunk 2 after a 5-frame window."""
    frame = {
        "image": np.zeros((64, 64, 3), dtype=np.uint8),
        "task": "adjust the bottle",
        "delta_action": np.zeros(14, dtype=np.float32),
    }
    for episode in range(2):
        np.save(Path(data_dir) / f"episode_{episode}.npy", np.array([frame, frame]))
    return {
        "env_type": "world_model",
        "backend": "remote",
        "wm_env_type": "robotwin",
        "seed": 0,
        "auto_reset": True,
        "ignore_terminations": True,
        "max_episode_steps": 4,
        "chunk": 2,
        "action_dim": 14,
        "reward_coef": 1.0,
        "initial_image_path": str(data_dir),
        "enable_kir": False,
        "use_fixed_reset_state_ids": True,
        "group_size": 1,
        "video_cfg": {"save_video": False},
        "condition_frame_length": 5,
        "num_frames": 7,
        "image_size": [64, 64],
        "remote": {
            "adapter": "videos_sync",
            "model": "bwm",
            "request_timeout_s": 30,
            "warmup_timeout_s": 30,
            "max_retries": 0,
        },
    }


# 7 frames lie off the 4n+1 grid, so the adapter asks for 9.
_SERVED_9 = _gray_mp4([15 * t + 10 for t in range(9)])


class _ConstructorRecordingBackend(_BlankFrameBackend):
    def __init__(self, cfg, device, **kwargs):
        super().__init__(cfg, device)
        self.kwargs = kwargs


@pytest.mark.parametrize("server_url", [None, "http://wm-1:8091"])
def test_world_model_env_passes_a_remote_server_url_only_when_given(
    tmp_path, monkeypatch, server_url
):
    from rlinf.envs.sim.world_model.env import WorldModelEnv

    monkeypatch.setitem(
        registry._REGISTRY, "unit_test_recording", _ConstructorRecordingBackend
    )
    cfg = OmegaConf.create(
        {
            **_served_env_cfg(tmp_path),
            "backend": "unit_test_recording",
            "unit_test_reward": 0.0,
        }
    )
    env = WorldModelEnv(
        cfg, num_envs=1, seed_offset=0, total_num_processes=1, server_url=server_url
    )

    expected = {} if server_url is None else {"server_url": server_url}
    assert env.backend.kwargs == expected


def test_world_model_env_steps_against_the_remote_server_it_was_given(
    tmp_path, monkeypatch
):
    from _fake_http_server import FakeHTTPServer

    from rlinf.envs.sim.world_model.env import WorldModelEnv

    monkeypatch.setitem(registry._REGISTRY, "remote", _ServedBackend)
    cfg = OmegaConf.create(_served_env_cfg(tmp_path))
    with FakeHTTPServer([(200, "video/mp4", _SERVED_9)]) as server:
        env = WorldModelEnv(
            cfg, num_envs=1, seed_offset=0, total_num_processes=1, server_url=server.url
        )
        env.reset()
        env.chunk_step(torch.zeros(1, 2, 14))

    assert [r["path"] for r in server.requests] == ["/v1/videos/sync"]


class _RecordingEnv:
    """Stands in for an env class and keeps the keywords it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = {
            k: kwargs[k] for k in ("seed_offset", "server_url") if k in kwargs
        }


class _PlacementEnvWorker(EnvWorker):
    """A real EnvWorker whose env classes are swapped inside the worker process."""

    def __init__(self, cfg, served_env: bool = False):
        super().__init__(cfg)
        if served_env:
            registry._REGISTRY["remote"] = _ServedBackend
        else:
            env_worker_module.get_env_cls = lambda env_type, env_cfg=None: _RecordingEnv

    def env_kwargs(self) -> dict:
        return {
            "train": [env.kwargs for env in self.env_list],
            "eval": [env.kwargs for env in self.eval_env_list],
        }

    def step_once(self) -> None:
        self.env_list[0].chunk_step(torch.zeros(1, 2, 14))


def _env_worker_cfg(train: dict, eval_: dict | None = None, *, workers, stages):
    split = {
        "total_num_envs": workers * stages,
        "rollout_epoch": 1,
        "max_steps_per_rollout_epoch": 2,
    }
    env = {"train": {**train, **split}}
    if eval_ is not None:
        env["eval"] = {**eval_, **split}
    return OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"env": f"0:0-{workers - 1}"},
            },
            "runner": {"only_eval": False, "val_check_interval": 1 if eval_ else -1},
            "rollout": {"pipeline_stage_num": stages},
            "actor": {"model": {"num_action_chunks": 2}},
            "env": env,
        }
    )


def _fake_split(backend: str) -> dict:
    return {
        "env_type": "world_model",
        "backend": backend,
        "auto_reset": False,
        "video_cfg": {"save_video": False},
    }


@pytest.fixture(scope="module")
def placement_cluster():
    from rlinf.scheduler import Cluster

    # Worker processes unpickle the test worker class by importing this module.
    tests_root = str(Path(__file__).resolve().parent)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [tests_root, previous]))
    yield Cluster(num_nodes=1)
    if previous is None:
        os.environ.pop("PYTHONPATH")
    else:
        os.environ["PYTHONPATH"] = previous


@contextlib.contextmanager
def _env_group(cluster, cfg, workers, served_env=False):
    from rlinf.scheduler import NodePlacementStrategy

    group = _PlacementEnvWorker.create_group(cfg, served_env).launch(
        cluster,
        name=f"EnvGroup-{uuid.uuid4().hex[:8]}",
        placement_strategy=NodePlacementStrategy([0] * workers),
    )
    try:
        yield group
    finally:
        group._close()


def _call_each_rank(group, method: str, *args):
    """Call a worker method with plain ``ray.get`` so a raise reaches the test.

    A failed ``WorkerGroup`` call signals the driver to shut the cluster down.
    """
    import ray

    return ray.get(
        [getattr(w.worker, method).remote(*args) for w in group.worker_info_list]
    )


@pytest.mark.placement
def test_remote_server_urls_reach_every_stage_of_each_env_worker(placement_cluster):
    cfg = _env_worker_cfg(
        _fake_split("remote"), _fake_split("wan"), workers=3, stages=2
    )
    cfg.env.train.remote = {"server_urls": ["http://url0", "http://url1"]}
    with _env_group(placement_cluster, cfg, workers=3) as group:
        push_world_model_server_urls(cfg, group)
        group.init_worker().wait()
        kwargs = group.env_kwargs().wait()

    urls = [[env["server_url"] for env in rank["train"]] for rank in kwargs]
    assert urls == [["http://url0"] * 2, ["http://url1"] * 2, ["http://url0"] * 2]
    assert all("server_url" not in env for rank in kwargs for env in rank["eval"])


@pytest.mark.placement
def test_local_world_model_envs_get_no_remote_server_url(placement_cluster):
    cfg = _env_worker_cfg(_fake_split("wan"), _fake_split("wan"), workers=2, stages=1)
    with _env_group(placement_cluster, cfg, workers=2) as group:
        push_world_model_server_urls(cfg, group)
        group.init_worker().wait()
        kwargs = group.env_kwargs().wait()

    assert all(
        "server_url" not in env
        for rank in kwargs
        for split in rank.values()
        for env in split
    )


@pytest.mark.placement
def test_remote_server_urls_must_arrive_before_init_worker(placement_cluster):
    cfg = _env_worker_cfg(_fake_split("remote"), workers=2, stages=1)
    with _env_group(placement_cluster, cfg, workers=2) as group:
        with pytest.raises(RuntimeError, match="set_world_model_server_urls"):
            _call_each_rank(group, "init_worker")


@pytest.mark.placement
def test_remote_server_urls_are_fixed_once_init_worker_ran(placement_cluster):
    cfg = _env_worker_cfg(_fake_split("remote"), workers=2, stages=1)
    with _env_group(placement_cluster, cfg, workers=2) as group:
        group.set_world_model_server_urls(["http://stale"]).wait()
        group.set_world_model_server_urls(["http://a", "http://b"]).wait()
        group.init_worker().wait()
        with pytest.raises(RuntimeError, match="cannot change"):
            _call_each_rank(group, "set_world_model_server_urls", ["http://late"])
        kwargs = group.env_kwargs().wait()

    assert [rank["train"][0]["server_url"] for rank in kwargs] == [
        "http://a",
        "http://b",
    ]


@pytest.mark.placement
def test_an_env_worker_steps_against_the_remote_server_at_its_rank(
    placement_cluster, tmp_path
):
    from _fake_http_server import FakeHTTPServer

    cfg = _env_worker_cfg(_served_env_cfg(tmp_path), workers=2, stages=1)
    with (
        FakeHTTPServer([(200, "video/mp4", _SERVED_9)]) as s0,
        FakeHTTPServer([(200, "video/mp4", _SERVED_9)]) as s1,
    ):
        cfg.env.train.remote.server_urls = [s0.url, s1.url]
        with _env_group(placement_cluster, cfg, workers=2, served_env=True) as group:
            push_world_model_server_urls(cfg, group)
            group.init_worker().wait()
            group.execute_on(1).step_once().wait()

    assert (len(s0.requests), len(s1.requests)) == (0, 1)
