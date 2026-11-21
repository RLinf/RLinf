"""Run the car-specific HIL checks against the dummy backend (parameterized 15 or 6 subtasks)."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.envs.realworld.tianji.car_monitor_tap import MonitorTap
from rlinf.envs.realworld.tianji.tianji_env import TianjiConfig, TianjiEnv
from rlinf.envs.realworld.tianji.tianji_subtasks import load_subtask_prompts
from rlinf.scheduler.worker.routing import validate_batch_size
from rlinf.workers.env.tianji_joint_eval_worker import TianjiVLAEnvWorker
from rlinf.workers.rollout.hf.tianji_vla_eval_worker import (
    TianjiVLAEvalWorker,
    joint_reference_actions,
)


def test_hil(root: Path, num_subtasks: int = 15, banner: str = "Stage2-30HZ-1118v1"):
    prompt_file = root / "prompts.json"
    prompts = [f"S{i:02} simulated instruction" for i in range(1, num_subtasks + 1)]
    prompt_file.write_text(json.dumps({"prompts": prompts}))
    assert load_subtask_prompts(prompt_file, limit=num_subtasks) == tuple(prompts)
    try:
        load_subtask_prompts(prompt_file)
    except ValueError:
        pass
    else:
        raise AssertionError("Box actor limit changed")
    output = root / "hil"
    env = TianjiEnv(
        TianjiConfig(
            backend="dummy",
            action_space="joint",
            hil_control=True,
            hil_port=0,
            hil_mode="inference",
            subtask_prompts_file=str(prompt_file),
            image_hw=(8, 8),
            step_frequency=30,
            manual_episode_control_only=True,
            recorder_dir=str(output),
            episode_archive=True,
            stream_chunks=True,
        )
    )
    try:
        assert env.episode_archive is not None
        assert banner in env.hil.page()
        with ThreadPoolExecutor(1) as pool:
            reset = pool.submit(env.reset)
            until = time.monotonic() + 3
            while env.hil.snapshot()["episode"] != 1:
                assert time.monotonic() < until
                time.sleep(0.01)
            env.hil.command("start")
            observation, _ = reset.result(timeout=3)

        original = int(observation["car_stage_meta"][4])
        env.set_action_window_steps(1)
        env.on_action_chunk_begin()
        env.hil.command("next_subtask")
        assert env.task_description == prompts[1]
        assert env.car_control_revision > original
        with_mock = Mock(wraps=env.backend.send_action)
        env.backend.send_action = with_mock
        info = env.step(env.get_hold_action())[-1]
        assert info["receipt"]["status"] == "hil_hold"
        with_mock.assert_not_called()

        worker = object.__new__(TianjiVLAEnvWorker)
        wrapped = SimpleNamespace(call=lambda *_: [env.car_control_revision])
        worker.eval_env_list = [SimpleNamespace(env=wrapped)]
        reply = test_policy_route(observation["car_stage_meta"])
        assert worker._reject_stale_policy(reply, 0)
        for i in range(3, num_subtasks + 1):
            env.hil.command("next_subtask")
            assert env.subtask == i and env.task_description == prompts[i - 1]
        env.hil.command("subtask", max(2, num_subtasks - 2))
        env.hil.command("pause")
        env.hil.command("subtask", num_subtasks)
        env.hil.command("human_start")
        assert env.hil.snapshot()["human_latched"]
        env.hil.command("human_stop")
        assert not env.hil.snapshot()["human_latched"]
        env.hil.command("success")
        env.step(env.get_hold_action())
    finally:
        env.close()

    events = [
        json.loads(line)
        for line in (output / "subtask_events.jsonl").read_text().splitlines()
    ]
    assert {r["stage"] for r in events} == set(range(1, num_subtasks + 1))
    assert {
        "first_advance",
        "backtrack",
        "revisit",
        "pause",
        "human_start",
        "human_stop",
        "success",
    }.issubset(r["name"] for r in events)
    assert any(output.rglob("steps.jsonl")), "No published episode step archive"


def test_policy_route(metadata):
    """Exercise the same first-request envelope as the live evaluation worker."""
    metadata = torch.as_tensor(metadata)[None]
    rollout = object.__new__(TianjiVLAEvalWorker)
    rollout.cfg = SimpleNamespace(rollout={"joint_actor_eval": True})
    rollout.model_cfg = SimpleNamespace(num_action_chunks=42)
    rollout.car_monitor = MonitorTap(None)
    rollout.hf_model = SimpleNamespace(
        extract_rlt_obs=lambda obs: {
            "ref_chunk": torch.zeros((1, 60, 16)),
            "z_rl": torch.zeros((1, 2048)),
            "proprio": obs["joint_states"],
        }
    )
    payload = EnvOutput(
        obs={
            "joint_states": torch.zeros((1, 16)),
            "task_descriptions": ["S01 simulated instruction"],
            "car_stage_meta": metadata,
        }
    ).to_dict()
    request = rollout._merge_obs_batches([payload])["obs"]
    actions, result = rollout._predict_rollout_actions(request)
    assert actions.shape == (1, 42, 16)
    reply = rollout._eval_policy_output(actions, result)
    env = object.__new__(TianjiVLAEnvWorker)
    env.cfg = OmegaConf.create(
        {"rollout": {"joint_actor_eval": True, "group_name": "RolloutGroup"}}
    )
    env.env_decoupled_mode = False
    env.eval_batch_size = 1
    shards = rollout._split_eval_policy_output(reply, [1])

    def recv_from(**kwargs):
        validate_batch_size(shards[0], 1, kwargs["infer_batch_size_fn"])
        return kwargs["merge_fn"](shards)

    env.recv_from = recv_from
    reply = env._recv_eval_policy_output(None, 0)
    torch.testing.assert_close(reply.forward_inputs["car_stage_meta"], metadata)
    return reply


def test_monitor(root: Path, num_subtasks: int = 15):
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        path = str(root / "monitor.sock")
        sock.bind(path)
        sock.settimeout(3)
        tap = MonitorTap(path)
        tap.publish(
            torch.tensor([[123, 1, num_subtasks, num_subtasks - 1, 20, 1]]),
            torch.zeros((1, 2048)),
            torch.zeros((1, 16)),
        )
        data = np.load(io.BytesIO(sock.recv(16384)))
        assert int(data["capture_ns"]) == 123 and int(data["stage"]) == num_subtasks
        assert data["rlt"].shape == (2048,) and data["joint"].shape == (16,)
    ref = torch.zeros((1, 60, 16))
    assert joint_reference_actions({"ref_chunk": ref}, 42, 18).shape == (1, 42, 16)


def main():
    num_subtasks = int(os.environ.get("TIANJI_CAR_NUM_SUBTASKS", "15"))
    default_banner = (
        "Stage2-30HZ-sub6-1118v1" if num_subtasks == 6 else "Stage2-30HZ-1118v1"
    )
    banner = os.environ.get("TIANJI_CAR_BANNER", default_banner)
    os.environ["TIANJI_CAR_1118"] = "1"
    os.environ["TIANJI_CAR_NUM_SUBTASKS"] = str(num_subtasks)
    os.environ["TIANJI_CAR_BANNER"] = banner
    with tempfile.TemporaryDirectory(prefix="car-hil-") as tmp:
        root = Path(tmp)
        test_hil(root, num_subtasks=num_subtasks, banner=banner)
        test_monitor(root, num_subtasks=num_subtasks)
    print(
        f"PASS: {num_subtasks}-stage HIL, stale policy rejection, handover, archive, monitor datagram"
    )


if __name__ == "__main__":
    main()
