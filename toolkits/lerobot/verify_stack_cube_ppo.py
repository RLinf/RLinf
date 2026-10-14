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

"""Probe GPU Stack Cube batching and SFT-to-RL parity without optimizer updates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def file_hash(path: Path) -> str:
    """Hash a source or input file for probe provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_requests(path: Path) -> list[dict]:
    """Reject test cohorts before a development probe can consume them."""
    requests = json.loads(path.read_text())["requests"]
    if not requests or any(request.get("split") != "val" for request in requests):
        raise ValueError("Preflight probes accept only validation requests.")
    return requests


def simulator_probe(args) -> dict:
    import torch
    from embodichain.lab.sim import SimulationManager
    from omegaconf import OmegaConf
    from PIL import Image

    from rlinf.envs.sim.embodichain.stack_cube_env import StackCubeEnv
    from toolkits.lerobot.evaluate_stack_cube_openpi import load_replay_episodes

    requests = validation_requests(args.requests)
    episodes = load_replay_episodes(args.collection, requests)
    cohort = [requests[index % len(requests)] for index in range(args.num_envs)]
    states = [
        episodes[request["seed"]]["report"]["initial_state"] for request in cohort
    ]
    actions = [episodes[request["seed"]]["actions"] for request in cohort]
    if len({len(action) for action in actions}) != 1:
        raise ValueError("Batched replay probe requires equal-length expert episodes.")
    steps = len(actions[0])
    if steps > 450:
        raise ValueError("Expert replay exceeds the 450-step evaluation budget.")
    manifest = args.output / "request_manifest.json"
    manifest.write_text(json.dumps({"requests": cohort}, indent=2))
    cfg = OmegaConf.create(
        {
            "gym_config_path": "rlinf/envs/sim/embodichain/configs/tasks/manipulation/stack_cube/task.panda.yaml",
            "panda_urdf": str(args.panda_urdf.resolve()),
            "request_manifest": str(manifest),
            "sim_device": "cuda:0",
            "stack_cube_protocol": "isaaclab",
            "auto_reset": False,
            "ignore_terminations": True,
            "max_episode_steps": 450,
            "use_fixed_reset_state_ids": True,
        }
    )
    environments = []
    snapshots = []
    snapshot_steps = []
    reward_trace = []
    position_trace = []
    finger_trace = []
    started = time.monotonic()
    free_before, total_memory = torch.cuda.mem_get_info()
    minimum_free = free_before
    try:
        runtime = StackCubeEnv(cfg, args.num_envs, 0, 1, None)
        environments.append(runtime)
        obs, _ = runtime.reset(options={"requests": cohort, "initial_states": states})
        assert np.isclose(runtime.env.step_dt, 0.05)
        initial = {
            key: value.detach().cpu().numpy().copy()
            for key, value in obs.items()
            if isinstance(value, torch.Tensor)
        }
        for key in ("main_images", "wrist_images"):
            assert initial[key].shape == (args.num_envs, 256, 256, 3)
        assert initial["states"].shape == (args.num_envs, 8)
        actual_qpos = runtime.env.robot.get_qpos().cpu().numpy()
        np.testing.assert_allclose(
            actual_qpos, [state["qpos"] for state in states], atol=1e-6
        )
        for uid in ("blue", "red", "green"):
            actual_pose = (
                runtime.env.sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)
                .cpu()
                .numpy()
            )
            np.testing.assert_allclose(
                actual_pose, [state["object_poses"][uid] for state in states], atol=1e-6
            )
        if args.num_envs > 1:
            try:
                runtime.reset(options={"env_idx": [0]})
            except NotImplementedError:
                pass
            else:
                raise AssertionError("Partial reset was accepted.")
            np.testing.assert_array_equal(
                runtime.env.robot.get_qpos().cpu().numpy(), actual_qpos
            )
        second_observation = None
        if args.second_num_envs:
            other_requests = [
                requests[i % len(requests)] for i in range(args.second_num_envs)
            ]
            other_path = args.output / "second_request_manifest.json"
            other_path.write_text(json.dumps({"requests": other_requests}))
            other_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            other_cfg.request_manifest = str(other_path)
            other = StackCubeEnv(other_cfg, args.second_num_envs, 0, 1, None)
            environments.append(other)
            second_observation, _ = other.reset()
            second_observation = second_observation["states"].cpu().clone()
            np.testing.assert_array_equal(
                runtime.env.robot.get_qpos().cpu().numpy(), actual_qpos
            )
        for row in range(min(8, args.num_envs)):
            Image.fromarray(
                np.concatenate(
                    (initial["main_images"][row], initial["wrist_images"][row]), axis=1
                )
            ).save(args.output / f"initial_{row:02d}.png")
        actions = torch.as_tensor(np.stack(actions), device="cuda:0")
        snapshots.append(initial)
        snapshot_steps.append(0)
        for step in range(steps):
            obs, reward, term, trunc, info = runtime.step(actions[:, step])
            measured = info["stack_metrics"]
            torch.testing.assert_close(
                reward, measured["baseline_success"].float(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                reward, info["rewards"]["success"], rtol=0, atol=0
            )
            reward_trace.append(reward.cpu().numpy().copy())
            position_trace.append(measured["positions"].cpu().numpy().copy())
            finger_trace.append(
                runtime.env.robot.get_qpos()[:, runtime.env.hand_ids]
                .cpu()
                .numpy()
                .copy()
            )
            assert not term.any()
            assert (trunc == (step + 1 == 450)).all()
            if step + 1 in {50, 150, 250, steps}:
                minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
                snapshots.append(
                    {
                        key: value.detach().cpu().numpy().copy()
                        for key, value in obs.items()
                        if isinstance(value, torch.Tensor)
                    }
                )
                snapshot_steps.append(step + 1)
        np.savez_compressed(
            args.output / "reward_trace.npz",
            rewards=np.stack(reward_trace),
            positions=np.stack(position_trace),
            fingers=np.stack(finger_trace),
        )
        np.savez_compressed(
            args.output / "observations.npz",
            **{
                key: np.stack([snapshot[key] for snapshot in snapshots])
                for key in initial
            },
            steps=np.array(snapshot_steps),
            seeds=np.array([request["seed"] for request in cohort]),
        )
        metrics = {
            key: value.cpu().tolist() for key, value in info["stack_metrics"].items()
        }
        second_unchanged = None
        if second_observation is not None:
            second_unchanged = bool(
                np.array_equal(
                    environments[1].env.dataset_states(), second_observation.numpy()
                )
            )
            if not second_unchanged:
                raise AssertionError(
                    "Stepping the first simulator altered the second simulator's state."
                )
            environments[1].close()
            environments.pop()
            SimulationManager.flush_cleanup_queue()
            if SimulationManager.get_instance_num() != 1:
                raise AssertionError(
                    "Closing the second simulator did not release its manager."
                )
            # The first simulator must remain usable after the second closes.
            if steps == 450:
                runtime.reset(options={"requests": cohort, "initial_states": states})
            hold = torch.zeros(args.num_envs, 7, device="cuda:0")
            hold[:, -1] = 1
            runtime.step(hold)
        return {
            "checks": {
                "all_replays_stable": bool(info["episode"]["success_at_end"].all()),
                "initial_front_views_are_distinct": len(
                    {
                        hashlib.sha256(row.tobytes()).hexdigest()
                        for row in initial["main_images"]
                    }
                )
                >= len({r["seed"] for r in cohort}),
            },
            "num_envs": args.num_envs,
            "distinct_validation_requests": len({r["seed"] for r in cohort}),
            "replicated_validation_rows": args.num_envs > len(requests),
            "steps": steps,
            "native_reward_matches_baseline": True,
            "episode_returns": np.stack(reward_trace).sum(axis=0).tolist(),
            "stable_at_end": info["episode"]["success_at_end"].cpu().tolist(),
            "metrics": metrics,
            "second_simulator_unchanged": second_unchanged,
            "seconds": time.monotonic() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "device_total_bytes": total_memory,
            "device_free_before_bytes": free_before,
            "device_minimum_observed_free_bytes": minimum_free,
            "initial_camera_hashes": {
                key: [hashlib.sha256(row.tobytes()).hexdigest() for row in initial[key]]
                for key in ("main_images", "wrist_images")
            },
        }
    finally:
        for environment in reversed(environments):
            environment.close()
        SimulationManager.flush_cleanup_queue()


def model_probe(args) -> dict:
    import torch
    from omegaconf import OmegaConf

    from rlinf.envs.sim.embodichain.stack_cube import STACK_PROMPT
    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights

    requests = validation_requests(args.requests)
    source = np.load(args.observations)
    permitted_seeds = {request["seed"] for request in requests}
    if not set(source["seeds"].tolist()).issubset(permitted_seeds):
        raise ValueError("Observation snapshot contains seeds outside validation.")
    count = min(
        args.max_observations, source["states"].shape[0] * source["states"].shape[1]
    )
    if count < 1:
        raise ValueError("Observation snapshot must contain at least one frame.")
    samples = []
    for index in range(count):
        frame, row = divmod(index, source["states"].shape[1])
        samples.append(
            {
                key: torch.from_numpy(source[key][frame, row : row + 1].copy())
                for key in ("main_images", "wrist_images", "states")
            }
        )
        samples[-1]["task_descriptions"] = [STACK_PROMPT]
    weights = resolve_full_weights(args.checkpoint)
    if weights is None:
        raise FileNotFoundError("Expected a full_weights.pt SFT export.")
    base = {
        "model_type": "openpi",
        "model_path": str(args.checkpoint),
        "precision": None,
        "is_lora": False,
        "action_dim": 7,
        "num_action_chunks": 5,
        "num_steps": 5,
        "add_value_head": False,
        "openpi_data": {"norm_stats_path": str(args.norm_stats)},
        "openpi": {
            "task": "eval",
            "config_name": "pi05_isaaclab_stack_cube",
            "action_horizon": 10,
            "action_chunk": 5,
            "action_env_dim": 7,
            "model_action_dim": 32,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "max_token_len": 200,
            "num_images_in_input": 2,
            "num_steps": 5,
            "discrete_state_input": False,
            "eval_sft_image_crop": True,
            "torch_compile": False,
        },
    }

    def policy_hash(model):
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            if name.startswith("value_head."):
                continue
            digest.update(name.encode())
            digest.update(
                tensor.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        return digest.hexdigest()

    def noise(index):
        return torch.randn(
            1,
            10,
            32,
            device="cuda:0",
            generator=torch.Generator(device="cuda:0").manual_seed(1000 + index),
        )

    model = get_model(OmegaConf.create(base)).to("cuda:0").eval()
    eval_hash = policy_hash(model)
    with torch.no_grad():
        reference = [
            model.predict_action_batch(obs, mode="eval", noise=noise(index))[0].cpu()
            for index, obs in enumerate(samples)
        ]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base["add_value_head"] = True
    base["openpi"].update(
        task="rl",
        noise_method="flow_sde",
        noise_level=0.5,
        joint_logprob=False,
        value_after_vlm=True,
        value_vlm_mode="mean_token",
        detach_critic_input=True,
        train_expert_only=True,
    )
    model = get_model(OmegaConf.create(base)).to("cuda:0").eval()
    rl_hash = policy_hash(model)
    if eval_hash != rl_hash:
        raise AssertionError("SFT-to-RL construction changed policy tensors.")
    comparisons = []
    for index, obs in enumerate(samples):
        with torch.no_grad():
            model.eval()
            eval_actions, _ = model.predict_action_batch(
                obs, mode="eval", noise=noise(index)
            )
            action_error = float((eval_actions.cpu() - reference[index]).abs().max())
            random.seed(2000 + index)
            _, rollout = model.predict_action_batch(
                obs,
                mode="train",
                noise=noise(index),
                rng=torch.Generator(device="cuda:0").manual_seed(3000 + index),
            )
            model.train()
            recomputed = model(
                forward_inputs=rollout["forward_inputs"], compute_values=True
            )
            delta = (recomputed["logprobs"] - rollout["prev_logprobs"]).float()
            comparisons.append(
                {
                    "index": index,
                    "seed": int(source["seeds"][index % len(source["seeds"])]),
                    "step": int(source["steps"][index // len(source["seeds"])]),
                    "denoise_indices": rollout["forward_inputs"]["denoise_inds"]
                    .cpu()
                    .tolist(),
                    "action_max_abs_error": action_error,
                    "logprob_max_abs_error": float(delta.abs().max()),
                    "ratio_max_abs_error": float((delta.exp() - 1).abs().max()),
                    "value_max_abs_error": float(
                        (
                            recomputed["values"].reshape(-1)
                            - rollout["prev_values"].reshape(-1)
                        )
                        .abs()
                        .max()
                    ),
                }
            )
    np.savez(
        args.output / "reference_actions.npz", actions=torch.cat(reference).numpy()
    )
    return {
        "checks": {
            "eval_actions_match": all(
                np.isfinite(item["action_max_abs_error"])
                and item["action_max_abs_error"] <= 1e-5
                for item in comparisons
            ),
            "sde_ratio_matches": all(
                np.isfinite(item["ratio_max_abs_error"])
                and item["ratio_max_abs_error"] <= 1e-3
                for item in comparisons
            ),
        },
        "thresholds": {"action_max_abs_error": 1e-5, "ratio_max_abs_error": 1e-3},
        "policy_weights_equal": eval_hash == rl_hash,
        "policy_sha256": rl_hash,
        "checkpoint_sha256": file_hash(weights),
        "norm_stats_sha256": file_hash(args.norm_stats),
        "observations_sha256": file_hash(args.observations),
        "comparisons": comparisons,
        "optimizer_updates": 0,
        "note": "Direct model preflight; distributed weight sync and checkpoint recovery require separate worker probes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim", "model"))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--second-num-envs", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--max-observations", type=int, default=40)
    args = parser.parse_args()
    required = (
        ("panda_urdf", "collection")
        if args.mode == "sim"
        else ("checkpoint", "norm_stats", "observations")
    )
    if any(getattr(args, key) is None for key in required):
        parser.error(f"{args.mode} requires {required}")
    if args.num_envs < 1 or args.second_num_envs < 0 or args.max_observations < 1:
        parser.error("Scene and observation counts must be positive.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    report = {
        "mode": args.mode,
        "requests_sha256": file_hash(args.requests),
        "probe_sha256": file_hash(Path(__file__)),
        "passed": False,
    }
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {
        name: file_hash(root / name)
        for name in (
            "rlinf/envs/sim/embodichain/stack_cube.py",
            "rlinf/envs/sim/embodichain/stack_cube_env.py",
            "rlinf/envs/sim/embodichain/embodichain_env.py",
            "rlinf/envs/sim/embodichain/stack_cube_metrics.py",
            "rlinf/models/embodiment/openpi/tasks/rl.py",
            "rlinf/envs/sim/embodichain/configs/components/embodiments/isaaclab_panda.yaml",
            "rlinf/envs/sim/embodichain/configs/tasks/manipulation/stack_cube/env.yaml",
        )
    }
    if args.panda_urdf is not None:
        report["panda_urdf_sha256"] = file_hash(args.panda_urdf)
    try:
        report.update(
            simulator_probe(args) if args.mode == "sim" else model_probe(args)
        )
        report["passed"] = all(report["checks"].values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(
            "Probe completed, but one or more checks failed; inspect report.json."
        )


if __name__ == "__main__":
    main()
