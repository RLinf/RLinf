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

"""Verify formal actor-to-rollout sync and validation, without training updates.

The launcher preserves production policy prediction and records its RNG state.
Use --dry-run while a training job owns the GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from toolkits.lerobot.verify_stack_cube_ppo import file_hash

REPO = Path(__file__).resolve().parents[2]


def fingerprints(state: dict[str, torch.Tensor]) -> dict[str, str]:
    """Hash the policy and critic independently without modifying tensors."""
    policy, critic = hashlib.sha256(), hashlib.sha256()
    for name, value in sorted(state.items()):
        digest = critic if name.startswith("value_head.") else policy
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return {"policy_sha256": policy.hexdigest(), "critic_sha256": critic.hexdigest()}


def cpu_json(value: Any) -> Any:
    """Convert worker evidence to JSON-compatible CPU values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: cpu_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_json(item) for item in value]
    return value


def loaded_sources(names: list[str]) -> dict[str, dict[str, str]]:
    """Record the source files actually imported inside each worker process."""
    return {
        name: {
            "path": sys.modules[name].__file__,
            "sha256": file_hash(Path(sys.modules[name].__file__)),
        }
        for name in names
    }


class PreflightActor(EmbodiedFSDPActor):
    """Observe production construction, including its zero-LR optimizer warmup."""

    def model_provider_func(self):
        model = super().model_provider_func()
        self.initial_weights = fingerprints(model.state_dict())
        return model

    def preflight_state(self):
        return {
            "before_fsdp_and_warmup": self.initial_weights,
            "current": fingerprints(self.get_rollout_state_dict()),
            "optimizer_steps": self.optimizer_steps,
            "adam_step_max": max(
                (
                    float(state["step"])
                    for state in self.optimizer.state.values()
                    if "step" in state
                ),
                default=0.0,
            ),
            "optimizer_state_entries": len(self.optimizer.state),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sources": loaded_sources(["rlinf.models.embodiment.openpi.tasks.rl"]),
        }


class PreflightEnv(EnvWorker):
    """Capture public evaluate-step outcomes while preserving production routing."""

    def _build_rollout_input_data(self, env_batch):
        data = super()._build_rollout_input_data(env_batch)
        infos = env_batch.get("env_infos") or {}
        if "reset_request_ids" in infos:
            if not hasattr(self, "preflight_resets"):
                self.preflight_resets = []
            self.preflight_resets.append(
                {
                    "request_ids": cpu_json(infos["reset_request_ids"]),
                    "request_seeds": cpu_json(infos["reset_request_seeds"]),
                }
            )
        index = getattr(self, "capture_input_index", 0)
        self.capture_input_index = index + 1
        if self.cfg.runner.get("capture_tensors", False) and index < 2:
            np.savez_compressed(
                Path(self.cfg.runner.logger.log_path)
                / f"environment_observation_{index:03d}.npz",
                **{
                    key: value.detach().cpu().numpy().copy()
                    for key, value in data["obs"].items()
                    if isinstance(value, torch.Tensor)
                },
            )
        return data

    def env_evaluate_step(self, chunk_actions, stage_id):
        output, info = super().env_evaluate_step(chunk_actions, stage_id)
        if not hasattr(self, "preflight_chunks"):
            self.preflight_chunks = []
        self.preflight_chunks.append(
            {
                "stage_id": stage_id,
                "states": cpu_json(output.obs["states"]),
                "stack_metrics": cpu_json(output.env_infos["stack_metrics"]),
                "episode": cpu_json(info),
                "received_actions": cpu_json(chunk_actions),
                "last_applied_action": cpu_json(output.env_infos["applied_action"]),
            }
        )
        return output, info

    def evaluation_evidence(self):
        return {
            "chunks": self.preflight_chunks,
            "resets": self.preflight_resets,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sources": loaded_sources(
                [
                    "rlinf.envs.sim.embodichain.stack_cube",
                    "rlinf.envs.sim.embodichain.embodichain_env",
                ]
            ),
        }

    def close_environments(self):
        from embodichain.lab.sim import SimulationManager

        for env in self.env_list + self.eval_env_list:
            env.close()
        SimulationManager.flush_cleanup_queue()
        return {"remaining_simulators": SimulationManager.get_instance_num()}


def rng_evidence(path: Path) -> dict:
    """Persist RNG states without consuming random draws or reseeding."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }
    torch.save(states, path)
    return {
        "state_file": str(path),
        "python_state_sha256": hashlib.sha256(
            pickle.dumps(states["python"])
        ).hexdigest(),
        "numpy_state_sha256": hashlib.sha256(pickle.dumps(states["numpy"])).hexdigest(),
        "cpu_initial_seed": torch.initial_seed(),
        "cuda_initial_seed": torch.cuda.initial_seed(),
        "torch_cpu_state_sha256": hashlib.sha256(
            states["torch_cpu"].numpy().tobytes()
        ).hexdigest(),
        "torch_cuda_state_sha256": hashlib.sha256(
            states["torch_cuda"].cpu().numpy().tobytes()
        ).hexdigest(),
    }


class FormalActor(PreflightActor):
    """Observe model construction and production zero-LR optimizer warmup."""

    def model_provider_func(self):
        self.rng_before_model = rng_evidence(
            Path(self.cfg.runner.logger.log_path) / "actor_rng_before_model.pt"
        )
        return super().model_provider_func()

    def formal_state(self, label: str):
        return self.preflight_state() | {
            "rng_before_model": self.rng_before_model,
            "rng": rng_evidence(
                Path(self.cfg.runner.logger.log_path) / f"actor_rng_{label}.pt"
            ),
            "configured_actor_seed": self.cfg.actor.seed,
            "rank": self._rank,
        }


class FormalRollout(MultiStepRolloutWorker):
    """Observe RNG and weights; prediction remains the inherited production method."""

    def init_worker(self):
        self.rng_before_model = rng_evidence(
            Path(self.cfg.runner.logger.log_path) / "rollout_rng_before_model.pt"
        )
        return super().init_worker()

    def formal_state(self, label: str):
        return {
            "weights": fingerprints(self.hf_model.state_dict()),
            "version": self.version,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "configured_rollout_seed": self.cfg.rollout.get("seed"),
            "rank": self._rank,
            "configured_actor_seed": self.cfg.actor.seed,
            "rng_before_model": self.rng_before_model,
            "rng": rng_evidence(
                Path(self.cfg.runner.logger.log_path) / f"rollout_rng_{label}.pt"
            ),
            "sources": loaded_sources(
                [
                    "rlinf.workers.rollout.hf.huggingface_worker",
                    "rlinf.models.embodiment.openpi.tasks.rl",
                ]
            ),
        }


def _ppo_checkpoint_integrity(actor: Path) -> dict:
    """Verify that a native PPO save contains its exported and resumable state."""
    from torch.distributed.checkpoint import FileSystemReader

    directory = actor / "dcp_checkpoint"
    metadata = FileSystemReader(directory).read_metadata()
    keys = metadata.state_dict_metadata
    for prefix in (
        "fsdp_checkpoint.model.",
        "fsdp_checkpoint.optimizers.",
        "fsdp_checkpoint.lr_schedulers.",
    ):
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError(f"PPO DCP checkpoint is missing {prefix}")
    if "fsdp_checkpoint.rng" not in keys or "fsdp_checkpoint.fsdp_version" not in keys:
        raise ValueError("PPO DCP checkpoint is missing rank RNG or FSDP version.")
    if not metadata.storage_data:
        raise ValueError("PPO DCP checkpoint has no backing storage.")
    files = {}
    for storage in metadata.storage_data.values():
        path = (directory / storage.relative_path).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("PPO DCP storage must remain inside its checkpoint.")
        size = path.stat().st_size
        if (
            storage.offset < 0
            or storage.length <= 0
            or size < storage.offset + storage.length
        ):
            raise ValueError(f"PPO DCP checkpoint has truncated storage: {path.name}")
        files[storage.relative_path] = size
    return {
        "dcp_metadata_sha256": file_hash(directory / ".metadata"),
        "dcp_backing_files_bytes": files,
        "dcp_components_present": [
            "model",
            "optimizers",
            "lr_schedulers",
            "rng",
            "fsdp_version",
        ],
        "dcp_storage_ranges_complete": True,
    }


def checkpoint_actor(path: Path, *, checkpoint_kind: str = "sft") -> tuple[Path, dict]:
    """Resolve an SFT export or native PPO checkpoint for policy evaluation.

    SFT requires its published completion marker. PPO additionally verifies
    DCP components and backing file ranges; unmarked native saves retain an
    explicit ``complete: null`` rather than inventing a producer marker.
    """
    if checkpoint_kind not in {"sft", "ppo"}:
        raise ValueError("checkpoint_kind must be 'sft' or 'ppo'.")
    path = path.resolve(strict=True)
    actor = path / "actor" if (path / "actor").is_dir() else path
    marker = actor / "checkpoint_metadata.json"
    if checkpoint_kind == "ppo" and (actor / "openpi_loader_state.json").exists():
        raise ValueError("PPO checkpoint contains an OpenPI SFT loader marker.")
    if checkpoint_kind == "sft" or marker.is_file():
        metadata = json.loads(marker.read_text())
    else:
        metadata = {
            "step": parse_global_step_from_checkpoint_path(actor.parent),
            "complete": None,
            "completion_marker_present": False,
            "checkpoint_format": "dcp",
        }
    step = metadata.get("step")
    if metadata.get("checkpoint_kind", checkpoint_kind) != checkpoint_kind:
        raise ValueError("Checkpoint kind differs from its published metadata.")
    if (
        (marker.is_file() and not metadata.get("complete"))
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 1
    ):
        raise ValueError(
            f"A complete {checkpoint_kind.upper()} actor export with integer step metadata is required."
        )
    weights = actor / "model_state_dict/full_weights.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError(
            "The complete actor export must contain model_state_dict/full_weights.pt."
        )
    if checkpoint_kind == "ppo":
        inference_only = (
            marker.is_file()
            and metadata.get("checkpoint_kind") == "ppo"
            and metadata.get("save_training_state") is False
        )
        metadata = {
            **metadata,
            **(
                {"inference_only_export": True}
                if inference_only
                else _ppo_checkpoint_integrity(actor)
            ),
            "checkpoint_kind": "ppo",
        }
    return actor, metadata


def request_cohorts(train_path: Path, val_path: Path) -> tuple[list, list]:
    train = json.loads(train_path.read_text())["requests"]
    val = json.loads(val_path.read_text())["requests"]
    if len(train) < 32 or any(
        r.get("split") != "train" or r.get("profile") != "formal" for r in train
    ):
        raise ValueError(
            "The training simulator requires at least 32 distinct formal train requests."
        )
    if len(val) != 27 or any(
        r.get("split") != "val" or r.get("profile") != "formal" for r in val
    ):
        raise ValueError(
            "Evaluation requires the frozen formal validation 27, never test/OOD."
        )
    for cohort in [train, val]:
        if any(
            not isinstance(r.get("request_id"), str) or not r["request_id"]
            for r in cohort
        ):
            raise ValueError("Every request needs a nonempty request_id.")
        if len({r["request_id"] for r in cohort}) != len(cohort):
            raise ValueError("Request IDs must be unique within each cohort.")
        if len({r["seed"] for r in cohort}) != len(cohort):
            raise ValueError("Request seeds must be unique within each cohort.")
    if {r["seed"] for r in train} & {r["seed"] for r in val}:
        raise ValueError("Train and validation seeds must be disjoint.")
    return train, val


def configuration(args: argparse.Namespace) -> DictConfig:
    """Compose the formal PPO worker layout without launching workers."""
    env_python = getattr(args, "env_python", None)
    os.environ["STACK_CUBE_MODEL_PYTHON"] = sys.executable
    os.environ["STACK_CUBE_ENV_PYTHON"] = str(env_python or sys.executable)
    os.environ["REPO_PATH"] = str(REPO)
    for key, value in {
        "STACK_CUBE_PANDA_URDF": args.panda_urdf,
        "STACK_CUBE_PPO_TRAIN_REQUESTS": args.train_requests,
        "STACK_CUBE_PPO_VAL_REQUESTS": args.val_requests,
        "STACK_CUBE_SFT_MODEL": args.checkpoint,
        "STACK_CUBE_NORM_STATS": args.norm_stats,
    }.items():
        os.environ[key] = str(value.resolve(strict=True))
    os.environ["RLINF_PATH_ENV_MERGE_MODE"] = "override"
    with initialize_config_dir(
        config_dir=str(REPO / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(config_name="embodichain_franka_stack_cube_ppo_openpi_pi05")
    tag = f"formal_val_{os.getpid()}"
    env_vars = {
        "PYTHONPATH": os.pathsep.join(
            str(path)
            for path in (args.dependency_path, REPO, os.environ.get("PYTHONPATH"))
            if path
        ),
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "RLINF_PATH_ENV_MERGE_MODE": "override",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_CUMEM_HOST_ENABLE": "0",
    }
    with open_dict(cfg):
        cfg.cluster.component_placement = {
            "actor,rollout,env": {"node_group": tag, "placement": str(args.gpu)}
        }
        cfg.cluster.node_groups = [
            {
                "label": tag,
                "node_ranks": "all",
                "env_configs": [
                    {
                        "node_ranks": "all",
                        "python_interpreter_path": sys.executable,
                        "env_vars": [{k: v} for k, v in env_vars.items()],
                    }
                ],
            }
        ]
        if env_python is not None:
            simulation_tag = tag + "_simulation"
            cfg.cluster.component_placement = {
                "actor,rollout": {"node_group": tag, "placement": str(args.gpu)},
                "env": {"node_group": simulation_tag, "placement": str(args.gpu)},
            }
            simulation_vars = {**env_vars, "PYTHONPATH": str(REPO)}
            cfg.cluster.node_groups.append(
                {
                    "label": simulation_tag,
                    "node_ranks": "all",
                    "env_configs": [
                        {
                            "node_ranks": "all",
                            "python_interpreter_path": str(env_python),
                            "env_vars": [{k: v} for k, v in simulation_vars.items()],
                        }
                    ],
                }
            )
        for component in ["actor", "rollout", "env"]:
            cfg[component].group_name = tag + "_" + component
        cfg.actor.seed = args.seed
        cfg.actor.micro_batch_size = 32
        cfg.actor.global_batch_size = 128
        cfg.actor.enable_offload = False
        cfg.rollout.enable_offload = False
        cfg.env.train.total_num_envs = 32
        cfg.env.eval.total_num_envs = 27
        cfg.env.train.seed = args.seed
        cfg.env.eval.seed = args.seed
        cfg.env.eval.video_cfg.save_video = True
        cfg.env.eval.video_cfg.info_on_video = (
            False  # Installed Pillow lacks sized default font; no policy effect.
        )
        cfg.env.eval.video_cfg.fps = 20
        cfg.runner.only_eval = (
            False  # Required for production weight-sync construction.
        )
        cfg.runner.max_steps = 0  # Defense in depth; run() is never invoked.
        cfg.runner.max_epochs = 0
        cfg.runner.resume_dir = None
        cfg.runner.ckpt_path = None
        cfg.runner.logger.log_path = str(args.output)
        cfg.runner.logger.experiment_name = "formal_validation_preflight"
        cfg.runner.logger.logger_backends = []
        cfg.runner.per_worker_log = False
    OmegaConf.resolve(cfg)
    if not args.dry_run:
        cfg = validate_cfg(cfg)
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
    assert cfg.actor.model.openpi.eval_sft_image_crop is True
    assert (
        cfg.env.eval.max_episode_steps
        == cfg.env.eval.max_steps_per_rollout_epoch
        == 450
    )
    assert cfg.env.eval.ignore_terminations and not cfg.env.eval.auto_reset
    return cfg


def evidence_sources(output: Path) -> dict:
    files = [
        Path(__file__),
        REPO
        / "examples/embodiment/config/embodichain_franka_stack_cube_ppo_openpi_pi05.yaml",
        REPO / "examples/embodiment/config/env/embodichain_stack_cube.yaml",
    ]
    files += [
        REPO / p
        for p in [
            "rlinf/models/__init__.py",
            "rlinf/models/embodiment/openpi/pi0.py",
            "rlinf/models/embodiment/openpi/tasks/rl.py",
            "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
            "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
            "rlinf/workers/rollout/hf/huggingface_worker.py",
            "rlinf/workers/env/env_worker.py",
            "rlinf/runners/embodied_runner.py",
            "rlinf/envs/sim/embodichain/stack_cube.py",
            "rlinf/envs/sim/embodichain/embodichain_env.py",
        ]
    ]
    config_root = REPO / "rlinf/envs/sim/embodichain/configs"
    files += list((config_root / "tasks/manipulation/stack_cube").rglob("*.yaml"))
    files += [
        config_root / "components/embodiments/isaaclab_panda.yaml",
        config_root / "components/execution_policies/stack_cube.yaml",
    ]
    result = {}
    for source in files:
        target = (
            output
            / "source_snapshot"
            / (
                source.relative_to(REPO)
                if source.is_relative_to(REPO)
                else Path("external") / source.name
            )
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        result[str(source)] = file_hash(source)
    return result


def main() -> None:
    """Evaluate an exported policy through production worker routing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=["sft", "ppo"], default="sft")
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--train-requests", type=Path, required=True)
    parser.add_argument("--val-requests", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--env-python",
        type=Path,
        help="Independent simulator interpreter; actor and rollout use this CLI's interpreter.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dependency-path",
        type=Path,
        default=None,
        help="Optional directory containing the isolated LeRobot v2.1 reader.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.env_python is not None:
        args.env_python = args.env_python.expanduser().absolute()
        if not os.access(args.env_python, os.X_OK):
            parser.error("--env-python must point to an executable interpreter")
    args.checkpoint, metadata = checkpoint_actor(
        args.checkpoint, checkpoint_kind=args.checkpoint_kind
    )
    train, val = request_cohorts(args.train_requests, args.val_requests)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml", resolve=True)
    report = {
        "passed": False,
        "dry_run": args.dry_run,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": metadata,
        "checkpoint_kind": args.checkpoint_kind,
        "runtime_interpreters": {
            "model": sys.executable,
            "simulation": str(args.env_python or sys.executable),
        },
        "train_simulators": 32,
        "eval_simulators": 27,
        "expected_episodes": 27,
        "control_steps": 450,
        "stable_dwell_steps": 20,
        "actor_micro_batch": 32,
        "actor_global_batch": 128,
        "policy_protocol": "Inherited production predict, global RNG, no per-episode noise override and no reseeding in launcher.",
        "optimizer_protocol": "Production zero-LR initialization warmup allowed; no optimizer training updates.",
        "optimizer_training_updates": 0,
        "train_requests_count": len(train),
        "val_request_seeds": [r["seed"] for r in val],
        "hashes": {
            k: file_hash(getattr(args, k))
            for k in ["norm_stats", "train_requests", "val_requests", "panda_urdf"]
        },
        "source_sha256": evidence_sources(args.output),
        "cleanup": {},
    }
    (args.output / "val_request_manifest.json").write_text(
        json.dumps({"requests": val}, indent=2)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if args.dry_run:
        assert not ray.is_initialized() and not torch.cuda.is_initialized()
        report["ray_initialized"] = False
        report["cuda_initialized"] = False
        report["configuration_composed"] = True
        report["runtime_validation_deferred"] = True
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        return
    weights = args.checkpoint / "model_state_dict/full_weights.pt"
    report["weights_sha256"] = file_hash(weights)
    groups = []
    env = runner = None
    try:
        cluster = Cluster(
            cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
        )
        placement = HybridComponentPlacement(cfg, cluster)
        workers = {}
        for name, cls in [
            ("actor", FormalActor),
            ("rollout", FormalRollout),
            ("env", PreflightEnv),
        ]:
            group = cls.create_group(cfg).launch(
                cluster,
                name=cfg[name].group_name,
                placement_strategy=placement.get_strategy(name),
            )
            groups.append(group)
            workers[name] = group
        env = workers["env"]
        runner = EmbodiedRunner(cfg, **workers)
        runner.init_workers()
        report["actor_before_sync"] = (
            workers["actor"].formal_state("before_sync").wait()[0]
        )
        report["rollout_before_sync"] = (
            workers["rollout"].formal_state("before_sync").wait()[0]
        )
        for component in ["actor", "rollout"]:
            state = report[f"{component}_before_sync"]
            base_seed = (
                state.get("configured_rollout_seed")
                if component == "rollout"
                else state["configured_actor_seed"]
            )
            if base_seed is None:
                base_seed = state["configured_actor_seed"]
            expected_seed = int(base_seed) + state["rank"]
            assert (
                state["rng"]["cpu_initial_seed"]
                == state["rng"]["cuda_initial_seed"]
                == expected_seed
            )
        original = report["actor_before_sync"]["current"]
        assert original == report["actor_before_sync"]["before_fsdp_and_warmup"], (
            "Actor initialization changed weights"
        )
        runner.update_rollout_weights()
        report["rollout_after_sync"] = (
            workers["rollout"].formal_state("after_sync").wait()[0]
        )
        assert report["rollout_after_sync"]["weights"] == original, (
            "Actor-to-rollout sync changed weights"
        )
        (args.output / "sync_report.json").write_text(json.dumps(report, indent=2))
        report["evaluation_metrics"] = cpu_json(runner.evaluate())
        report["environment"] = env.evaluation_evidence().wait()[0]
        assert report["environment"]["resets"] == [
            {
                "request_ids": [request["request_id"] for request in val],
                "request_seeds": [request["seed"] for request in val],
            }
        ], "Evaluated requests differ from the frozen validation cohort."
        report["actor_after_eval"] = (
            workers["actor"].formal_state("after_eval").wait()[0]
        )
        report["rollout_after_eval"] = (
            workers["rollout"].formal_state("after_eval").wait()[0]
        )
        assert (
            report["actor_after_eval"]["current"]
            == report["rollout_after_eval"]["weights"]
            == original
        )
        assert (
            report["actor_after_eval"]["optimizer_steps"]
            == report["actor_after_eval"]["adam_step_max"]
            == 0
        )
        assert report["evaluation_metrics"]["num_trajectories"] == 27
        assert len(report["environment"]["chunks"]) == 90
        for chunk in report["environment"]["chunks"]:
            assert np.asarray(chunk["states"]).shape == (27, 8)
            assert np.isfinite(np.asarray(chunk["states"])).all()
            assert np.isfinite(np.asarray(chunk["received_actions"])).all()
        assert (
            report["environment"]["chunks"][-1]["episode"]["episode_len"] == [450] * 27
        )
        final = report["environment"]["chunks"][-1]["episode"]["success_at_end"]
        assert len(final) == 27
        report["stable_success_count"] = sum(bool(value) for value in final)
        report["stable_success_rate"] = report["stable_success_count"] / 27
        assert (
            abs(
                report["stable_success_rate"]
                - report["evaluation_metrics"]["success_at_end"]
            )
            < 1e-6
        )
        report["formal_gate_met"] = report["stable_success_count"] >= 6
        assert report["weights_sha256"] == file_hash(weights), (
            "Checkpoint changed during validation"
        )
        report["passed"] = (
            True  # Interface checks; the success gate is reported separately.
        )
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        if env is not None:
            try:
                report["cleanup"]["environment"] = env.close_environments().wait()
            except Exception as exc:
                report["cleanup"]["environment_error"] = str(exc)
        for group in reversed(groups):
            for worker in group.worker_info_list:
                try:
                    ray.kill(worker.worker, no_restart=True)
                except Exception as exc:
                    report["cleanup"].setdefault("worker_errors", []).append(str(exc))
        report["cleanup"]["own_workers_killed"] = True
        if "environment_error" in report["cleanup"] or report["cleanup"].get(
            "worker_errors"
        ):
            report["passed"] = False
        if any(
            item.get("remaining_simulators") != 0
            for item in report["cleanup"].get("environment", [])
        ):
            report["passed"] = False
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        if runner is not None:
            runner.metric_logger.finish()
        ray.shutdown()
        report["cleanup"]["ray_shutdown_returned"] = True
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    if not report["passed"]:
        raise SystemExit("Formal worker validation failed; inspect report.json.")


if __name__ == "__main__":
    main()
