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

"""Isolated simulator process owned by ``RoboUniverseEnv``.

Runs under the configured ``sim_python`` interpreter (typically Isaac Sim 5.1.0's
``/isaac-sim/python.sh`` or a CPU test interpreter) so Isaac Sim's pinned Python
and NumPy versions never leak into RLinf's main worker process.
"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# Ensure sibling protocol module can be imported even when launched under an
# external simulator interpreter whose PYTHONPATH does not include RLinf.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from protocol import receive, send  # noqa: E402


def _load_entrypoint(entrypoint: str) -> Any:
    module_name, attr_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _evict_workspace_modules(workspace: Path) -> None:
    """Remove modules loaded from ``workspace`` so the next task imports cleanly."""
    workspace_resolved = workspace.resolve()
    stale: list[str] = []
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            resolved = Path(module_file).resolve()
        except OSError:
            continue
        if resolved == workspace_resolved or workspace_resolved in resolved.parents:
            stale.append(name)
    for name in stale:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _select_camera_ids(spec: dict[str, Any]) -> tuple[str, str | None]:
    cameras = spec["observation"]["cameras"]
    main_cam = next(
        (c["id"] for c in cameras if c.get("role") == "external"),
        cameras[0]["id"],
    )
    wrist_cam = next(
        (c["id"] for c in cameras if c.get("role") == "wrist"),
        cameras[1]["id"] if len(cameras) > 1 else None,
    )
    return main_cam, wrist_cam


class _TaskSession:
    """Own one active RoboUniverse ``operable-env/0.1`` backend instance."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.device: int = int(settings.get("device_index", 0))
        self.max_episode_steps: int = int(settings["max_episode_steps"])
        self.clip_actions: bool = bool(settings.get("clip_actions", True))
        self.include_joint_velocity: bool = bool(
            settings.get("include_joint_velocity", False)
        )
        self.grade_on_done: bool = bool(settings.get("grade_on_done", True))
        self.keep_evidence: bool = bool(settings.get("keep_evidence", False))

        self.workspace: Path | None = None
        self.backend: Any = None
        self.evaluator: Any = None
        self.spec: dict[str, Any] | None = None
        self.case: str = ""
        self.instruction: str = ""
        self.run_dir: Path | None = None
        self.main_cam: str = "external"
        self.wrist_cam: str | None = "wrist"
        self.action_lower: np.ndarray | None = None
        self.action_upper: np.ndarray | None = None

        self.steps: int = 0
        self.finished: bool = False
        self.success: bool = False
        self.infra_error: bool = False
        self.infra_error_message: str | None = None
        self.verdict: str | None = None
        self.episode_return: float = 0.0
        self.last_obs: dict[str, Any] | None = None

    def close_backend(self) -> None:
        """Close the active task backend and evict its workspace modules."""
        if self.backend is not None:
            try:
                self.backend.close()
            finally:
                self.backend = None
        self.evaluator = None
        if self.workspace is not None:
            workspace_str = str(self.workspace)
            if workspace_str in sys.path:
                sys.path.remove(workspace_str)
            _evict_workspace_modules(self.workspace)
            self.workspace = None

    def load_task(
        self,
        *,
        case: str,
        workspace_dir: str,
        runtime_dir: str,
        instruction: str,
        env_spec: dict[str, Any],
        run_dir: str,
    ) -> dict[str, Any]:
        """Load or reset a task workspace and return its initial observation."""
        workspace = Path(workspace_dir).resolve()
        runtime = Path(runtime_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.case = case
        self.instruction = instruction

        same_workspace = (
            self.backend is not None
            and self.workspace is not None
            and self.workspace == workspace
        )
        if not same_workspace:
            self.close_backend()
            sys.path.insert(0, str(workspace))
            os.chdir(workspace)
            (workspace / "output").mkdir(exist_ok=True)
            self.workspace = workspace
            self.spec = env_spec
            self.main_cam, self.wrist_cam = _select_camera_ids(env_spec)
            self.action_lower = np.asarray(
                env_spec["action"]["lower"], dtype=np.float32
            )
            self.action_upper = np.asarray(
                env_spec["action"]["upper"], dtype=np.float32
            )
            factory = _load_entrypoint(env_spec["backend_entrypoint"])
            self.backend = factory(
                workspace=workspace,
                runtime_root=runtime,
                device=self.device,
            )
            self.evaluator = _load_entrypoint(env_spec["evaluator_entrypoint"])

        supported_seeds = self.spec["reset"]["supported_seeds"]
        reset_seed = int(supported_seeds[0])
        raw_obs = self.backend.reset(seed=reset_seed)

        self.steps = 0
        self.finished = False
        self.success = False
        self.infra_error = False
        self.infra_error_message = None
        self.verdict = None
        self.episode_return = 0.0
        self.last_obs = self._pack_obs(raw_obs)
        return {
            "observation": self.last_obs,
            "metadata": {
                "case": self.case,
                "action_dim": int(self.action_lower.shape[0]),
                "state_dim": int(self.last_obs["states"].shape[0]),
                "action_lower": self.action_lower,
                "action_upper": self.action_upper,
                "control_dt_s": float(self.spec["timing"]["control_dt_s"]),
            },
        }

    def _pack_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        rgb = raw_obs["rgb"]
        main_img = np.ascontiguousarray(rgb[self.main_cam], dtype=np.uint8)
        proprio = raw_obs["proprio"]
        q = np.asarray(proprio["joint_position"], dtype=np.float32)
        if self.include_joint_velocity:
            qd = np.asarray(proprio["joint_velocity"], dtype=np.float32)
            state = np.concatenate([q, qd], axis=-1)
        else:
            state = q
        packed: dict[str, Any] = {
            "main_images": main_img,
            "states": np.ascontiguousarray(state, dtype=np.float32),
            "task_descriptions": self.instruction,
        }
        if self.wrist_cam is not None and self.wrist_cam in rgb:
            packed["wrist_images"] = np.ascontiguousarray(
                rgb[self.wrist_cam], dtype=np.uint8
            )
        return packed

    def _grade_episode(self) -> tuple[bool, str | None, bool, str | None]:
        """Run ``export_evidence`` and the task's verifier once at episode end."""
        if not self.grade_on_done:
            return False, None, False, None
        evidence_dir = self.run_dir / "evidence"
        if evidence_dir.exists():
            shutil.rmtree(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        try:
            exported = self.backend.export_evidence(evidence_dir)
            if not exported or not exported.get("files"):
                raise ValueError("Backend export_evidence returned no files")
            verdict_payload = self.evaluator(run_dir=evidence_dir)
            verdict = verdict_payload.get("verdict")
            if verdict not in ("success", "failure", "inconclusive"):
                raise ValueError(
                    f"Expected verdict in ('success', 'failure', 'inconclusive'), "
                    f"found {verdict!r}"
                )
            return verdict == "success", verdict, False, None
        except Exception as exc:
            return False, "error", True, f"{type(exc).__name__}: {exc}"
        finally:
            if not self.keep_evidence:
                shutil.rmtree(evidence_dir, ignore_errors=True)

    def chunk_step(self, actions: np.ndarray) -> dict[str, Any]:
        """Execute an action chunk ``[horizon, action_dim]``."""
        if self.backend is None or self.last_obs is None:
            raise RuntimeError("Call load_task/reset before stepping RoboUniverse")

        actions = np.asarray(actions, dtype=np.float32)
        expected_dim = int(self.action_lower.shape[0])
        if actions.ndim != 2 or actions.shape[1] != expected_dim:
            raise ValueError(
                f"Expected chunk actions shaped [horizon, {expected_dim}], "
                f"found {actions.shape}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("RoboUniverse actions must contain only finite values.")

        if self.clip_actions:
            actions = np.clip(actions, self.action_lower, self.action_upper)
        elif np.any(actions < self.action_lower) or np.any(actions > self.action_upper):
            raise ValueError(
                "RoboUniverse action outside declared env_spec.json bounds "
                "(set clip_actions=true to clamp)."
            )

        horizon = actions.shape[0]
        rewards = np.zeros(horizon, dtype=np.float32)
        terminations = np.zeros(horizon, dtype=bool)
        truncations = np.zeros(horizon, dtype=bool)
        observations: list[dict[str, Any]] = []

        for idx in range(horizon):
            if self.finished:
                observations.append(self.last_obs)
                continue
            try:
                raw_obs = self.backend.step(actions[idx].astype(np.float64))
                self.last_obs = self._pack_obs(raw_obs)
                self.steps += 1
            except Exception as exc:
                self.finished = True
                self.infra_error = True
                self.infra_error_message = f"{type(exc).__name__}: {exc}"
                self.verdict = "error"
                truncations[idx] = True
                observations.append(self.last_obs)
                continue

            observations.append(self.last_obs)
            if self.steps >= self.max_episode_steps:
                self.finished = True
                success, verdict, infra_error, error_msg = self._grade_episode()
                self.success = success
                self.verdict = verdict
                self.infra_error = infra_error
                self.infra_error_message = error_msg
                reward = 1.0 if success else 0.0
                rewards[idx] = reward
                self.episode_return += reward
                terminations[idx] = success
                truncations[idx] = not success

        episode = {
            "success_once": float(self.success),
            "return": float(self.episode_return),
            "episode_len": float(self.steps),
            "reward": float(self.episode_return / max(self.steps, 1)),
            "infra_error": float(self.infra_error),
        }
        return {
            "observations": observations,
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
            "episode": episode,
            "verdict": self.verdict,
            "infra_error_message": self.infra_error_message,
            "case": self.case,
        }


def main() -> None:
    """Serve RoboUniverse simulator commands until the owning adapter closes."""
    sock = socket.socket(fileno=int(sys.argv[1]))
    session: _TaskSession | None = None
    try:
        init_msg = receive(sock)
        if init_msg.get("command") != "init":
            raise ValueError(
                f"Expected initial command 'init', found {init_msg.get('command')!r}."
            )
        session = _TaskSession(init_msg["settings"])
        send(sock, {"status": "ready"})

        while True:
            msg = receive(sock)
            command = msg.get("command")
            if command == "close":
                session.close_backend()
                break
            if command == "load_task":
                send(sock, session.load_task(**msg["task"]))
            elif command == "chunk_step":
                send(sock, session.chunk_step(msg["actions"]))
            else:
                raise ValueError(f"Unknown RoboUniverse worker command: {command!r}.")
    except EOFError:
        pass
    except BaseException:
        try:
            send(sock, {"error": traceback.format_exc()})
        except OSError:
            pass
    finally:
        if session is not None:
            try:
                session.close_backend()
            except Exception:
                pass
        sock.close()


if __name__ == "__main__":
    main()
