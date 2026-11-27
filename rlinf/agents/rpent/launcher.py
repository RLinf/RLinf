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

"""RPent episode subprocesses and their simulator startup permits."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import AsyncExitStack
from pathlib import Path

from omegaconf import DictConfig


class RPentEpisodeError(RuntimeError):
    """A failed service or simulator must never become a zero-reward trajectory."""

    def __init__(self, message: str, *, kind: str):
        super().__init__(f"[{kind}] {message}")
        self.kind = kind


_EXIT_SETUP_FAILED = 75
_STOP_GRACE_S = 60


class RPentEpisodeLauncher:
    """Own episode processes, launch arguments, and bounded simulator startup."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._env_initializations = asyncio.Semaphore(
            cfg.tools.rpent.get("max_env_initializations_per_worker", 8)
        )

    async def run(self, episode_id: str, task: dict, planner: dict, gpu: int) -> dict:
        """Run the episode's RPent process and return the outcome it wrote."""
        rpent = self.cfg.tools.rpent
        output = Path(rpent.output_dir) / episode_id
        output.mkdir(parents=True)
        request = {"task": task, "planner": planner}
        (output / "request.json").write_text(json.dumps(request))
        command = self.command(output, gpu)
        log = output / "episode.log"
        with log.open("wb") as log_file:
            await self._env_initializations.acquire()
            permit_held = True

            def release_permit() -> None:
                nonlocal permit_held
                if permit_held:
                    permit_held = False
                    self._env_initializations.release()

            async with AsyncExitStack() as cleanup:
                cleanup.callback(release_permit)
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                    env=self.environment(gpu),
                )
                cleanup.push_async_callback(self._stop, process)
                startup = asyncio.create_task(
                    self._watch_environment_startup(process, output, release_permit)
                )
                cleanup.push_async_callback(self._cancel_startup, startup)
                try:
                    returncode = await asyncio.wait_for(
                        process.wait(), rpent.process_timeout_s
                    )
                except asyncio.TimeoutError:
                    raise RPentEpisodeError(
                        f"RPent episode {episode_id} ran past "
                        f"tools.rpent.process_timeout_s; see {log}",
                        kind="process",
                    ) from None
        if returncode == _EXIT_SETUP_FAILED:
            raise RPentEpisodeError(
                f"RPent episode {episode_id} did not start; see {log}",
                kind="env_setup",
            )
        if returncode < 0:
            raise RPentEpisodeError(
                f"RPent episode {episode_id} was stopped by signal "
                f"{-returncode}; see {log}",
                kind="process",
            )
        if returncode != 0:
            raise RuntimeError(
                f"RPent episode {episode_id} failed with exit status "
                f"{returncode}; see {log}"
            )
        outcome = json.loads((output / "outcome.json").read_text())
        keep = rpent.keep_episodes == "all" or (
            rpent.keep_episodes == "failures" and not outcome["success"]
        )
        if not keep:
            shutil.rmtree(output)
        return outcome

    @staticmethod
    async def _cancel_startup(startup: asyncio.Task) -> None:
        startup.cancel()
        await asyncio.gather(startup, return_exceptions=True)

    async def _watch_environment_startup(
        self, process, output: Path, release_permit
    ) -> None:
        """Release a cold-start permit after reset, process exit, or cancellation."""
        try:
            while process.returncode is None and not (output / "states.json").is_file():
                await asyncio.sleep(0.5)
        finally:
            release_permit()

    def environment(self, gpu: int) -> dict[str, str]:
        """Bind MuJoCo to the selected physical GPU in the driver's EGL device order."""
        from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

        egl = NvidiaGPUManager.get_egl_device_id(gpu)
        if egl is None:
            raise ValueError(f"No EGL device found for RPent simulator GPU {gpu}")
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["MUJOCO_EGL_DEVICE_ID"] = str(egl)
        env["EGL_DEVICE_ID"] = str(egl)
        config_path = self.cfg.tools.rpent.get("libero_config_path")
        if config_path is not None:
            env["LIBERO_CONFIG_PATH"] = str(config_path)
        return env

    def command(self, output: Path, gpu: int) -> list[str]:
        """Command line of an episode process writing to ``output``."""
        rpent = self.cfg.tools.rpent
        command = [
            rpent.python,
            "-m",
            "robots.rl.rl_episode",
            "--output-dir",
            str(output),
            "--request",
            str(output / "request.json"),
            "--observation-mode",
            "adaptive",
            "--prompt",
            self.cfg.agentloop.get("prompt", "simple"),
            "--workflow",
            "episode",
            "--action-mode",
            "batch",
            "--max-actions-per-call",
            "3",
        ]
        if rpent.get("sam3_endpoint") is not None:
            command += ["--sam3-endpoint", rpent.sam3_endpoint]
        if rpent.get("libero_type") is not None:
            command += ["--libero-type", rpent.libero_type]
        command += [
            "--max-episode-steps",
            str(rpent.max_episode_steps),
            "--cuda-device",
            str(gpu),
            "--tokenizer-path",
            rpent.get("tokenizer_path", self.cfg.rollout.model.model_path),
        ]
        return command

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        """Stop an episode process, giving it time to release its simulator."""
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), _STOP_GRACE_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
