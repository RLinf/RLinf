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
from pathlib import Path

from omegaconf import DictConfig

from rlinf.agents.rpent.episode import Episode, RPentEpisodeError

_EXIT_SETUP_FAILED = 75
_STOP_GRACE_S = 60


class RPentEpisodeLauncher:
    """Own episode processes, launch arguments, and bounded simulator startup."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._env_initializations = asyncio.Semaphore(
            cfg.tools.rpent.get("max_env_initializations_per_worker", 8)
        )

    def endpoint(self, task: dict) -> str | None:
        """Return the toolkit that owns this task's scene."""
        simulators = (self.cfg.get("env") or {}).get("rpent")
        if simulators is not None:
            return simulators[task["environment"]].endpoint
        return self.cfg.tools.rpent.get("remote_toolkit_endpoint")

    async def run(
        self,
        episode: Episode,
        planner: dict,
        gpu: int | None,
        *,
        candidate: dict | None = None,
    ) -> dict:
        """Run the episode's RPent process and return the outcome it wrote."""
        rpent = self.cfg.tools.rpent
        output = Path(rpent.output_dir) / episode.episode_id
        output.mkdir(parents=True)
        if candidate is not None:
            (output / "sampling-candidate.json").write_text(
                json.dumps({**candidate, "episode_id": episode.episode_id}) + "\n"
            )
        task = episode.task
        (output / "request.json").write_text(
            json.dumps({"task": task, "planner": planner})
        )
        environment = self.environment()
        if candidate is not None:
            environment["RPENT_REMOTE_OWNER_ID"] = episode.episode_id
            simulators = (self.cfg.get("env") or {}).get("rpent")
            if simulators and simulators[task["environment"]].get(
                "group_router", False
            ):
                environment["RPENT_REMOTE_GROUP"] = (
                    f"{candidate['cohort_id']}:{candidate['group_id']}"
                )
        command = self.command(output, gpu, task=task)
        log = output / "episode.log"
        with log.open("wb") as log_file:
            await self._env_initializations.acquire()
            start_task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                    env=environment,
                )
            )
            try:
                process = await asyncio.shield(start_task)
            except asyncio.CancelledError:
                try:
                    process = await start_task
                    await self.stop(process)
                finally:
                    self._env_initializations.release()
                raise
            except BaseException:
                self._env_initializations.release()
                raise
            startup = asyncio.create_task(
                self._watch_environment_startup(process, output)
            )
            try:
                returncode = await asyncio.wait_for(
                    process.wait(), rpent.process_timeout_s
                )
            except asyncio.TimeoutError:
                raise RPentEpisodeError(
                    f"RPent episode {episode.episode_id} ran past "
                    f"tools.rpent.process_timeout_s; see {log}",
                    kind="process",
                ) from None
            finally:
                try:
                    await self.stop(process)
                finally:
                    startup.cancel()
                    await asyncio.gather(startup, return_exceptions=True)
        if returncode == _EXIT_SETUP_FAILED:
            raise RPentEpisodeError(
                f"RPent episode {episode.episode_id} did not start; see {log}",
                kind="env_setup",
            )
        if returncode < 0:
            raise RPentEpisodeError(
                f"RPent episode {episode.episode_id} was stopped by signal "
                f"{-returncode}; see {log}",
                kind="process",
            )
        if returncode != 0:
            raise RPentEpisodeError(
                f"RPent episode {episode.episode_id} failed with exit status "
                f"{returncode}; see {log}",
                kind="process",
            )
        outcome = json.loads((output / "outcome.json").read_text())
        keep = rpent.keep_episodes == "all" or (
            rpent.keep_episodes == "failures" and not outcome["success"]
        )
        if not keep and candidate is None:
            shutil.rmtree(output)
        return outcome

    async def _watch_environment_startup(self, process, output: Path) -> None:
        """Release a cold-start permit after reset, process exit, or cancellation."""
        try:
            while process.returncode is None and not (output / "states.json").is_file():
                await asyncio.sleep(0.5)
        finally:
            self._env_initializations.release()

    def environment(self) -> dict[str, str]:
        """Environment shared by the episode and cleanup processes.

        Agent workers hold every GPU, so the worker's own environment names
        GPU 0 as the render device. The episode process picks its device from
        ``--cuda-device`` instead, and ``configure_egl_device`` only overrides
        an inherited value when it is unset, so the inherited one has to go.
        """
        env = dict(os.environ)
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        env.pop("EGL_DEVICE_ID", None)
        repo_path = self.cfg.tools.rpent.get("repo_path")
        if repo_path is not None:
            env["PYTHONPATH"] = os.pathsep.join(
                [str(repo_path), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        return env

    def command(
        self, output: Path, gpu: int | None, *, task: dict | None = None
    ) -> list[str]:
        """Command line of an episode process writing to ``output``."""
        rpent = self.cfg.tools.rpent
        task = task or {}
        command = [
            rpent.python,
            "-m",
            "robots.rl.rl_episode",
            "--output-dir",
            str(output),
            "--request",
            str(output / "request.json"),
            "--remote-toolkit-endpoint",
            self.endpoint(task),
            "--max-episode-steps",
            str(rpent.max_episode_steps),
            "--tokenizer-path",
            self.cfg.rollout.model.model_path,
        ]
        if task.get("environment", "libero") == "libero":
            command += ["--libero-type", rpent.libero_type]
        for option in (
            "image_size",
            "max_actions_per_call",
            "reasoning_history",
            "image_history",
        ):
            flag = (
                "--image_size"
                if option == "image_size"
                else "--" + option.replace("_", "-")
            )
            command += [flag, str(rpent[option])]
        if gpu is not None:
            command += ["--cuda-device", str(gpu)]
        return command

    @staticmethod
    async def stop(process: asyncio.subprocess.Process) -> None:
        """Stop an episode process, giving it time to release its simulator."""
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), _STOP_GRACE_S)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
