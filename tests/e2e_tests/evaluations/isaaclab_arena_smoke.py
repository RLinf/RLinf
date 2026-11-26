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

"""Check DROID cameras, reset observations and timeout bootstrapping on one RTX GPU."""

import csv
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import torch
from hydra import compose, initialize_config_dir

from rlinf.envs import get_env_cls
from rlinf.scheduler import AcceleratorType


def check_cameras():
    root = Path(__file__).resolve().parents[3]
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    with initialize_config_dir(
        config_dir=str(root / "evaluations/isaaclab_arena"), version_base="1.1"
    ):
        config = compose(
            config_name="isaaclab_arena_pi05_eval",
            overrides=[
                "env.eval.is_eval=false",
                "env.eval.auto_reset=true",
                "env.eval.max_episode_steps=12",
            ],
        )
    cfg = config.env.eval
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    index, uuid = next(csv.reader([gpu.strip()], skipinitialspace=True))
    placement = SimpleNamespace(
        accelerator_type=AcceleratorType.NV_GPU, accelerator_rank=int(index)
    )
    with get_env_cls(cfg.env_type)(cfg, 1, 0, 1, placement) as env:
        first, _ = env.reset()
        # Move the arm so the terminal pose is distinguishable in both views.
        action = first["states"].clone()
        action[:, 0] += 0.5
        for step in range(cfg.max_episode_steps):
            reset_obs, _, terminated, truncated, info = env.step(action)
            assert not terminated.item(), (
                f"The arm motion unexpectedly terminated the task at step {step + 1}"
            )
            assert truncated.item() == (step + 1 == cfg.max_episode_steps), (
                f"Unexpected timeout at {step + 1}"
            )
        assert not info["episode"]["success_once"].item(), (
            "A timeout must not count as success"
        )
        final = info["final_observation"]
        assert (
            final["states"][:, 0] - reset_obs["states"][:, 0]
        ).abs().item() > 0.05, "Expected the arm to return to its reset pose"
        settled, *_ = env.step(reset_obs["states"].clone())
        for key in ("main_images", "wrist_images"):
            image = reset_obs[key]
            assert (
                image.ndim == 4
                and image.shape[-1] == 3
                and image.dtype == torch.uint8
                and image.max() > image.min()
            ), f"Missing rendered RGB data in {key}"
            after_reset = (image.float() - settled[key]).abs().mean()
            before_reset = (final[key].float() - settled[key]).abs().mean()
            assert before_reset > after_reset, (
                f"{key} did not refresh on reset ({before_reset=}, {after_reset=})"
            )

        check_kit_devices(env._process.pid, uuid)


def check_kit_devices(pid: int, expected_uuid: str) -> None:
    """Check both CUDA and graphics contexts against the physical worker GPU."""
    inventory = ElementTree.fromstring(
        subprocess.check_output(["nvidia-smi", "-q", "-x"])
    )
    for kind, label in (("C", "compute"), ("G", "graphics")):
        devices = {
            gpu.findtext("uuid")
            for gpu in inventory.findall("gpu")
            for process in gpu.findall("processes/process_info")
            if process.findtext("pid") == str(pid)
            and kind in process.findtext("type", "")
        }
        assert devices == {expected_uuid}, (
            f"Expected Kit {label} on {expected_uuid}, got {devices}"
        )


if __name__ == "__main__":
    check_cameras()
