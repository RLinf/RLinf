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

"""SFT runner checkpoint selection and resume with a local training actor."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.runners.sft_runner import SFTRunner


class LocalResult:
    """Completed result at the worker RPC boundary."""

    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value

    def consume_duration(self):
        return 0.0


class LocalSFTActor:
    """Train and checkpoint a small model without a remote scheduler."""

    def __init__(self, monitor="val_loss"):
        self.model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.model.weight.fill_(0.1)
            self.model.bias.zero_()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.monitor = monitor
        self.step = 0

    def get_max_steps_per_epoch(self):
        return LocalResult([8])

    def init_worker(self):
        return LocalResult()

    def run_training(self):
        self.optimizer.zero_grad()
        loss = (self.model(torch.tensor([[1.0, 2.0]])) - 1.0).square().mean()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return LocalResult([{"loss": loss.item()}])

    def run_eval(self):
        # Other metrics improve while the monitored metric exhausts patience.
        value = 1.0 if self.monitor == "val_loss" else -0.5
        return LocalResult([{self.monitor: value, "val_accuracy": self.step / 10}])

    def save_checkpoint(self, path, step):
        assert step == self.step
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
            },
            Path(path) / "train.pt",
        )
        return LocalResult()

    def load_checkpoint(self, path):
        state = torch.load(Path(path) / "train.pt", weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["step"]
        return LocalResult()


def make_runner(
    root, max_steps=8, resume_dir=None, monitor="val_loss", save_interval=2, patience=3
):
    cfg = OmegaConf.create(
        {
            "runner": {
                "max_steps": max_steps,
                "val_check_interval": 1,
                "save_interval": save_interval,
                "resume_dir": str(resume_dir) if resume_dir else None,
                "logger": {
                    "log_path": str(root),
                    "experiment_name": "sft",
                    "logger_backends": [],
                },
                "early_stop": {
                    "enabled": True,
                    "monitor": monitor,
                    "mode": "min" if monitor == "val_loss" else "max",
                    "patience": patience,
                },
            }
        }
    )
    return SFTRunner(cfg, LocalSFTActor(monitor))


def run_to_checkpoint(root, max_steps, resume_dir, monitor):
    runner = make_runner(root, max_steps, resume_dir, monitor)
    runner.init_workers()
    runner.run()
    Path(root, "result.json").write_text(json.dumps({"step": runner.global_step}))


def checkpoint(root, name):
    return Path(root, "sft", "checkpoints", name)


@pytest.mark.parametrize("monitor", ["val_loss", "value_spearman"])
def test_sft_best_checkpoint_and_stopping_state(tmp_path, monitor):
    runner = make_runner(tmp_path, monitor=monitor, save_interval=6)
    runner.init_workers()
    runner.run()
    assert runner.global_step == 4
    best = checkpoint(tmp_path, "best_model")
    assert torch.load(best / "actor/train.pt", weights_only=True)["step"] == 1
    state = json.loads(
        (checkpoint(tmp_path, "global_step_4") / "runner_state.json").read_text()
    )
    assert state["early_stop"]["counter"] == 3
    assert state["global_step"] == 4


@pytest.mark.parametrize("monitor", ["val_loss", "value_spearman"])
@pytest.mark.parametrize("resume_name", ["global_step_2", "best_model"])
def test_sft_resume_in_new_process_matches_uninterrupted(
    tmp_path, monitor, resume_name
):
    def run(root, max_steps=8, resume=None):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(Path(__file__).parent),
                str(Path(__file__).parents[2]),
                env.get("PYTHONPATH", ""),
            ]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from test_sft import run_to_checkpoint; import sys; run_to_checkpoint(sys.argv[1], int(sys.argv[2]), sys.argv[3] or None, sys.argv[4])",
                str(root),
                str(max_steps),
                str(resume) if resume else "",
                monitor,
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    whole, split = tmp_path / "whole", tmp_path / "split"
    run(whole)
    run(split, max_steps=2)
    saved = json.loads(
        (checkpoint(split, "global_step_2") / "runner_state.json").read_text()
    )
    assert saved["early_stop"]["counter"] == 1
    run(split, resume=checkpoint(split, resume_name))
    assert json.loads((split / "result.json").read_text())["step"] == 4
    assert json.loads((whole / "result.json").read_text())["step"] == 4
    reference = torch.load(
        checkpoint(whole, "global_step_4") / "actor/train.pt", weights_only=True
    )
    resumed = torch.load(
        checkpoint(split, "global_step_4") / "actor/train.pt", weights_only=True
    )
    for key, value in reference["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    for key, states in reference["optimizer"]["state"].items():
        for name, value in states.items():
            torch.testing.assert_close(
                value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    # Resuming a checkpoint that has stopped must not perform another update.
    run(split, resume=checkpoint(split, "global_step_4"))
    assert json.loads((split / "result.json").read_text())["step"] == 4


def test_sft_legacy_checkpoint_without_runner_state(tmp_path, caplog):
    runner = make_runner(tmp_path, max_steps=2)
    runner.init_workers()
    runner.run()
    saved = checkpoint(tmp_path, "global_step_2")
    (saved / "runner_state.json").unlink()
    resumed = make_runner(tmp_path, resume_dir=saved, max_steps=3)
    resumed.init_workers()
    assert resumed.global_step == 2
    assert resumed.actor.step == 2
    assert "early stopping starts fresh" in caplog.text
    resumed.run()
    assert (
        torch.load(
            checkpoint(tmp_path, "best_model") / "actor/train.pt", weights_only=True
        )["step"]
        == 3
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"version": 2}, "version"),
        ({"global_step": 3}, "differs"),
        ({"global_step": -1}, "non-negative"),
    ],
)
def test_sft_rejects_invalid_runner_state(tmp_path, change, match):
    runner = make_runner(tmp_path, max_steps=2)
    runner.init_workers()
    runner.run()
    saved = checkpoint(tmp_path, "global_step_2")
    state_path = saved / "runner_state.json"
    state = json.loads(state_path.read_text()) | change
    state_path.write_text(json.dumps(state))
    resumed = make_runner(tmp_path, resume_dir=saved)
    with pytest.raises(ValueError, match=match):
        resumed.init_workers()


@pytest.mark.parametrize("patience,stop_step", [(1, 2), (5, 6)])
def test_sft_resume_uses_configured_patience(tmp_path, patience, stop_step):
    first = make_runner(tmp_path, max_steps=2)
    first.init_workers()
    first.run()
    resumed = make_runner(
        tmp_path, resume_dir=checkpoint(tmp_path, "global_step_2"), patience=patience
    )
    resumed.init_workers()
    resumed.run()
    assert resumed.global_step == stop_step
    assert resumed.actor.step == stop_step


@pytest.mark.parametrize("failed_directory", ["best_model", "global_step_2"])
def test_sft_rejects_interrupted_checkpoint_save(tmp_path, failed_directory):
    class InterruptedActor(LocalSFTActor):
        def run_eval(self):
            return LocalResult([{"val_loss": 1 / self.step}])

        def save_checkpoint(self, path, step):
            result = super().save_checkpoint(path, step)
            if step == 2 and Path(path).parent.name == failed_directory:
                raise RuntimeError("checkpoint write interrupted")
            return result

    runner = make_runner(tmp_path, max_steps=2)
    runner.actor = InterruptedActor()
    runner.init_workers()
    with pytest.raises(RuntimeError, match="write interrupted"):
        runner.run()
    saved = checkpoint(tmp_path, failed_directory)
    assert (saved / "runner_state.incomplete").is_file()
    assert torch.load(saved / "actor/train.pt", weights_only=True)["step"] == 2
    if failed_directory == "best_model":
        assert json.loads((saved / "runner_state.json").read_text())["global_step"] == 1
    resumed = make_runner(tmp_path, resume_dir=saved)
    with pytest.raises(ValueError, match="checkpoint save is incomplete"):
        resumed.init_workers()
    assert resumed.actor.step == 0


def test_sft_without_early_stop_keeps_regular_checkpoint_resume(tmp_path):
    runner = make_runner(tmp_path, max_steps=2)
    runner.early_stop = None
    runner.init_workers()
    runner.run()
    saved = checkpoint(tmp_path, "global_step_2")
    state = json.loads((saved / "runner_state.json").read_text())
    assert state["early_stop"] is None
    assert not checkpoint(tmp_path, "best_model").exists()
    resumed = make_runner(tmp_path, resume_dir=saved, max_steps=3)
    resumed.early_stop = None
    resumed.init_workers()
    resumed.run()
    assert resumed.global_step == resumed.actor.step == 3
