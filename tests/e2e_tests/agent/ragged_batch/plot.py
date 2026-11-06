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

"""Render the recorded batch-validation figures without starting training."""

import json
from pathlib import Path


def main() -> None:
    """Plot historical measurements with separate communication and run scopes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent
    evidence = json.loads((root / "evidence.json").read_text())
    comparison = evidence["comparison"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, key, metric, scale, title, unit in (
        (
            axes[0],
            "peak_increment_bytes",
            None,
            1048576,
            "DP exchange temporary GPU memory",
            "MiB",
        ),
        (axes[1], "warm_mean_seconds", None, 1, "DP exchange elapsed time", "seconds"),
        (
            axes[2],
            "mean_step_seconds",
            "comparison",
            1,
            "Three-step mean elapsed time",
            "seconds",
        ),
    ):
        values = [
            (
                comparison[key][case]
                if metric
                else comparison["frozen_communication"][case][key]
            )
            / scale
            for case in ("baseline", "ragged")
        ]
        ax.bar(["Dense", "Ragged"], values, color=["#667085", "#07828b"])
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.20))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("8 x A100 | GLM-4.7-Flash | 512 questions x 4 trajectories")
    fig.savefig(figures / "performance.png", dpi=170)
    plt.close(fig)

    panels = [
        ("rollout/reward_scores_traj", "Trajectory reward", False),
        ("train/actor/policy_loss", "Policy loss (symlog)", True),
        ("train/actor/grad_norm", "Gradient norm before clipping (log)", True),
    ]
    for name, cases, title in (
        (
            "three_step",
            ["baseline", "ragged"],
            "Paired inputs; independently sampled trajectories",
        ),
        (
            "longrun",
            ["longrun"],
            "Ragged path: 46 rollout steps / 184 optimizer updates; no long-run control",
        ),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for ax, (key, label, update_metric) in zip(axes, panels):
            for case in cases:
                rows = [
                    row
                    for row in evidence["training_series"][case]
                    if key in row["metrics"]
                ]
                x = [
                    row["step"] / 4 + 1 if update_metric else row["step"] + 1
                    for row in rows
                ]
                y = [row["metrics"][key] for row in rows]
                ax.plot(
                    x,
                    y,
                    label=case,
                    linewidth=1,
                    marker="." if not update_metric else None,
                )
            if "grad_norm" in key:
                ax.set_yscale("log")
            elif "policy_loss" in key:
                ax.set_yscale("symlog", linthresh=1)
            ax.set(title=label, xlabel="Rollout step")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend()
        fig.suptitle(title)
        fig.savefig(figures / f"{name}.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
