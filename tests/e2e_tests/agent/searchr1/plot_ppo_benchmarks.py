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

"""Report reward trends, optimizer counts, and complete-run training speed."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_run(directory: Path, expected_steps: int) -> tuple[dict, np.ndarray]:
    """Read complete event histories and validate async ownership invariants."""
    result = json.loads((directory / "benchmark.json").read_text())
    assert result["steps"] == expected_steps, result
    events = EventAccumulator(
        str(directory / "tensorboard"), size_guidance={"scalars": 0}
    )
    events.Reload()
    points = events.Scalars("rollout/reward_scores_traj")
    assert [point.step for point in points] == list(range(expected_steps))
    rewards = np.array([point.value for point in points])
    assert np.isfinite(rewards).all()
    window = min(10, expected_steps // 2)
    rng = np.random.default_rng(2026)
    differences = rng.choice(rewards[-window:], (10000, window)).mean(
        axis=1
    ) - rng.choice(rewards[:window], (10000, window)).mean(axis=1)
    interval = np.quantile(differences, [0.025, 0.975])
    result.update(
        {
            "first_reward_mean": float(rewards[:window].mean()),
            "last_reward_mean": float(rewards[-window:].mean()),
            "reward_improvement_ci95": interval.tolist(),
            "reward_increased": bool(interval[0] > 0),
            "window": window,
        }
    )
    actor_updates = events.Scalars("train/actor/update_success")
    assert [point.step for point in actor_updates] == list(range(expected_steps))
    result["actor_successful_updates"] = sum(point.value for point in actor_updates)
    result["actor_skipped_learner_steps"] = [
        point.step + 1 for point in actor_updates if not point.value
    ]
    critic_tag = (
        "train/critic/update_success"
        if result["synchronous"]
        else "critic_update/critic/update_success"
    )
    critic_updates = events.Scalars(critic_tag)
    assert len(critic_updates) == expected_steps * (1 if result["synchronous"] else 2)
    result["critic_successful_updates"] = sum(point.value for point in critic_updates)
    result["critic_skipped_optimizer_calls"] = [
        index + 1 for index, point in enumerate(critic_updates) if not point.value
    ]
    if not result["synchronous"]:
        trace = [
            json.loads(line)
            for line in (directory / "async_trace.jsonl").read_text().splitlines()
        ]
        sample_ids = [sample for step in trace for sample in step["sample_ids"]]
        assert sorted(sample_ids) == list(range(expected_steps * result["batch_size"]))
        assert all(
            step["oldest_consumed_version"]
            <= step["consumed_version"]
            <= step["published_version"]
            for step in trace
        )
        result["generated_during_training"] = sum(
            step["generated_during_training"] for step in trace
        )
        result["retracted_requests"] = sum(step["retracted_requests"] for step in trace)
        assert trace[-1]["actor_updates"] == result["actor_successful_updates"]
        assert trace[-1]["critic_updates"] == result["critic_successful_updates"]
    return result, rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="searchr1-ppo-bsz1024-200")
    parser.add_argument("--output-dir", type=Path, default=Path("logs"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["shared", "split"],
        default=["shared", "split"],
    )
    args = parser.parse_args()
    modes = [
        f"{execution}-{layout}"
        for layout in args.layouts
        for execution in ("sync", "async")
    ]
    report_dir = args.output_dir / args.name
    report_dir.mkdir(parents=True, exist_ok=True)
    results, curves = {}, {}
    for mode in modes:
        results[mode], curves[mode] = read_run(
            args.output_dir / f"{args.name}-{mode}", args.steps
        )
    batch_sizes = {result["batch_size"] for result in results.values()}
    if len(batch_sizes) != 1:
        raise ValueError(f"Expected matching batch sizes, found {sorted(batch_sizes)}.")
    batch_size = batch_sizes.pop()
    if "shared" in args.layouts:
        assert results["async-shared"]["generated_during_training"] == 0
        assert results["async-shared"]["retracted_requests"] > 0
    if "split" in args.layouts:
        assert results["async-split"]["generated_during_training"] > 0
    speedups = {
        layout: results[f"sync-{layout}"]["training_seconds"]
        / results[f"async-{layout}"]["training_seconds"]
        for layout in args.layouts
    }
    accepted = all(run["reward_increased"] for run in results.values()) and all(
        value > 1 for value in speedups.values()
    )
    (report_dir / "results.json").write_text(
        json.dumps(
            {"runs": results, "speedups": speedups, "accepted": accepted}, indent=2
        )
        + "\n"
    )
    fig, axes = plt.subplots(
        1,
        len(args.layouts),
        figsize=(6 * len(args.layouts), 4),
        sharey=True,
        squeeze=False,
    )
    for axis, layout in zip(axes[0], args.layouts, strict=True):
        for execution, color in [("sync", "#2878b5"), ("async", "#e57c23")]:
            reward = curves[f"{execution}-{layout}"]
            axis.plot(
                np.arange(1, len(reward) + 1),
                reward,
                color=color,
                alpha=0.15,
                linewidth=0.8,
            )
            axis.plot(
                np.arange(10, len(reward) + 1),
                np.convolve(reward, np.ones(10) / 10, mode="valid"),
                color=color,
                label="PPO (sync)" if execution == "sync" else "SAO (async)",
            )
        axis.set(
            title=f"{layout.capitalize()} GPUs", xlabel="Learner step", ylim=(0, 1)
        )
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0, 0].set_ylabel("Training reward (10-batch mean)")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(report_dir / f"reward.{suffix}", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(
        modes,
        [results[mode]["training_seconds"] / 60 for mode in modes],
        color=["#2878b5", "#e57c23"] * len(args.layouts),
    )
    axis.bar_label(bars, fmt="%.1f min")
    axis.set(
        ylabel="Elapsed minutes",
        title=f"{args.steps} learner steps, batch {batch_size}, eight A100 GPUs",
    )
    axis.set_ylim(0, max(bar.get_height() for bar in bars) * 1.15)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(report_dir / f"training_time.{suffix}", dpi=180)
    plt.close(fig)

    lines = [
        "# Search-R1 PPO comparison",
        "",
        f"{args.steps} learner steps per run; {batch_size} prompts per batch; eight A100-SXM4-80GB GPUs.",
        "",
        "| Run | First 10 reward | Last 10 reward | Improvement 95% CI | Training minutes | Actor / critic updates |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for mode, result in results.items():
        low, high = result["reward_improvement_ci95"]
        lines.append(
            f"| {mode} | {result['first_reward_mean']:.4f} | {result['last_reward_mean']:.4f} | [{low:.4f}, {high:.4f}] | {result['training_seconds'] / 60:.2f} | {result['actor_successful_updates']:.0f} / {result['critic_successful_updates']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Async speedup: "
            + "; ".join(
                f"{layout} {speedup:.3f}×" for layout, speedup in speedups.items()
            )
            + ".",
            "",
            "These are single seeded runs. Small timing gains do not establish a speed advantage across repeated runs or seeds.",
            "",
            "Timing includes initial rollout/buffer fill, offload/onload, training, weight transfers, and checkpoints every 50 steps and at the end. Model initialization is reported separately in results.json.",
            "",
            "Reward intervals bootstrap the first and last ten batch means independently with 10,000 resamples. These are training rewards, not held-out evaluation scores.",
            "",
            "Both modes use the same model, prompt budget, seeds, raw rollout logprobs, sampling, packing, learning rates, role offloading, and checkpoint cadence. SAO uses DIS, adaptive GAE, unclipped value regression, and two critic optimizer calls per actor optimizer call.",
            "",
            "Update counts show successful optimizer steps. FP16 dynamic loss scaling can skip an optimizer call without discarding the learner batch; results.json records skipped calls using one-based indices.",
            "",
            "![Reward curves](reward.png)",
            "",
            "![Training time](training_time.png)",
            "",
            "An AI assistant was used to implement and validate this change.",
            "",
            f"Acceptance criteria met: {accepted}.",
        ]
    )
    trial_path = report_dir / "trials.json"
    if trial_path.is_file():
        lines.extend(["", "Earlier trials are preserved in trials.json:", ""])
        for trial in json.loads(trial_path.read_text()):
            lines.append(
                f"- {trial['mode']}, concurrency {trial['max_in_flight_rollouts']}: "
                f"{trial['training_seconds'] / 60:.2f} minutes, "
                f"{trial['speedup_over_sync']:.4f}× sync speed. {trial['reason']}"
            )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    if not accepted:
        raise SystemExit(
            "At least one reward trend or speed comparison did not meet acceptance; see results.json."
        )


if __name__ == "__main__":
    main()
