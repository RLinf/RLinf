# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot WAM PPO sampled and analytic KL at each optimizer step of a rollout."""

import argparse
import csv
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

METRIC_FIELDS = ("chunks", "approx_kl", "ratio", "ratio_abs", "ratio_abs_max")
CSV_FIELDS = (
    "global_step",
    "optimizer_step",
    "total_optimizer_steps",
    "status",
    "analytic_kl",
    *METRIC_FIELDS,
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class OptimizerStepKL:
    analytic_kl: float
    status: str
    metrics: dict[str, str]

    @property
    def approx_kl(self) -> float:
        return float(self.metrics.get("approx_kl", "nan"))


@dataclass
class RolloutKL:
    total_steps: int
    points: dict[int, OptimizerStepKL] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.points) == self.total_steps + 1


def _read_log_fields(lines: Iterator[str]) -> Iterator[dict[str, str]]:
    for line in lines:
        for payload in line.split("[WAM PPO]")[1:]:
            fields = dict(
                token.split("=", 1)
                for token in ANSI_ESCAPE.sub(" ", payload).split()
                if "=" in token
            )
            try:
                fields["global_step"] = str(int(fields["rollout"]) + 1)
                fields["optimizer_step"], fields["total_optimizer_steps"] = fields[
                    "optimizer_step"
                ].split("/")
            except (KeyError, ValueError):
                continue
            yield fields


def read_optimizer_kl(input_path: Path) -> dict[int, RolloutKL]:
    """Read optimizer diagnostics from the original log or its CSV export."""
    rollouts: dict[int, RolloutKL] = {}
    from_csv = input_path.suffix.lower() == ".csv"
    with input_path.open(encoding="utf-8", errors="replace", newline="") as source:
        rows = csv.DictReader(source) if from_csv else _read_log_fields(source)
        for fields in rows:
            try:
                global_step = int(fields["global_step"])
                step = int(fields["optimizer_step"])
                total_steps = int(fields["total_optimizer_steps"])
                kl = float(fields["analytic_kl"])
            except (KeyError, TypeError, ValueError):
                continue
            if global_step < 1 or not 0 <= step <= total_steps:
                continue
            if global_step not in rollouts or (step == 0 and not from_csv):
                rollouts[global_step] = RolloutKL(total_steps)
            metrics = {}
            for name in METRIC_FIELDS:
                value = fields.get(name)
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue
                metrics[name] = value
            rollouts[global_step].points[step] = OptimizerStepKL(
                kl, fields.get("status") or "", metrics
            )
    return rollouts


@contextmanager
def _atomic_output(output: Path) -> Iterator[Path]:
    with NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as file:
        temporary_path = Path(file.name)
    try:
        yield temporary_path
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_optimizer_csv(rollouts: dict[int, RolloutKL], output: Path) -> None:
    """Write one row for each recorded optimizer step."""
    with (
        _atomic_output(output) as temporary_path,
        temporary_path.open("w", encoding="utf-8", newline="") as file,
    ):
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for global_step, series in sorted(rollouts.items()):
            for step, point in sorted(series.points.items()):
                writer.writerow(
                    {
                        "global_step": global_step,
                        "optimizer_step": step,
                        "total_optimizer_steps": series.total_steps,
                        "status": point.status,
                        "analytic_kl": point.analytic_kl,
                        **point.metrics,
                    }
                )


def plot_optimizer_kl(rollouts: dict[int, RolloutKL], output: Path) -> None:
    """Save the sampled and analytic KL curves for each rollout as a PNG."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    panels = [
        ("analytic_kl", "Analytic KL", "Analytic KL vs pre-update actor policy"),
    ]
    if any(
        math.isfinite(point.approx_kl)
        for series in rollouts.values()
        for point in series.points.values()
    ):
        panels.insert(
            0,
            (
                "approx_kl",
                "Sampled approx_kl",
                "Sampled approx_kl vs original rollout logprobs\n"
                "Fixed probe scored after each update",
            ),
        )
    figure = Figure(figsize=(9.5, 7.8 if len(panels) == 2 else 5.2), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(nrows=len(panels), sharex=True, squeeze=False)[:, 0]
    use_legend = len(rollouts) <= 12
    cmap = colormaps["viridis"]
    if use_legend:
        cmap = colormaps["tab10" if len(rollouts) <= 10 else "tab20"]
    norm = Normalize(vmin=min(rollouts), vmax=max(rollouts))
    incomplete_count = sum(not series.is_complete for series in rollouts.values())
    has_skipped_steps = False
    sorted_rollouts = sorted(rollouts.items())
    for axis, (metric, ylabel, title) in zip(axes, panels):
        axis.axhline(0, color="0.4", linewidth=0.9, linestyle="--", zorder=0)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
        for index, (global_step, series) in enumerate(sorted_rollouts):
            points = series.points
            steps = range(max(points) + 1)
            values = [
                getattr(points[step], metric) if step in points else math.nan
                for step in steps
            ]
            color = cmap(index) if use_legend else cmap(norm(global_step))
            label = str(global_step) if series.is_complete else f"{global_step} (partial)"
            axis.plot(steps, values, color=color, marker=".", linewidth=1.3, label=label)
            skipped = [
                (step, values[step])
                for step, point in points.items()
                if point.status.startswith("skipped") and math.isfinite(values[step])
            ]
            if skipped:
                has_skipped_steps = True
                axis.scatter(*zip(*skipped), color=color, marker="x", zorder=4)

    title = "WAM PPO optimizer KL by global step"
    notes = []
    if incomplete_count and not use_legend:
        notes.append(f"{incomplete_count} partial rollout(s)")
    if has_skipped_steps:
        notes.append("× Skipped optimizer step")
    if notes:
        title += "\n" + " · ".join(notes)
    figure.suptitle(title)
    axes[-1].set_xlabel("Optimizer step within rollout (0 = before update)")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    if use_legend:
        axes[0].legend(title="Global Step", loc="upper left", bbox_to_anchor=(1.02, 1))
    else:
        colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes)
        colorbar.set_label("Global Step")
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
    with _atomic_output(output) as temporary_path:
        figure.savefig(temporary_path, format="png", dpi=160)


def main() -> None:
    """Plot optimizer KL and export the raw log as a CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory, log file, or CSV export")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()
    input_path = args.run
    if input_path.is_dir():
        input_path = args.run / "run_embodiment.log"
        csv_fallback = args.run / "optimizer_kl.csv"
        if not input_path.exists() and csv_fallback.is_file():
            input_path = csv_fallback
    output = args.output or input_path.with_name("optimizer_kl.png")
    csv_output = input_path.with_name("optimizer_kl.csv")
    export_csv = input_path.suffix.lower() != ".csv"
    try:
        rollouts = read_optimizer_kl(input_path)
        if not any(
            math.isfinite(point.analytic_kl) or math.isfinite(point.approx_kl)
            for series in rollouts.values()
            for point in series.points.values()
        ):
            parser.error(f"No finite WAM PPO KL values in {input_path}")
        if output.resolve() == input_path.resolve() or (
            export_csv and output.resolve() == csv_output.resolve()
        ):
            parser.error("The PNG path must differ from the input and CSV paths.")
        if export_csv:
            write_optimizer_csv(rollouts, csv_output)
        plot_optimizer_kl(rollouts, output)
        point_count = sum(len(series.points) for series in rollouts.values())
        complete = sum(series.is_complete for series in rollouts.values())
        print(
            f"Plotted {len(rollouts)} global steps ({complete} complete, "
            f"{len(rollouts) - complete} partial) and {point_count} optimizer records "
            f"from {input_path}."
        )
        print(f"Saved {output}")
        if export_csv:
            print(f"Saved {csv_output}")
    except ModuleNotFoundError as error:
        parser.error(f"Plotting requires Matplotlib ({error}).")
    except OSError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
