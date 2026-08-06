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

"""Visualise a STEAM ensemble advantages parquet (offline diagnostics).

Reads ``meta/advantages_{tag}.parquet`` and writes distribution, per-member,
uncertainty, per-episode positive-rate, and episode-timeline plots plus a
``summary.json`` — no value model is loaded and nothing is recomputed; every
score comes from the parquet. See the STEAM pipeline docs for the output gallery
and CLI options.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_advantages_parquet(dataset_path: Path, tag: str) -> tuple[pd.DataFrame, Path]:
    parquet_path = dataset_path / "meta" / f"advantages_{tag}.parquet"
    if not parquet_path.exists():
        available = sorted(
            p.name for p in (dataset_path / "meta").glob("advantages*.parquet")
        )
        raise FileNotFoundError(
            f"Advantage parquet not found: {parquet_path}\nAvailable: {available}"
        )
    df = pd.read_parquet(parquet_path)
    required_cols = {
        "episode_index",
        "frame_index",
        "advantage",
        "advantage_continuous",
        "p_progress_mean",
        "p_progress_min",
        "p_progress_variance",
        "member_values",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet {parquet_path} missing required columns: {sorted(missing)}"
        )
    return df, parquet_path


def load_threshold_from_mixture(dataset_path: Path, tag: str) -> float | None:
    """Read positive_threshold from meta/mixture_config.yaml if present."""
    cfg_path = dataset_path / "meta" / "mixture_config.yaml"
    if not cfg_path.exists():
        return None
    import yaml

    with open(cfg_path, "r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return None
    tag_entry = (loaded.get("tags") or {}).get(tag)
    if not isinstance(tag_entry, dict):
        return None
    th = tag_entry.get("positive_threshold")
    return float(th) if th is not None else None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _stack_member_values(df: pd.DataFrame) -> np.ndarray:
    """Return a [K, N] array of per-member signed-progress values in [-1, 1]."""
    rows = [np.asarray(row, dtype=np.float32) for row in df["member_values"].tolist()]
    if not rows:
        raise RuntimeError("Empty parquet — nothing to plot")
    K = rows[0].shape[0]
    if any(r.shape[0] != K for r in rows):
        raise RuntimeError(
            "member_values rows have inconsistent length — every row must have "
            "the same ensemble size"
        )
    return np.stack(rows, axis=1)  # [K, N]


def _data_driven_xrange(values: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    """``(lo, hi)`` for histogram / axis bounds, padded so extreme bins stay
    visible and corrected values below ``-1`` are never clipped."""
    if values.size == 0:
        return -1.05, 1.05
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def plot_distribution(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    adv = df["advantage_continuous"].to_numpy()
    xlo, xhi = _data_driven_xrange(adv)
    ax.hist(
        adv,
        bins=80,
        range=(xlo, xhi),
        color="steelblue",
        edgecolor="black",
        alpha=0.75,
        label="advantage_continuous",
    )
    ax.axvline(
        x=float(np.mean(adv)),
        color="green",
        linestyle="-",
        linewidth=1.5,
        label=f"mean = {np.mean(adv):.4f}",
    )
    if threshold is not None:
        n_pos = int((adv > threshold).sum())
        ax.axvline(
            x=threshold,
            color="orange",
            linestyle="--",
            linewidth=2.0,
            label=f"threshold = {threshold:.3f}  (positive: {n_pos}/{len(adv)})",
        )
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("advantage_continuous (aggregated signed progress)")
    ax.set_ylabel("count")
    ax.set_title(f"Advantage distribution — {dataset_name}\ntag = {tag}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_member_distributions(
    df: pd.DataFrame, out_path: Path, dataset_name: str, tag: str
) -> None:
    members = _stack_member_values(df)  # [K, N]
    K = members.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 80)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(K, 3)))
    for k in range(K):
        ax.hist(
            members[k],
            bins=bins,
            histtype="step",
            linewidth=1.7,
            color=colors[k],
            label=f"member {k}  (mean={members[k].mean():.3f})",
        )
    ax.set_xlabel("signed progress")
    ax.set_ylabel("count")
    ax.set_title(
        f"Per-member signed-progress distributions — {dataset_name}\n"
        f"tag = {tag} (K = {K})"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_uncertainty_scatter(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
) -> None:
    mean = df["p_progress_mean"].to_numpy()
    var = df["p_progress_variance"].to_numpy()
    pos = df["advantage"].to_numpy().astype(bool)

    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(
        mean[~pos],
        var[~pos],
        s=4,
        alpha=0.5,
        color="tab:red",
        label="advantage = False",
    )
    ax_main.scatter(
        mean[pos], var[pos], s=4, alpha=0.5, color="tab:green", label="advantage = True"
    )
    if threshold is not None:
        ax_main.axvline(threshold, color="orange", linestyle="--", linewidth=1.4)
        ax_top.axvline(threshold, color="orange", linestyle="--", linewidth=1.4)
    ax_main.set_xlim(-1.0, 1.0)
    ax_main.set_ylim(0.0, max(float(var.max()) * 1.05, 1e-4))
    ax_main.set_xlabel("ensemble mean signed progress")
    ax_main.set_ylabel("ensemble variance of signed progress")
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    ax_top.hist(mean, bins=80, range=(-1.0, 1.0), color="steelblue", alpha=0.75)
    ax_top.set_ylabel("count")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, alpha=0.3)

    ax_right.hist(var, bins=60, orientation="horizontal", color="indianred", alpha=0.75)
    ax_right.set_xlabel("count")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(
        f"Ensemble disagreement vs mean — {dataset_name}\ntag = {tag}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_positive_rate_per_episode(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
) -> None:
    pos_per_ep = df.groupby("episode_index")["advantage"].mean().sort_index()
    counts_per_ep = df.groupby("episode_index").size()
    n_eps = len(pos_per_ep)
    fig, ax = plt.subplots(figsize=(max(8, n_eps * 0.18), 5))
    ax.bar(
        pos_per_ep.index,
        pos_per_ep.values,
        color=["tab:green" if v > 0.5 else "tab:red" for v in pos_per_ep.values],
        alpha=0.85,
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode_index")
    ax.set_ylabel("fraction of frames with advantage = True")
    ax.set_title(
        f"Positive rate per episode (threshold = "
        f"{threshold:.3f}) — {dataset_name}\ntag = {tag}"
        if threshold is not None
        else f"Positive rate per episode — {dataset_name}\ntag = {tag}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    # annotate frame count for short episodes (the rest get crowded)
    if n_eps <= 60:
        for ep, rate in zip(pos_per_ep.index, pos_per_ep.values):
            ax.text(
                ep,
                rate + 0.02,
                str(int(counts_per_ep.loc[ep])),
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _pick_representative_episodes(df: pd.DataFrame, n: int) -> list[int]:
    """Pick episodes spanning the spectrum of average ensemble disagreement,
    via evenly-spaced quantile picks over mean variance."""
    var_per_ep = df.groupby("episode_index")["p_progress_variance"].mean().sort_values()
    if len(var_per_ep) <= n:
        return [int(ep) for ep in var_per_ep.index.tolist()]
    quantile_idx = np.linspace(0, len(var_per_ep) - 1, n).round().astype(int)
    return [int(var_per_ep.index[i]) for i in quantile_idx]


def plot_episode_timelines(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
    n_episodes: int = 9,
) -> None:
    selected = _pick_representative_episodes(df, n_episodes)
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.5, rows * 3.0), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax_idx, ep in enumerate(selected):
        ax = axes.flat[ax_idx]
        ax.axis("on")
        sub = df[df["episode_index"] == ep].sort_values("frame_index")
        x = sub["frame_index"].to_numpy()
        members = np.stack(
            [np.asarray(r, dtype=np.float32) for r in sub["member_values"].tolist()],
            axis=1,
        )  # [K, T]
        member_min = members.min(axis=0)
        member_max = members.max(axis=0)
        mean = sub["p_progress_mean"].to_numpy()
        agg = sub["advantage_continuous"].to_numpy()
        ax.fill_between(
            x,
            member_min,
            member_max,
            color="steelblue",
            alpha=0.18,
            label="member min/max",
        )
        ax.plot(x, mean, color="steelblue", linewidth=1.2, label="member mean")
        ax.plot(
            x,
            agg,
            color="tab:purple",
            linewidth=1.5,
            linestyle="--",
            label="aggregated (worst-of-N)",
        )
        if threshold is not None:
            ax.axhline(threshold, color="orange", linestyle=":", linewidth=1.0)
        pos_rate = float(sub["advantage"].mean())
        ax.set_title(
            f"episode {ep}  (T={len(sub)}, pos_rate={pos_rate:.2f})", fontsize=10
        )
        ax.set_xlabel("frame_index")
        ax.set_ylabel("signed progress / advantage_continuous")
        # Data-driven y-range so out-of-range values are not clipped, but never
        # shrink below the native [-1, 1] envelope.
        y_lo, y_hi = _data_driven_xrange(
            np.concatenate([member_min, member_max, agg]), pad=0.02
        )
        ax.set_ylim(min(y_lo, -1.02), max(y_hi, 1.02))
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Episode timelines (sorted by mean ensemble variance) — {dataset_name}\n"
        f"tag = {tag}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------


def write_summary(
    df: pd.DataFrame,
    parquet_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
    out_path: Path,
) -> None:
    members = _stack_member_values(df)  # [K, N]
    summary: dict = {
        "dataset": dataset_name,
        "tag": tag,
        "parquet": str(parquet_path),
        "positive_threshold": threshold,
        "ensemble_size": int(members.shape[0]),
        "total_anchors": int(len(df)),
        "num_episodes": int(df["episode_index"].nunique()),
        "num_positive": int(df["advantage"].sum()),
        "positive_rate": float(df["advantage"].mean()),
        "advantage_continuous": {
            "mean": float(df["advantage_continuous"].mean()),
            "std": float(df["advantage_continuous"].std()),
            "min": float(df["advantage_continuous"].min()),
            "max": float(df["advantage_continuous"].max()),
        },
        "p_progress_mean": {
            "mean": float(df["p_progress_mean"].mean()),
            "std": float(df["p_progress_mean"].std()),
        },
        "p_progress_variance": {
            "mean": float(df["p_progress_variance"].mean()),
            "max": float(df["p_progress_variance"].max()),
        },
        "per_member_mean_p_progress": [
            float(members[k].mean()) for k in range(members.shape[0])
        ],
    }
    if "ensemble_signed_score" in df.columns:
        summary["ensemble_signed_score"] = {
            "mean": float(df["ensemble_signed_score"].mean()),
            "std": float(df["ensemble_signed_score"].std()),
            "min": float(df["ensemble_signed_score"].min()),
            "max": float(df["ensemble_signed_score"].max()),
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="LeRobot dataset path (the dir that contains meta/advantages_{tag}.parquet).",
    )
    parser.add_argument("--tag", type=str, required=True, help="Advantage tag.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the visualisation PNGs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override positive_threshold. Defaults to the value recorded in "
            "meta/mixture_config.yaml under tags.<tag>.positive_threshold."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=9,
        help="Number of episodes in the timeline grid (default: 9, fits 3x3).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)

    df, parquet_path = load_advantages_parquet(args.dataset, args.tag)
    threshold = (
        args.threshold
        if args.threshold is not None
        else load_threshold_from_mixture(args.dataset, args.tag)
    )
    if threshold is None:
        raise ValueError(
            f"positive_threshold not provided and not found under "
            f"tags.{args.tag} in meta/mixture_config.yaml"
        )

    dataset_name = args.dataset.name
    print(f"Loaded {len(df)} anchors from {parquet_path}")
    print(f"Threshold = {threshold}")
    print(f"Output dir = {args.output}")

    plot_distribution(
        df, args.output / "distribution.png", threshold, dataset_name, args.tag
    )
    plot_member_distributions(df, args.output / "members.png", dataset_name, args.tag)
    plot_uncertainty_scatter(
        df, args.output / "uncertainty.png", threshold, dataset_name, args.tag
    )
    plot_positive_rate_per_episode(
        df,
        args.output / "positive_rate_per_episode.png",
        threshold,
        dataset_name,
        args.tag,
    )
    plot_episode_timelines(
        df,
        args.output / "timeline_episodes.png",
        threshold,
        dataset_name,
        args.tag,
        n_episodes=args.episodes,
    )
    write_summary(
        df,
        parquet_path,
        threshold,
        dataset_name,
        args.tag,
        args.output / "summary.json",
    )

    print("Wrote:")
    for p in sorted(args.output.rglob("*")):
        if p.is_file():
            print(f"  {p}")


if __name__ == "__main__":
    main()
