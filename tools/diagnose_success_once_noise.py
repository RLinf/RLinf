# Copyright 2025 The RLinf Authors.
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

"""Diagnose whether fluctuations in ``success_once`` are statistical noise.

``eval/env/success_once`` is logged as the plain mean of a binary outcome
over all evaluation episodes (``rlinf.utils.metric_utils.compute_evaluate_metrics``).
With a small evaluation batch this mean is a binomial proportion, so it
oscillates between consecutive validation steps even when the policy has not
changed in any meaningful way.

This script quantifies that noise:

1. Given the number of evaluation episodes and an observed success rate, it
   prints the standard deviation and confidence interval of the estimate.
2. Given two consecutive evaluation results, it runs a two-proportion z-test
   to decide whether the change is statistically significant.
3. It recommends an episode count for a target confidence-interval width.

Examples
--------
Noise band of a single 0.55 success rate over 16 episodes::

    python tools/diagnose_success_once_noise.py single --rate 0.55 --episodes 16

Compare two consecutive eval points (0.50 vs 0.60, 16 episodes each)::

    python tools/diagnose_success_once_noise.py compare \
        --rate-a 0.50 --episodes-a 16 --rate-b 0.60 --episodes-b 16

How many episodes for a +/-0.03 confidence interval?::

    python tools/diagnose_success_once_noise.py plan --target-width 0.06 --rate 0.55
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from a source checkout without installing
# the package: python tools/diagnose_success_once_noise.py ...
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ruff: noqa: E402 (repo root must be added to sys.path before this import)
from rlinf.utils.eval_stats import (  # noqa: E402
    episodes_needed_for_width,
    is_within_noise,
    normal_interval,
    success_rate_std,
    two_proportion_z_test,
    wilson_interval,
)


def _add_rate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Observed mean success rate (e.g. the logged eval/env/success_once value).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help=(
            "Number of evaluation episodes the rate was averaged over. In RLinf "
            "this is the logged eval/num_trajectories, i.e. "
            "env.eval.total_num_envs * env.eval.rollout_epoch * num rollout stages."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single", help="Noise band of a single success-rate estimate."
    )
    _add_rate_args(single)
    single.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95).",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Two-proportion z-test between two consecutive evaluation results.",
    )
    compare.add_argument("--rate-a", type=float, required=True)
    compare.add_argument("--episodes-a", type=int, required=True)
    compare.add_argument("--rate-b", type=float, required=True)
    compare.add_argument("--episodes-b", type=int, required=True)
    compare.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95).",
    )

    plan = subparsers.add_parser(
        "plan",
        help="Episode count needed for a target confidence-interval width.",
    )
    plan.add_argument(
        "--target-width",
        type=float,
        required=True,
        help="Target full width of the confidence interval (e.g. 0.06 for +/-0.03).",
    )
    plan.add_argument(
        "--rate",
        type=float,
        default=0.5,
        help="Expected success rate used for planning (default: 0.5, the conservative worst case).",
    )
    plan.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95).",
    )

    args = parser.parse_args(argv)

    if args.command == "single":
        std = success_rate_std(args.rate, args.episodes)
        wald = normal_interval(args.rate, args.episodes, args.confidence)
        wilson = wilson_interval(args.rate, args.episodes, args.confidence)
        print(f"success rate      : {args.rate:.4f}")
        print(f"episodes          : {args.episodes}")
        print(f"std of estimate   : ±{std:.4f}")
        print(
            f"{args.confidence:.0%} Wald CI    : "
            f"[{max(0.0, wald[0]):.4f}, {min(1.0, wald[1]):.4f}]"
        )
        print(f"{args.confidence:.0%} Wilson CI  : [{wilson[0]:.4f}, {wilson[1]:.4f}]")
        print(
            "\nInterpretation: run-to-run fluctuations of this metric that stay "
            "inside the interval above are expected sampling noise, not "
            "necessarily training instability."
        )
    elif args.command == "compare":
        z, p_value = two_proportion_z_test(
            args.rate_a, args.episodes_a, args.rate_b, args.episodes_b
        )
        noise = is_within_noise(
            args.rate_a,
            args.episodes_a,
            args.rate_b,
            args.episodes_b,
            args.confidence,
        )
        print(f"rate A = {args.rate_a:.4f} over {args.episodes_a} episodes")
        print(f"rate B = {args.rate_b:.4f} over {args.episodes_b} episodes")
        print(f"two-proportion z = {z:.4f}, two-sided p = {p_value:.4f}")
        if noise:
            print(
                f"Result: the difference is NOT statistically significant at the "
                f"{args.confidence:.0%} level. Treat the fluctuation as sampling "
                f"noise; consider a larger eval batch before concluding instability."
            )
        else:
            print(
                f"Result: the difference IS statistically significant at the "
                f"{args.confidence:.0%} level. The change is unlikely to be pure "
                f"sampling noise; investigate training stability."
            )
    else:  # plan
        half_width = args.target_width / 2.0
        n = episodes_needed_for_width(half_width, args.rate, args.confidence)
        print(
            f"To estimate a success rate of {args.rate:.2f} with a "
            f"{args.confidence:.0%} CI of width {args.target_width:.4f} "
            f"(±{half_width:.4f}):"
        )
        print(f"  -> evaluate at least {n} episodes per validation step.")
        print(
            "In RLinf, set env.eval.total_num_envs and/or env.eval.rollout_epoch "
            "so that their product meets this episode count."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
