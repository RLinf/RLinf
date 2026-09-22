# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check an SO-101 RLinf policy service with a deterministic zero request."""

from __future__ import annotations

import argparse

import numpy as np

from rlinf.workers.rollout.grpc.grpc_policy_adapter import GRPCPolicyAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default="127.0.0.1:50051")
    parser.add_argument("--policy-id", default="so101-pi05")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--task-description",
        default="抓取青色目标物体并放到盒子里面",
        help="Task text sent with the deterministic policy request.",
    )
    args = parser.parse_args()

    policy = GRPCPolicyAdapter(
        args.server_address,
        action_dim=6,
        num_action_chunks=20,
        policy_id=args.policy_id,
        timeout=args.timeout,
    )
    try:
        actions, _ = policy.predict_action_batch(
            {
                "states": np.zeros((1, 6), dtype=np.float32),
                "main_images": np.zeros((1, 224, 224, 3), dtype=np.uint8),
                "task_descriptions": [args.task_description],
            }
        )
        values = actions.numpy()
        if tuple(values.shape) != (1, 20, 6):
            raise RuntimeError(f"unexpected action shape: {values.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError("policy returned non-finite actions")
        print(
            f"PASS gRPC {args.server_address}: shape={values.shape} "
            f"dtype={values.dtype} max_abs={np.abs(values).max():.6f}"
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
