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

"""Check per-simulator admission, native readiness and cleanup for RPent cohorts."""

import time
from concurrent.futures import ThreadPoolExecutor


def rpc(endpoint: str, method: str, *args, timeout_s: float = 10):
    """Call one operation on the simulator owner."""
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    client = HttpRpcClient(endpoint)
    try:
        return client.call(method, args=args, timeout_s=timeout_s)
    finally:
        client.close()


def validate_routers(simulators, group_size: int, *, strict: bool) -> None:
    """Verify router contract and capacity before scheduling any environment."""
    for name, domain in simulators.items():
        if not domain.group_router:
            if strict:
                raise ValueError(
                    f"{name} needs a toolkit group router for full readiness"
                )
            continue
        groups = rpc(domain.endpoint, "backend.group_status")
        if groups["group_size"] != group_size:
            raise ValueError(
                f"{name} router group size {groups['group_size']} differs from {group_size}"
            )
        if strict and groups["capacity"] < domain.sampling_groups:
            raise ValueError(
                f"{name} needs {domain.sampling_groups} concurrent groups, router has {groups['capacity']}"
            )


def wait_until_ready(
    cohort: str, simulators, group_size: int, timeout_s: float
) -> dict:
    """Wait for real native processes belonging to this cohort, before generation."""
    deadline = time.monotonic() + timeout_s
    last = {}
    with ThreadPoolExecutor(max_workers=len(simulators)) as pool:
        while (remaining := deadline - time.monotonic()) > 0:

            def measure(domain):
                args = (cohort,)
                return rpc(
                    domain.endpoint,
                    "backend.status",
                    *args,
                    timeout_s=min(remaining, 10),
                )

            last = dict(
                zip(simulators, pool.map(measure, simulators.values()), strict=True)
            )
            if all(
                row["ready"]
                == row["reserved"]
                == simulators[name].sampling_groups * group_size
                for name, row in last.items()
            ):
                return last
            time.sleep(min(remaining, 0.5))
    observed = {name: row["ready"] for name, row in last.items()}
    raise TimeoutError(
        f"Timed out waiting for native cohort {cohort}; ready slots: {observed}"
    )


def verify_released(
    cohort: str, simulators, direct_owners: dict[str, list[str]]
) -> dict:
    """Check native slots after agent cleanup and the generation fence."""
    snapshots = {}
    for name, domain in simulators.items():
        if domain.group_router:
            status = rpc(domain.endpoint, "backend.status", cohort)
            groups = rpc(domain.endpoint, "backend.group_status", cohort)
            if groups["waiting"] or groups["admitted"]:
                raise RuntimeError(f"{name} still owns cohort groups: {groups}")
        else:
            status = rpc(domain.endpoint, "backend.status", direct_owners.get(name, []))
        if status["reserved"] or status["ready"]:
            raise RuntimeError(f"{name} still owns simulator slots: {status}")
        snapshots[name] = status
    return snapshots
