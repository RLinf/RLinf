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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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


class SimulatorCohort:
    """Own simulator quotas and RPC checks for the runner's current batch."""

    def __init__(self, cfg, simulators):
        self.cfg = cfg
        self.simulators = simulators
        self.group_size = cfg.algorithm.group_size
        self.ready_timeout = cfg.tools.rpent.get("full_cohort_ready_timeout_s")
        self.groups = []
        self.selected = Counter()

    def validate(self):
        validate_routers(
            self.simulators, self.group_size, strict=self.ready_timeout is not None
        )

    def sampler(self, dataset):
        from rlinf.agents.rpent.simulators import SimulatorBatchSampler

        return SimulatorBatchSampler(
            [len(pool) for pool in dataset.datasets],
            [entry.sampling_groups for entry in self.simulators.values()],
            shuffle=self.cfg.data.shuffle,
            seed=self.cfg.data.get("seed", 1),
        )

    def prepare(self, batch, requests):
        from rlinf.agents.rpent.simulators import simulator_for_answer

        self.groups = [simulator_for_answer(answer) for answer in batch["answer"]]
        self.selected.clear()
        expected = {
            name: domain.sampling_groups for name, domain in self.simulators.items()
        }
        actual = Counter(self.groups)
        if actual != expected:
            raise ValueError(
                f"Mixed rollout expected task groups {expected}, got {dict(actual)}"
            )
        if self.ready_timeout is not None:
            capacity = self.cfg.tools.rpent.episodes_per_worker
            needed = max(len(request.input_ids) * request.n for request in requests)
            if capacity < needed:
                raise ValueError(
                    f"Full cohort requires {needed} planner slots per worker, got {capacity}"
                )

    def wait(self, cohort_id, agent_loop):
        if self.ready_timeout is None:
            return None
        ready = wait_until_ready(
            cohort_id, self.simulators, self.group_size, self.ready_timeout
        )
        agent_loop.release_rpent_rollout(cohort_id).wait()
        return ready

    def select(self, group_id):
        domain = self.groups[group_id]
        if self.selected[domain] == self.simulators[domain].rollout_groups:
            return False
        self.selected[domain] += 1
        return True

    def verify_release(self, cohort_id, cleanup):
        owners = defaultdict(list)
        for worker in cleanup:
            for candidate in worker["candidates"]:
                name = self.groups[candidate["candidate_id"] // self.group_size]
                owners[name].extend(
                    Path(path).name for path in candidate.get("episode_paths", ())
                )
        return verify_released(cohort_id, self.simulators, owners)

    def record_selection(self, audit):
        audit.update(
            group_simulators=self.groups,
            selected_groups_by_simulator=dict(self.selected),
        )
