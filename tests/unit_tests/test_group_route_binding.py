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

import pytest

from rlinf.scheduler.worker.routing import GroupRouteBinding, balanced_group_id


@pytest.mark.parametrize(
    ("env_world_size", "rollout_world_size", "expected"),
    [
        (2, 1, [0, 0]),
        (4, 1, [0, 0, 0, 0]),
        (4, 3, [0, 0, 1, 2]),
        (8, 3, [0, 0, 0, 1, 1, 1, 2, 2]),
    ],
)
def test_balanced_group_id_covers_every_rollout(
    env_world_size: int,
    rollout_world_size: int,
    expected: list[int],
) -> None:
    group_ids = [
        balanced_group_id(env_rank, env_world_size, rollout_world_size)
        for env_rank in range(env_world_size)
    ]

    assert group_ids == expected
    assert set(group_ids) == set(range(rollout_world_size))


def test_disabled_binding_preserves_global_pool() -> None:
    env_binding = GroupRouteBinding.for_env(
        enabled=False,
        decoupled_mode=False,
        env_rank=0,
        env_world_size=4,
        rollout_world_size=1,
    )
    rollout_binding = GroupRouteBinding.for_rollout(
        enabled=False,
        decoupled_mode=False,
        rollout_rank=0,
        env_world_size=4,
        rollout_world_size=1,
    )

    assert env_binding.route_key is None
    assert rollout_binding.route_key is None


@pytest.mark.parametrize("role", ["env", "rollout"])
def test_enabled_binding_requires_decoupled_mode(role: str) -> None:
    kwargs = {
        "enabled": True,
        "decoupled_mode": False,
        "env_world_size": 4,
        "rollout_world_size": 1,
    }

    with pytest.raises(
        ValueError,
        match=r"enable_group_route_binding=true requires .*enable_decoupled_mode=true",
    ):
        if role == "env":
            GroupRouteBinding.for_env(env_rank=0, **kwargs)
        else:
            GroupRouteBinding.for_rollout(rollout_rank=0, **kwargs)


def test_train_eval_request_reply_route_keys_match() -> None:
    env_world_size = 8
    rollout_world_size = 3

    for env_rank in range(env_world_size):
        env_binding = GroupRouteBinding.for_env(
            enabled=True,
            decoupled_mode=True,
            env_rank=env_rank,
            env_world_size=env_world_size,
            rollout_world_size=rollout_world_size,
        )
        rollout_binding = GroupRouteBinding.for_rollout(
            enabled=True,
            decoupled_mode=True,
            rollout_rank=env_binding.group_id,
            env_world_size=env_world_size,
            rollout_world_size=rollout_world_size,
        )

        # The mode and direction do not alter the logical queue key. Worker
        # train/eval request and reply sites all consume this same property.
        train_request_key = env_binding.route_key
        train_reply_key = rollout_binding.route_key
        eval_request_key = env_binding.route_key
        eval_reply_key = rollout_binding.route_key
        assert {
            train_request_key,
            train_reply_key,
            eval_request_key,
            eval_reply_key,
        } == {f"grp{env_binding.group_id}"}


def test_group_route_rejects_empty_rollout_groups() -> None:
    with pytest.raises(ValueError, match="env_world_size >= rollout_world_size"):
        balanced_group_id(
            env_rank=0,
            env_world_size=2,
            rollout_world_size=3,
        )
