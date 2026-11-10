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

"""Assembly of one group's payload into a training result, on the read side.

The payload carries the engine view (what was sent and generated, per
call); the finalize part carries the agent's verdicts (kept lengths,
per-trajectory rewards, continuity). This module turns the two into one
:class:`DynamicRolloutResult` in the layout every turn's generate call
produces one sequence — the layout the agent loop's
``get_rollout_result`` produced on the channel path.
"""

from typing import Any

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.workers.rollout.session_server.payload import GroupPayload, PayloadPart

__all__ = ["assemble_group_payload", "verify_group_assembly"]

# Compared against the agent-side reference, field by field. The version
# fields stay out: the reference carries None for them (the agent never saw
# per-sequence versions), so comparing them would only test the None.
_COMPARED_FIELDS = (
    "num_sequence",
    "group_size",
    "idx_to_traj",
    "input_ids",
    "prompt_lengths",
    "response_lengths",
    "rollout_logprobs",
    "rewards",
    "is_end",
    "extra_fields_train",
)


def assemble_group_payload(
    payload: GroupPayload, return_logprobs: bool
) -> DynamicRolloutResult:
    """Assemble one group's parts into a `DynamicRolloutResult`.

    One sequence per generate call: the sequence's input is the call's
    input ids followed by the kept part of its output, its logprobs and
    length fields sliced the same way, its reward the trajectory's reward
    from the finalize part, and its weight versions the call's
    `version_start` and `weight_version`. A turn the agent declared
    continuous with its predecessor stays its own sequence; continuity is
    only re-checked, not merged. A structurally inconsistent payload fails
    an assert instead of assembling wrong training data.

    Args:
        payload: The group's emitted container: the per-turn parts and the
            finalize part.
        return_logprobs: Whether the result must carry per-token logprobs;
            the parts must then have recorded them.

    Returns:
        The assembled result, without group identity — the caller sets it
        from the delivery it fetched the payload for.
    """
    finalize_parts = [part for part in payload.parts if part.last]
    assert len(finalize_parts) == 1, (
        f"a group payload must hold exactly one finalize part, group "
        f"{payload.group_id} has {len(finalize_parts)}"
    )
    finalize = finalize_parts[0]
    verdicts = finalize.verdicts
    assert verdicts is not None, (
        f"the finalize part of group {payload.group_id} carries no verdicts"
    )
    assert finalize.rewards is not None, (
        f"the finalize part of group {payload.group_id} carries no rewards"
    )

    parts = sorted(
        (part for part in payload.parts if not part.last),
        key=lambda part: (part.traj_index, part.turn_index),
    )
    turns_by_traj: dict[int, list[PayloadPart]] = {}
    for part in parts:
        turns_by_traj.setdefault(part.traj_index, []).append(part)

    idx_to_traj: list[int] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    input_ids: list[list[int]] = []
    rollout_logprobs = None
    if return_logprobs:
        rollout_logprobs = []
    is_end: list[bool] = []
    rewards: list[Any] = []
    version_start: list[int] = []
    version_end: list[int] = []

    extra_fields_train: dict[str, list] = {}
    declared_fields = None
    for traj_index in sorted(verdicts):
        verdict = verdicts[traj_index]
        traj_parts = turns_by_traj.get(traj_index, [])
        assert [part.turn_index for part in traj_parts] == list(
            range(len(traj_parts))
        ), (
            f"trajectory {traj_index} of group {payload.group_id} has turn "
            f"indexes {[part.turn_index for part in traj_parts]}, expected "
            f"consecutive from 0"
        )
        kept_lens = verdict["kept_lens"]
        continues_prev = verdict["continues_prev"]
        traj_fields = verdict.get("extra_fields_train", {})
        names = set(traj_fields)
        if declared_fields is None:
            declared_fields = names
        assert names == declared_fields, (
            f"the trajectories of group {payload.group_id} declare different "
            f"extra_fields_train names: {sorted(names)} vs "
            f"{sorted(declared_fields)}"
        )
        for name, values in traj_fields.items():
            assert len(values) == len(traj_parts), (
                f"extra_fields_train {name!r} of trajectory {traj_index} in "
                f"group {payload.group_id} covers {len(values)} turns, the "
                f"payload holds {len(traj_parts)}"
            )
            extra_fields_train.setdefault(name, []).extend(values)
        previous: PayloadPart | None = None
        previous_kept: int | None = None
        for part in traj_parts:
            kept = kept_lens.get(part.turn_index)
            assert kept is not None, (
                f"trajectory {traj_index} turn {part.turn_index} of group "
                f"{payload.group_id} has no declared kept length"
            )
            if previous is not None and continues_prev[part.turn_index]:
                # Cheap re-check of the write side's classification: a turn
                # declared continuous must start with the previous call's
                # input followed by its kept output.
                expected_prefix = (
                    previous.input_ids + previous.output_ids[:previous_kept]
                )
                assert part.input_ids[: len(expected_prefix)] == expected_prefix, (
                    f"trajectory {traj_index} turn {part.turn_index} of group "
                    f"{payload.group_id} is declared continuous but does not "
                    f"start with the previous call's input plus its kept "
                    f"output: starts with {part.input_ids[: len(expected_prefix)]!r}, "
                    f"expected {expected_prefix!r}"
                )
            previous = part
            previous_kept = kept

            response_ids = part.output_ids[:kept]
            idx_to_traj.append(traj_index)
            input_ids.append(part.input_ids + response_ids)
            prompt_lengths.append(len(part.input_ids))
            response_lengths.append(kept)
            if return_logprobs:
                assert part.logprobs is not None, (
                    f"trajectory {traj_index} turn {part.turn_index} of group "
                    f"{payload.group_id} has no logprobs but the result must "
                    f"carry them"
                )
                rollout_logprobs.append(part.logprobs[:kept])
            is_end.append(False)
            rewards.append(finalize.rewards[traj_index])
            version_start.append(part.version_start)
            version_end.append(part.weight_version)

    for name, values in extra_fields_train.items():
        assert len(values) == len(idx_to_traj), (
            f"extra_fields_train {name!r} of group {payload.group_id} covers "
            f"{len(values)} sequences, the payload holds {len(idx_to_traj)}"
        )
    return DynamicRolloutResult(
        num_sequence=len(idx_to_traj),
        group_size=len(verdicts),
        idx_to_traj=idx_to_traj,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        input_ids=input_ids,
        rollout_logprobs=rollout_logprobs,
        is_end=is_end,
        rewards=rewards,
        version_start=version_start or None,
        version_end=version_end or None,
        extra_fields_train=extra_fields_train,
    )


def verify_group_assembly(result: DynamicRolloutResult, verify: dict, cfg) -> None:
    """Compare an assembled result against the agent-side reference.

    The reference rode with the group's last close; the finalize part
    carried it together with the sampling params the server forwarded.
    Every compared field must match exactly, and each forwarded param set
    must equal the config defaults merged with the caller's per-turn
    override — a mismatch means the two paths would train on different
    data or sample from different distributions. A disagreement fails an
    assert naming the field or the call.

    Args:
        result: The payload-assembled result.
        verify: The finalize part's verify record: ``result`` (the
            reference fields), ``caller_sampling_params`` and
            ``merged_sampling_params``, each mapping trajectory to the
            per-turn param dicts.
        cfg: The run config; its sampling-params block, resolved the same
            way a rollout worker resolves it, is merged over the caller's
            overrides to build the expectation.
    """
    reference = verify["result"]
    for field in _COMPARED_FIELDS:
        expected = reference[field]
        actual = getattr(result, field)
        assert actual == expected, (
            f"assembled {field} disagrees with the agent reference: "
            f"assembled {actual!r}, reference {expected!r}"
        )
    # Imported here: the sglang worker module is only importable where the
    # rollout stack is installed, and this check never runs without it.
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    defaults = SGLangWorker.get_default_sampling_params(cfg)
    caller = {
        int(traj): turns for traj, turns in verify["caller_sampling_params"].items()
    }
    forwarded = {
        int(traj): turns for traj, turns in verify["merged_sampling_params"].items()
    }
    assert set(caller) == set(forwarded), (
        f"the verify record's caller and forwarded sampling params cover "
        f"different trajectories: caller {sorted(caller)}, forwarded "
        f"{sorted(forwarded)}"
    )
    for traj, caller_turns in caller.items():
        forwarded_turns = forwarded[traj]
        assert len(caller_turns) == len(forwarded_turns), (
            f"trajectory {traj} declared {len(caller_turns)} generate calls "
            f"but the server forwarded {len(forwarded_turns)}"
        )
        for turn_index, (asked, got) in enumerate(zip(caller_turns, forwarded_turns)):
            expected = dict(defaults)
            if asked is not None:
                expected.update(asked)
            assert got == expected, (
                f"sampling params forwarded for trajectory {traj} turn "
                f"{turn_index} disagree with the config defaults merged "
                f"with the caller's override: forwarded {got!r}, expected "
                f"{expected!r}"
            )
