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

import base64
import io
from typing import Any, Optional

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.workers.rollout.session_server.payload import GroupPayload, PayloadPart

__all__ = ["assemble_group_payload", "verify_group_assembly"]


def _decode_png_b64(encoded: str) -> Any:
    """Decode one base64-encoded image string into an RGB PIL image."""
    from PIL import Image

    raw = base64.b64decode(encoded)
    with Image.open(io.BytesIO(raw)) as img:
        return img.convert("RGB")


def _assemble_multimodal_trajectory(
    group_id: int,
    traj_index: int,
    trained_turns: list[PayloadPart],
    kept_lens: dict,
    ended: dict,
    reward: Any,
    return_logprobs: bool,
    processor: Any,
    response_budget: Optional[int],
) -> dict[str, Any]:
    """Fold one multimodal streaming trajectory into a single episode sequence."""
    assert processor is not None, (
        f"group {group_id} trajectory {traj_index} carries multimodal delta_text "
        "parts, which require a HuggingFace processor for read-side assembly"
    )
    import torch

    from rlinf.data.multimodal_episode import IncrementalEpisodeTokens

    # Use an unbounded budget during replay unless explicitly configured so
    # turns that were already generated and kept are never dropped at assembly.
    budget = int(response_budget) if response_budget is not None else 10**9
    episode = IncrementalEpisodeTokens(response_budget=budget)
    has_topk = all(
        part.topk_ids is not None and part.topk_logprobs is not None
        for part in trained_turns
    )
    v_starts: list[int] = []
    v_ends: list[int] = []
    turn_is_end = False
    had_truncation = False

    for turn_pos, part in enumerate(trained_turns):
        kept = kept_lens.get(part.turn_index)
        assert kept is not None, (
            f"trajectory {traj_index} turn {part.turn_index} of group "
            f"{group_id} has no declared kept length"
        )
        assert part.delta_text is not None, (
            f"trajectory {traj_index} turn {part.turn_index} of group "
            f"{group_id} is missing delta_text in multimodal trajectory"
        )
        turn_images = (
            [_decode_png_b64(encoded) for encoded in part.image_data]
            if part.image_data
            else None
        )
        if turn_pos == 0:
            inputs = processor(
                text=[part.delta_text],
                images=turn_images,
                return_tensors="pt",
                padding=True,
            )
            episode.start_episode(prompt_text=part.delta_text, inputs=inputs)
        else:
            delta_inputs = processor(
                text=[part.delta_text],
                images=turn_images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            episode.append_user_observation(
                delta_text=part.delta_text,
                delta_inputs=delta_inputs,
            )

        if (
            not had_truncation
            and part.prompt_tokens is not None
            and part.prompt_tokens > 0
            and len(episode.input_ids) != part.prompt_tokens
        ):
            raise ValueError(
                f"Multimodal prompt token count mismatch for group {group_id} "
                f"trajectory {traj_index} turn {part.turn_index}: actor "
                f"processor produced {len(episode.input_ids)} tokens, expected "
                f"engine prompt_tokens={part.prompt_tokens}."
            )

        if return_logprobs:
            assert part.logprobs is not None, (
                f"trajectory {traj_index} turn {part.turn_index} of group "
                f"{group_id} has no logprobs but the result must carry them"
            )
            turn_lps = part.logprobs[:kept]
        else:
            turn_lps = part.logprobs[:kept] if part.logprobs is not None else None

        episode.append_assistant_response(
            response_ids=part.output_ids[:kept],
            response_text=part.output_text or "",
            token_logprobs=turn_lps,
            topk_ids=part.topk_ids[:kept] if has_topk else None,
            topk_logprobs=part.topk_logprobs[:kept] if has_topk else None,
        )
        if kept < len(part.output_ids):
            had_truncation = True
        v_starts.append(part.version_start)
        v_ends.append(part.weight_version)
        turn_is_end = bool(ended.get(part.turn_index, False))

    seq_ids = list(episode.input_ids)
    prompt_len = episode.initial_prompt_length
    resp_len = len(seq_ids) - prompt_len
    resp_mask = [False] * prompt_len + [bool(m) for m in episode.response_mask]

    seq_logprobs: Optional[list[float]] = None
    if return_logprobs:
        seq_logprobs = [0.0] * resp_len
        for pos, lp in zip(episode.response_positions, episode.response_logprobs):
            offset = pos - prompt_len
            if 0 <= offset < resp_len:
                seq_logprobs[offset] = float(lp)

    mm_inputs: dict[str, Any] = {
        "pixel_values": (
            torch.cat(episode.pixel_chunks, dim=0)
            if len(episode.pixel_chunks) > 1
            else episode.pixel_chunks[0].clone()
        ),
        "image_grid_thw": (
            torch.cat(episode.grid_chunks, dim=0)
            if len(episode.grid_chunks) > 1
            else episode.grid_chunks[0].clone()
        ),
    }
    if episode.has_mm_token_types:
        types = list(episode.mm_token_type_ids)
        if len(types) < len(seq_ids):
            types.extend([0] * (len(seq_ids) - len(types)))
        mm_inputs["mm_token_type_ids"] = torch.tensor(
            types[: len(seq_ids)], dtype=torch.long
        )

    return {
        "input_ids": seq_ids,
        "prompt_length": prompt_len,
        "response_length": resp_len,
        "response_mask": resp_mask,
        "rollout_logprobs": seq_logprobs,
        "multi_modal_inputs": mm_inputs,
        "topk_ids": list(episode.response_topk_ids) if has_topk else None,
        "topk_logprobs": list(episode.response_topk_logprobs) if has_topk else None,
        "is_end": turn_is_end,
        "reward": reward,
        "version_start": min(v_starts),
        "version_end": max(v_ends),
    }


def _merge_token_id_runs(
    group_id: int,
    traj_index: int,
    trained_turns: list[PayloadPart],
    kept_lens: dict,
    continues_prev: list[bool],
    ended: dict,
    reward: Any,
    return_logprobs: bool,
) -> list[dict[str, Any]]:
    """Merge continuous token-id turns of one trajectory into episode runs."""
    runs: list[list[PayloadPart]] = []
    for part in trained_turns:
        if not runs or not continues_prev[part.turn_index]:
            runs.append([part])
        else:
            runs[-1].append(part)

    sequences: list[dict[str, Any]] = []
    for run in runs:
        first = run[0]
        first_kept = kept_lens.get(first.turn_index)
        assert first_kept is not None, (
            f"trajectory {traj_index} turn {first.turn_index} of group "
            f"{group_id} has no declared kept length"
        )
        prompt_len = len(first.input_ids)
        seq_ids = list(first.input_ids) + list(first.output_ids[:first_kept])
        resp_mask = [False] * prompt_len + [True] * first_kept
        seq_lps: Optional[list[float]] = None
        if return_logprobs:
            assert first.logprobs is not None, (
                f"trajectory {traj_index} turn {first.turn_index} of group "
                f"{group_id} has no logprobs but the result must carry them"
            )
            seq_lps = [float(x) for x in first.logprobs[:first_kept]]
        has_topk = all(
            p.topk_ids is not None and p.topk_logprobs is not None for p in run
        )
        topk_ids: Optional[list[list[int]]] = (
            list(first.topk_ids[:first_kept]) if has_topk else None
        )
        topk_lps: Optional[list[list[float]]] = (
            list(first.topk_logprobs[:first_kept]) if has_topk else None
        )
        v_starts = [first.version_start]
        v_ends = [first.weight_version]
        last_is_end = bool(ended.get(first.turn_index, False))

        for part in run[1:]:
            kept = kept_lens.get(part.turn_index)
            assert kept is not None, (
                f"trajectory {traj_index} turn {part.turn_index} of group "
                f"{group_id} has no declared kept length"
            )
            assert part.input_ids[: len(seq_ids)] == seq_ids, (
                f"trajectory {traj_index} turn {part.turn_index} of group "
                f"{group_id} is declared continuous but does not start with "
                "the accumulated sequence"
            )
            obs_ids = part.input_ids[len(seq_ids) :]
            resp_ids = part.output_ids[:kept]
            seq_ids.extend(obs_ids)
            seq_ids.extend(resp_ids)
            resp_mask.extend([False] * len(obs_ids))
            resp_mask.extend([True] * len(resp_ids))
            if return_logprobs:
                assert part.logprobs is not None, (
                    f"trajectory {traj_index} turn {part.turn_index} of group "
                    f"{group_id} has no logprobs but the result must carry them"
                )
                seq_lps.extend([0.0] * len(obs_ids))
                seq_lps.extend(float(x) for x in part.logprobs[:kept])
            if has_topk:
                topk_ids.extend(part.topk_ids[:kept])
                topk_lps.extend(part.topk_logprobs[:kept])
            v_starts.append(part.version_start)
            v_ends.append(part.weight_version)
            last_is_end = bool(ended.get(part.turn_index, False))

        sequences.append(
            {
                "input_ids": seq_ids,
                "prompt_length": prompt_len,
                "response_length": len(seq_ids) - prompt_len,
                "response_mask": resp_mask,
                "rollout_logprobs": seq_lps,
                "multi_modal_inputs": None,
                "topk_ids": topk_ids,
                "topk_logprobs": topk_lps,
                "is_end": last_is_end,
                "reward": reward,
                "version_start": min(v_starts),
                "version_end": max(v_ends),
            }
        )
    return sequences


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
    payload: GroupPayload,
    return_logprobs: bool,
    *,
    merge_continuous_turns: bool = False,
    processor: Any = None,
    response_budget: Optional[int] = None,
) -> DynamicRolloutResult:
    """Assemble one group's parts into a `DynamicRolloutResult`.

    By default (``merge_continuous_turns=False``), produces one sequence per
    generate call: the sequence's input is the call's input ids followed by
    the kept part of its output, its logprobs and length fields sliced the
    same way, its reward the trajectory's reward from the finalize part, and
    its weight versions the call's ``version_start`` and ``weight_version``.

    When ``merge_continuous_turns=True`` (or when a trajectory carries
    multimodal ``delta_text`` parts), continuous turns of a trajectory are
    folded into one episode sequence with a multi-turn ``response_mask``,
    concatenated ``multi_modal_inputs``, aligned ``rollout_logprobs``, and
    optional top-K heads (``rollout_topk_ids``, ``rollout_topk_logprobs``).

    Args:
        payload: The group's emitted container: the per-turn parts and the
            finalize part.
        return_logprobs: Whether the result must carry per-token logprobs;
            the parts must then have recorded them.
        merge_continuous_turns: Fold continuous turns of each trajectory into
            one sequence with an explicit multi-turn ``response_mask``.
        processor: HuggingFace multimodal processor used to tokenize
            incremental ``delta_text`` / ``image_data`` turns on read-side
            assembly.
        response_budget: Optional episode token budget forwarded to
            ``IncrementalEpisodeTokens`` when assembling multimodal turns.

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

    has_multimodal = any(part.delta_text is not None for part in parts)
    use_merged = merge_continuous_turns or has_multimodal

    idx_to_traj: list[int] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    input_ids: list[list[int]] = []
    rollout_logprobs = [] if return_logprobs else None
    response_mask: Optional[list[list[bool]]] = [] if use_merged else None
    multi_modal_inputs: Optional[list[dict[str, Any]]] = [] if has_multimodal else None
    rollout_topk_ids: Optional[list[list[list[int]]]] = None
    rollout_topk_logprobs: Optional[list[list[list[float]]]] = None
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
        ended = verdict["is_end"]
        dropped = verdict.get("not_training") or {}
        traj_fields = verdict.get("extra_fields_train", {})
        names = set(traj_fields)
        if declared_fields is None:
            declared_fields = names
        assert names == declared_fields, (
            f"the trajectories of group {payload.group_id} declare different "
            f"extra_fields_train names: {sorted(names)} vs "
            f"{sorted(declared_fields)}"
        )
        trained_turns = [part for part in traj_parts if part.turn_index not in dropped]

        if use_merged:
            if has_multimodal:
                merged_seqs = [
                    _assemble_multimodal_trajectory(
                        payload.group_id,
                        traj_index,
                        trained_turns,
                        kept_lens,
                        ended,
                        finalize.rewards[traj_index],
                        return_logprobs,
                        processor,
                        response_budget,
                    )
                ]
            else:
                merged_seqs = _merge_token_id_runs(
                    payload.group_id,
                    traj_index,
                    trained_turns,
                    kept_lens,
                    continues_prev,
                    ended,
                    finalize.rewards[traj_index],
                    return_logprobs,
                )
            for name, values in traj_fields.items():
                if len(values) == len(merged_seqs):
                    extra_fields_train.setdefault(name, []).extend(values)
                else:
                    assert len(values) == len(trained_turns), (
                        f"extra_fields_train {name!r} of trajectory {traj_index} in "
                        f"group {payload.group_id} covers {len(values)} turns, "
                        f"expected {len(merged_seqs)} or {len(trained_turns)}"
                    )
                    # When per-turn scalar extra fields are provided on a merged
                    # episode, take the final turn's value for the episode.
                    extra_fields_train.setdefault(name, []).append(values[-1])
            for seq in merged_seqs:
                idx_to_traj.append(traj_index)
                input_ids.append(seq["input_ids"])
                prompt_lengths.append(seq["prompt_length"])
                response_lengths.append(seq["response_length"])
                response_mask.append(seq["response_mask"])
                if return_logprobs:
                    rollout_logprobs.append(seq["rollout_logprobs"])
                if multi_modal_inputs is not None:
                    multi_modal_inputs.append(seq["multi_modal_inputs"])
                if seq["topk_ids"] is not None and seq["topk_logprobs"] is not None:
                    if rollout_topk_ids is None:
                        rollout_topk_ids = []
                        rollout_topk_logprobs = []
                    rollout_topk_ids.append(seq["topk_ids"])
                    rollout_topk_logprobs.append(seq["topk_logprobs"])
                is_end.append(seq["is_end"])
                rewards.append(seq["reward"])
                version_start.append(seq["version_start"])
                version_end.append(seq["version_end"])
            continue

        for name, values in traj_fields.items():
            assert len(values) == len(trained_turns), (
                f"extra_fields_train {name!r} of trajectory {traj_index} in "
                f"group {payload.group_id} covers {len(values)} turns, the "
                f"payload holds {len(trained_turns)} non-dropped turns"
            )
            extra_fields_train.setdefault(name, []).extend(values)
        previous: PayloadPart | None = None
        previous_kept: int | None = None
        for part in trained_turns:
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
            is_end.append(ended[part.turn_index])
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
        response_mask=response_mask,
        multi_modal_inputs=multi_modal_inputs,
        rollout_topk_ids=rollout_topk_ids,
        rollout_topk_logprobs=rollout_topk_logprobs,
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
