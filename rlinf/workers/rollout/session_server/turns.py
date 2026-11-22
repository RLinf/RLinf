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

"""Turn bookkeeping of one session: prefix classification and kept lengths.

The session server records one :class:`TurnRecord` per successful generate
call and classifies the turns against each other when the agent closes the
trajectory. The functions here take the data they classify rather than a
session, so the session server keeps owning the session state, and they
report what they found instead of writing metrics.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "TurnRecord",
    "TurnCounts",
    "UnderreportedTruncation",
    "new_turn_record",
    "kept_len",
    "classify_turns",
]


@dataclass
class TurnRecord:
    """Prefix bookkeeping for one successful generate call of a session.

    ``lcp`` is the longest common prefix length of this turn's input with the
    previous turn's input+output, None on the first turn. ``kept_len`` and
    ``continues_prev`` are filled when the agent declares them (close, or the
    per-turn ``prev_kept_len`` declaration); until then the kept length is the
    full output and the classification is pending. ``is_end`` is the agent's
    declaration, at close, of whether this turn's sequence ended (the agent's
    ``AgentLoopOutput.is_end``); it feeds the properly-ended fraction of the
    rollout metrics, and an agent that declares nothing leaves it False. In
    verify mode ``merged_sampling_params`` holds the params this call was
    forwarded with.
    """

    input_len: int
    output_len: int = 0
    lcp: Optional[int] = None
    declared_prev_kept_len: Optional[int] = None
    kept_len: Optional[int] = None
    continues_prev: Optional[bool] = None
    is_end: bool = False
    merged_sampling_params: Optional[dict] = None


@dataclass
class TurnCounts:
    """How many turn pairs the classification found in each shape."""

    continuous: int = 0
    rewritten: int = 0


class UnderreportedTruncation(ValueError):
    """A turn breaks inside the previous turn's declared kept output.

    Raised by :func:`classify_turns`, which changes no state: the caller
    counts the truncation and fails the close, and the agent's retried close
    re-runs the whole classification.
    """


def new_turn_record(
    input_ids: list[int],
    previous_input: Optional[list[int]],
    previous_output: Optional[list[int]],
    prev_kept_len: Optional[int],
) -> TurnRecord:
    """The turn's prefix bookkeeping, classified now where possible.

    The lcp with the previous turn's input+output is computed against the
    ids still held for the previous turn. A break inside the previous
    input is a history rewrite and classifies immediately — the record's
    ``continues_prev`` comes back False, for the caller to count — and with
    a declared ``prev_kept_len`` the classification completes now too.
    """
    if previous_input is None:
        return TurnRecord(
            input_len=len(input_ids), declared_prev_kept_len=prev_kept_len
        )
    previous = list(previous_input) + list(previous_output or [])
    lcp = 0
    for cur, prev in zip(input_ids, previous):
        if cur != prev:
            break
        lcp += 1
    record = TurnRecord(
        input_len=len(input_ids),
        lcp=lcp,
        declared_prev_kept_len=prev_kept_len,
    )
    if lcp < len(previous_input):
        record.continues_prev = False
    return record


def kept_len(turn_index: int, turns: list[TurnRecord], kept_lens: dict) -> int:
    """The declared kept length of a turn, defaulting to the full output."""
    raw = kept_lens.get(turn_index, kept_lens.get(str(turn_index)))
    turn = turns[turn_index]
    if raw is None:
        return turn.output_len
    if not isinstance(raw, int) or raw < 0:
        raise ValueError(
            f"kept_lens[{turn_index}] must be a non-negative int, got {raw!r}"
        )
    if raw > turn.output_len:
        raise ValueError(
            f"kept_lens[{turn_index}] is {raw} but the turn's output has "
            f"only {turn.output_len} tokens"
        )
    return raw


def classify_turns(
    turns: list[TurnRecord],
    kept: list[int],
    not_training_turns: set,
    entry: Optional[str],
    traj_index: int,
    group_id: int,
) -> tuple[list[bool], TurnCounts]:
    """Classify each turn pair, failing loud on under-reporting.

    For turn k the pair is its ``lcp`` against turn k-1's input length
    ``L`` and kept length ``kept``: ``lcp >= L + kept`` continues the
    previous turn, ``L <= lcp < L + kept`` means the agent used less
    output than it declared (an under-reported truncation), and
    ``lcp < L`` rewrites history and starts a new segment. A turn whose
    input already broke inside the previous input was classified as a
    rewrite when it arrived. A turn dropped for not training never pairs:
    it yields no sequence, and the turn after it compares against the last
    kept turn before it.

    Args:
        turns: The session's turn records, in turn order.
        kept: The kept length of each turn, as declared at close.
        not_training_turns: Turn indexes the close dropped from the training
            data.
        entry: The session's entry (``"chat"`` for the chat completions
            entry), which relaxes a rewritten prefix to a new segment.
        traj_index: Trajectory the turns belong to, for error messages.
        group_id: Group the turns belong to, for error messages.

    Returns:
        ``continues_prev`` per turn — turn 0 is False — and the pair counts.

    Raises:
        UnderreportedTruncation: On an under-reported truncation.
        ValueError: On a per-turn ``prev_kept_len`` declaration that
            disagrees with the close declaration. No state is changed.
    """
    continues: list[bool] = []
    counts = TurnCounts()
    previous_index: Optional[int] = None
    for turn_index, turn in enumerate(turns):
        if turn_index in not_training_turns:
            # Dropped from the training data: it never pairs, so it
            # neither continues nor becomes the previous turn for the
            # turn after it. Checked before the arrive-time
            # classification: a rewrite that arrived before the drop
            # must not anchor the next turn either.
            continues.append(False)
            continue
        if turn.continues_prev is not None:
            continues.append(turn.continues_prev)
            previous_index = turn_index
            continue
        if previous_index is None:
            continues.append(False)
            previous_index = turn_index
            continue
        previous = turns[previous_index]
        boundary = previous.input_len + kept[previous_index]
        if turn.lcp >= boundary:
            continues.append(True)
            counts.continuous += 1
        elif turn.lcp >= previous.input_len and entry == "chat":
            # A chat client rebuilt its history over the previous
            # turn's output: the conversation rewrites and starts a new
            # segment instead of failing the close. The generate entry
            # has no such client and keeps the strict check.
            continues.append(False)
            counts.rewritten += 1
        elif turn.lcp >= previous.input_len:
            raise UnderreportedTruncation(
                f"turn {turn_index} of trajectory {traj_index} in group "
                f"{group_id} breaks inside the previous turn's declared kept "
                f"output: the agent used {turn.lcp - previous.input_len} "
                f"of the {kept[previous_index]} tokens it declared"
            )
        else:
            continues.append(False)
            counts.rewritten += 1
        previous_index = turn_index

    for turn_index in range(1, len(turns)):
        declared = turns[turn_index].declared_prev_kept_len
        if declared is not None and declared != kept[turn_index - 1]:
            raise ValueError(
                f"turn {turn_index} of trajectory {traj_index} in group "
                f"{group_id} declared prev_kept_len {declared} but close "
                f"declared {kept[turn_index - 1]} for the previous turn"
            )
    return continues, counts
