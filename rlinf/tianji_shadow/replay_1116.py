"""Rescore saved Policy samples, reconstructing both 1116 stages on the source clock."""
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from .inference import Predictor
from .parallel import PROFILES, RISK_1116, VERSION
from .rules import SECOND, TIMEOUTS
from .settings import FT_VERSION, overrides
from .variants import progress_variant
from .worker import Episode


class ArchivedTouch:
    def arm(self, ns):
        pass


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_sample(path, row):
    with np.load(path / row["input"], allow_pickle=False) as data:
        capture = row["capture"]
        source = capture if 0 < capture <= row["read"] else 0
        return dict(capture=source or row["read"] or row["arrival"], source_capture=source,
                    oid=row["oid"], rlt=data["a_rlt"][0, -1], state=data["a_state"][0, -1],
                    rgb=data["a_rgb"][0, -1], refs=data["a_refs"][0], meta=data["a_meta"][0, -1],
                    alignment=row["alignment"] if "alignment" in row else row["windows"][0][-1]["tactile"])


def replay_episode(source, output, predictor, profiles=None):
    notes = json.loads((source / "identity.json").read_text())
    if predictor.identity.get("version") not in (VERSION, FT_VERSION):
        raise ValueError("Replay requires a supported 1116-family Monitor")
    source_version = notes.get("model", {}).get("version", VERSION)
    if source_version != predictor.identity.get("version"):
        raise ValueError(f"Replay requires the recorded version: {source_version}")
    trained_policy = Path(notes["policy"]["path"]).name
    if trained_policy != "box_r5b_step4000":
        raise ValueError(f"Expected archived r5b step4000 features, got {trained_policy}")
    outcome = json.loads((source / "comparison.json").read_text())["outcome"]
    human = jsonl(source / "human.jsonl")
    rows = jsonl(source / "observations.jsonl")
    if not human or human[0]["name"] != "start" or not any(r["result"] == "ok" for r in rows):
        raise ValueError(f"Missing episode start or inputs: {source}")
    # New archives record the last event drained by the worker. An operator
    # command can arrive during inference, before the score is stamped.
    sequenced = all("seq" in event for event in human) and all("hil_event_seq" in row for row in rows)
    clocks = [(e["seq"] if sequenced else e["ns"], 0, i, e["ns"], e)
              for i, e in enumerate(human[1:])]
    human_times = [event["ns"] for event in human]
    fields = ("revision", "collection", "stage")
    for i, row in enumerate(rows):
        ns = row.get("decision", row.get("arrival", row.get("ns", human[0]["ns"])))
        order = row.get("hil_event_seq") if sequenced else row.get("arrival", ns)
        if not sequenced and row["result"] == "ok":
            index = bisect_right(human_times, order) - 1
            context = tuple(row[k] for k in fields)
            # The legacy worker can accept a sample immediately before it drains a
            # concurrent operator command, even if that command has a lower clock stamp.
            while (index > 0 and tuple(human[index][k] for k in fields) != context
                   and 0 <= order - human_times[index] <= 100_000_000):
                order = human_times[index] - 1
                index -= 1
        clocks.append((order, 1, i, ns, row))
    original_profiles = notes.get("rule_profiles")
    profiles = profiles or original_profiles or PROFILES
    identity = dict(schema="tianji.monitor_shadow.1116.v1", model=predictor.identity,
                    policy=notes["policy"], variant=notes["variant"], rule_profiles=profiles,
                    risk_threshold=RISK_1116, timeouts_s=TIMEOUTS, observation_age_gate=False,
                    source_episode=str(source), time_basis="recorded Policy decisions on the original monotonic clock")
    episode = Episode(output, human[0], ArchivedTouch(), predictor, identity)
    for _, kind, _, ns, record in sorted(clocks):
        if episode.allowed() and episode.active_since is not None:
            deadline = episode.active_since + TIMEOUTS[episode.stage-1]*SECOND - episode.elapsed_ns
            if deadline <= ns:
                episode.alarm_clock(deadline)
        if kind == 0:
            episode.on_event(record)
        elif record["result"] == "ok":
            if (record["revision"], record["collection"], record["stage"]) != (
                episode.revision, episode.collection, episode.stage
            ):
                raise ValueError(f"Archived decisions and HIL disagree at {record['oid']}")
            original = {key: record[key] for key in ("oid","episode","stage","revision","collection",
                        "capture","read","arrival","prompt")}
            sample = read_sample(source, record)
            episode.feature_1116(sample, original, recorded_decision=ns)
        else:
            reason = record["result"]
            episode.reject(record, reason, reset=reason not in {
                "state_changed", "duplicate_or_out_of_order", "not_policy"
            })
    episode.close(outcome)
    if original_profiles == profiles:
        def events(directory):
            return [row for row in jsonl(directory/"predictions.jsonl") if row["name"] != "timeout_h"]
        if events(source) != events(episode.directory):
            raise ValueError(f"Recomputed prediction events differ in {source}")
    return episode.directory


def replay_1116(sources, output, assets, device, thresholds=None):
    predictor = Predictor(assets, device=device)
    profiles = overrides(json.loads(Path(thresholds).read_text())) if thresholds else None
    variants = {VERSION: predictor}
    for source in sources:
        folder = Path(source)
        version = json.loads((folder / "identity.json").read_text()).get("model", {}).get("version", VERSION)
        if version == FT_VERSION and version not in variants:
            variants[version] = progress_variant(predictor, assets)
        if version not in variants:
            raise ValueError(f"Unsupported archived Monitor version: {version}")
        print(replay_episode(folder, Path(output), variants[version], profiles), flush=True)
    return 0
