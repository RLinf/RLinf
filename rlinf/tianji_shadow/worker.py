"""Episode recorder and GPU1 inference loop."""
import json
import logging
import os
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .inference import Predictor, pack
from . import control as auto_control
from .rules import Evidence, RISK, SECOND, TASKS, THRESHOLDS, TIMEOUTS, compare
from .parallel import PROFILES, RISK_1116, Track, VERSION, stage_rows
from .settings import BASE_VERSION, FT_VERSION, defaults, overrides
from .variants import progress_variant
from .review import append_review, episode_file, monitor_row
from .touch import TouchFeed
from .transport import selected


def append(file, row):
    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    file.flush()


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


class Episode:
    def __init__(self, root, event, touch, predictor, identity):
        self.episode = event["episode"]
        self.auto_mode = event.get("monitor_mode", "human")
        self.auto_runtime = os.environ.get("TIANJI_SHADOW_RUNTIME") if auto_control.enabled() else None
        self.auto_revision = event.get("control_revision")
        self.auto_timeout_ack = False
        self.auto_timeout_send_ns = 0
        if "monitor_mode" in event:
            identity = dict(identity, monitor_mode=self.auto_mode, effective_rule=event.get("monitor_rule"))
        review_start_ns = event.get("review_start_ns")
        self.review_file = (
            episode_file(root.parent, self.episode, review_start_ns)
            if review_start_ns is not None else None
        )
        self.directory = root / f"episode_{self.episode:04d}_{event['ns']}"
        self.directory.mkdir(parents=True)
        self.files = {name: (self.directory / f"{name}.jsonl").open("w", encoding="utf-8")
                      for name in ("human", "observations", "predictions")}
        atomic(self.directory / "identity.json", identity)
        self.predictor, self.touch = predictor, touch
        self.actual, self.independent = Evidence(), Evidence()
        self.profiles = identity.get("rule_profiles")
        if self.profiles:
            self.actual_tracks = {name: Track(rule) for name, rule in self.profiles.items()}
            self.shadow_tracks = {name: Track(rule) for name, rule in self.profiles.items()}
            self.stage_predictions = []
        self.stage = 1
        self.shadow_stage = 1
        self.run_state = None
        self.active_since = None
        self.elapsed_ns = 0
        self.timed_out = False
        self.events, self.predicts = [], []
        self.counts = Counter()
        self.decision_times = defaultdict(list)
        self.rejections = Counter()
        self.last_capture = 0
        self.last_seq = event.get("seq", 1)-1
        self.event_gaps = 0
        self.last_drop = 0
        self.seen_oids = set()
        self.won = set()
        self.won_independent = set()
        self.revision = event["revision"]
        self.collection = event["collection"]
        self.touch.arm(event["ns"])
        self.on_event(event)

    def allowed(self):
        return self.run_state == "policy"

    def reset_evidence(self):
        self.actual.reset()
        self.independent.reset()
        if self.profiles:
            for track in (*self.actual_tracks.values(), *self.shadow_tracks.values()):
                track.evidence.reset()

    def _elapsed(self, ns):
        return self.elapsed_ns + (max(0, ns-self.active_since) if self.active_since is not None else 0)

    def _timeout_due(self, ns):
        return self.allowed() and self._elapsed(ns) >= TIMEOUTS[self.stage-1]*SECOND

    def _submit_control(self, ns, event_id, **decision):
        if self.auto_mode in {"semi", "full"} and self.allowed():
            auto_control.send(self.auto_runtime, dict(id=event_id, ns=ns, episode=self.episode,
                             stage=self.stage, stage_revision=self.revision, collection=self.collection,
                             control_revision=self.auto_revision, **decision))

    def alarm_clock(self, ns, *, submit=True):
        if not self._timeout_due(ns):
            return
        if not self.timed_out:
            self.timed_out = True
            self.prediction("timeout_h", ns, self.stage, "actual", elapsed_s=self._elapsed(ns)/SECOND)
        if (submit and self.auto_mode == "full" and not self.auto_timeout_ack
                and ns - self.auto_timeout_send_ns >= SECOND // 2):
            self._submit_control(ns, f"timeout:{ns}", timeout=True)
            self.auto_timeout_send_ns = ns

    def on_event(self, event):
        ns = event["ns"]
        if "seq" in event:
            self.event_gaps += max(0, event["seq"] - self.last_seq - 1)
            self.last_seq = event["seq"]
        self.alarm_clock(ns, submit=False)
        self.auto_revision = event.get("control_revision")
        self.auto_mode = event.get("monitor_mode", self.auto_mode)
        changed_stage = event["revision"] != self.revision
        if event["name"] == "monitor_pause" and event.get("alarm", {}).get("timeout"):
            self.auto_timeout_ack = True
        new_state = event.get("control", self.run_state)
        changed_control = new_state != self.run_state and {new_state, self.run_state} != {"hold", "policy"}
        if changed_stage or changed_control or event["collection"] != self.collection:
            self.reset_evidence()
        interrupted = not self.allowed() or new_state != "policy"
        if self.active_since is not None and (changed_stage or interrupted):
            self.elapsed_ns += max(0, ns-self.active_since)
            self.active_since = None
        if changed_stage or event["name"] in ("human_stop", "reset_failed"):
            self.elapsed_ns = 0
            self.timed_out = False
            self.auto_timeout_ack = False
            self.auto_timeout_send_ns = 0
        self.stage, self.revision, self.collection, self.run_state = (
            event["stage"], event["revision"], event["collection"], new_state)
        if self.allowed() and self.active_since is None:
            self.active_since = ns
        self.events.append(event)
        append(self.files["human"], event)

    def prediction(self, name, ns, stage, timeline, *, review_input=None, **extra):
        row = dict(name=name, ns=ns, stage=stage, revision=self.revision,
                   actual_stage=self.stage, timeline=timeline, counterfactual=stage != self.stage,
                   **extra)
        append(self.files["predictions"], row)
        if self.review_file is not None:
            model_input = (
                f"{self.directory.name}/{review_input}" if review_input is not None else None
            )
            try:
                append_review(self.review_file, monitor_row(self.episode, row, model_input))
            except OSError:
                logging.getLogger(__name__).exception("Could not write Monitor review event")
        if self.profiles and name == "complete":
            self.stage_predictions.append(row)
        elif timeline == "actual":
            self.predicts.append(row)

    def reject(self, record, reason, reset=True):
        self.rejections[reason] += 1
        if self.profiles:
            record = dict(record, hil_event_seq=self.last_seq)
        append(self.files["observations"], dict(record, result=reason))
        if reset:
            self.reset_evidence()

    def feature(self, message, body):
        now = time.monotonic_ns()
        header = dict(message, arrival=now)
        self.last_drop = max(self.last_drop, message.get("dropped", 0))
        if (message["revision"], message["collection"], message["stage"]) != (
            self.revision, self.collection, self.stage
        ):
            self.reject(header, "state_changed", reset=False)
            return
        # The first planning observation can precede the first executed chunk.
        if self.run_state not in ("policy", "hold"):
            self.reject(header, "not_policy", reset=False)
            return
        cap, read, oid = message["capture"], message["read"], message["oid"]
        # Age is diagnostic only. A missing source clock disables tactile
        # pairing for this sample, but the actual Policy features remain usable.
        capture_ns = cap if 0 < cap <= read else 0
        if oid in self.seen_oids or 0 < capture_ns <= self.last_capture:
            self.reject(header, "duplicate_or_out_of_order", reset=False)
            return
        self.seen_oids.add(oid)
        if capture_ns:
            self.last_capture = capture_ns
        expected = next((x.get("prompt") for x in reversed(self.events) if x.get("prompt")), None)
        if expected != message["prompt"]:
            self.reject(header, "prompt_mismatch")
            return
        values = np.frombuffer(body, dtype=np.float32)
        if values.shape != (2064,) or not np.isfinite(values).all():
            self.reject(header, "invalid_features")
            return
        rgb, refs, meta, aligned = self.touch.align(capture_ns)
        history_ns = capture_ns or (read if 0 < read <= now else now)
        sample = dict(capture=history_ns, source_capture=capture_ns, rlt=values[:2048],
                      state=values[2048:], rgb=rgb,
                      refs=refs, meta=meta, oid=oid, alignment=aligned)
        if self.profiles:
            self.feature_1116(sample, header)
            return
        self.actual.append(sample)
        self.independent.append(sample)
        independent_stage = self.shadow_stage
        independent_active = independent_stage <= 4
        a = pack(self.actual.history, TASKS[self.stage-1])
        b = pack(self.independent.history, TASKS[min(independent_stage, 4)-1])
        try:
            actual = self.predictor(a)[0]
            if np.array_equal(a["valid"], b["valid"]) and a["task"][0] == b["task"][0]:
                independent = actual
            else:
                independent = self.predictor(b)[0] if independent_active else actual
        except Exception as exc:
            self.reject(header, "inference_error")
            raise RuntimeError("1113 inference failed") from exc
        decision = time.monotonic_ns()
        actual, independent = [{k: float(x[k][0]) for k in ("progress", "risk", "logit")}
                               for x in (actual, independent)]
        if not np.isfinite(list(actual.values())+list(independent.values())).all():
            self.reject(header, "invalid_scores")
            return
        self.counts[self.revision] += 1
        self.decision_times[self.revision].append(decision)
        sequence = sum(self.counts.values())
        path = f"obs_{sequence:06d}.npz"
        np.savez_compressed(self.directory / path, **{"a_"+k: v for k, v in a.items()},
                            **{"b_"+k: v for k, v in b.items()})
        windows = [
            [dict(oid=v["oid"], capture=v["source_capture"], history_ns=v["capture"],
                  tactile=v["alignment"]) for v in evidence.history]
            for evidence in (self.actual, self.independent)
        ]
        append(self.files["observations"], dict(
            header, decision=decision, input=path, actual=actual, shadow=independent if independent_active else None,
            shadow_stage=independent_stage, counterfactual=independent_stage!=self.stage,
            actual_task=TASKS[self.stage-1], shadow_task=TASKS[min(independent_stage, 4)-1],
            windows=windows, actual_epoch=self.actual.epoch, shadow_epoch=self.independent.epoch,
            result="ok",
        ))
        for label, scores, evidence, stage in (
            ("actual", actual, self.actual, self.stage),
            ("independent", independent, self.independent, independent_stage),
        ):
            if stage > 4:
                continue
            complete, alarm = evidence.score(stage, scores["progress"], scores["risk"])
            if alarm:
                self.prediction("model_h", decision, stage, label, review_input=path, oid=oid, **scores)
            key = self.revision if label == "actual" else stage
            won = self.won if label == "actual" else self.won_independent
            if complete and key not in won:
                won.add(key)
                self.prediction("complete", decision, stage, label, review_input=path, oid=oid, **scores)
                if label == "independent":
                    self.shadow_stage += 1
                    self.independent.reset()

    def feature_1116(self, sample, header, recorded_decision=None):
        self.actual.append(sample)
        actual_input = pack(self.actual.history, TASKS[self.stage - 1])
        saved = {"a_" + key: value for key, value in actual_input.items()}
        scores = self.score_1116(actual_input)
        simulations = {}
        for name, track in self.shadow_tracks.items():
            if track.stage > 4:
                simulations[name] = dict(stage=track.stage, scores=None, counterfactual=True)
                continue
            track.evidence.append(sample)
            simulated = pack(track.evidence.history, TASKS[track.stage - 1])
            same_inputs = track.stage == self.stage and len(track.evidence.history) == len(self.actual.history)
            simulated_scores = scores if same_inputs else self.score_1116(simulated)
            simulations[name] = dict(stage=track.stage, scores=simulated_scores,
                                     counterfactual=track.stage != self.stage)
        decision = time.monotonic_ns() if recorded_decision is None else recorded_decision
        if recorded_decision is None:
            self._submit_control(decision, "sample:" + header["oid"], progress=scores["progress"],
                                 risk=scores["risk"], history=self.actual.epoch,
                                 touch=[side["reason"] for side in sample["alignment"]],
                                 timeout=self.auto_mode == "full" and self._timeout_due(decision))
        self.counts[self.revision] += 1
        self.decision_times[self.revision].append(decision)
        path = f"obs_{sum(self.counts.values()):06d}.npz"
        np.savez_compressed(self.directory / path, **saved)
        append(self.files["observations"], dict(header, decision=decision, input=path, actual=scores,
               profiles=simulations, alignment=sample["alignment"], hil_event_seq=self.last_seq, result="ok"))
        oid = header["oid"]
        risk = scores["risk"]
        if risk > RISK_1116 and not self.actual.risk:
            self.prediction("model_h", decision, self.stage, "actual", review_input=path, oid=oid, **scores)
        if risk != RISK_1116:
            self.actual.risk = risk > RISK_1116
        for name, track in self.actual_tracks.items():
            track.evidence.append(sample)
            if track.confirm(self.stage, self.revision, scores["progress"]):
                self.prediction("complete", decision, self.stage, "actual",
                                review_input=path, oid=oid, rule_profile=name, **scores)
        for name, sim in simulations.items():
            track = self.shadow_tracks[name]
            stage = track.stage
            if sim["scores"] and track.confirm(stage, stage, sim["scores"]["progress"]):
                self.prediction("complete", decision, stage, "independent",
                                review_input=path, oid=oid, rule_profile=name, **sim["scores"])
                track.stage += 1
                track.evidence.reset()

    def score_1116(self, arrays):
        result = self.predictor(arrays)[0]
        scores = {key: float(result[key][0]) for key in ("progress", "risk", "logit")}
        if not np.isfinite(list(scores.values())).all():
            raise ValueError("1116v1 produced non-finite scores")
        return scores

    def close(self, outcome):
        for file in self.files.values():
            file.close()
        result = compare(self.events, self.predicts, self.counts, self.decision_times)
        if self.profiles:
            result.pop("stages")
            rows, differences = stage_rows(self.events, self.stage_predictions)
            result.update(stages=rows, rule_differences=differences,
                          independent_last_stages={k: v.stage for k, v in self.shadow_tracks.items()})
        atomic(self.directory / "comparison.json", dict(
            outcome=outcome, accepted=sum(self.counts.values()), rejected=dict(self.rejections),
            accepted_by_stage_revision=self.counts, human_event_sequence_gaps=self.event_gaps,
            feature_queue_drop_counter=self.last_drop, independent_last_stage=self.shadow_stage,
            **result,
        ))


class Session:
    def __init__(self, root, touch, predictor, identity, variants=None):
        self.root, self.touch, self.predictor, self.identity = root, touch, predictor, identity
        self.variants = variants or {}
        self.episode = None
        self.closed_episode = 0

    def handle(self, record, body):
        kind = record["kind"]
        if kind == "event" and record["name"] == "start":
            if self.episode:
                self.close("next_start")
            name = record.get("monitor_version", self.predictor.identity.get("version"))
            predictor = self.predictor if name == self.predictor.identity.get("version") else self.variants.get(name)
            if predictor is None:
                logging.getLogger(__name__).error("Requested Monitor version %r is unavailable", name)
                return
            identity = dict(self.identity, model=predictor.identity)
            if name == FT_VERSION:
                identity["rule_profiles"] = defaults(FT_VERSION)
            if "rule_profiles" in identity and "monitor_thresholds" in record:
                identity = dict(identity, rule_profiles=overrides(record["monitor_thresholds"]))
            self.episode = Episode(self.root, record, self.touch, predictor, identity)
            return
        episode = self.episode
        if episode is None or episode.episode != record["episode"]:
            return
        if kind == "event":
            episode.on_event(record)
            if record["name"] in ("success", "failure", "discard", "fault", "reset_failed"):
                self.close(record["name"])
        elif kind == "feature":
            episode.feature(record, body)
        elif kind == "invalid":
            episode.reject(record, record["reason"])

    def close(self, outcome):
        if self.episode:
            self.episode.close(outcome)
            self.closed_episode = self.episode.episode
            self.episode = None


def drain_events(reader, session):
    while True:
        offset = reader.tell()
        line = reader.readline()
        if not line:
            return
        if not line.endswith(b"\n"):
            reader.seek(offset)
            return
        session.handle(json.loads(line), b"")


def run(runtime, root, assets, policy, variant, stopped):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(runtime / "feed"))
    sock.settimeout(.05)
    touch = TouchFeed(os.environ.get("TIANJI_SHADOW_COLLECTOR", "http://127.0.0.1:8780"), selected)
    session = None
    try:
        start = time.monotonic()
        predictor = Predictor(assets)
        probe = dict(rlt=np.zeros(2048, np.float32), state=np.zeros(16, np.float32),
                     rgb=np.zeros((2,3,140,80), np.uint8), refs=None, meta=np.zeros((2,3), np.float32))
        predictor(pack([probe], TASKS[0]))
        policy = Path(policy).resolve()
        model_manifest = policy / "model_state_dict/full_weights.bf16.safetensors.json"
        from .inference import sha
        model_receipt = dict(path=str(policy), manifest_sha256=sha(model_manifest) if model_manifest.is_file() else None)
        identity = dict(schema="tianji.monitor_shadow.1113.v1", model=predictor.identity, policy=model_receipt,
                        variant=variant, thresholds=THRESHOLDS, risk_threshold=RISK, timeouts_s=TIMEOUTS,
                        upstream_rlt="v8s_step10000", live_device="GPU1", observation_age_gate=False)
        version = predictor.identity.get("version", "1113")
        requested = os.environ.get("TIANJI_SHADOW_VERSION", "1113")
        if version != requested:
            raise ValueError(f"Expected {requested} Monitor, got {version}")
        if version == VERSION:
            identity.update(schema="tianji.monitor_shadow.1116.v1", rule_profiles=PROFILES,
                            risk_threshold=RISK_1116, upstream_rlt="r5b_step4000")
            identity.pop("thresholds")
        variants, availability = {}, {}
        if auto_control.enabled() and version == BASE_VERSION:
            availability[BASE_VERSION] = {"ready": True}
            try:
                candidate = progress_variant(predictor, assets)
                candidate(pack([probe], TASKS[0]))
                variants[FT_VERSION] = candidate
                availability[FT_VERSION] = {"ready": True}
            except (OSError, ValueError, RuntimeError) as exc:
                availability[FT_VERSION] = {"ready": False, "error": str(exc)}
                logging.getLogger(__name__).exception("Could not load optional Monitor version")
        import torch
        atomic(runtime / "status.json", dict(state="ready", model=f"{version} Joint E4 + Hover C E30", version=version, device="GPU1",
                                             gpu=torch.cuda.get_device_name(0), load_count=1, pid=os.getpid(),
                                             peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2, 1),
                                             load_s=round(time.monotonic()-start, 3), versions=availability))
        session = Session(root, touch, predictor, identity, variants)
        live_control = auto_control.enabled()
        beat = 0
        journal = runtime / "events.jsonl"
        journal.touch()
        with journal.open("rb") as reader:
            while not stopped.is_set() and os.getppid() != 1:
                try:
                    packet = sock.recv(65536)
                except socket.timeout:
                    packet = None
                drain_events(reader, session)
                if packet is not None and selected():
                    prefix, body = packet.split(b"\0", 1)
                    session.handle(json.loads(prefix), body)
                now = time.monotonic_ns()
                if live_control and now - beat >= SECOND // 2:
                    auto_control.heartbeat(runtime, now)
                    beat = now
                if session.episode:
                    session.episode.alarm_clock(now)
            drain_events(reader, session)
    finally:
        if session is not None:
            session.close("monitor_exit")
        touch.close()
        sock.close()
