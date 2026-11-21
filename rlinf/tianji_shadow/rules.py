"""1113 shadow rules and deterministic ground-truth matching."""
from collections import deque

SECOND = 1_000_000_000
THRESHOLDS = (1.00, 2.01, 3.02, 3.96)
TIMEOUTS = (50, 90, 100, 15)
RISK = .9999223947525024
TASKS = (0, 2, 1, 3)


class Evidence:
    def __init__(self):
        self.epoch = -1
        self.reset()

    def reset(self):
        self.epoch += 1
        self.history = deque(maxlen=5)
        self.hits = 0
        self.candidate = False
        self.risk = False

    def score(self, stage, progress, risk):
        self.hits = self.hits + 1 if progress > THRESHOLDS[stage-1] else 0
        complete = self.hits >= 2 and not self.candidate
        self.candidate |= complete
        alarm = risk > RISK and not self.risk
        if risk != RISK:
            self.risk = risk > RISK
        return complete, alarm

    def append(self, sample):
        if self.history and sample["capture"] - self.history[-1]["capture"] > 1_200_000_000:
            self.reset()
        self.history.append(sample)


def compare(events, predicted, observations, decision_times=None):
    """Use only confirmed operator anchors; times are on the same monotonic clock."""
    stage_results = []
    handovers = []
    model_h = sorted((x for x in predicted if x["name"] == "model_h"), key=lambda x: x["ns"])
    completions = [x for x in predicted if x["name"] == "complete"]
    for event in events:
        name = event.get("name")
        if name in ("first_advance", "success") and event.get("source") != "monitor":
            stage = event.get("from_stage", event["stage"])
            revision = event.get("from_revision", event["revision"])
            choices = [x for x in completions if x["stage"] == stage and x["revision"] == revision]
            hit = min(choices, key=lambda x: abs(x["ns"] - event["ns"]), default=None)
            dt = (hit["ns"] - event["ns"]) / SECOND if hit else None
            available = observations.get(revision, 0)
            stage_results.append(dict(stage=stage, revision=revision, anchor=event["ns"],
                                      predicted=hit["ns"] if hit else None, delta_s=dt,
                                      result=("insufficient" if available < 2 else "missing") if dt is None
                                      else "match" if abs(dt) <= 2 else "early" if dt < 0 else "late"))
        if name == "human_start" and not event.get("monitor_assisted"):
            handovers.append(event)
    # Sort by human time and use earliest eligible prediction to maximize
    # one-to-one matches over fixed-width, monotonic windows.
    pairs = []
    used = set()
    for h in sorted(handovers, key=lambda x: x["ns"]):
        eligible = [
            i for i, m in enumerate(model_h)
            if m["revision"] == h["revision"] and m["stage"] == h["stage"]
            and 0 < h["ns"] - m["ns"] <= 2 * SECOND
        ]
        match = next((i for i in eligible if i not in used), None)
        if match is not None:
            used.add(match)
        enough = (observations.get(h["revision"]) if decision_times is None else any(
            0 < h["ns"]-ns <= 2*SECOND for ns in decision_times.get(h["revision"], ())
        ))
        pairs.append(dict(human=h, model=model_h[match] if match is not None else None,
                          lead_s=(h["ns"]-model_h[match]["ns"])/SECOND if match is not None else None,
                          result="match" if match is not None else
                          "insufficient" if not enough else "unmatched"))
    unused = []
    for i, m in enumerate(model_h):
        if i not in used:
            duplicate = any(
                h["revision"] == m["revision"] and h["stage"] == m["stage"]
                and 0 < h["ns"] - m["ns"] <= 2 * SECOND for h in handovers
            )
            unused.append(dict(event=m, result="duplicate" if duplicate else "unmatched"))
    return dict(stages=stage_results, human_h=pairs, unused_model_h=unused)
