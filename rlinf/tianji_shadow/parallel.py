"""1116 release profiles, independent states and human anchor comparison."""
from .rules import Evidence, SECOND

VERSION = "1116v1"
PROFILES = {
    "single": {"thresholds": [1.02, 2.02, 3.07, 3.96], "confirm_points": 1},
    "two": {"thresholds": [1.00, 2.01, 3.04, 3.94], "confirm_points": 2},
}
RISK_1116 = 0.9988200068473816


class Track:
    def __init__(self, rule):
        self.rule = rule
        self.stage = 1
        self.evidence = Evidence()
        self.completed = set()

    def confirm(self, stage, occurrence, progress):
        evidence = self.evidence
        threshold = self.rule["thresholds"][stage - 1]
        evidence.hits = evidence.hits + 1 if progress > threshold else 0
        if evidence.hits < self.rule["confirm_points"] or occurrence in self.completed:
            return False
        self.completed.add(occurrence)
        return True


def stage_rows(events, predictions):
    anchors = {}
    for event in events:
        if event["name"] == "first_advance" and event.get("source") != "monitor":
            anchors.setdefault(event["from_stage"], (event["ns"], event["from_revision"]))
        elif event["name"] == "success" and event["stage"] == 4:
            anchors[4] = (event["ns"], event["revision"])
    rows = []
    timings = {}
    for name in PROFILES:
        for timeline in ("actual", "independent"):
            for stage in range(1, 5):
                anchor, revision = anchors.get(stage, (None, None))
                match = next((p for p in predictions if p.get("rule_profile") == name
                              and p["timeline"] == timeline and p["stage"] == stage
                              and (timeline == "independent" or revision is None or p["revision"] == revision)), None)
                decision = match["ns"] if match else None
                timings[name, timeline, stage] = decision
                delta = (decision - anchor) / SECOND if decision is not None and anchor is not None else None
                result = ("no_human_anchor" if anchor is None else "missing" if decision is None
                          else "early" if delta < -2 else "late" if delta > 2 else "match")
                rows.append(dict(rule_profile=name, timeline=timeline, stage=stage, human_ns=anchor,
                                 decision_ns=decision, delta_s=delta, result=result,
                                 counterfactual=match["counterfactual"] if match else None))
    differences = []
    for timeline in ("actual", "independent"):
        for stage in range(1, 5):
            single = timings["single", timeline, stage]
            two = timings["two", timeline, stage]
            differences.append(dict(timeline=timeline, stage=stage, single_ns=single, two_ns=two,
                                    two_minus_single_s=(two-single)/SECOND if single is not None and two is not None else None))
    return rows, differences
