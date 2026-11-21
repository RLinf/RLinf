"""Session-local control decisions for the r5b Stage2 panel."""
import json
import logging
import math
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path

from .parallel import RISK_1116
from .rules import SECOND

MODES = ("human", "semi", "full")
RULES = ("single", "two")
HEARTBEAT_NS = 3 * SECOND
MAX_DECISION_NS = 2 * SECOND
LOG = logging.getLogger(__name__)


def enabled():
    return (os.getenv("TIANJI_R5B_MONITOR_CONTROL") == "1"
            and os.getenv("TIANJI_STAGE2_SHADOW_VARIANT") == "DAgger5-r5b-4000"
            and os.getenv("TIANJI_SHADOW_VERSION") == "1116v1")


def send(runtime, row):
    if runtime is None:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.setblocking(False)
            channel.sendto(json.dumps(row, allow_nan=False).encode(), str(Path(runtime, "control")))
    except (OSError, ValueError):
        LOG.debug("Stage2 monitor decision delivery failed", exc_info=True)


def heartbeat(runtime, ns):
    path = Path(runtime, "heartbeat")
    tmp = path.with_suffix(".pending")
    tmp.write_text(str(ns))
    tmp.replace(path)


class AutoControl:
    """All state methods run under the HIL condition lock."""

    def __init__(self, runtime):
        self.runtime = Path(runtime)
        self.mode = "human"
        self.rule = "single"
        self.active = "human"
        self.profiles = None
        self.human_shadow = None
        self.pending = None
        self.hits = 0
        self.history = None
        self.high = False
        self.timed_out = False
        self.fault = False
        self.alarm = None
        self.assisted = False
        self.candidate = False
        self.notice = None
        self.seen = deque(maxlen=256)
        self.last_score_ns = 0
        self.touch = None
        self.listener = None
        self.thread = None

    def reset(self, *, execution=False):
        self.pending = None
        self.hits = 0
        self.history = None
        if execution:
            self.high = False
            self.timed_out = False
            self.candidate = False
            self.alarm = None

    def new_episode(self):
        self.active = "human"
        self.profiles = None
        self.reset(execution=True)
        self.fault = self.assisted = False
        self.notice = None
        self.last_score_ns = 0
        self.touch = None
        self.seen.clear()

    def configure(self, hil, shadow, mode, rule):
        if hil._phase != "waiting" or mode not in MODES or rule not in RULES:
            raise ValueError("只能在等待开始时选择有效模式和规则")
        if mode != "human":
            if not self.ready(shadow) or hil._monitor_settings_error:
                raise ValueError("Monitor 尚未就绪或阈值配置无效")
            if self.mode == "human":
                self.human_shadow = hil._shadow_enabled
            shadow.selection(True)
            hil._shadow_enabled = True
        elif self.mode != "human":
            shadow.selection(self.human_shadow)
            hil._shadow_enabled = self.human_shadow
        self.mode, self.rule = mode, rule

    def ready(self, shadow, now=None):
        if (shadow.status() or {}).get("state") != "ready":
            return False
        try:
            tick = int((self.runtime / "heartbeat").read_text())
        except (OSError, ValueError):
            return False
        # The heartbeat may be replaced after the caller captured its clock.
        # Read the clock after reading the heartbeat to avoid a false future timestamp.
        observed_now = time.monotonic_ns()
        now = observed_now if now is None else max(now, observed_now)
        return 0 <= now - tick <= HEARTBEAT_NS

    def start(self, hil, shadow):
        if self.mode != "human" and (not self.ready(shadow) or hil._monitor_settings_error):
            raise ValueError("Monitor 未就绪，请选择人类全部负责或等待加载")
        self.new_episode()
        self.active = self.mode
        self.profiles = hil._monitor_profiles

    def public(self, phase, ready):
        return dict(mode=self.mode if phase == "waiting" else self.active,
                    rule=self.rule, ready=ready, pending=self.pending is not None,
                    notice=self.notice, alarm=self.alarm, fault=self.fault,
                    s4_candidate=self.candidate, touch=self.touch)

    def matches(self, hil, row, now):
        ns = row.get("ns")
        return (
            self.active != "human" and hil._phase == "running" and not hil._human_latched
            and hil._runtime.get("control_mode") == "policy"
            and type(ns) is int and 0 <= now - ns <= MAX_DECISION_NS
            and all(type(row.get(k)) is int and row[k] == value for k, value in (
                ("episode", hil._episode), ("stage", hil._subtask),
                ("stage_revision", hil._subtask_revision), ("collection", hil._collection_revision),
                ("control_revision", hil._revision)))
        )

    def receive(self, hil, row, now):
        if not self.matches(hil, row, now):
            return
        event_id = row.get("id")
        if not isinstance(event_id, str) or not 0 < len(event_id) <= 180 or event_id in self.seen:
            return
        self.seen.append(event_id)
        timeout = self.active == "full" and row.get("timeout") is True and not self.timed_out
        model = False
        progress = row.get("progress")
        risk = row.get("risk")
        if progress is not None:
            if not all(type(v) in (int, float) and math.isfinite(v) for v in (progress, risk)):
                return
            if not 0 <= progress <= 4 or not 0 <= risk <= 1 or not isinstance(row.get("history"), int):
                return
            if row["ns"] <= self.last_score_ns:
                return
            self.last_score_ns = row["ns"]
            touch = row.get("touch")
            if (isinstance(touch, list) and len(touch) == 2
                    and all(isinstance(side, str) and len(side) <= 32 for side in touch)):
                self.touch = touch
            if self.history != row["history"]:
                self.hits = 0
                self.history = row["history"]
            model = self.active == "full" and risk > RISK_1116 and not self.high
            if risk != RISK_1116:
                self.high = risk > RISK_1116
        if model or timeout:
            self.timed_out |= timeout
            self.alarm = dict(model=model, timeout=timeout)
            hil._pause_for_monitor(row, {"model": model, "timeout": timeout})
            return
        if progress is None or self.pending is not None:
            return
        rule = self.profiles[self.rule]
        self.hits = self.hits + 1 if progress > rule["thresholds"][hil._subtask-1] else 0
        if self.hits < rule["confirm_points"]:
            return
        if hil._subtask == 4:
            if not self.candidate:
                self.candidate = True
                self.notice = "S4 完成候选，等待人工结束"
                hil._auto_event("monitor_s4_candidate", row)
        else:
            self.pending = row
            self.notice = "阶段切换待执行"
            hil._auto_event("monitor_stage_requested", row)

    def boundary(self, hil):
        row, self.pending = self.pending, None
        if row is None:
            return False
        if not self.matches(hil, row, time.monotonic_ns()):
            self.reset()
            self.notice = "阶段请求已失效，等待新证据"
            hil._auto_event("monitor_stage_expired", row)
            return False
        hil._revision += 1
        hil.select_subtask(hil._subtask+1, source="monitor", decision=row)
        self.notice = f"Monitor 已切换到 S{hil._subtask}"
        return True

    def listen(self, hil, shadow):
        if self.listener is not None:
            return
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.listener.bind(str(self.runtime / "control"))
        self.listener.settimeout(.15)

        def run():
            while not hil._closed:
                try:
                    data = self.listener.recv(2048)
                    row = json.loads(data)
                except (ValueError, socket.timeout):
                    row = None
                except OSError:
                    break
                now = time.monotonic_ns()
                with hil._condition:
                    if hil._closed:
                        break
                    if self.fault and self.ready(shadow, now):
                        self.fault = False
                        self.reset()
                        self.notice = "Monitor 已恢复，请按 P 或继续 Policy"
                        hil._auto_event("monitor_recovered", None, control="human" if hil._human_latched else "paused")
                    if self.active != "human" and hil._phase in {"running", "paused"} and not self.fault:
                        if not self.ready(shadow, now):
                            self.fault = True
                            self.notice = "Monitor 已离线，可将本条转人工"
                            if hil._phase == "running" and not hil._human_latched:
                                hil._pause_for_monitor(None, {"offline": True})
                    if isinstance(row, dict) and not self.fault:
                        self.receive(hil, row, now)

        self.thread = threading.Thread(target=run, name="tianji-monitor-control", daemon=True)
        self.thread.start()

    def close(self):
        if self.listener is not None:
            self.listener.close()
            self.thread.join(timeout=1.)
            (self.runtime / "control").unlink(missing_ok=True)
