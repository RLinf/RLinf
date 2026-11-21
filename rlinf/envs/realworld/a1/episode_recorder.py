"""Stage-2 运行时录制带图像的原始数据,供之后离线转成 lerobot 做 DAgger / SFT。

控制循环里只做一次非阻塞入队(不编码、不写盘、不等待);后台线程把图像经管道交给
ffmpeg 子进程编码成 h264(编码在独立进程里,不占控制进程的 GIL),数值数据定期落盘。

每段一个目录 <root>/<开始时间>_epNNNN/:
  cam_high.mp4 / cam_left_wrist.mp4 / cam_right_wrist.mp4   分段 mp4,中途断电也可读
  data.npz    t, state(A1 原生 14 维), action(实际下发给 A1 的指令,A1 原生 14 维),
              intervene, stage, reward, rlt_switch   —— 每 30 秒覆盖写一次,段结束时写最终版
  meta.json   起止时间、帧数、fps、结局(success/fail/discard/unknown)、最高阶段、丢帧数
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.utils.logging import get_logger

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class A1EpisodeRecorder(gym.Wrapper):
    def __init__(self, env: gym.Env, root: str, fps: float, task: str = "", max_queue: int = 150):
        super().__init__(env)
        self._log = get_logger()
        self.root, self.fps, self.task = root, float(fps), task
        os.makedirs(root, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._ep_open = False
        self._after_end = False   # s/f/r 之后、复位之前:当前动作块剩余几步仍会执行,不属于任何一段,不录
        self._n_ep = len([d for d in os.listdir(root) if d[:1].isdigit()])
        self._th = threading.Thread(target=self._writer, name="a1-episode-recorder", daemon=True)
        self._th.start()
        self._log.info("[Recorder] 录制到 %s(%.0f fps,后台线程 + ffmpeg 子进程编码)", root, self.fps)

    # ---------------- 控制循环侧:只入队 ----------------
    def _put(self, item) -> bool:
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            return False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._after_end:
            return obs, reward, terminated, truncated, info
        try:
            if not self._ep_open:
                self._ep_open = self._put(("start", time.time()))
            frames = obs.get("frames", {}) if isinstance(obs, dict) else {}
            state = obs.get("state", {}).get("qpos") if isinstance(obs, dict) else None
            cmd = getattr(self.env.unwrapped, "last_cmd_env", lambda: None)()
            rec = dict(
                t=time.time(), frames={k: frames.get(k) for k in CAMS},
                state=None if state is None else np.asarray(state, np.float32).copy(),
                action=None if cmd is None else np.asarray(cmd, np.float32),
                intervene="intervene_action" in info, stage=int(info.get("stage", 0)),
                reward=float(reward), rlt_switch=int(info.get("rlt_switch_flags", 0)),
            )
            if not self._put(("frame", rec)):
                # 只在后台写盘卡住(磁盘满/未挂载/ffmpeg 崩溃)时发生:宁可丢录制也不阻塞控制、不撑爆内存。
                # 本段会在 meta.json 标记 has_gap,离线转换时整段排除。
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 150 == 0:
                    self._log.error("[Recorder] 后台写盘跟不上,本段已丢 %d 步录制(本段将标记为有缺口)", self._dropped)
            if terminated or truncated:
                outcome = "discard" if info.get("discard") else (
                    "success" if info.get("episode_success") is True else
                    "fail" if info.get("episode_success") is False else "unknown")
                self._put(("end", outcome, self._dropped)); self._ep_open = False; self._dropped = 0
                self._after_end = True
        except Exception as e:  # 录制出错绝不影响控制
            self._log.error("[Recorder] 入队失败: %r", e)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._after_end = False
        if self._ep_open:                       # 没经过 s/f/r 就被复位(如 g 键、异常)
            self._put(("end", "unknown", self._dropped)); self._ep_open = False; self._dropped = 0
        return self.env.reset(**kwargs)

    def close(self):
        if self._ep_open:
            self._put(("end", "unknown", self._dropped))
        self._put(("stop",))
        self._th.join(timeout=30)
        return self.env.close()

    # ---------------- 后台线程:编码 + 落盘 ----------------
    def _open_ffmpeg(self, path, h, w):
        cmd = ["nice", "-n", "10", "ffmpeg", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h), "-r", "%g" % self.fps, "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
               "-movflags", "frag_keyframe+empty_moov+default_base_moof", path]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _writer(self):
        ep = None
        while True:
            item = self._q.get()
            kind = item[0]
            try:
                if kind == "stop":
                    if ep: self._finish(ep, "unknown", 0)
                    return
                if kind == "start":
                    if ep: self._finish(ep, "unknown", 0)
                    self._n_ep += 1
                    d = os.path.join(self.root, "%s_ep%04d" % (time.strftime("%Y%m%d_%H%M%S", time.localtime(item[1])), self._n_ep))
                    os.makedirs(d, exist_ok=True)
                    ep = dict(dir=d, t0=item[1], procs={}, cols={k: [] for k in ("t", "state", "action", "intervene", "stage", "reward", "rlt_switch")},
                              last_dump=time.time(), bad_frames=0)
                elif kind == "frame" and ep is not None:
                    rec = item[1]
                    for cam in CAMS:
                        img = rec["frames"].get(cam)
                        if img is None:
                            ep["bad_frames"] += 1; continue
                        if cam not in ep["procs"]:
                            ep["procs"][cam] = self._open_ffmpeg(os.path.join(ep["dir"], cam + ".mp4"), img.shape[0], img.shape[1])
                        ep["procs"][cam].stdin.write(np.ascontiguousarray(img, np.uint8).data)
                    c = ep["cols"]
                    c["t"].append(rec["t"]); c["intervene"].append(rec["intervene"]); c["stage"].append(rec["stage"])
                    c["reward"].append(rec["reward"]); c["rlt_switch"].append(rec["rlt_switch"])
                    c["state"].append(rec["state"] if rec["state"] is not None else np.full(14, np.nan, np.float32))
                    c["action"].append(rec["action"] if rec["action"] is not None else np.full(14, np.nan, np.float32))
                    if time.time() - ep["last_dump"] > 30:
                        self._dump(ep); ep["last_dump"] = time.time()
                elif kind == "end" and ep is not None:
                    self._finish(ep, item[1], item[2]); ep = None
            except Exception as e:
                self._log.error("[Recorder] 写入失败(%s): %r", kind, e)

    def _dump(self, ep):
        c = ep["cols"]
        if not c["t"]:
            return
        tmp = os.path.join(ep["dir"], "data.tmp.npz")
        np.savez(tmp, t=np.asarray(c["t"], np.float64), state=np.stack(c["state"]), action=np.stack(c["action"]),
                 intervene=np.asarray(c["intervene"], bool), stage=np.asarray(c["stage"], np.int16),
                 reward=np.asarray(c["reward"], np.float32), rlt_switch=np.asarray(c["rlt_switch"], np.int16))
        os.replace(tmp, os.path.join(ep["dir"], "data.npz"))

    def _finish(self, ep, outcome, dropped):
        for p in ep["procs"].values():
            try:
                p.stdin.close(); p.wait(timeout=60)
            except Exception:
                pass
        self._dump(ep)
        c = ep["cols"]; n = len(c["t"])
        meta = dict(task=self.task, fps=self.fps, n_frames=n, start=ep["t0"], end=c["t"][-1] if n else ep["t0"],
                    outcome=outcome, max_stage=int(max(c["stage"])) if n else 0,
                    intervene_ratio=float(np.mean(c["intervene"])) if n else 0.0,
                    dropped_steps=int(dropped), bad_frames=ep["bad_frames"],
                    has_gap=bool(dropped or ep["bad_frames"]))
        with open(os.path.join(ep["dir"], "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        self._log.info("[Recorder] 段已保存 %s:%d 帧,结局 %s,最高阶段 %d,介入 %.0f%%,丢 %d 步",
                       os.path.basename(ep["dir"]), n, outcome, meta["max_stage"], 100 * meta["intervene_ratio"], dropped)
