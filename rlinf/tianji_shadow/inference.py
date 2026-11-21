"""Strict loading and the published 1113 transform."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import Hover, Monitor, route
from .rules import RISK, THRESHOLDS
from .parallel import PROFILES, RISK_1116, VERSION

DEFAULT_ASSETS = Path.home() / "tianji_monitor_assets" / "shadow1113"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Predictor:
    def __init__(self, assets=DEFAULT_ASSETS, device="cuda:0"):
        assets = Path(assets)
        self.identity = json.loads((assets / "identity.json").read_text())
        version = self.identity.get("version", "1113")
        if version == VERSION:
            if self.identity.get("profiles") != PROFILES or self.identity["risk"] != RISK_1116:
                raise ValueError("1116v1 rule identity mismatch")
            if sha(assets / "thresholds.json") != self.identity["stage_rules_sha256"]:
                raise ValueError("1116v1 stage-rule checksum mismatch")
        elif (version != "1113" or tuple(self.identity["progress"]) != THRESHOLDS
              or self.identity["risk"] != RISK or self.identity["confirm"] != 2):
            raise ValueError("1113 rule identity mismatch")
        for key in ("joint.pt", "hover.pt", "statistics.json"):
            if sha(assets / key) != self.identity["assets"][key]:
                raise ValueError(f"{version} checksum mismatch: {key}")
        self.stats = json.loads((assets / "statistics.json").read_text())
        self.device = torch.device(device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.joint = Monitor(self.stats, joint=True)
        self.joint.load_state_dict(torch.load(assets / "joint.pt", map_location="cpu", weights_only=True), strict=True)
        self.joint.eval().to(self.device)
        self.hover = Hover(self.joint.risk).cpu()
        self.hover.load_state_dict(torch.load(assets / "hover.pt", map_location="cpu", weights_only=True), strict=True)
        self.hover.eval()
        self.mean = self.tensor(self.stats["image_mean"])[None, None, :, :, None, None]
        self.std = self.tensor(self.stats["image_std"])[None, None, :, :, None, None]
        self.lut = torch.arange(256, dtype=torch.float32, device=self.device) / 255

    def tensor(self, array):
        return torch.as_tensor(array, device=self.device)

    def transform(self, inputs):
        rlt, state, valid, task, rgb, refs, meta = (
            self.tensor(inputs[k]).clone() for k in ("rlt", "state", "valid", "task", "rgb", "refs", "meta")
        )
        x = self.lut[rgb.long()]
        ref = self.lut[refs.long()][:, None]
        touch = (torch.cat((x, x - ref), dim=3) - self.mean) / self.std
        mask = meta[..., 0].bool()
        touch = touch.masked_fill(~mask[..., None, None, None], 0.)
        age = (meta[..., 2] - self.stats["age_mean"]) / self.stats["age_std"]
        meta[..., 2] = torch.where(mask, age, 0.)
        return rlt, state, valid, task, touch, meta

    @torch.inference_mode()
    def __call__(self, inputs, alternate_task=None):
        x = self.transform(inputs)
        hidden = self.joint.encode(*x)
        tasks = [x[3]]
        if alternate_task is not None:
            tasks.append(torch.full_like(x[3], alternate_task))
        h = hidden.cpu()
        scores = []
        for task in tasks:
            base = route(self.joint.heads, hidden, task).cpu()
            progress, logit = self.hover(h, task.cpu(), base)
            scores.append(dict(progress=progress.numpy(), risk=logit.sigmoid().numpy(),
                               logit=logit.numpy(), hidden=h.numpy(), base=base.numpy()))
        return scores


def pack(history, task):
    rlt = np.zeros((1, 5, 2048), np.float32)
    state = np.zeros((1, 5, 16), np.float32)
    rgb = np.zeros((1, 5, 2, 3, 140, 80), np.uint8)
    valid = np.zeros((1, 5), bool)
    meta = np.zeros((1, 5, 2, 3), np.float32)
    refs = np.zeros((1, 2, 3, 140, 80), np.uint8)
    for i, sample in enumerate(history, 5-len(history)):
        rlt[0, i], state[0, i], rgb[0, i], meta[0, i] = (
            sample["rlt"], sample["state"], sample["rgb"], sample["meta"]
        )
        if sample["refs"] is not None:
            observed = meta[0, i, :, 0].astype(bool)
            refs[0, observed] = sample["refs"][observed]
        valid[0, i] = True
    return dict(rlt=rlt, state=state, valid=valid, task=np.array([task], np.int64),
                rgb=rgb, refs=refs, meta=meta)
