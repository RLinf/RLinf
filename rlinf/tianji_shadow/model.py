"""Touch-add monitor and task-conditioned progress readouts."""
import torch
from torch import nn

from .network_base import ProgressNet


def route(heads, h, task):
    scores = torch.stack([head(h).squeeze(-1) for head in heads], dim=1)
    return scores.gather(1, task[:, None]).squeeze(1)


class Monitor(ProgressNet):
    def __init__(self, stats, joint=False):
        super().__init__('touch_add')
        self.state_mean.copy_(torch.tensor(stats['state_mean']))
        self.state_std.copy_(torch.tensor(stats['state_std']))
        self.risk = nn.Sequential(nn.Linear(832, 128), nn.GELU(), nn.Linear(128, 1)) if joint else None
        if joint:
            nn.init.normal_(self.risk[-1].weight, std=.01)
            nn.init.zeros_(self.risk[-1].bias)

    def encode(self, rlt, state, valid, task, touch, meta):
        z = torch.cat((self.rlt_mlp(rlt), self.state_mlp((state - self.state_mean) / self.state_std)), -1)
        tz = self.touch(touch, meta).flatten(-2)
        dz = self.touch_projection(torch.cat((tz, meta.flatten(-2)), -1))
        z = z + dz * meta[..., 0].any(-1)[..., None]
        z = z.masked_fill(~valid[..., None], 0.) + self.positions
        return self.temporal(z, src_key_padding_mask=~valid)[:, -1]

    def forward(self, *x):
        h = self.encode(*x)
        return 4 * route(self.heads, h, x[3]).sigmoid(), self.risk(h).squeeze(-1) if self.risk else None


class Hover(nn.Module):
    def __init__(self, risk):
        import copy

        super().__init__()
        self.risk = copy.deepcopy(risk)
        self.adapter = nn.Sequential(nn.Linear(832, 64), nn.GELU(), nn.Linear(64, 832))
        self.progress = nn.Sequential(nn.Linear(832, 64), nn.GELU(), nn.Linear(64, 4))
        for last in (self.adapter[-1], self.progress[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def progress_only(self, h, task, progress_logit):
        delta = self.progress(h).gather(1, task[:, None]).squeeze(1)
        return 4 * (progress_logit + delta).sigmoid()

    def risk_only(self, h):
        return self.risk(h + self.adapter(h)).squeeze(-1)

    def forward(self, h, task, progress_logit):
        return self.progress_only(h, task, progress_logit), self.risk_only(h)
