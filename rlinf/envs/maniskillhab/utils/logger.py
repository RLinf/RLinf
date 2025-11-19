import os
import os.path as osp
import pickle
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import wandb as wb
from omegaconf import OmegaConf

import numpy as np


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


@dataclass
class LoggerConfig:
    workspace: str
    exp_name: str = "default_exp"
    clear_out: bool = False

    checkpoint_logger: bool = True

    tensorboard: bool = False
    wandb: bool = False
    project_name: Optional[str] = None
    wandb_cfg: Dict = field(default_factory=dict)

    exp_cfg: Dict = field(default_factory=dict)
    best_stats_cfg: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.exp_path = Path(self.workspace) / self.exp_name
        self.model_path = self.exp_path / "models"
        self.train_video_path = self.exp_path / "train_videos"
        self.eval_video_path = self.exp_path / "eval_videos"
        self.log_path = self.exp_path / "logs"


class Logger:
    def __init__(
        self,
        logger_cfg: LoggerConfig,
        save_fn: Callable = None,
    ):
        self.tensorboard = logger_cfg.tensorboard
        self.tb_writer = None
        self.wandb = logger_cfg.wandb
        self.wb_run = None

        self.exp_path = logger_cfg.exp_path
        self.model_path = logger_cfg.model_path
        self.train_video_path = logger_cfg.train_video_path
        self.eval_video_path = logger_cfg.eval_video_path
        self.log_path = logger_cfg.log_path

        self.checkpoint_logger = logger_cfg.checkpoint_logger

        self.best_stats = dict()
        self.last_log_step = 0
        if logger_cfg.clear_out:
            if osp.exists(self.exp_path):
                shutil.rmtree(self.exp_path, ignore_errors=True)
        elif (
            self.checkpoint_logger
            and (self.log_path / "logger_state_dict.pkl").exists()
        ):
            with open(self.log_path / "logger_state_dict.pkl", "rb") as f:
                self.load(pickle.load(f))

        for x in [
            self.exp_path,
            self.model_path,
            self.train_video_path,
            self.eval_video_path,
            self.log_path,
        ]:
            os.makedirs(x, exist_ok=True)

        self._init_tb()
        self._init_wb(
            (
                logger_cfg.project_name
                if logger_cfg.project_name is not None
                else logger_cfg.workspace
            ),
            logger_cfg.exp_name,
            logger_cfg.wandb_cfg,
            logger_cfg.exp_cfg,
        )
        self._save_config(logger_cfg.exp_cfg)

        self.data = defaultdict(dict)
        self.data_log_summary = defaultdict(dict)
        self.stats = dict()
        self.best_stats_cfg = logger_cfg.best_stats_cfg
        self.save_fn = save_fn

    def _init_wb(self, project_name, exp_name, wandb_cfg, exp_cfg):
        if self.wandb:
            if "wandb_id" in exp_cfg and exp_cfg["wandb_id"] is not None:
                self.wb_run = wb.init(
                    project=project_name,
                    name=exp_name,
                    id=exp_cfg["wandb_id"],
                    resume="must",
                    **wandb_cfg,
                )
                self.last_log_step = max(self.last_log_step, self.wb_run.step)
            else:
                self.wb_run = wb.init(project=project_name, name=exp_name, **wandb_cfg)
                exp_cfg["wandb_id"] = self.wb_run.id

    def _init_tb(self):
        if self.tensorboard:
            from tensorboardX import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=self.log_path)

    def _save_config(self, config: Union[Dict, OmegaConf], verbose=2):
        if type(config) == type(OmegaConf.create()):
            config = OmegaConf.to_container(config)
        if self.wandb:
            wb.config.update(config, allow_val_change=True)
        config_path = self.exp_path / "config.yml"
        if verbose > 1:
            self.print("Saving config:\n", color="cyan", bold=True)
            self.print(config)
        with open(config_path, "w") as out:
            out.write(OmegaConf.to_yaml(config))

    def store(self, tag="default", log_summary=False, **kwargs):
        for k, v in kwargs.items():
            self.data[tag][k] = v
            self.data_log_summary[tag][k] = log_summary

    def get_data(self, tag=None):
        if tag is None:
            data_dict = {}
            for tag in self.data.keys():
                for k, v in self.data[tag].items():
                    data_dict[f"{tag}/{k}"] = v
            return data_dict
        return self.data[tag]

    def print(self, *msg, file=sys.stdout, color=None, bold=False, flush=False):
        if color is None:
            print(*msg, file=file, flush=flush)
        else:
            print(
                *[colorize(m, color=color, bold=bold) for m in msg],
                file=file,
                flush=flush,
            )

    def pretty_print_table(
        self, data, color=None, bold=False, flush=False, print_borders=False
    ):
        # Code from spinning up
        vals = []
        key_lens = [len(key) for key in data.keys()]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        if print_borders:
            print("-" * n_slashes)
        for key in sorted(data.keys()):
            val = data[key]
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            self.print(
                fmt % (key, valstr),
                color=color,
                bold=bold,
                flush=flush and not print_borders,
            )
            vals.append(val)
        if print_borders:
            print("-" * n_slashes, flush=flush)

    def pretty_print_borderless_table(self, data, color=None, bold=False, flush=False):
        # Code from spinning up
        max_key_len = 30
        keystr = "%" + "%d" % max_key_len
        fmt = keystr + "s"
        for key in sorted(data.keys()):
            val = data[key]
            if not isinstance(val, list):
                val = [val]
            this_fmt = "  ".join([fmt] * (len(val) + 1))
            self.print(
                this_fmt % (key, *[str(v)[:max_key_len] for v in val]),
                color=color,
                bold=bold,
                flush=flush,
            )

    def log(self, step, local_only=False):
        assert (
            step >= self.last_log_step
        ), f"logged at step {step} but previously logged at step {self.last_log_step}"
        self.last_log_step = step

        for tag in self.data.keys():
            data_dict = self.data[tag]
            for k, v in data_dict.items():
                key_vals = dict()
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    if len(v) > 0:
                        vals = np.array(v)
                        vals_sum, n = vals.sum(), len(vals)
                        avg = vals_sum / n
                        key_vals = {f"{tag}/{k}_avg": avg}
                        if self.data_log_summary[tag][k]:
                            sum_sq = np.sum((vals - avg) ** 2)
                            std = np.sqrt(sum_sq / n)
                            minv = np.min(vals)
                            maxv = np.max(vals)
                            key_vals = {
                                **key_vals,
                                f"{tag}/{k}_std": std,
                                f"{tag}/{k}_min": minv,
                                f"{tag}/{k}_max": maxv,
                            }
                else:
                    key_vals = {f"{tag}/{k}": v}
                for name, scalar in key_vals.items():
                    if name in self.best_stats_cfg:
                        sort_order = self.best_stats_cfg[name]
                        update_val = False
                        if name not in self.best_stats:
                            update_val = True
                        else:
                            prev_val = self.best_stats[name]["val"]
                            if (sort_order == 1 and prev_val < scalar) or (
                                sort_order == -1 and prev_val > scalar
                            ):
                                update_val = True
                        if update_val:
                            self.best_stats[name] = dict(
                                val=scalar, step=self.last_log_step
                            )
                            fmt_name = name.replace("/", "_")
                            if self.save_fn is not None:
                                self.save_fn(
                                    self.model_path / f"best_{fmt_name}_ckpt.pt"
                                )
                            if self.checkpoint_logger:
                                self.save()
                            self.print(
                                f"{name} new best at {self.last_log_step}: {scalar}",
                                color="cyan",
                                bold=True,
                            )
                    if self.tensorboard and not local_only:
                        self.tb_writer.add_scalar(name, scalar, self.last_log_step)
                    self.stats[name] = scalar
                if self.wandb and not local_only:
                    self.wb_run.log(data=key_vals, step=self.last_log_step)

        return self.stats

    def reset(self):
        self.data = defaultdict(dict)
        self.stats = {}

    def state_dict(self):
        return dict(best_stats=self.best_stats, last_log_step=self.last_log_step)

    def load(self, data):
        self.best_stats = data["best_stats"]
        self.last_log_step = data["last_log_step"]
        return self

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.log_path / "logger_state_dict.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def close(self):
        if self.checkpoint_logger:
            self.save()
        if self.tensorboard:
            self.tb_writer.close()
        if self.wandb:
            wb.finish()
