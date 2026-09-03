# Copyright 2025 The RLinf Authors.
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

import os
from collections.abc import Iterable

from omegaconf import DictConfig, OmegaConf


class _Backend:
    """A metric backend whose teardown runs at most once."""

    def __init__(self):
        self._finished = False

    def finish(self) -> None:
        if not self._finished:
            self._close()
            self._finished = True

    def _close(self) -> None:
        raise NotImplementedError


class _TensorboardLogger(_Backend):
    def __init__(self, log_path):
        from torch.utils.tensorboard import SummaryWriter

        super().__init__()
        self.writer = SummaryWriter(log_path)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def _close(self) -> None:
        self.writer.close()


class _RunLogger(_Backend):
    """A backend bound to the run object its ``init`` returned.

    Logging through the run instead of the module keeps concurrently active
    runs of the same backend from writing into whichever one was created last.
    """

    def __init__(self, run):
        super().__init__()
        self.run = run

    def log(self, data: dict, step: int) -> None:
        self.run.log(data, step=step)

    def _close(self) -> None:
        self.run.finish()


class _WandbLogger(_RunLogger):
    def __init__(self, module, run):
        super().__init__(run)
        self.module = module

    def log_table(self, df_data, name: str, step: int) -> None:
        self.run.log({name: self.module.Table(dataframe=df_data)}, step=step)


def _finish_backends(backends: Iterable[_Backend]) -> None:
    """Close every backend, re-raising the first failure once all are closed."""
    first_error = None
    for backend in backends:
        try:
            backend.finish()
        except Exception as exc:  # noqa: BLE001 - close the remaining backends
            first_error = first_error or exc
    if first_error is not None:
        raise first_error


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(self, cfg: DictConfig):
        self._all_loggers = []
        self.cfg = cfg
        logger_cfg = cfg.runner.logger

        self.log_path = logger_cfg.get("log_path", "logs")
        self.project_name = logger_cfg.get("project_name", "rlinf")
        self.experiment_name = logger_cfg.get("experiment_name", "default")
        self.per_worker_log = bool(cfg.runner.get("per_worker_log", False))
        self.per_worker_log_root = cfg.runner.get(
            "per_worker_log_path", os.path.join(self.log_path, "worker_logs")
        )

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        self.wandb_proxy = logger_cfg.get("wandb_proxy", None)
        self.wandb_entity = logger_cfg.get("wandb_entity", None)
        self.swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"
        if self.per_worker_log and "swanlab" in self.logger_backends:
            raise ValueError(
                "SwanLab supports only one active run per process; "
                "disable runner.per_worker_log or drop the swanlab backend."
            )

        self.config = OmegaConf.to_container(cfg, resolve=True)
        self._worker_loggers: dict[tuple[str, int], dict] = {}
        self.logger = self._create_logger_bundle(
            log_path=self.log_path,
            experiment_name=self.experiment_name,
            log_path_suffix="all" if self.per_worker_log else "",
        )

    def _create_logger_bundle(
        self, log_path: str, experiment_name: str, log_path_suffix: str = ""
    ) -> dict:
        logger = {}
        try:
            if "wandb" in self.logger_backends:
                import wandb

                wandb_log_path = os.path.join(log_path, "wandb", log_path_suffix)
                os.makedirs(wandb_log_path, exist_ok=True)

                settings = None
                if self.wandb_proxy:
                    settings = wandb.Settings(https_proxy=self.wandb_proxy)
                run = wandb.init(
                    entity=self.wandb_entity,
                    project=self.project_name,
                    name=experiment_name,
                    config=self.config,
                    settings=settings,
                    dir=wandb_log_path,
                    # "create_new" keeps per-worker runs alive side by side; it
                    # needs wandb >= 0.19.10.
                    reinit="create_new" if self.per_worker_log else True,
                )
                if run is None:
                    raise RuntimeError("wandb.init() returned no run to log to")
                logger["wandb"] = _WandbLogger(wandb, run)

            if "swanlab" in self.logger_backends:
                import swanlab

                swanlab_log_path = os.path.join(log_path, "swanlab", log_path_suffix)
                os.makedirs(swanlab_log_path, exist_ok=True)

                run = swanlab.init(
                    project=self.project_name,
                    experiment_name=experiment_name,
                    config=self.config,
                    logdir=swanlab_log_path,
                    mode=self.swanlab_mode,
                )
                if run is None:
                    raise RuntimeError("swanlab.init() returned no run to log to")
                logger["swanlab"] = _RunLogger(run)

            if "tensorboard" in self.logger_backends:
                tensorboard_log_path = os.path.join(
                    log_path, "tensorboard", log_path_suffix
                )
                os.makedirs(tensorboard_log_path, exist_ok=True)

                config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
                OmegaConf.save(self.cfg, config_yaml_path, resolve=True)

                logger["tensorboard"] = _TensorboardLogger(tensorboard_log_path)
        except BaseException:
            try:
                _finish_backends(logger.values())
            except Exception:  # noqa: BLE001 - keep the construction failure
                pass
            raise

        self._all_loggers.append(logger)
        return logger

    def _get_scoped_logger(self, worker_group_name: str, rank: int) -> dict:
        key = (worker_group_name, int(rank))
        if key in self._worker_loggers:
            return self._worker_loggers[key]

        scoped_log_path = os.path.join(
            self.per_worker_log_root,
            worker_group_name,
            f"rank_{int(rank)}",
        )
        scoped_experiment_name = (
            f"{self.experiment_name}-{worker_group_name}-rank_{int(rank)}"
        )
        scoped_logger = self._create_logger_bundle(
            log_path=scoped_log_path,
            experiment_name=scoped_experiment_name,
        )
        self._worker_loggers[key] = scoped_logger
        return scoped_logger

    def log(
        self,
        data,
        step,
        backend=None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ):
        target_logger = self.logger
        if self.per_worker_log and worker_group_name is not None and rank is not None:
            target_logger = self._get_scoped_logger(
                worker_group_name=worker_group_name,
                rank=rank,
            )
        for default_backend, logger_instance in target_logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def log_table(self, df_data, name, step):
        if "wandb" in self.logger_backends:
            self.logger["wandb"].log_table(df_data=df_data, name=name, step=step)
        else:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")

    def __del__(self):
        try:
            self.finish()
        except Exception:  # noqa: BLE001 - a destructor must not raise
            pass

    def finish(self):
        """Close every backend. Safe to call more than once."""
        _finish_backends(
            backend for logger in self._all_loggers for backend in logger.values()
        )
