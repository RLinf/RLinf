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

from omegaconf import DictConfig, OmegaConf


class _TensorboardLogger:
    def __init__(self, log_path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_path)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(self, cfg: DictConfig):
        logger_cfg = cfg.runner.logger

        log_path = logger_cfg.get("log_path", "logs")
        project_name = logger_cfg.get("project_name", "rlinf")
        experiment_name = logger_cfg.get("experiment_name", "default")

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        wandb_proxy = logger_cfg.get("wandb_proxy", None)
        swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"

        self.logger = {}
        config = OmegaConf.to_container(cfg, resolve=True)

        if "wandb" in self.logger_backends:
            import wandb

            wandb_log_path = os.path.join(log_path, "wandb")
            os.makedirs(wandb_log_path, exist_ok=True)

            settings = None
            if wandb_proxy:
                settings = wandb.Settings(https_proxy=wandb_proxy)
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                settings=settings,
                dir=wandb_log_path,
            )
            self.logger["wandb"] = wandb

        if "swanlab" in self.logger_backends:
            import swanlab

            swanlab_log_path = os.path.join(log_path, "swanlab")
            os.makedirs(swanlab_log_path, exist_ok=True)

            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config=config,
                logdir=swanlab_log_path,
                mode=swanlab_mode,
            )
            self.logger["swanlab"] = swanlab

        if "tensorboard" in self.logger_backends:
            tensorboard_log_path = os.path.join(log_path, "tensorboard")
            os.makedirs(tensorboard_log_path, exist_ok=True)

            config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
            OmegaConf.save(cfg, config_yaml_path, resolve=True)

            self.logger["tensorboard"] = _TensorboardLogger(tensorboard_log_path)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def log_table(self, df_data, name, step):
        if "wandb" in self.logger_backends:
            table = self.logger["wandb"].Table(dataframe=df_data)
            self.logger["wandb"].log({name: table}, step=step)
        else:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")

    def __del__(self):
        self.finish()

    def finish(self):
        for logger_instance in self.logger.values():
            logger_instance.finish()
