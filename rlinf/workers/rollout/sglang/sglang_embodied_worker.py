# Copyright 2026 The RLinf Authors.
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

"""Embodied sglang rollout worker: spawn a ``sglang serve`` and drive a
registered action policy over channels (no worker-owned HTTP server).

Used by embodied action-policy models (e.g. Cosmos3). The worker spawns (or
connects to) a standalone ``sglang serve`` subprocess driven by
``rollout.sglang``, loads the action policy registered for
``rollout.model.model_type``, and is driven by ``EmbodiedEvalRunner`` over
channels (``recv_from``/``send_to``) — it does NOT host its own HTTP server
(the agent path uses :class:`SGLangAgentWorkerWithHTTPServer`).
"""

import os
import tempfile
from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig, open_dict as _open_dict

from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker


class SGLangEmbodiedWorker(SGLangWorker):
    """Spawn ``sglang serve`` + action policy + channel eval (no own HTTP)."""

    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        weight_reload="sync",
        config_rollout: Optional[DictConfig] = None,
    ):
        super().__init__(config, placement, weight_reload, config_rollout)
        # Embodied action-policy path. model_type selects the action policy (see
        # rlinf/workers/rollout/sglang/action_policies); the policy turns env obs
        # into action chunks by calling the launched sglang serve.
        self.model_type = str(
            getattr(getattr(self._cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.action_policy = None
        self.sglang_server_url = None
        self.sglang_proc = None
        self.sglang_log_path = None
        self.sglang_log_fh = None
        self.obs_tmpdir = None
        # NOTE: no _setup_http_routes() — this worker does not host HTTP; the
        # embodied eval is driven over channels, not via the /evaluate route.

        # Channel-eval attrs (mirror MultiStepRolloutWorker.__init__ so
        # EmbodiedEvalRunner can drive this worker). Set at construction, not in
        # init_worker, so they're available before the serve is spawned.
        cfg = self._cfg
        self.cfg = cfg  # MultiStepRolloutWorker exposes self.cfg; mirror it
        self.model_cfg = self._cfg_rollout.model
        # This worker is eval-only (spawn a serve + channel eval; no training).
        assert cfg.runner.get("only_eval", True), (
            "SGLangEmbodiedWorker is eval-only; set runner.only_eval: true"
        )
        self.only_eval = True
        eval_env_cfg = cfg.env.get("eval", None)
        self.num_pipeline_stages = int(cfg.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.total_num_eval_envs = total_eval
        self.eval_batch_size = (
            total_eval // self.num_pipeline_stages if self.num_pipeline_stages else total_eval
        )
        self.eval_rollout_epoch = (
            int(eval_env_cfg.rollout_epoch) if eval_env_cfg else 1
        )
        if eval_env_cfg is not None:
            self.n_eval_chunk_steps = int(
                eval_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
        else:
            self.n_eval_chunk_steps = 0
        self.env_decoupled_mode = cfg.runner.get("enable_decoupled_mode", False)
        self.collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)

    async def init_worker(self):
        # Spawn (or connect to) a standalone ``sglang serve`` (driven by
        # ``rollout.sglang``), then load the model's registered action policy.
        # No worker-owned HTTP server (the eval loop is channel-based); eval
        # attrs are already set in __init__.
        self._init_sglang_server()
        policy_cls = None
        if self.model_type:
            from rlinf.workers.rollout.sglang.action_policies import (
                get_action_policy_cls,
            )

            try:
                policy_cls = get_action_policy_cls(self.model_type)
            except ValueError:
                policy_cls = None
        if policy_cls is None:
            raise RuntimeError(
                f"no action policy registered for model_type "
                f"'{self.model_type}'; cannot run the embodied sglang path"
            )
        self.action_policy = policy_cls(
            self._cfg, self.sglang_server_url, self._rank
        )

    def shutdown(self):
        """Kill the spawned sglang serve subprocess."""
        self.shutdown_sglang_server()

    def _init_sglang_server(self) -> None:
        """Spawn a ``sglang serve`` subprocess and wait for ``/health``.

        Driven by ``rollout.sglang`` (model-agnostic). CUDA_VISIBLE_DEVICES is
        already set by Ray (isolate_gpu), so the subprocess inherits the
        assigned GPU(s).
        """
        import shutil
        import subprocess

        import requests

        # The local sglang serve is reached over 127.0.0.1; never route those
        # localhost calls through an upstream HTTP(S)_PROXY captured in the
        # launch env, or /health polling and action requests would be proxied
        # to a host that cannot reach this worker's loopback and hang/retry.
        _local_hosts = "127.0.0.1,localhost,::1"
        _no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
        if not any(h in _no_proxy for h in ("127.0.0.1", "localhost")):
            os.environ["NO_PROXY"] = (
                f"{_no_proxy},{_local_hosts}".strip(",") if _no_proxy else _local_hosts
            )

        # All serve-launch fields live flat under ``rollout.sglang`` (host,
        # port_base, num_gpus, tp_size, ...). 0 / null / empty = omit the flag
        # (sglang default).
        sglang_cfg = self._cfg_rollout.get("sglang") or {}

        # Resolve the checkpoint path: prefer rollout.model_path (convenience
        # override directly under rollout:), fall back to rollout.model.model_path.
        model_cfg = self._cfg_rollout.model
        rollout_model_path = self._cfg_rollout.get("model_path", None)
        model_path = rollout_model_path or model_cfg.model_path
        with _open_dict(model_cfg):
            model_cfg.model_path = model_path

        self.obs_tmpdir = tempfile.mkdtemp(prefix="sglang_serve_obs_")
        self.sglang_proc = None

        # Always spawn a ``sglang serve`` subprocess (internal-only).
        host = str(getattr(sglang_cfg, "host", "127.0.0.1"))
        port = int(getattr(sglang_cfg, "port_base", 30010))
        master_port = int(getattr(sglang_cfg, "master_port_base", 30100))
        sglang_bin = str(getattr(sglang_cfg, "sglang_bin", "sglang"))
        if not shutil.which(sglang_bin):
            # fall back to `python -m sglang` if the console script is absent
            sglang_bin = "sglang"

        cmd = [
            sglang_bin, "serve",
            "--model-path", model_path,
            "--host", host,
            "--port", str(port),
            "--master-port", str(master_port),
        ]
        if (v := int(getattr(sglang_cfg, "num_gpus", 0) or 0)) > 0:
            # GPUs the serve occupies
            cmd += ["--num-gpus", str(v)]
        if (v := int(getattr(sglang_cfg, "tp_size", 0) or 0)) > 0:
            # = num_gpus, shard the DiT
            cmd += ["--tp-size", str(v)]
        if (v := int(getattr(sglang_cfg, "ulysses_degree", 0) or 0)) > 0:
            # 1 = no sequence parallelism
            cmd += ["--ulysses-degree", str(v)]
        if (v := int(getattr(sglang_cfg, "ring_degree", 0) or 0)) > 0:
            cmd += ["--ring-degree", str(v)]
        if (v := int(getattr(sglang_cfg, "data_parallel_size", 0) or 0)) > 0:
            cmd += ["--data-parallel-size", str(v)]
        if isinstance(getattr(sglang_cfg, "enable_cfg_parallel", None), bool):
            cmd += ["--enable-cfg-parallel",
                    "true" if getattr(sglang_cfg, "enable_cfg_parallel") else "false"]
        # --- performance preset ---
        if (v := getattr(sglang_cfg, "performance_mode", None)):
            cmd += ["--performance-mode", str(v)]
        env = os.environ.copy()
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        self.sglang_log_path = os.path.join(
            self.obs_tmpdir, f"sglang_serve_rank{self._rank}.log"
        )
        self.sglang_log_fh = open(self.sglang_log_path, "ab")
        self.log_info(f"sglang serve: launching: {' '.join(cmd)}")
        self.log_info(f"sglang serve: log -> {self.sglang_log_path}")
        self.sglang_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.sglang_log_fh,
            stderr=subprocess.STDOUT,
        )
        self.sglang_server_url = f"http://{host}:{port}"

        # Wait for /health (the server loads the model on first boot, so allow
        # a generous timeout).
        spawn_timeout = float(getattr(sglang_cfg, "spawn_timeout", 1800.0))
        self.wait_sglang_server_ready(self.sglang_server_url, spawn_timeout, requests)
        self.log_info(f"sglang serve ready at {self.sglang_server_url}")

    def wait_sglang_server_ready(self, url: str, timeout: float, requests_mod) -> None:
        """Poll the serve's ``/health`` until it returns 200 (model loaded + warmup done).

        Two failure modes caught early (before the full timeout):
        - Subprocess crashed: read the last 8 KB of the serve log and raise with
          the tail so the error is actionable (we hit NO_PROXY, PIPE deadlock,
          cfg-parallel rejection etc. this way).
        - /health never 200 within timeout: raise with the last HTTP error.
        """
        import time as _time

        deadline = _time.monotonic() + timeout
        last_err = None
        while _time.monotonic() < deadline:
            # (1) If the subprocess exited, the serve will never come up — fail
            # fast with the log tail instead of waiting the full timeout.
            proc = self.sglang_proc
            if proc is not None and proc.poll() is not None:
                tail = b""
                try:
                    log_path = getattr(self, "sglang_log_path", None)
                    if log_path and os.path.exists(log_path):
                        with open(log_path, "rb") as f:
                            tail = f.read()[-8192:]
                    elif proc.stdout:
                        tail = proc.stdout.read()
                except Exception:
                    pass
                raise RuntimeError(
                    f"sglang server subprocess exited (code={proc.returncode}) "
                    f"before /health. Tail:\n{tail.decode(errors='replace')[-4000:]}"
                )
            # (2) Poll /health; 200 means the model is loaded and the serve is
            # ready to accept /v1/videos requests.
            try:
                r = requests_mod.get(f"{url}/health", timeout=5.0)
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = str(e)
            _time.sleep(2.0)
        raise RuntimeError(
            f"sglang server did not become ready at {url}/health within {timeout}s "
            f"(last error: {last_err})"
        )

    def shutdown_sglang_server(self) -> None:
        """Terminate the spawned serve subprocess (SIGTERM → wait 30s → SIGKILL)."""
        proc = getattr(self, "sglang_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.sglang_proc = None
        log_fh = getattr(self, "sglang_log_fh", None)
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass
        self.sglang_log_fh = None

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            b["obs"] if "obs" in b else b for b in obs_batches
        ]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0].keys():
            values = [d[key] for d in obs_dicts]
            first = next((v for v in values if v is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for sub in values for item in sub]
            else:
                merged[key] = values
        return {"obs": merged}

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """env_obs -> action chunks [N, num_action_chunks, action_dim].

        Delegates to the registered action policy (selected by model_type);
        the policy builds the model-specific request, calls the launched
        sglang serve, and parses the returned action.
        """
        if self.action_policy is None:
            raise RuntimeError(
                "no action policy loaded (init_worker not called, or model_type "
                f"'{self.model_type}' has no registered policy)"
            )
        return self.action_policy.infer(env_obs, mode=mode)

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner.

        Mirrors MultiStepRolloutWorker.evaluate's non-decoupled path: recv an
        obs batch from env, predict actions, send them back. Inherited from the
        SGLangWorker base's channel machinery (recv_from/send_to).
        """
        from tqdm import tqdm

        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=self._infer_env_batch_size,
                    ).async_wait()
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    if isinstance(actions, torch.Tensor):
                        actions = actions.detach().cpu().contiguous()
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )