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

"""Exercise real SGLang retraction, weight transport, and exact prefix retention."""

import asyncio
import json
from pathlib import Path

import hydra
from agent.searchr1 import train
from anyio import fail_after
from omegaconf import DictConfig

from rlinf.runners.async_agent_ppo_runner import AsyncAgentPPORunner
from rlinf.utils.logging import get_logger


async def verify(runner: AsyncAgentPPORunner) -> dict:
    """Publish three versions while long requests are still decoding/prefilling."""
    await asyncio.to_thread(runner._publish, initial=True)
    calls = []
    for rank in range(runner.batch_split_num):
        calls.append(
            runner.rollout.execute_on(rank).async_generate(
                input_ids=[
                    [200 + rank, 300 + index, *([1234] * 5998)] for index in range(8)
                ],
                sampling_params={
                    "max_new_tokens": 1024,
                    "min_new_tokens": 1024,
                    "ignore_eos": True,
                    "stop": [],
                    "temperature": 1.0,
                    "top_k": -1,
                    "top_p": 1.0,
                },
                return_logprob=True,
            )
        )
    snapshots = []
    generated = [0] * runner.batch_split_num
    for version in range(1, 4):
        with fail_after(90):
            while True:
                state = await runner.rollout.get_async_rollout_state().async_wait()
                if all(
                    s["generated_tokens"] >= previous + 64
                    for s, previous in zip(state, generated, strict=True)
                ):
                    break
                await asyncio.sleep(0.1)
        await runner.rollout.pause_generation().async_wait()
        paused = await runner.rollout.get_async_rollout_state(
            include_requests=True
        ).async_wait()
        assert all(s["paused"] and s["parked_requests"] > 0 for s in paused), paused
        for state in paused:
            for request in state["requests"].values():
                assert (
                    len(request["output_ids"])
                    == len(request["logprobs"])
                    == len(request["weight_versions"])
                )
        snapshots.extend(state["requests"] for state in paused)
        if runner.release_rollout:
            await runner.rollout.offload_engine().async_wait()
        if version == 1 and runner.cfg.get("verification_checkpoint"):
            await runner.actor.load_checkpoint(
                runner.cfg.verification_checkpoint
            ).async_wait()
        await asyncio.to_thread(runner._sync_weights, weight_version=version)
        installed = await runner.rollout.get_async_rollout_state(
            include_requests=True
        ).async_wait()
        assert all(
            state["paused"] and state["weight_version"] == version
            for state in installed
        )
        for before, after in zip(paused, installed, strict=True):
            assert before["requests"] == after["requests"]
        generated = [state["generated_tokens"] for state in installed]
        await runner.rollout.resume_generation().async_wait()

    completed = {}
    for call in calls:
        outputs, _ = (await call.async_wait())[0]
        for output in outputs:
            info = output["meta_info"]
            ids = output["output_ids"]
            logprobs = info["output_token_logprobs"]
            versions = info["rollout_weight_versions"]
            assert len(ids) == len(logprobs) == len(versions) == 1024
            assert ids == [entry[1] for entry in logprobs]
            assert versions == sorted(versions) and versions[-1] == 3
            completed[info["id"]] = (ids, [entry[0] for entry in logprobs], versions)
    checked = 0
    for snapshot in snapshots:
        for rid, prefix in snapshot.items():
            ids, logprobs, versions = completed[rid]
            length = len(prefix["output_ids"])
            assert ids[:length] == prefix["output_ids"]
            assert logprobs[:length] == prefix["logprobs"]
            assert versions[:length] == prefix["weight_versions"]
            checked += length
    assert any(len(set(versions)) > 1 for _, _, versions in completed.values())
    observed_versions = sorted(
        {version for _, _, versions in completed.values() for version in versions}
    )
    assert observed_versions == [0, 1, 2, 3], observed_versions
    return {
        "requests": len(completed),
        "verified_prefix_tokens": checked,
        "publications": 3,
        "observed_versions": observed_versions,
        "used_trained_checkpoint": bool(runner.cfg.get("verification_checkpoint")),
    }


class _EngineVerificationRunner(AsyncAgentPPORunner):
    """Run the engine probe after the training entry point initializes workers."""

    def run(self) -> None:
        result = asyncio.run(verify(self))
        output = Path(self.cfg.runner.logger.log_path) / "engine_verification.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        get_logger().info(f"SGLang async verification passed: {result}")
        self.metric_logger.finish()


@hydra.main(version_base="1.1")
def main(cfg: DictConfig) -> None:
    if cfg.runner.synchronous:
        raise ValueError(
            "Engine verification requires an asynchronous SearchR1 config."
        )
    async_runner = train.AsyncAgentPPORunner
    train.AsyncAgentPPORunner = _EngineVerificationRunner
    try:
        train.main(cfg)
    finally:
        train.AsyncAgentPPORunner = async_runner


if __name__ == "__main__":
    main()
