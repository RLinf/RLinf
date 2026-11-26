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

import json
import random

import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from rlinf.agents.envscaler.agent_loop import EnvScalerAgentLoopWorker
from rlinf.data.datasets.reasoning.envscaler import EnvScalerDataset
from rlinf.data.datasets.reasoning.math_prompt import format_dapo_math_prompt
from rlinf.runners.agent_training import run_agent_training
from rlinf.utils.utils import output_redirector
from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput


class EnvScalerDAPODataset(EnvScalerDataset):
    """Pair equal numbers of EnvScaler scenarios and DAPO questions per batch."""

    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizerBase):
        self.math_path = cfg.data.math_data_path
        super().__init__(cfg.data.train_data_paths, cfg, tokenizer)
        batch_size = cfg.data.rollout_batch_size
        if (
            cfg.data.shuffle
            or batch_size < 2
            or batch_size % 2
            or cfg.runner.max_steps < 1
        ):
            raise ValueError(
                "Expected sequential sampling, an even batch size and positive max_steps"
            )
        per_source = batch_size // 2
        count = cfg.runner.max_steps * per_source
        tooluse = [row for row in self.data if row[self.route_field] == "tooluse"]
        math_by_prompt = {
            row[self.prompt_key]: row
            for row in self.data
            if row[self.route_field] == "math"
        }
        if len(tooluse) != self.scenario_count:
            raise ValueError(
                f"Prompt filtering retained {len(tooluse)} of {self.scenario_count} EnvScaler scenarios"
            )
        if not tooluse or len(math_by_prompt) < count:
            raise ValueError(
                f"Expected EnvScaler scenarios and at least {count} distinct eligible DAPO questions, got {len(tooluse)} and {len(math_by_prompt)}"
            )
        rng = random.Random(cfg.data.seed)
        rng.shuffle(tooluse)
        math = rng.sample(list(math_by_prompt.values()), count)
        self.data = []
        for start in range(0, count, per_source):
            batch = math[start : start + per_source]
            batch.extend(
                tooluse[i % len(tooluse)] for i in range(start, start + per_source)
            )
            rng.shuffle(batch)
            self.data.extend(batch)

    def _load_data(self) -> list[dict]:
        rows = super()._load_data()
        self.scenario_count = len(rows)
        for row in rows:
            row[self.route_field] = "tooluse"
        with open(self.math_path, encoding="utf-8") as source:
            for line in source:
                question = json.loads(line)
                for message in question["prompt"]:
                    content = message.get("content")
                    if message["role"] == "user" and isinstance(content, str):
                        message["content"] = format_dapo_math_prompt(content)
                prompt = self.tokenizer.apply_chat_template(
                    question["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.apply_chat_template_kwargs,
                )
                rows.append(
                    {
                        self.prompt_key: prompt,
                        self.answer_key: [
                            str(question["reward_model"]["ground_truth"])
                        ],
                        self.route_field: "math",
                    }
                )
        return rows


class EnvScalerDAPOAgentLoopWorker(EnvScalerAgentLoopWorker):
    """Run EnvScaler scenarios interactively and DAPO questions for one turn."""

    async def run_one_query(
        self, prompt_ids: list[int], answer: dict | list[str]
    ) -> MultiAgentLoopOutput:
        """Return policy-generated turns for the routed teacher to score."""
        if isinstance(answer, dict):
            result = await super().run_one_query(prompt_ids, answer)
            for turn in result.single_turn_outputs:
                turn.reward_score = None
            return result
        seq_length = min(
            self.cfg.runner.seq_length,
            self.cfg.agentloop.get("math_seq_length", self.cfg.runner.seq_length),
        )
        budget = min(
            self.cfg.algorithm.sampling_params.max_new_tokens,
            seq_length - len(prompt_ids),
        )
        if budget <= 0:
            raise ValueError(
                f"DAPO prompt has {len(prompt_ids)} tokens, expected fewer than {seq_length}"
            )
        generation = await self.generate(
            prompt_ids, sampling_params={"max_new_tokens": budget}
        )
        if not generation["output_ids"]:
            raise RuntimeError("Expected a nonempty DAPO response")
        turn = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=generation["output_ids"],
            response_logprobs=generation["logprobs"] if self.return_logprobs else None,
            response_topk_token_ids=generation.get("topk_token_ids"),
            response_mc_samples=generation.get("mc_samples"),
            version_start=generation.get("version_start"),
            version_end=generation.get("version_end"),
            is_end=generation.get("finish_reason") != "length",
        )
        return MultiAgentLoopOutput(single_turn_outputs=[turn])


def create_mopd_datasets(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerBase
) -> tuple[EnvScalerDAPODataset, None]:
    """Build the fixed mixture for a pure distillation run."""
    if cfg.distillation.distillation_loss.use_task_rewards:
        raise ValueError("EnvScaler/DAPO MOPD requires use_task_rewards=false")
    return EnvScalerDAPODataset(cfg, tokenizer), None


@hydra.main(version_base="1.1", config_path="config", config_name="train_qwen3_8b_mopd")
@output_redirector
def main(cfg: DictConfig) -> None:
    """Train Qwen3 on EnvScaler and DAPO with a teacher per task."""
    run_agent_training(
        cfg, EnvScalerDAPOAgentLoopWorker, create_datasets=create_mopd_datasets
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
