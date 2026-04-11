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

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Final, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_REWARD: Final[float] = 0.4


def fim_llm_as_judge_verify_call(
    responses: list[str],
    references: list[str],
    prompts: list[str],
) -> list:
    assert len(responses) == len(references) == len(prompts), (
        len(responses),
        len(references),
        len(prompts),
    )

    # Deduplicate keys
    unique_keys = []
    key_to_indices: dict[str, list[int]] = {}
    for i, (rp, rs, rf) in enumerate(zip(prompts, responses, references)):
        key = json.dumps([rp, rs, rf], ensure_ascii=False, sort_keys=True)
        unique_keys.append(key)
        key_to_indices.setdefault(key, []).append(i)

    unique_requests = []
    for key in key_to_indices.keys():
        try:
            rp, rs, rf = json.loads(key)
        except Exception:
            rp, rs, rf = prompts[0], responses[0], references[0]
        unique_requests.append((rp, rs, rf))

    results: list[Optional[dict[str, Any]]] = [None] * len(responses)
    rewards: list[float] = []
    success_cnt = 0
    fail_cnt = 0

    max_workers = 32
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(process_single_request, req): key
            for key, req in zip(key_to_indices.keys(), unique_requests)
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
                for idx in key_to_indices[key]:
                    results[idx] = result
            except Exception as e:
                for idx in key_to_indices[key]:
                    results[idx] = {
                        "success": False,
                        "reward": DEFAULT_REWARD,
                        "error": str(e),
                    }

    for i, result in enumerate(results):
        if result and result.get("success"):
            success_cnt += 1
            rewards.append(result["reward"])
        else:
            fail_cnt += 1
            rewards.append(DEFAULT_REWARD)

    return rewards


def create_session_with_retry(
    max_retries: int = 3, backoff_factor: float = 0.3
) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _build_prompt(
    raw_prompt: str, response: str, reference: Optional[str] = None
) -> str:
    prefix_token = "<|fim_prefix|>"
    suffix_token = "<|fim_suffix|>"
    middle_token = "<|fim_middle|>"
    try:
        prefix = raw_prompt.split(suffix_token)[0].split(prefix_token)[-1]
        suffix = raw_prompt.split(middle_token)[0].split(suffix_token)[-1]
    except Exception:
        prefix, suffix = "", ""

    # Do not use reference as scoring reference
    judge_prompt = """
请你作为代码质量评估专家，对给定的代码补全结果进行质量评分。这份评分将用于强化学习训练中的奖励信号，因此请确保评分客观、一致且有区分度。

评估依据信息
<prefix>{prefix}</prefix>
<suffix>{suffix}</suffix>
<completion>{response}</completion>

信息项描述
prefix: 代码的前半部分
suffix: 代码的后半部分
completion: LLM 提供的待评估补全内容（即 Prompt 和 Suffix 之间的部分）。

评分标准如下，采用 0-10 分制，分为 5 个等级（0, 3, 6, 8, 10）：
正确性和功能性（correctness_and_functionality）：
0 分：代码完全不能实现预期功能，存在根本性逻辑错误
3 分：代码能实现部分功能，但存在严重逻辑缺陷或无法处理常见情况
6 分：代码能实现核心功能，但存在一些边缘情况处理不当或 minor 错误
8 分：代码能正确实现所有功能，仅存在极少可忽略的问题
10 分：代码完美实现所有功能，逻辑严谨，能妥善处理各种边缘情况

请基于以上标准对提供的代码补全结果进行评分，并按照以下 XML 格式输出，确保分数为指定的五个等级之一，理由简短具体且有针对性：
```xml
<evaluation>
<criteria_scores>
    <correctness_and_functionality>
    <score>[SCORE]</score>
    <justification>[简短具体的理由]</justification>
    </correctness_and_functionality>
</criteria_scores>
</evaluation>
```
"""
    return judge_prompt.format(prefix=prefix, suffix=suffix, response=response)


def send_reward_request(
    raw_prompt: str,
    response: str,
    reference: str,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
) -> dict[str, Any]:
    url = os.getenv(
        "LLMASJUDGE_API_URL", "https://cloud.infini-ai.com/maas/v1/chat/completions"
    )
    if session is None:
        session = create_session_with_retry()

    api_key = os.getenv("LLMASJUDGE_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    model = os.getenv("LLMASJUDGE_MODEL", "deepseek-v3.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": _build_prompt(raw_prompt, response, reference)},
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
        "seed": 42,
    }

    # Unified default fallback score to avoid undefined due to early exception

    try:
        r = session.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        result = r.json()

        # Parse the first <score>...</score> value; fallback to DEFAULT_REWARD on failure
        try:
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                import re

                xml_match = re.search(r"<evaluation>[\s\S]*?</evaluation>", content)
                if xml_match:
                    xml_str = xml_match.group(0)
                    score_match = re.search(
                        r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", xml_str
                    )
                    if score_match:
                        score_val = float(score_match.group(1))
                        if 0 <= score_val <= 10:
                            reward = score_val / 10.0
        except Exception:
            reward = DEFAULT_REWARD

        return {
            "success": True,
            "reward": reward,
            "raw_response": result,
            "error": None,
        }

    except Exception as e:
        error_msg = f"Request or parsing failed: {e}"
        return {
            "success": False,
            "reward": DEFAULT_REWARD,
            "raw_response": None,
            "error": error_msg,
        }


def process_single_request(args: tuple) -> dict[str, Any]:
    raw_prompt, response, reference = args
    reward_response = send_reward_request(raw_prompt, response, reference)
    return reward_response
