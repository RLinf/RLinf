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

import re

from rlinf.config import SupportedModel


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("boxed")
    if idx < 0:
        return None
    s = text[idx + len("boxed") :].strip()
    if not s:
        return None
    if s[0] != "{":
        return s.split("$")[0].strip() or None

    depth = 0
    out = []
    for ch in s:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(ch)
    ans = "".join(out).strip()
    return ans or None


def vlm_extract_answer(text: str, model_type) -> str:
    if SupportedModel(model_type) not in [
        SupportedModel.QWEN2_5_VL_SFT,
        SupportedModel.QWEN3_VL_SFT,
        SupportedModel.QWEN3_VL_MOE_SFT,
    ]:
        raise ValueError(f"not support such model type {model_type} for SFT right now.")

    if not text:
        return ""

    # 1) Get the last assistant span from common chat templates.
    patterns = [
        r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
        r"<\|assistant\|>\s*(.*?)(?:<\|end\|>|$)",
    ]
    body = None
    for p in patterns:
        matches = re.findall(p, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            body = matches[-1].strip()
            break
    if body is None:
        body = text.strip()

    # 2) Remove reasoning blocks if present.
    body = re.sub(
        r"<think>.*?</think>", "", body, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # 3) Remove chat special tokens (e.g., <|im_end|>, <|endoftext|>)
    body = re.sub(r"<\|[^>]+?\|>", " ", body).strip()

    # 4) Try explicit "final answer" markers.
    marker_patterns = [
        r"(?:final answer is|the answer is)\s*[:：]?\s*(.+)$",
        r"(?:answer)\s*[:：]\s*(.+)$",
    ]
    for p in marker_patterns:
        m = re.search(p, body, flags=re.IGNORECASE | re.DOTALL)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"\n|<\|im_end\|>", cand)[0].strip()
            if cand:
                body = cand
                break

    # 5) Math-style boxed fallback.
    boxed = _extract_boxed(body)
    if boxed:
        body = boxed

    # 6) Last non-empty line fallback.
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if lines:
        body = lines[-1]

    # final cleanup
    body = body.strip().strip("`").strip()
    body = body.rstrip(".").rstrip("/")
    return body


def vlm_normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())
