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

import asyncio
import copy
import json
import re
from io import StringIO

import pandas as pd
from omegaconf import DictConfig

from rlinf.agents.wideseek_r1.utils.sglang_client import SGLangClient
from rlinf.workers.agent.agent_loop import AgentLoopOutput


def credit_assignment(
    agentloop_config: DictConfig,
    output_buffer: list[AgentLoopOutput],
    llm_reward,
    succ_end,
    answer_format,
):
    """Assign trajectory reward and select trainable turns for policy updates.

    Args:
        agentloop_config: Agent-loop config containing reward shaping weights.
        output_buffer: All turns generated in one trajectory.
        llm_reward: End-of-trajectory reward from answer evaluation.
        succ_end: Whether the agent stopped naturally without tool calls.
        answer_format: Whether final-answer extraction/format validation succeeded.

    Returns:
        Tuple of `(output_buffer, train_buffer, final_answer_format, reward_score)`.
    """
    final_answer_format = 0
    search_credit = 0.0
    length_penalty = 0.0

    format_reward = agentloop_config.get("format_reward", 0.0)
    call_search_reward = agentloop_config.get("call_search_reward", 0.0)
    length_limit = agentloop_config.get("length_limit", 5000)
    max_length_limit = agentloop_config.get("max_length_limit", 7000)
    length_p = agentloop_config.get("length_penalty", 0.0)

    for turn_output in output_buffer:
        # Reward search behavior when at least one access call exists in trajectory.
        tool_call_info = turn_output.tool_call_info
        if tool_call_info is None:
            continue
        if tool_call_info.get("access", 0) > 0:
            search_credit = call_search_reward
            break

    max_response_len = max(
        len(turn_output.response_ids) for turn_output in output_buffer
    )
    if max_response_len > length_limit:
        t = (max_response_len - length_limit) / (max_length_limit - length_limit)
        t = max(0.0, min(1.0, t))
        length_penalty = t * length_p

    one_turn_failed = False

    for turn in output_buffer:
        if turn.extra_fields["turn_repeat_failed"]:
            one_turn_failed = True

    train_buffer: list[AgentLoopOutput] = []
    if answer_format:
        flag = False
        for turn in output_buffer:
            if (
                turn.extra_fields["context_failed"]
                or turn.extra_fields["max_turn_limit_failed"]
            ) and turn.extra_fields["role"] != "worker":
                # main agent or sa failed but extract good format
                flag = True

        if not flag:
            for turn in output_buffer:
                if not (
                    turn.extra_fields["context_failed"]
                    or turn.extra_fields["max_turn_limit_failed"]
                ):
                    train_buffer.append(turn)

            reward_score = llm_reward + format_reward + search_credit - length_penalty
            final_answer_format = 1
        else:
            for turn in output_buffer:
                if (
                    turn.extra_fields["context_failed"]
                    or turn.extra_fields["max_turn_limit_failed"]
                ) and turn.extra_fields["role"] != "worker":
                    train_buffer.append(turn)

            reward_score = 0.0

    else:
        reward_score = 0.0
        if succ_end:
            train_buffer.append(output_buffer[-1])

        if one_turn_failed:
            for turn in output_buffer:
                if turn.extra_fields["turn_repeat_failed"]:
                    if turn not in train_buffer:
                        train_buffer.append(turn)
        else:
            for turn in output_buffer:
                if (
                    turn.extra_fields["max_turn_limit_failed"]
                    or turn.extra_fields["context_failed"]
                ):
                    assert not turn.extra_fields["turn_repeat_failed"]
                    if turn not in train_buffer:
                        train_buffer.append(turn)

    return output_buffer, train_buffer, final_answer_format, reward_score


async def get_final_reward_score(
    origin_question,
    extract_answer,
    label_answer,
    is_markdown,
    norm_column,
    sgl_client: SGLangClient,
):
    """Compute final reward score for boxed answers or markdown-table answers.

    Args:
        origin_question: Original user question text.
        extract_answer: Parsed model answer (string or DataFrame).
        label_answer: Ground-truth answer payload from dataset.
        is_markdown: Whether to evaluate in markdown-table mode.
        norm_column: Whether to normalize markdown column names aggressively.
        sgl_client: Shared SGLang client used by LLM judges.

    Returns:
        Tuple of `(reward_score, format_ok)`.
    """
    format = True
    if is_markdown:
        reward_score, format = await evaluate_markdown(
            extract_answer, label_answer, sgl_client, norm_column
        )
        return reward_score, format

    label_answer = label_answer["answer"]
    if label_answer is not None and extract_answer is not None:
        # Use LLM as judge
        llm_score = await verify_answer_with_llm_judge(
            question=origin_question,
            predicted_answer=extract_answer,
            correct_answer=label_answer,
            sgl_client=sgl_client,
        )
        reward_score = llm_score

    else:
        reward_score = 0.0

    return reward_score, format


async def verify_answer_with_llm_judge(
    question: str, predicted_answer: str, correct_answer: list, sgl_client: SGLangClient
) -> float:
    """Use an LLM judge to score equivalence between prediction and reference.

    Args:
        question: Original user question.
        predicted_answer: Model-predicted boxed answer.
        correct_answer: Reference answer list from dataset.
        sgl_client: SGLang client used to query the judge model.

    Returns:
        `1.0` if judged correct, otherwise `0.0`.
    """
    from rlinf.agents.wideseek_r1.utils.prompt import LLM_JUDGE_PROMPT

    if len(correct_answer) == 1:
        # Format the judge prompt
        judge_prompt_text = LLM_JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer[0],
            response=predicted_answer,
        )
    else:
        judge_prompt_text = LLM_JUDGE_PROMPT.format(
            question=question, correct_answer=correct_answer, response=predicted_answer
        )

    judge_messages = [
        {
            "role": "system",
            "content": "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.",
        },
        {"role": "user", "content": judge_prompt_text},
    ]
    judge_response_text = await sgl_client.call_sglang_api(judge_messages)
    judge_response_clean = judge_response_text.strip().lower()
    if "correct" in judge_response_clean and "incorrect" not in judge_response_clean:
        return 1.0
    else:
        return 0.0


async def evaluate_markdown(
    extract_answer, label_answer, sgl_client, norm_column_=False
):
    """Evaluate markdown-table answers with schema checks and LLM cell matching.

    Args:
        extract_answer: Parsed prediction DataFrame.
        label_answer: Ground-truth markdown payload or DataFrame.
        sgl_client: SGLang client used for semantic matching.
        norm_column_: Whether to normalize spaces in column names.

    Returns:
        Tuple of `(score, format_ok)` where `score` is item-level F1.
    """

    # Helper function to normalize column names
    def norm_column(col: str) -> str:
        """Normalize column names to improve schema alignment robustness."""
        if not norm_column_:
            return col.strip().lower()
        else:
            return col.strip().lower().replace(" ", "")

    # Helper function to calculate F1 score
    def calc_f1(precision, recall):
        """Compute a numerically stable F1 score."""
        epsilon = 1e-9
        return (
            (2 * precision * recall / (precision + recall))
            if (precision + recall > epsilon)
            else 0.0
        )

    def normalize_series_to_str(s: pd.Series) -> pd.Series:
        """Normalize a series to stripped canonical strings for matching."""
        s0 = s.astype(str).str.strip()
        num = pd.to_numeric(s0, errors="coerce")
        if num.notna().any():
            return num.map(lambda x: "" if pd.isna(x) else f"{x:g}")
        else:
            return s0

    # Initialize metrics
    score = 0.0
    precision_by_row = 0.0
    recall_by_row = 0.0
    f1_by_row = 0.0
    precision_by_item = 0.0
    recall_by_item = 0.0
    f1_by_item = 0.0

    try:
        # Parse label_answer
        if isinstance(label_answer, dict):
            answer_markdown = label_answer.get("answer", "")
            unique_columns = label_answer.get("unique_columns", [])
            required_columns = label_answer.get("required", [])
        else:
            # If label_answer is a string, assume it's markdown
            answer_markdown = label_answer
            unique_columns = []
            required_columns = []

        # Convert answer_markdown to DataFrame if it's a string
        if isinstance(answer_markdown, str):
            answer_df = extract_final_answer(
                answer_markdown, mode="markdown", strict=False
            )
            if answer_df is None:
                # print("Failed to parse label answer markdown")
                return 0.0, False
        elif isinstance(answer_markdown, pd.DataFrame):
            answer_df = answer_markdown
        else:
            # print(f"Invalid label answer type: {type(answer_markdown)}")
            return 0.0, False

        # Validate extract_answer
        if not isinstance(extract_answer, pd.DataFrame) or extract_answer.empty:
            # print(f"Extracted answer is None or not a DataFrame, it's {extract_answer}")
            return 0.0, False

        response_df = copy.deepcopy(extract_answer)

        # Normalize column names
        answer_df.columns = [norm_column(col) for col in answer_df.columns]
        response_df.columns = [norm_column(col) for col in response_df.columns]

        # Normalize unique_columns and required_columns
        unique_columns = [norm_column(col) for col in unique_columns]

        if not required_columns:
            required_columns = list(answer_df.columns)
        else:
            required_columns = [
                norm_column(col) for col in required_columns
            ]  # widesearch requir columns: " " -> ""

        # Check if response has required columns
        if not set(required_columns).issubset(set(response_df.columns)):
            # Try primary key preprocessing to map column names
            column_map = await primary_key_preprocess(
                list(response_df.columns), required_columns, sgl_client
            )
            response_df.rename(columns=column_map, inplace=True)

        if not set(required_columns).issubset(set(response_df.columns)):
            # print(f"required_columns {required_columns} != response_df {list(response_df.columns)}")
            return 0.0, False

        for col in required_columns:
            answer_df[col] = normalize_series_to_str(answer_df[col])
            response_df[col] = normalize_series_to_str(response_df[col])

        # Remove duplicates based on unique columns
        if unique_columns:
            response_df.drop_duplicates(subset=unique_columns, inplace=True)
            answer_df.drop_duplicates(subset=unique_columns, inplace=True)

            # Preprocess primary keys using LLM
            for col in unique_columns:
                primary_key_map = await primary_key_preprocess(
                    response_df[col].tolist(),
                    answer_df[col].tolist(),
                    sgl_client,
                )
                response_df[col + "_before_map"] = response_df[col]
                response_df[col] = response_df[col].apply(
                    lambda x: primary_key_map.get(x, x)
                )

        # Inner join over unique keys to align comparable rows.
        df_inner = pd.merge(
            answer_df,
            response_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )

        # Initialize score DataFrames for each metric type in reward_eval
        df_inner_scores = pd.DataFrame(index=df_inner.index)

        llm_tasks = []
        llm_columns = []

        # Process each column
        for col in required_columns:
            if col in unique_columns:
                df_inner_scores[f"{col}_score"] = 1.0
            else:
                responses = df_inner[col + "_response"].tolist()
                targets = df_inner[col + "_query"].tolist()
                llm_tasks.append(llm_judge_column(responses, targets, sgl_client))
                llm_columns.append(col)

        # Execute LLM semantic checks in parallel per non-key column.
        if llm_tasks:
            llm_results = await asyncio.gather(*llm_tasks)
            # Assign results back to df_inner_scores["LLM"]
            for col, scores in zip(llm_columns, llm_results):
                df_inner_scores[f"{col}_score"] = scores

        # Calculate metrics for each evaluation method
        num_pred_rows = len(response_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(required_columns)
        num_gt_items = num_gt_rows * len(required_columns)

        row_scores = df_inner_scores.min(axis=1)
        tp_by_row = row_scores.sum()
        precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
        recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
        f1_by_row = calc_f1(precision_by_row, recall_by_row)

        # Item-level metrics
        tp_by_item = df_inner_scores.sum().sum()
        precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
        f1_by_item = calc_f1(precision_by_item, recall_by_item)

        if (
            precision_by_item == recall_by_item == 1.0
            and precision_by_row == recall_by_row == 1.0
        ):
            score = 1.0

    except Exception:
        # print(f"Evaluation error: {traceback.format_exc()}")
        return 0.0, False

    return f1_by_item, True


async def llm_judge_column(
    responses: list, targets: list, sgl_client: SGLangClient
) -> list:
    """Score non-key markdown table cells using semantic LLM comparison.

    Args:
        responses: Predicted cell values for one column.
        targets: Ground-truth cell values for one column.
        sgl_client: SGLang client used for judge inference.

    Returns:
        List of float scores aligned with `responses`.
    """
    criterion = "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence."

    if not responses:
        return []

    # Widesearch's eval_column_prompt
    eval_column_prompt = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by step according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
"idx_0": score,
"idx_1": score,
...
}}
```

{criterion}
"""

    user_prompt = """Here is the response you need to judge, please make sure to analyze each item step by step before providing the final scoring results.

{response}
"""

    # Build response dict
    response_dict = {}
    for idx, (resp, tar) in enumerate(zip(responses, targets)):
        response_dict[f"idx_{idx}"] = {"response": str(resp), "target": str(tar)}

    # Format prompt
    system_prompt = eval_column_prompt.format(
        criterion=criterion,
    )

    user_prompt = user_prompt.format(response=response_dict)
    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result_text = await sgl_client.call_sglang_api(messages)

    try:
        pat = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pat, result_text, re.DOTALL)
        if matches:
            json_str = matches[-1]
            score_dict = json.loads(json_str)
            score_list = [
                float(score_dict.get(f"idx_{idx}", 0)) for idx in range(len(responses))
            ]
        else:
            # Parsing failed, default to 0
            score_list = [0.0] * len(responses)
    except Exception:
        # If any error, default to 0
        score_list = [0.0] * len(responses)

    # Ensure correct length
    if len(score_list) != len(responses):
        score_list = [0.0] * len(responses)
    return score_list


async def primary_key_preprocess(
    response_list, reference_list, sgl_client: SGLangClient
):
    """Align predicted primary-key values to reference canonical forms.

    Args:
        response_list: Candidate values from prediction side.
        reference_list: Reference values used as canonical vocabulary.
        sgl_client: SGLang client used for semantic alignment.

    Returns:
        Mapping from predicted string to aligned reference string.
    """
    primary_key_map = {}

    # The prompt template from widesearch
    primary_key_preprocess_prompt = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
"origin_str1": "transform_str1",
"origin_str2": "transform_str2"
}}
```
"""

    user_prompt = """
The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}
"""

    # Format the prompt
    system_prompt = primary_key_preprocess_prompt

    user_prompt = user_prompt.format(response=response_list, reference=reference_list)

    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result_text = await sgl_client.call_sglang_api(messages)

    # Parse JSON from result
    try:
        pat = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pat, result_text, re.DOTALL)
        if matches:
            json_str = matches[-1]
            transform_map = json.loads(json_str)
            primary_key_map.update(transform_map)
    except Exception:
        pass

    return primary_key_map


def extract_final_answer(text: str, mode: bool = "boxed", strict=True):
    """Extract final answer from generated text using a specific parsing mode.

    Args:
        text: Raw generated text that may include reasoning/tool wrappers.
        mode: Parsing mode (`tag`, `boxed`, or `markdown`).
        strict: For markdown mode, require fenced markdown blocks when True.

    Returns:
        For `tag`/`boxed`: extracted string or None.
        For `markdown`: parsed `pd.DataFrame` or None.
    """
    text = text.split("</think>")[-1].strip()
    if mode == "tag":
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.finditer(answer_pattern, text, re.DOTALL)
        matches = list(match)

        if len(matches) < 1:
            return None
        return matches[-1].group(1).strip()
    elif mode == "boxed":
        if not text:
            return None

        matches = []
        i = 0

        while i < len(text):
            boxed_start = text.find(r"\boxed{", i)
            if boxed_start == -1:
                break

            content_start = boxed_start + 7  # len(r'\boxed{') = 7
            if content_start >= len(text):
                break

            # Count balanced braces
            brace_count = 1
            content_end = content_start

            while content_end < len(text) and brace_count > 0:
                char = text[content_end]
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                content_end += 1

            if brace_count == 0:
                content = text[content_start : content_end - 1]
                matches.append(content)
                i = content_end
            else:
                i = content_start

        return matches[-1] if matches else None
    elif mode == "markdown":
        if not text or not isinstance(text, str):
            return None

        response_df = None
        markdown_str = re.findall(r"```markdown(.*?)```", text, re.DOTALL)
        if not markdown_str:
            # Fallback parser for answers that forgot markdown fences.
            if strict:
                return None
            pipe_positions = [m.start() for m in re.finditer(r"\|", text)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = text.rfind("\n", 0, first_pipe)
                start = 0 if start == -1 else start
                end = text.find("\n", last_pipe)
                end = len(text) if end == -1 else end
                table_candidate = text[start:end]
                markdown_str = re.findall(r"((?:\|.*\n?)+)", table_candidate)
        if markdown_str:
            markdown_str = markdown_str[-1].strip()
            lines = markdown_str.split("\n")
            # lines[0] = lines[0].replace(" ", "").lower()
            lines = [line.strip() for line in lines]

            new_lines = []
            for line in lines:
                if set(line.strip()).issubset(set("|- :")) or "|" not in line:
                    continue
                new_lines.append("|".join([_line.strip() for _line in line.split("|")]))

            if not new_lines:
                return None
            markdown_str = "\n".join(new_lines)
            try:
                response_df = pd.read_csv(
                    StringIO(markdown_str), sep="|", keep_default_na=False
                )
                response_df = response_df.loc[
                    :, ~response_df.columns.str.startswith("Unnamed")
                ]

                for col in response_df.columns:  # FIXME: check if need？
                    if response_df[col].dtype == "object":
                        response_df[col] = response_df[col].apply(
                            lambda x: x.replace("<br>", "\n")
                            if isinstance(x, str) and x
                            else x
                        )
                    response_df[col] = response_df[col].replace("", "nan")

                return response_df
            except Exception:
                # print(f"Error parsing markdown table: {e}")
                return None

        return response_df
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'tag', 'boxed', or 'markdown'")
