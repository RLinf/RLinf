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

from rlinf.agents.wideseek_r1.utils.prompt import (
    BOXED_FORMAT_EN,
    BOXED_FORMAT_ZH,
    MARKDOWN_FORMAT_EN,
    MARKDOWN_FORMAT_ZH,
    SYSTEM_PROMPT_PLANNER,
    SYSTEM_PROMPT_PLANNER_NOSHOT,
    SYSTEM_PROMPT_PLANNER_ZH,
    SYSTEM_PROMPT_PLANNER_ZH_NOSHOT,
    SYSTEM_PROMPT_SINGLE_AGENT,
    SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT,
    SYSTEM_PROMPT_SINGLE_AGENT_ZH,
    SYSTEM_PROMPT_SINGLE_AGENT_ZH_NOSHOT,
    SYSTEM_PROMPT_WORKER,
    SYSTEM_PROMPT_WORKER_ZH,
    USER_PROMPT_PLANNER,
    USER_PROMPT_PLANNER_ZH,
    USER_PROMPT_SINGLE_AGENT,
    USER_PROMPT_SINGLE_AGENT_ZH,
    USER_PROMPT_WORKER,
    USER_PROMPT_WORKER_ZH,
)


def get_prompt_planner(question: str, is_markdown: bool, language: str) -> str:
    if language == "zh":
        return get_prompt_planner_zh(question, is_markdown)
    else:
        return get_prompt_planner_en(question, is_markdown)


def get_prompt_planner_en(question: str, is_markdown: bool) -> str:
    # Add fewshot only for markdown questions
    add_few_shot = is_markdown

    if add_few_shot:
        if is_markdown:
            system = SYSTEM_PROMPT_PLANNER.format(MARKDOWN_FORMAT_EN)
        else:
            system = SYSTEM_PROMPT_PLANNER.format(BOXED_FORMAT_EN)
    else:
        if is_markdown:
            system = SYSTEM_PROMPT_PLANNER_NOSHOT.format(MARKDOWN_FORMAT_EN)
        else:
            system = SYSTEM_PROMPT_PLANNER_NOSHOT.format(BOXED_FORMAT_EN)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_PLANNER.format(question)},
    ]


def get_prompt_planner_zh(question: str, is_markdown: bool) -> str:
    # Add fewshot only for markdown questions
    add_few_shot = is_markdown

    if add_few_shot:
        if is_markdown:
            system = SYSTEM_PROMPT_PLANNER_ZH.format(MARKDOWN_FORMAT_ZH)
        else:
            system = SYSTEM_PROMPT_PLANNER_ZH.format(BOXED_FORMAT_ZH)
    else:
        if is_markdown:
            system = SYSTEM_PROMPT_PLANNER_ZH_NOSHOT.format(MARKDOWN_FORMAT_ZH)
        else:
            system = SYSTEM_PROMPT_PLANNER_ZH_NOSHOT.format(BOXED_FORMAT_ZH)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_PLANNER_ZH.format(question)},
    ]


def get_prompt_worker(origin_question: str, subtask: str, language="en") -> str:
    if language == "zh":
        text = USER_PROMPT_WORKER_ZH.format(origin_question, subtask)
    else:
        text = USER_PROMPT_WORKER.format(origin_question, subtask)
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_WORKER_ZH
            if language == "zh"
            else SYSTEM_PROMPT_WORKER,
        },
        {"role": "user", "content": text},
    ]


def get_prompt_single_agent(question: str, is_markdown: bool, language) -> str:
    if language == "zh":
        return get_prompt_single_agent_zh(question, is_markdown)
    else:
        return get_prompt_single_agent_en(question, is_markdown)


def get_prompt_single_agent_en(question: str, is_markdown: bool) -> str:
    # Add fewshot only for markdown questions
    add_few_shot = is_markdown

    if add_few_shot:
        if is_markdown:
            system = SYSTEM_PROMPT_SINGLE_AGENT.format(MARKDOWN_FORMAT_EN)
        else:
            system = SYSTEM_PROMPT_SINGLE_AGENT.format(BOXED_FORMAT_EN)
    else:
        if is_markdown:
            system = SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT.format(MARKDOWN_FORMAT_EN)
        else:
            system = SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT.format(BOXED_FORMAT_EN)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_SINGLE_AGENT.format(question)},
    ]


def get_prompt_single_agent_zh(question: str, is_markdown: bool) -> str:
    # Add fewshot only for markdown questions
    add_few_shot = is_markdown

    if add_few_shot:
        if is_markdown:
            system = SYSTEM_PROMPT_SINGLE_AGENT_ZH.format(MARKDOWN_FORMAT_ZH)
        else:
            system = SYSTEM_PROMPT_SINGLE_AGENT_ZH.format(BOXED_FORMAT_ZH)
    else:
        if is_markdown:
            system = SYSTEM_PROMPT_SINGLE_AGENT_ZH_NOSHOT.format(MARKDOWN_FORMAT_ZH)
        else:
            system = SYSTEM_PROMPT_SINGLE_AGENT_ZH_NOSHOT.format(BOXED_FORMAT_ZH)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_SINGLE_AGENT_ZH.format(question)},
    ]


def get_access_summary_messages(info_to_extract, page_content):
    system_prompt = (
        "You are an information extraction assistant.\n"
        "You MUST base your output ONLY on the provided webpage content.\n"
        "You are strictly forbidden from using any prior knowledge, assumptions, or external information.\n\n"
        "Your task is NOT to answer the question directly, but to extract and summarize all information from the webpage that is relevant to the specified information requirement.\n\n"
        "If the webpage does NOT contain the exact requested information:\n"
        "- Extract the most closely related information from the webpage and explain its relevance.\n"
        '- If there is truly nothing related, explicitly state: "This webpage contains no information relevant to the request."\n\n'
        "You must NOT hallucinate, infer, or guess.\n"
        "You must NOT answer from your own knowledge.\n\n"
        "Your output MUST be a clear, complete, and well-structured summary report.\n"
        "The report should:\n"
        "- Be organized with headings or bullet points when appropriate\n"
        "- Include concrete facts, statements, or quotations from the webpage as evidence\n"
        "- Focus exclusively on information relevant to the request\n"
        "- Exclude any general summaries or unrelated content\n"
        "- Exclude any meta-commentary about your process\n"
    )

    user_prompt = (
        f"INFORMATION TO EXTRACT:\n{info_to_extract}\n\n"
        f"CONTENT TO ANALYZE:\n{page_content}\n\n"
        "Extract and summarize only the information relevant to the request above.\n"
        "Follow all instructions strictly."
    )

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return message


def get_first_turn_hint(max_turns: int, language: str) -> str:
    if language == "en":
        return (
            "\n\nThis is your first turn to answer the question. "
            f"You must finish your answer within {max_turns} turns"
        )
    return f"\n\n这是你回答该问题的第一轮。你必须在 {max_turns} 轮之内完成你的回答"


def get_next_turn_hint(next_turn_idx: int, max_turns: int, language: str) -> str:
    if language == "en":
        return (
            f"\n\nYour next answer will be on turn {next_turn_idx}. "
            f"You MUST finish the entire answer by turn {max_turns}."
        )
    return (
        f"\n\n你的下一次回答将是第 {next_turn_idx} 轮。"
        f"你必须在第 {max_turns} 轮之内完成整个回答。"
    )


def get_planner_subtask_result_message(
    subtask_idx: int,
    subtask_text: str,
    worker_summary: str,
    language: str,
) -> str:
    if language == "en":
        return f"# Subtask {subtask_idx}:\n{subtask_text}\n# Result:\n{worker_summary}"
    return f"# 子任务 {subtask_idx}:\n{subtask_text}\n# 结果:\n{worker_summary}"


def get_planner_subtask_failed_message(
    subtask_idx: int,
    subtask_text: str,
    language: str,
) -> str:
    if language == "en":
        return (
            f"# Subtask {subtask_idx}:\n{subtask_text}\n# Result:\n"
            "The current subagent exceeded its context window limit while "
            "executing this subtask, which caused the failure. Please retry."
        )
    return (
        f"# 子任务 {subtask_idx}:\n{subtask_text}\n# 结果:\n"
        "当前子智能体在执行该子任务时超出其上下文窗口限制，导致失败。请重试。"
    )


def get_search_tool_message(query: str, search_result: str, language: str) -> str:
    if language == "en":
        return f"# Search query:\n{query}\n# Result:\n{search_result}"
    return f"# 搜索查询:\n{query}\n# 结果:\n{search_result}"


def get_access_tool_message(url: str, page_content: str, language: str) -> str:
    if language == "en":
        return f"# Access URL:\n{url}\n# Result:\n{page_content}"
    return f"# 访问URL:\n{url}\n# 结果:\n{page_content}"


def get_access_summary_tool_message(
    url: str,
    info_to_extract: str | None,
    summary: str,
    language: str,
) -> str:
    if language == "en":
        return (
            f"# Access URL:\n{url}\n# Info to extract:\n{info_to_extract}\n"
            f"# Result:\n{summary}"
        )
    return (
        f"# 访问URL:\n{url}\n# 需要提取的信息:\n{info_to_extract}\n# 结果:\n{summary}"
    )
