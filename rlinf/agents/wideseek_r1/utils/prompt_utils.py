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
