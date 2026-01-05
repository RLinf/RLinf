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

"""Summary utilities for multi-agent workflow.

Adapted from MiroFlow's summary_utils.py for RLinf project.
Provides hint extraction and final answer extraction functionality.
All functions use local LLM via agent's generate() method.
"""

import re


def get_hint_extraction_prompt(question: str) -> str:
    """Generate prompt for hint extraction.

    Args:
        question: The task question to analyze

    Returns:
        Prompt text for hint generation
    """
    from rlinf.agents.prompt.prompt import HINT_PROMPT
    # Create message for hint generation
    hint_messages = [
        {"role": "system", "content": HINT_PROMPT},
        {"role": "user", "content": "Here is the original question:\n" + question},
    ]

    return hint_messages


from typing import List, Dict

def get_final_answer_extraction_prompt(task_description: str, summary: str, is_markdown: False) -> List[Dict[str, str]]:
    system_prompt = """You are an expert analytical reasoning agent. Your job is to extract the most accurate final answer from a provided summary by reconstructing and validating the reasoning, without introducing any outside knowledge.

# Task Instructions

1. Independently derive the best possible answer, step by step, using only the evidence and reasoning contained in the Agent Summary.
   - Ignore any “Final Answer” field within the summary during this derivation phase.
2. Compare your derived answer with the summary's provided final answer:
   - If both are well-supported, select the one with stronger or clearer justification.
   - If only one is supported, use that one.
3. Revise your selected answer to meet all phrasing and formatting requirements.
4. If neither is clearly supported, provide a justified, evidence-based educated guess.

# Output Format

**Step-by-step Analysis:**
Provide your detailed reasoning process.

**Final Answer:**
"""

    extraction_boxed = """Please ensure your final answer meets the following two key requirements:
1. **Must be enclosed in a box** — Place the final answer inside \\boxed{...}.
2. **Must be concise and clear** — The boxed content should be a single, short phrase using the fewest words possible."""

    extraction_markdown = """Please ensure your final answer meets the following two key requirements:
1. **Follow Markdown table syntax* — Wrap the final answer in: ```markdown\n{data_content}\n```.
2. **Be concise and clear** — The item in data_content should be a single, short phrase using the fewest words possible."""

    system_prompt += extraction_markdown if is_markdown else extraction_boxed

    user_prompt = f"""**Original Question:**
{task_description}

**Agent Summary:**
{summary}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

# def get_final_answer_extraction_prompt(task_description: str, summary: str) -> str:
#     """Generate prompt for final answer extraction.

#     Args:
#         task_description: Original task question
#         summary: Agent's summary of work history and findings

#     Returns:
#         Prompt text for answer extraction
#     """
#     prompt = f"""# Inputs

# * **Original Question**: `{task_description}`
# * **Agent Summary**: `{summary}`

# ---

# # Task

# 1. **Independently derive** the best possible answer, step by step, based solely on evidence and reasoning from the Agent Summary. **Ignore the summary's "Final Answer" field** at this stage.
# 2. **Compare** your derived answer to the final answer provided in the Agent Summary (ignoring formatting and phrasing requirements at this stage).
#    - If both are well supported by the summary's evidence, choose the one with stronger or clearer support.
#    - If only one is well supported, use that one.
# 3. **Revise** your chosen answer to fully satisfy all formatting and phrasing requirements.
# 4. If no answer is clearly supported by the evidence, provide a well-justified educated guess. 

# **Always wrap your final answer in a non-empty `\\boxed{{...}}`.**

# ---

# # Output Format

# Provide your analysis in the following format:

# **Step-by-step Analysis:**
# [Your detailed reasoning process]

# **Final Answer:**
# Please ensure your final answer meets the following two key requirements:
# 1. **Must be enclosed in a box** — Place the final answer inside `\\boxed{{...}}`.
# 2. **Must be concise and clear** — The boxed content should be a single, short phrase using the fewest words possible.

# **Confidence:** [0-100 integer]

# **Supporting Evidence:** [Brief summary of evidence that supports this answer]

# Focus on extracting a clear, properly formatted final answer that directly addresses the original question."""

#     return [
#         {"role": "system", "content": "You are a helpful and harmless assistant."},
#         {"role": "user", "content": prompt},
#     ]



def generate_summarize_prompt(
    task_description: str,
    main_task: str,
    task_failed: bool = False,
    final_extraction: bool = False, 
    role: str = 'planner',
    is_markdown: bool = False,
) -> str:
    """Generate prompt for final summary generation.

    Args:
        task_description: Original task description
        task_failed: Whether the task failed (hit context/turn limits)

    Returns:
        Prompt text for summary generation
    """
    if final_extraction:
        summarize_prompt = (
            (
                "This is a direct instruction to you (the assistant), not the result of a tool call.\n\n"
            )
            + (
                "**Important: You have either exhausted the context token limit or reached the maximum number of interaction turns without arriving at a conclusive answer. Therefore, you failed to complete the task. You Must explicitly state that you failed to complete the task in your response.**\n\n"
                if task_failed
                else ""
            )
            + (
                "We are now ending this session, and your conversation history will be deleted. "
                "You must NOT initiate any further tool use. This is your final opportunity to report "
                "*all* of the information gathered during the session.\n\n"
                "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
                "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
                "simply extract that answer and reformat it to match the required format below.\n"
                "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
                "The original question is repeated here for reference:\n\n"
                f"---\n{task_description}\n---\n\n"
                "Summarize ALL working history for this task, including your step-by-step thoughts, all tool calls, and all tool results (i.e., the full solving trajectory so far).\n"
                "Output the FINAL ANSWER and detailed supporting information of the task given to you.\n\n"
                "If you found any useful facts, data, or quotes directly relevant to the original task, include them clearly and completely.\n"
                "If you reached a conclusion or answer, include it as part of the response.\n"
                "If the task could not be fully answered, return all partially relevant findings, search results, quotes, and observations that might help a downstream agent solve the problem.\n"
                "If partial, conflicting, or inconclusive information was found, clearly indicate this in your response.\n\n"
                "Your final response should be a clear, complete, and structured report.\n"
                "Organize the content into logical sections with appropriate headings.\n"
                "Do NOT include any tool call instructions, speculative filler, or vague summaries.\n"
                "Focus on factual, specific, and well-organized information."
            )
        )
    else:
        if role == 'planner':
            if is_markdown:
                summarize_prompt = (
                    "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
                    "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
                    "simply extract that answer and reformat it to match the required format below.\n"
                    "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
                    "The original question is repeated here for reference:\n\n"
                    f'"{task_description}"\n\n'
                    "Wrap your final answer in the format:\nmarkdown{data content}.\n"
                )
            else:
                summarize_prompt = (
                    "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
                    "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
                    "simply extract that answer and reformat it to match the required format below.\n"
                    "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
                    "The original question is repeated here for reference:\n\n"
                    f'"{task_description}"\n\n'
                    "Wrap your final answer in \\boxed{}.\n"
                    "Your final answer should be:\n"
                    "- a number, OR\n"
                    "- as few words as possible, OR\n"
                    "- a comma-separated list of numbers and/or strings.\n\n"
                    "ADDITIONALLY, your final answer MUST strictly follow any formatting instructions in the original question — "
                    "such as alphabetization, sequencing, units, rounding, decimal places, etc.\n"
                    "If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.\n"
                    "If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.\n"
                    "If you are asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.\n"
                    "Do NOT include any punctuation such as '.', '!', or '?' at the end of the answer.\n"
                    "Do NOT include any invisible or non-printable characters in the answer output."
                )
        elif role == 'worker':
            summarize_prompt = (
                "This is a direct instruction to you (the assistant), not the result of a tool call.\n\n"
                "We are now ending this session, and your conversation history will be deleted. "
                "You must NOT initiate any further tool use. This is your final opportunity to report "
                "*all* of the information gathered during the session.\n\n"
                "The main task and your current task is repeated here for reference:\n\n"
                f'[MAIN TASK]"{main_task}"\n\n'
                f'[YOUR ASSIGNED SUBTASK]"{task_description}"\n\n'
                "Summarize the above search and browsing history. Output the FINAL RESPONSE and detailed supporting information of the task given to you.\n\n"
                "If you found any useful facts, data, quotes, or answers directly relevant to the original task, include them clearly and completely.\n"
                "If you reached a conclusion or answer, include it as part of the response.\n"
                "If the task could not be fully answered, do NOT make up any content. Instead, return all partially relevant findings, "
                "Search results, quotes, and observations that might help a downstream agent solve the problem.\n"
                "If partial, conflicting, or inconclusive information was found, clearly indicate this in your response.\n\n"
                "Your final response should be a clear, complete, and structured report.\n"
                "Organize the content into logical sections with appropriate headings.\n"
                "Do NOT include any tool call instructions, speculative filler, or vague summaries.\n"
                "Focus on factual, specific, and well-organized information."
            )            

    return summarize_prompt
