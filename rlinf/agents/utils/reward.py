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

from typing import List, Dict

from omegaconf import DictConfig

import random
import re
import string
from io import StringIO
import pandas as pd


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_text(text: str) -> str:
    """Preprocess text for NQ dataset scoring.

    Processing steps:
    1. Convert to lowercase
    2. Remove punctuation (.,!?;:'"()[]{}...)
    3. Remove extra spaces
    """
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, ' ')

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing spaces
    text = text.strip().lower()
    return text

def bool_mapping(s):
    """Map boolean string values to yes/no."""
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def contains_chinese(text):
    """Check if the given text contains Chinese characters."""
    for char in text:
        # Check for common Chinese characters (CJK Unified Ideographs)
        if '\u4e00' <= char <= '\u9fff':
            return True
        # Check for rare characters (CJK Unified Ideographs Extension A)
        if '\u3400' <= char <= '\u4dbf':
            return True
        # Check for compatibility characters
        if '\uf900' <= char <= '\ufaff':
            return True
    return False


def em_check(prediction, golden_answer):
    # if isinstance(golden_answers, str):
    #     golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    score = 0
    golden_answer = normalize_answer(bool_mapping(golden_answer))
    if golden_answer == normalized_prediction:
        score = 1
    return score


def f1_score(answer_content, gt):
    """Compute F1 score between answer and ground truth."""
    answer_content = normalize_text(bool_mapping(answer_content))
    gt = normalize_text(bool_mapping(gt))

    # Tokenize answer and reference answer
    if contains_chinese(gt):
        def parse_chinese_str(s):
            # parse consecutive numbers
            numbers = []
            for i, c in enumerate(s):
                if c.isdigit():
                    if i > 0 and s[i-1].isdigit():
                        numbers[-1] = numbers[-1] + c
                    else:
                        numbers.append(c)
            for c in "0123456789，。 ,.-":
                s = s.replace(c, "")
            s = set(list(s) + numbers)
            return s
        pred_tokens = parse_chinese_str(answer_content)
        gt_tokens = parse_chinese_str(gt)
    else:
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt.split())

    if not gt_tokens:  # Avoid division by zero
        return 0
    if not pred_tokens:
        return 0

    # Calculate common tokens
    common_tokens = pred_tokens & gt_tokens

    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

    # Calculate F1 score
    f1 = 0
    if precision + recall > 0:  # Avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


# def subem_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer in normalized_prediction:
#             score = 1
#             break
#     return score


# def extract_solution(
#     text: str,
#     is_boxed = 0
# ) -> tuple[list[int], str]:
#     if not is_boxed:
#         answer_pattern = r"<answer>(.*?)</answer>"
#         match = re.finditer(answer_pattern, text, re.DOTALL)
#         matches = list(match)

#         # If there are 0  matches, return None
#         if len(matches) < 1:
#             return None

#         # If there are 2 or more matches, return the last one
#         return matches[-1].group(1).strip()            
#     else:
#         if not text:
#             return ""

#         matches = []
#         i = 0

#         while i < len(text):
#             # Find the next \boxed{ pattern
#             boxed_start = text.find(r"\boxed{", i)
#             if boxed_start == -1:
#                 break

#             # Start after the opening brace
#             content_start = boxed_start + 7  # len(r'\boxed{') = 7
#             if content_start >= len(text):
#                 break

#             # Count balanced braces
#             brace_count = 1
#             content_end = content_start

#             while content_end < len(text) and brace_count > 0:
#                 char = text[content_end]
#                 if char == "{":
#                     brace_count += 1
#                 elif char == "}":
#                     brace_count -= 1
#                 content_end += 1

#             # If we found a balanced match (brace_count == 0)
#             if brace_count == 0:
#                 content = text[
#                     content_start : content_end - 1
#                 ]  # -1 to exclude the closing brace
#                 matches.append(content)
#                 # Continue searching from after this complete match
#                 i = content_end
#             else:
#                 # If braces are unbalanced, skip this \boxed{ and continue searching
#                 i = content_start

#         return matches[-1] if matches else ""



# def count_answer_tags(text):
#     opening_tags = text.count("<answer>")
#     closing_tags = text.count("</answer>")

#     return opening_tags, closing_tags


def compute_score_em(predict, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        predict: the predicted text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer

    Returns:
        tuple: (extracted_answer, score)
    """
    if isinstance(ground_truth, list):
        # answer = extract_solution(text=solution_str)
        return max([compute_score_em(predict, g) for g in ground_truth])

    # answer = extract_solution(text=solution_str)


    if em_check(predict, ground_truth):
        return score
    else:
        return format_score


def compute_score_f1(predict, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for F1 score.

    Args:
        predict: the predicted text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer (max F1 score)

    Returns:
        tuple: (extracted_answer, f1_score)
    """
    if isinstance(ground_truth, list):
        return max([compute_score_f1(predict, g) for g in ground_truth])

    ret_score = f1_score(predict, ground_truth)
    return ret_score


# # TODO: 把extract部分放到agentloop里面，这样agentloop如果没有extract到，直接设置reward=0
# class MASReward:
#     def __init__(self, config: DictConfig):
#         self.scale = config.get("reward_scale", 1.0)
#         self.reward_type = config.get("reward", "EM").upper()  # Default to EM

#         if self.reward_type not in ["EM", "F1"]:
#             raise ValueError(f"Invalid reward type: {self.reward_type}. Must be 'EM' or 'F1'")

#         print(f"[INFO] MASReward: Using {self.reward_type} reward metric")

#     def get_reward(
#         self,
#         response: List[str],
#         reference: List[List[str]]
#     ) -> List[float]:
#         if self.reward_type == "F1":
#             # Use F1 score - returns (answer, score) tuple, we only need the score
#             rewards = [compute_score_f1(sol, gt)[1] for sol, gt in zip(response, reference)]
#         else:
#             # Use EM score - returns (answer, score) tuple, we only need the score
#             rewards = [compute_score_em(sol, gt)[1] for sol, gt in zip(response, reference)]

#         return rewards

def extract_final_answer(text: str, mode: bool = "boxed"):
    """Extract final answer from text based on mode.

    Args:
        text: The text to extract answer from
        mode: Extraction mode - 'tag', 'boxed', or 'markdown'

    Returns:
        For 'tag' and 'boxed': str (the extracted answer)
        For 'markdown': pd.DataFrame or None
    """
    text = text.split('</think>')[-1].strip()
    if mode == 'tag':
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.finditer(answer_pattern, text, re.DOTALL)
        matches = list(match)

        if len(matches) < 1:
            return None
        return matches[-1].group(1).strip()
    elif mode == 'boxed':
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
                content = text[content_start:content_end - 1]
                matches.append(content)
                i = content_end
            else:
                i = content_start

        return matches[-1] if matches else None
    elif mode == 'markdown':
        if not text or not isinstance(text, str):
            return None

        response_df = None
        markdown_str = re.findall(r"```markdown(.*?)```", text, re.DOTALL)
        if not markdown_str:
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
            markdown_str = markdown_str[0].strip()
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
                response_df = pd.read_csv(StringIO(markdown_str), sep="|", keep_default_na=False)
                response_df = response_df.loc[
                    :, ~response_df.columns.str.startswith("Unnamed")
                ]

                for col in response_df.columns: # FIXME: check if need？
                    if response_df[col].dtype == 'object':
                        response_df[col] = response_df[col].apply(
                            lambda x: x.replace('<br>', '\n') if isinstance(x, str) and x else x
                        )
                    response_df[col] = response_df[col].replace('', 'nan')

                return response_df
            except Exception as e:
                print(f"Error parsing markdown table: {e}")
                return None
            
        return response_df
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'tag', 'boxed', or 'markdown'")
