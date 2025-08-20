import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:  # noqa: E722
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


PROMPT_TEMPLATES = {
    "r1-distilled-qwen": (
        "<｜User｜>{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n",
        "{output}",
        "\n\n",
    ),
    "r1-distilled-qwen-gpqa": (
        "<｜User｜>{input}\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{{}} in the end.<｜Assistant｜><think>\n",
        "{output}",
        "\n\n",
    ),
}


def construct_prompt(example, data_name, args):
    demos = []

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )

    demo_prompt = splitter.join(
        [
            input_template.format(input=q) + output_template.format(output=a)
            for q, a in demos
        ]
    )

    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        full_prompt = demo_prompt + splitter + context

    return full_prompt.strip(" ")  # important!
