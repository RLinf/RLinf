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


"""
Preprocess the MATH 500 dataset to jsonl format
"""

import json
import os

import datasets

if __name__ == "__main__":
    data_source = "open-r1/DAPO-Math-17k-Processed"
    dataset = datasets.load_dataset(data_source, "all")

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
            solution = example.pop("solution")
            source_prompt = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{}\n\nRemember to put your answer on its own line after "Answer:".'.format(
                question
            )

            for key in [
                "data_source",
                "source_prompt",
                "ability",
                "reward_model",
                "extra_info",
            ]:
                example.pop(key, None)

            data = {
                "prompt": [
                    {
                        "content": source_prompt,
                        "role": "user",
                    },
                ],
                "task": "math",
                "query_id": f"{idx:08d}",
                "solutions": [solution],
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    output_file = "train.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in train_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Dataset saved to {output_file}")
    print(f"Total number of examples: {len(train_dataset)}")
