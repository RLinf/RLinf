# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Preprocess the MATH 500 dataset to jsonl format
"""

import os
import json

import datasets


if __name__ == "__main__":

    data_source = "open-r1/DAPO-Math-17k-Processed"
    dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            solution = example.pop("answer")
            source_prompt = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{}\n\nRemember to put your answer on its own line after "Answer:".'.format(question)

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
    local_dir = "dapo"
    
    os.makedirs(local_dir, exist_ok=True)

    output_file = os.path.join(local_dir, "train.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in train_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Dataset saved to {output_file}")
    print(f"Total number of examples: {len(train_dataset)}")
