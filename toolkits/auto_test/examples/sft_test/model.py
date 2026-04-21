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

"""Supervised fine-tuning model example"""

import argparse
import sys


def train():
    """Supervised fine-tuning model"""
    print("Starting supervised fine-tuning model...")
    # TODO: Implement your fine-tuning logic
    # Example:
    # import torch
    # from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    #
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    #
    # # ... fine-tuning loop
    print("Fine-tuning completed")


def serve(model_path: str | None = None):
    """Start inference service"""
    print(f"Starting inference service, model path: {model_path}")
    # TODO: Implement your inference service logic
    print("Inference service started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised fine-tuning model")
    parser.add_argument(
        "command", choices=["train", "serve"], help="Command: train or serve"
    )
    parser.add_argument(
        "--model_path", type=str, help="Model path (required for serve command)"
    )

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "serve":
        if not args.model_path:
            print(
                "Error: serve command requires --model_path argument", file=sys.stderr
            )
            sys.exit(1)
        serve(args.model_path)
