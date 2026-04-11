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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/download.py


import argparse

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(
    description="Download files from a Hugging Face dataset repository."
)

parser.add_argument(
    "--save_path", type=str, required=True, help="Local directory to save files"
)

args = parser.parse_args()

repo_id = "inclusionAI/ASearcher-Local-Knowledge"
hf_hub_download(
    repo_id=repo_id,
    filename="wiki_corpus.jsonl",
    repo_type="dataset",
    local_dir=args.save_path,
)

repo_id = "inclusionAI/ASearcher-Local-Knowledge"
hf_hub_download(
    repo_id=repo_id,
    filename="wiki_webpages.jsonl",
    repo_type="dataset",
    local_dir=args.save_path,
)
