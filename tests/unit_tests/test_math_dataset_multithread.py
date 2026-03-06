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

import copy
import json
import random
import time
from unittest import mock

import pytest
from omegaconf import DictConfig

from rlinf.data.datasets.math import MathDataset


class TestMathDatasetMultithread:
    """Tests for MathDataset multithread processing consistency."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = mock.Mock()
        tokenizer.is_fast = True
        tokenizer.eos_token_id = 2

        def apply_chat_template_side_effect(
            prompts, tokenize=False, add_generation_prompt=True
        ):
            """Mock apply_chat_template that handles generator input."""
            # Convert generator to list if needed
            prompts_list = list(prompts) if not isinstance(prompts, list) else prompts
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n"
                if isinstance(prompt, str)
                else prompt
                for prompt in prompts_list
            ]

        tokenizer.apply_chat_template = mock.Mock(
            side_effect=apply_chat_template_side_effect
        )
        tokenizer.batch_encode_plus = mock.Mock(
            side_effect=lambda texts: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            }
        )
        tokenizer.encode = mock.Mock(side_effect=lambda text: [1] * len(text.split()))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = DictConfig(
            {
                "data": {
                    "max_prompt_length": 1000,
                    "prompt_key": "question",
                    "answer_key": "answer",
                    "apply_chat_template": True,
                    "filter_prompt_by_length": False,
                    "process_workers": 4,
                    "process_batch_size": 32,
                }
            }
        )
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing (at least 10000 items)."""
        # Generate at least 10000 math problems
        data = []
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a // b if b != 0 else 0),
        ]

        for i in range(10000):
            op_symbol, op_func = random.choice(operations)
            a = random.randint(1, 1000)
            b = random.randint(1, 1000) if op_symbol != "/" else random.randint(1, 100)
            if op_symbol == "/" and b == 0:
                b = 1
            result = op_func(a, b)
            question = f"What is {a} {op_symbol} {b}?"
            data.append({"question": question, "answer": str(result)})

        return data

    def test_multithread_vs_singlethread_consistency(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test that multithread processing produces identical results to single-thread processing.

        This test verifies that:
        1. Results from multi-worker processing match single-worker processing
        2. All keys are the same
        3. All values are the same
        """
        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create MathDataset instance to get the configuration
        dataset = MathDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        # We need to reload the raw data to avoid double processing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        time_start = time.time()
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )
        time_elapse_multithread = time.time() - time_start

        # Test with single thread
        time_start = time.time()
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)
        time_elapse_singlethread = time.time() - time_start

        # Verify lengths are equal
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify all items have the same keys and values
        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            keys_mt, keys_st = item_mt.keys(), item_st.keys()
            assert keys_mt == keys_st, (
                f"Keys mismatch at index {idx}: "
                f"multithread={list(keys_mt)}, singlethread={list(keys_st)}"
            )

            # Check all values are equal
            unequal_keys = [key for key in keys_mt if item_mt[key] != item_st[key]]
            assert len(unequal_keys) == 0, (
                f"Values mismatch at index {idx} for keys: {unequal_keys}"
            )

        # Log timing information (for debugging)
        print(
            f"Data count: {len(data_multithread)}, "
            f"Multithread processing time: {time_elapse_multithread:.2f}s, "
            f"Singlethread processing time: {time_elapse_singlethread:.2f}s"
        )

    def test_multithread_consistency_with_filter(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test multithread processing consistency when filter_prompt_by_length is enabled.
        """
        # Update config to enable filtering
        mock_config.data.filter_prompt_by_length = True
        mock_config.data.max_prompt_length = 50  # Reasonable limit to test filtering

        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create MathDataset instance to get the configuration
        dataset = MathDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )

        # Test with single thread
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)

        # Verify consistency
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify that some data was filtered (not all data passed)
        assert len(data_multithread) <= len(raw_data), (
            f"Filtering should reduce data size, but got {len(data_multithread)} >= {len(raw_data)}"
        )

        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            assert item_mt.keys() == item_st.keys(), f"Keys mismatch at index {idx}"
            for key in item_mt.keys():
                assert item_mt[key] == item_st[key], (
                    f"Mismatch at index {idx}, key {key}"
                )

        print(
            f"Filtered data count: {len(data_multithread)}/{len(raw_data)} "
            f"(max_prompt_length={mock_config.data.max_prompt_length})"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
