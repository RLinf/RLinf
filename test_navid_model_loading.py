#!/usr/bin/env python3
"""
Smoke test for NaVid model loading and basic inference.

This script verifies that:
1. NaVid model can be loaded via `get_model()`
2. `preprocess_env_obs()` works correctly
3. `predict_action_batch()` can generate actions from observations

Usage:
    python test_navid_model_loading.py
"""

# Copyright 2026 RLinf contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rlinf.models import get_model  # noqa: E402


def create_dummy_env_obs(batch_size: int = 2):
    """Create dummy environment observations for testing."""
    return {
        "main_images": torch.randint(
            0, 255, (batch_size, 224, 224, 3), dtype=torch.uint8
        ),
        "task_descriptions": [f"Test task {i}" for i in range(batch_size)],
        "states": torch.randn(batch_size, 7),  # 7D state for example
    }


def test_model_loading():
    """Test that model can be loaded."""
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)

    # Try to find the actual model directory
    base_path = Path("/data/RLinf/VLN-CE/models/Navid_weight")
    model_dirs = list(base_path.glob("navid-*"))
    if model_dirs:
        model_path = str(model_dirs[0])
        print(f"Found model directory: {model_path}")
    else:
        model_path = str(base_path)
        print("Warning: Using base path, may need to specify exact subdirectory")

    cfg_dict = {
        "model_type": "navid",
        "model_path": model_path,
        "precision": "bf16",
        "action_dim": 7,
        "num_action_chunks": 8,
        "max_new_tokens": 32,
        "conversation_template": "imgsp_v1",
        # RLinf get_model expects these keys to exist
        "is_lora": False,
    }
    cfg = OmegaConf.create(cfg_dict)

    try:
        model = get_model(cfg)
        print("✓ Model loaded successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Action dim: {model.action_dim}")
        print(f"  - Num action chunks: {model.num_action_chunks}")
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        error_msg = str(e)
        if "vision tower" in error_msg.lower():
            print("\n" + "=" * 60)
            print("NOTE: Vision Tower Checkpoint Missing")
            print("=" * 60)
            print("The NaVid model requires a vision tower checkpoint (EVA-ViT).")
            print("The config specifies: ./model_zoo/eva_vit_g.pth")
            print("\nPossible solutions:")
            print("1. Download the EVA-ViT checkpoint and place it at:")
            print(f"   {cfg.model_path}/model_zoo/eva_vit_g.pth")
            print("2. Or update the config.json to point to an existing checkpoint")
            print("3. Or set an environment variable with the checkpoint path")
            print("\nFor now, this smoke test cannot proceed without the vision tower.")
            print("=" * 60)
        import traceback

        traceback.print_exc()
        raise


def test_preprocess_env_obs(model):
    """Test that preprocess_env_obs works."""
    print("\n" + "=" * 60)
    print("Test 2: preprocess_env_obs()")
    print("=" * 60)

    env_obs = create_dummy_env_obs(batch_size=2)
    print(f"Input obs keys: {list(env_obs.keys())}")
    print(f"  - main_images shape: {env_obs['main_images'].shape}")
    print(f"  - states shape: {env_obs['states'].shape}")

    try:
        processed_obs = model.preprocess_env_obs(env_obs)
        print("✓ preprocess_env_obs() succeeded")
        print(f"Output obs keys: {list(processed_obs.keys())}")
        return processed_obs
    except Exception as e:
        print(f"✗ preprocess_env_obs() failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_predict_action_batch(model):
    """Test that predict_action_batch works."""
    print("\n" + "=" * 60)
    print("Test 3: predict_action_batch()")
    print("=" * 60)

    env_obs = create_dummy_env_obs(batch_size=2)
    print(f"Input batch size: {env_obs['main_images'].shape[0]}")

    try:
        chunk_actions, result = model.predict_action_batch(
            env_obs=env_obs,
            mode="eval",  # Use eval mode for deterministic output
            return_obs=True,
            do_sample=False,  # Deterministic
            temperature=1.0,
            max_new_tokens=32,
        )
        print("✓ predict_action_batch() succeeded")
        print(f"  - chunk_actions shape: {chunk_actions.shape}")
        print(
            f"    Expected: (batch_size={env_obs['main_images'].shape[0]}, "
            f"num_action_chunks={model.num_action_chunks}, action_dim={model.action_dim})"
        )
        print(f"  - prev_logprobs shape: {result['prev_logprobs'].shape}")
        print(f"  - prev_values shape: {result['prev_values'].shape}")
        print(f"  - forward_inputs keys: {list(result['forward_inputs'].keys())}")

        # Check shapes
        expected_chunk_shape = (
            env_obs["main_images"].shape[0],
            model.num_action_chunks,
            model.action_dim,
        )
        assert chunk_actions.shape == expected_chunk_shape, (
            f"chunk_actions shape mismatch: got {chunk_actions.shape}, "
            f"expected {expected_chunk_shape}"
        )
        print("  ✓ Shape checks passed")

        # Print sample generated text if available
        if "generated_text" in result["forward_inputs"]:
            gen_texts = result["forward_inputs"]["generated_text"]
            print("\n  Sample generated texts:")
            for i, text in enumerate(gen_texts[:2]):
                print(
                    f"    [{i}]: {text[:100]}..."
                    if len(text) > 100
                    else f"    [{i}]: {text}"
                )

        return chunk_actions, result
    except Exception as e:
        print(f"✗ predict_action_batch() failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_train_mode(model):
    """Test that predict_action_batch works in train mode."""
    print("\n" + "=" * 60)
    print("Test 4: predict_action_batch() in train mode")
    print("=" * 60)

    env_obs = create_dummy_env_obs(batch_size=1)

    try:
        chunk_actions, result = model.predict_action_batch(
            env_obs=env_obs,
            mode="train",
            return_obs=True,
            do_sample=True,
            temperature=1.6,
            max_new_tokens=32,
        )
        print("✓ predict_action_batch() in train mode succeeded")
        print(f"  - chunk_actions shape: {chunk_actions.shape}")
        return chunk_actions, result
    except Exception as e:
        print(f"✗ predict_action_batch() in train mode failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("NaVid Model Loading Smoke Test")
    print("=" * 60)
    print("Model base path: /data/RLinf/VLN-CE/models/Navid_weight")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    try:
        # Test 1: Load model
        model = test_model_loading()

        # Test 2: preprocess_env_obs
        _ = test_preprocess_env_obs(model)

        # Test 3: predict_action_batch (eval mode)
        chunk_actions, result = test_predict_action_batch(model)

        # Test 4: predict_action_batch (train mode)
        train_actions, train_result = test_train_mode(model)

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nThe NaVid model is ready to use with RLinf rollout worker.")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Tests failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
