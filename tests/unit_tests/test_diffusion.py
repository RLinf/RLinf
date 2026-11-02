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

"""Diffusion datasets, differentiable rewards, and SD3 training forwards."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.sim.diffusion import build_reward_dataset
from rlinf.models.embodiment.base_policy import ForwardType


def test_prompt_dataset_excludes_eval_and_preserves_unique_order(tmp_path):
    train = tmp_path / "train.json"
    held_out = tmp_path / "test.json"
    train.write_text(
        json.dumps(
            [
                {"text": " a fox "},
                {"prompt": "a cat"},
                {"text": "a fox"},
                {"text": "a dog"},
            ]
        )
    )
    held_out.write_text(json.dumps([{"prompt": "a cat"}]))
    dataset = build_reward_dataset(
        {
            "type": "visual.prompt_dataset",
            "path": str(train),
            "exclude_path": str(held_out),
        }
    )

    assert [row["task_description"] for row in dataset] == ["a fox", "a dog"]
    obs, records = dataset.build_grouped_env_batch([1, 0], group_size=2, num_envs=4)
    assert obs["task_descriptions"] == ["a dog", "a dog", "a fox", "a fox"]
    assert len(records) == 4


def test_prompt_dataset_without_exclusion(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{"text": "a fox"}, {"prompt": "a cat"}]))
    dataset = build_reward_dataset({"type": "visual.prompt_dataset", "path": str(path)})
    assert len(dataset) == 2
    assert dataset[0] == {"task_description": "a fox"}


def test_image_text_dataset_preprocessing_and_batches(tmp_path):
    pytest.importorskip("torchvision")
    from torch.utils.data import DataLoader, DistributedSampler

    for index, caption in enumerate(("The image shows a fox", "a cat")):
        directory = tmp_path / str(index)
        directory.mkdir()
        Image.new("RGB", (12, 8), color=(255, 0, 0)).save(directory / "image.png")
        (directory / "image.txt").write_text(caption)
    dataset = build_reward_dataset(
        {
            "type": "visual.image_text_dataset",
            "path": str(tmp_path),
            "resolution": 8,
            "caption_prefix": "The image shows ",
        }
    )
    assert len(dataset) == 2
    assert [dataset[i]["prompts"] for i in range(2)] == ["a fox", "a cat"]
    torch.testing.assert_close(
        dataset[0]["images"],
        torch.tensor([1.0, -1.0, -1.0]).view(3, 1, 1).expand(3, 8, 8),
    )

    # Recreate the same CPU loader at a dataset boundary, as Lpre does.
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, drop_last=True)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["images"].shape == (2, 3, 8, 8)
        assert set(batches[0]["prompts"]) == {"a fox", "a cat"}


@pytest.mark.parametrize("pre_weight", [0.0, 0.1, 0.01])
def test_refl_config_composes(monkeypatch, pre_weight):
    from hydra import compose, initialize_config_dir

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(repo))
    monkeypatch.setenv("DIFFUSION_PATH", str(repo / "examples/diffusion"))
    with initialize_config_dir(
        config_dir=str(repo / "examples/diffusion/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="sd3_refl_imagereward",
            overrides=[
                f"algorithm.refl_pre_loss_weight={pre_weight}",
                "actor.refl_pre_data.num_workers=0",
                "runner.max_epochs=200",
                "runner.max_steps=200",
            ],
        )
        OmegaConf.resolve(cfg)
    assert cfg.algorithm.loss_type == "embodied_refl"
    assert cfg.actor.model.sd3.rl_mode == "refl"
    assert cfg.actor.refl_pre_data.num_workers == 0
    assert cfg.algorithm.refl_pre_loss_weight == pre_weight
    assert cfg.env.train.reward.model_path == cfg.actor.refl_reward_model.model_path
    assert cfg.env.eval.sequential_eval
    assert cfg.runner.max_steps == 200


class _ExternalImageReward(torch.nn.Module):
    """Small replacement for the external checkpoint, not the RLinf scorer."""

    def __init__(self, tokenizer):
        super().__init__()
        self.projection = torch.nn.Linear(3, 1)
        self.blip = SimpleNamespace(tokenizer=tokenizer)

    def score_gard(self, input_ids, attention_mask, pixels):
        return self.projection(pixels.mean(dim=(-2, -1)))


def test_image_reward_freezes_weights_but_preserves_image_gradients(
    tmp_path, monkeypatch
):
    pytest.importorskip("torchvision")
    from transformers import BertTokenizer

    from rlinf.envs.sim.diffusion.rewards.visual.image_reward import (
        ImageRewardBackend,
        ImageRewardScorer,
    )

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nfox\ncat\n")
    external_model = _ExternalImageReward(BertTokenizer(vocab_file=str(vocab)))
    reward_package = ModuleType("ImageReward")
    reward_package.load = lambda *args, **kwargs: external_model
    monkeypatch.setitem(sys.modules, "ImageReward", reward_package)
    scorer = ImageRewardScorer.from_config(
        {
            "model_path": "local-checkpoint",
            "med_config": "local-config",
            "precision": "fp32",
        },
        device="cpu",
    )
    images = torch.full((2, 3, 16, 16), 0.5, requires_grad=True)
    tokens = scorer.tokenize(["a fox", "a cat"])
    rewards = scorer.score(images, *tokens)
    assert rewards.shape == (2,)
    assert tokens[0].shape == (2, 35)
    rewards.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert all(
        not p.requires_grad and p.grad is None for p in external_model.parameters()
    )

    backend = ImageRewardBackend(scorer, batch_size=1)
    scores = backend.score(
        np.full((2, 16, 16, 3), 128, dtype=np.uint8),
        [{"task_description": "a fox"}, {"task_description": "a cat"}],
    )
    expected = scorer.score(torch.full((2, 3, 16, 16), 128 / 255), *tokens).detach()
    torch.testing.assert_close(scores["image_reward"], expected)
    torch.testing.assert_close(scores["avg"], expected)
    assert not scores["image_reward"].requires_grad


@pytest.fixture
def tiny_sd3(tmp_path):
    pytest.importorskip("torchvision")
    diffusers = pytest.importorskip("diffusers", minversion="0.38.0")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertTokenizer,
        CLIPTextConfig,
        CLIPTextModelWithProjection,
        PreTrainedTokenizerFast,
        T5Config,
        T5EncoderModel,
    )

    from rlinf.models.diffusion.sd3 import StableDiffusion3, StableDiffusion3Config
    from rlinf.models.diffusion.sd3.utils import configure_pipeline_trainability

    torch.manual_seed(42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "a", "fox", "cat"]
        )
    }
    tokenizer_backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=8,
    )
    clip_config = CLIPTextConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=8,
        projection_dim=16,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    transformer = diffusers.SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        caption_projection_dim=16,
        pooled_projection_dim=32,
        pos_embed_max_size=8,
    )
    vae = diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        latent_channels=16,
        norm_num_groups=4,
        use_quant_conv=False,
        use_post_quant_conv=False,
        scaling_factor=1.5305,
        shift_factor=0.0609,
    )
    pipeline = diffusers.StableDiffusion3Pipeline(
        transformer=transformer,
        vae=vae,
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0),
        text_encoder=CLIPTextModelWithProjection(clip_config),
        tokenizer=tokenizer,
        text_encoder_2=CLIPTextModelWithProjection(clip_config),
        tokenizer_2=tokenizer,
        text_encoder_3=T5EncoderModel(
            T5Config(
                vocab_size=len(vocab),
                d_model=32,
                d_ff=32,
                d_kv=8,
                num_layers=1,
                num_heads=2,
            )
        ),
        tokenizer_3=tokenizer,
    )
    configure_pipeline_trainability(pipeline, train_transformer=True)
    bert_dir = tmp_path / "bert"
    bert_dir.mkdir()
    bert_vocab = bert_dir / "vocab.txt"
    bert_vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nfox\ncat\n")
    BertTokenizer(vocab_file=str(bert_vocab)).save_pretrained(bert_dir)
    config = StableDiffusion3Config(
        resolution=16,
        num_steps=4,
        max_sequence_length=8,
        rl_mode="refl",
        cfg=False,
        guidance_scale=1.0,
        train_sampler="ode",
        refl_late_fraction=0.5,
        reward_tokenizer_path=str(bert_dir),
        use_lora=False,
        offload_auxiliary_modules=False,
    )
    try:
        yield StableDiffusion3(config, pipeline)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
def test_refl_rollout_and_differentiable_forward(tiny_sd3, cfg, use_lora):
    model = tiny_sd3
    if use_lora:
        from peft import LoraConfig, get_peft_model

        model.transformer = get_peft_model(
            model.transformer,
            LoraConfig(r=2, lora_alpha=4, target_modules=model.config.target_modules),
        )
        model.pipeline.transformer = model.transformer
    model.config.use_lora = use_lora
    model.config.cfg = cfg
    images, output = model.predict_action_batch({"task_descriptions": ["a fox"]})
    forward_inputs = output["forward_inputs"]
    assert forward_inputs["latents"].shape == (1, 2, 16, 8, 8)
    assert forward_inputs["reward_input_ids"].shape == (1, 35)
    assert all(not tensor.requires_grad for tensor in forward_inputs.values())
    assert "next_latents" not in forward_inputs

    predicted = model(
        forward_type=ForwardType.REFL,
        forward_inputs=forward_inputs,
        refl_inputs={
            "x_t": forward_inputs["latents"][:, -1],
            "timesteps": forward_inputs["timesteps"][:, -1] / 1000,
        },
    )["images"]
    torch.testing.assert_close(predicted, images, atol=1e-5, rtol=1e-5)
    predicted.square().mean().backward()
    grads = [p.grad for p in model.transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0
    assert all(p.grad is None for p in model.pipeline.vae.parameters())
    if use_lora:
        assert all(
            p.grad is None
            for name, p in model.transformer.named_parameters()
            if "lora_" not in name
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in model.transformer.named_parameters()
            if "lora_" in name
        )


def test_sd3_pretraining_forward_backpropagates_only_to_transformer(tiny_sd3):
    model = tiny_sd3
    loss = model(
        forward_type=ForwardType.SFT,
        data={"images": torch.zeros(1, 3, 16, 16), "prompts": ["a fox"]},
    )
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    grads = [p.grad for p in model.transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0
    for module in (
        model.pipeline.vae,
        model.pipeline.text_encoder,
        model.pipeline.text_encoder_2,
        model.pipeline.text_encoder_3,
    ):
        assert all(p.grad is None for p in module.parameters())


def test_sd3_pretraining_matches_flow_velocity_target(tiny_sd3):
    model = tiny_sd3
    model.eval()
    for module in (
        model.pipeline.vae,
        model.pipeline.text_encoder,
        model.pipeline.text_encoder_2,
        model.pipeline.text_encoder_3,
    ):
        module.eval()
    with torch.no_grad():
        for parameter in model.transformer.parameters():
            parameter.zero_()
        model.transformer.proj_out.bias.fill_(0.25)

    images = torch.zeros(1, 3, 16, 16)
    vae = model.pipeline.vae
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            torch.randn(1)  # The logit-normal time draw precedes the image noise.
            noise = torch.randn_like(latents)
            expected = ((0.25 - (noise - latents)) ** 2).mean()
        torch.manual_seed(7)
        actual = model(
            forward_type=ForwardType.SFT,
            data={"images": images, "prompts": ["a fox"]},
        )
    torch.testing.assert_close(actual, expected)
