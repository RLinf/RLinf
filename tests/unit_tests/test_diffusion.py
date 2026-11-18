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

from rlinf.algorithms.refl_scorers import build_refl_scorer, register_refl_scorer
from rlinf.envs.sim.diffusion import build_reward_dataset
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


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
                {"text": "A CAT"},
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
@pytest.mark.parametrize("image_size,max_tokens", [(224, 35), (32, 7)])
def test_refl_config_composes(monkeypatch, pre_weight, image_size, max_tokens):
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
                f"actor.refl_reward_model.image_size={image_size}",
                f"actor.refl_reward_model.max_tokens={max_tokens}",
                "runner.max_epochs=200",
                "runner.max_steps=200",
            ],
        )
        OmegaConf.resolve(cfg)
    from rlinf.config import validate_refl_cfg

    validate_refl_cfg(cfg)
    assert cfg.algorithm.loss_type == "embodied_refl"
    assert "refl_sampling" not in cfg.algorithm
    assert cfg.actor.model.sd3.rl_mode == "refl"
    assert cfg.actor.refl_reward_model.type == "image_reward"
    assert cfg.actor.refl_reward_model.image_size == image_size
    assert cfg.actor.refl_reward_model.max_tokens == max_tokens
    assert cfg.env.eval.reward.image_size == image_size
    assert cfg.env.eval.reward.max_tokens == max_tokens
    assert cfg.actor.refl_pre_data.num_workers == 0
    assert cfg.algorithm.refl_pre_loss_weight == pre_weight
    assert cfg.env.train is None
    assert cfg.actor.refl_data.type == "visual.prompt_dataset"
    assert cfg.actor.refl_samples_per_step == 512
    assert cfg.env.eval.reward.model_path == cfg.actor.refl_reward_model.model_path
    assert cfg.env.eval.sequential_eval
    assert cfg.runner.max_steps == 200


class _VectorPolicy(torch.nn.Module, BasePolicy):
    forward = BasePolicy.forward

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)

    def default_forward(self, features):
        return self.projection(features)

    def predict_action_batch(self, env_obs, **kwargs):
        return self.default_forward(env_obs["features"]), {}

    def refl_forward(self, forward_inputs, xt_source="rollout", **kwargs):
        return {"samples": self.default_forward(forward_inputs["features"])}


class _VectorScorer(torch.nn.Module):
    def __init__(self, target, device):
        super().__init__()
        self.target = torch.nn.Parameter(
            torch.full((4,), target, device=device), requires_grad=False
        )

    def forward(self, samples, forward_inputs):
        error = samples - self.target - forward_inputs["offset"]
        return -error.square().flatten(1).mean(1)


def test_refl_scorer_supports_non_image_samples_and_policy_gradients():
    register_refl_scorer(
        "test_vector_reward", lambda cfg, device: _VectorScorer(cfg["target"], device)
    )
    scorer = build_refl_scorer({"type": "test_vector_reward", "target": 0.5}, "cpu")
    policy = _VectorPolicy()
    samples = policy(
        forward_type=ForwardType.REFL,
        forward_inputs={"features": torch.ones(2, 5, 3)},
    )["samples"]
    rewards = scorer(samples, {"offset": torch.zeros(2, 5, 4)})
    assert rewards.shape == (2,) and torch.isfinite(rewards).all()
    torch.nn.functional.relu(2 - rewards).mean().backward()
    assert policy.projection.weight.grad is not None
    assert policy.projection.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in scorer.parameters())
    with pytest.raises(ValueError, match="already registered"):
        register_refl_scorer("test_vector_reward", lambda cfg, device: scorer)


def test_refl_scorer_rejects_invalid_registration_and_unknown_types():
    with pytest.raises(ValueError, match="non-empty"):
        register_refl_scorer("", lambda cfg, device: None)
    with pytest.raises(TypeError, match="callable"):
        register_refl_scorer("invalid_builder", None)
    with pytest.raises(ValueError, match="Unknown ReFL scorer 'unknown'"):
        build_refl_scorer({"type": "unknown"}, "cpu")


class _ExternalImageReward(torch.nn.Module):
    """Small replacement for the external checkpoint, not the RLinf scorer."""

    def __init__(self, tokenizer):
        super().__init__()
        self.projection = torch.nn.Linear(3, 1)
        self.blip = SimpleNamespace(tokenizer=tokenizer)
        self.image_shapes = []

    def score_gard(self, input_ids, attention_mask, pixels):
        self.image_shapes.append(pixels.shape)
        return self.projection(pixels.mean(dim=(-2, -1)))


@pytest.fixture
def image_reward_package(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    from transformers import BertTokenizer

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nfox\ncat\n")
    external_model = _ExternalImageReward(BertTokenizer(vocab_file=str(vocab)))
    reward_package = ModuleType("ImageReward")
    reward_package.load = lambda *args, **kwargs: external_model
    monkeypatch.setitem(sys.modules, "ImageReward", reward_package)
    reward_cfg = {
        "model_path": "local-checkpoint",
        "med_config": "local-config",
        "precision": "fp32",
    }
    return reward_cfg, external_model


@pytest.mark.parametrize("scorer_type", [None, "image_reward"])
@pytest.mark.parametrize("preprocess_cfg", [{}, {"image_size": 32, "max_tokens": 7}])
def test_image_reward_freezes_weights_but_preserves_image_gradients(
    image_reward_package, scorer_type, preprocess_cfg
):
    from rlinf.envs.sim.diffusion.rewards.visual.image_reward import ImageRewardBackend

    reward_cfg, external_model = image_reward_package
    reward_cfg.update(preprocess_cfg)
    if scorer_type is not None:
        reward_cfg["type"] = scorer_type
    scorer = build_refl_scorer(reward_cfg, device="cpu")
    images = torch.full((2, 3, 16, 16), 0.5, requires_grad=True)
    tokens = scorer.tokenize(["a fox", "a cat"])
    rewards = scorer(images, {"prompts": ["a fox", "a cat"]})
    torch.testing.assert_close(
        rewards,
        scorer(
            images, {"reward_input_ids": tokens[0], "reward_attention_mask": tokens[1]}
        ),
    )
    torch.testing.assert_close(rewards, scorer.score(images, *tokens))
    assert rewards.shape == (2,)
    assert tokens[0].shape == (2, preprocess_cfg.get("max_tokens", 35))
    rewards.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert all(
        not p.requires_grad and p.grad is None for p in external_model.parameters()
    )

    backend = ImageRewardBackend.from_config(
        {**reward_cfg, "use_gpu": False, "batch_size": 1}
    )
    scores = backend.score(
        np.full((2, 16, 16, 3), 128, dtype=np.uint8),
        [{"task_description": "a fox"}, {"task_description": "a cat"}],
    )
    expected = scorer.score(torch.full((2, 3, 16, 16), 128 / 255), *tokens).detach()
    torch.testing.assert_close(scores["image_reward"], expected)
    torch.testing.assert_close(scores["avg"], expected)
    assert not scores["image_reward"].requires_grad
    image_size = preprocess_cfg.get("image_size", 224)
    assert all(
        shape[-2:] == (image_size, image_size) for shape in external_model.image_shapes
    )


@pytest.fixture
def tiny_sd3(tmp_path):
    pytest.importorskip("torchvision")
    diffusers = pytest.importorskip("diffusers", minversion="0.38.0")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
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
            [
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]",
                "a",
                "fox",
                "cat",
            ]
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
    config = StableDiffusion3Config(
        resolution=16,
        num_steps=4,
        max_sequence_length=8,
        rl_mode="refl",
        cfg=False,
        guidance_scale=1.0,
        train_sampler="ode",
        refl_late_fraction=0.5,
        use_lora=False,
        offload_auxiliary_modules=False,
    )
    try:
        yield StableDiffusion3(config, pipeline)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("native_time_scale", [1000, 2000])
def test_refl_explicit_state_forward_preserves_gradients(
    tiny_sd3, cfg, use_lora, native_time_scale
):
    from diffusers import FlowMatchEulerDiscreteScheduler

    model = tiny_sd3
    model.pipeline.scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=native_time_scale
    )
    if use_lora:
        from peft import LoraConfig, get_peft_model

        model.transformer = get_peft_model(
            model.transformer,
            LoraConfig(r=2, lora_alpha=4, target_modules=model.config.target_modules),
        )
        model.pipeline.transformer = model.transformer
    model.config.use_lora = use_lora
    model.config.cfg = cfg
    embeds, pooled = model.encode_prompts(["a fox"])
    x_t = torch.randn(1, 16, 8, 8)
    t = torch.tensor([0.25])
    with torch.no_grad():
        velocity = model.transformer(
            hidden_states=x_t,
            timestep=t * native_time_scale,
            encoder_hidden_states=embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
        x0 = x_t - t.view(-1, 1, 1, 1) * velocity
        vae = model.pipeline.vae
        latents = x0 / vae.config.scaling_factor + vae.config.shift_factor
        expected = (vae.decode(latents, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)
    output = model(
        forward_type=ForwardType.REFL,
        forward_inputs={"prompt_embeds": embeds, "pooled_prompt_embeds": pooled},
        refl_inputs={"x_t": x_t, "timesteps": t},
    )
    predicted = output["samples"]
    torch.testing.assert_close(predicted, expected, atol=1e-5, rtol=1e-5)
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


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_refl_inference_returns_images(tiny_sd3, mode):
    model = tiny_sd3
    model.config.eval_num_steps = 6
    images, output = model.predict_action_batch(
        {"task_descriptions": ["a fox", "a cat"]}, mode=mode
    )
    assert images.shape == (2, 3, 16, 16)
    assert torch.isfinite(images).all() and not images.requires_grad
    assert output == {"prev_values": None}


@pytest.mark.parametrize("rl_mode", ["flow-grpo", "nft", "art"])
def test_sd3_other_algorithms_keep_rollout_training_inputs(tiny_sd3, rl_mode):
    model = tiny_sd3
    model.config.rl_mode = rl_mode
    images, output = model.predict_action_batch({"task_descriptions": ["a fox"]})
    assert images.shape == (1, 3, 16, 16)
    assert output["prev_logprobs"].shape == (1, 3)
    inputs = output["forward_inputs"]
    if rl_mode == "art":
        assert inputs["art_states"].shape == (1, 4, 1, 16, 8, 8)
        assert inputs["art_sigmas"].shape == (1, 4)
        assert inputs["art_sigmas_next"].shape == (1, 4)
        assert inputs["art_endpoint"].shape == (1, 1, 16, 8, 8)
        assert not {"latents", "next_latents", "timesteps"}.intersection(inputs)
    else:
        assert inputs["latents"].shape == (1, 3, 16, 8, 8)
        assert inputs["next_latents"].shape == inputs["latents"].shape
    assert all(not value.requires_grad for value in inputs.values())
    if rl_mode == "nft":
        assert inputs["nft_x0"].shape == (1, 1, 16, 8, 8)


def test_refl_model_rejects_unknown_state_sources(tiny_sd3):
    with pytest.raises(ValueError, match="xt_source must be 'rollout' or 'resample'"):
        tiny_sd3(
            forward_type=ForwardType.REFL,
            forward_inputs={"prompts": ["a fox"]},
            xt_source="unknown",
        )


def test_refl_model_requires_prompts_or_an_explicit_state(tiny_sd3):
    with pytest.raises(ValueError, match="Expected prompts.*or explicit refl_inputs"):
        tiny_sd3(forward_type=ForwardType.REFL, forward_inputs={})


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


@pytest.mark.parametrize("xt_source", ["rollout", "resample"])
@pytest.mark.parametrize("native_time_scale", [1000, 2000])
def test_refl_batch_sampling_preserves_gradients(
    tiny_sd3, xt_source, native_time_scale
):
    from diffusers import FlowMatchEulerDiscreteScheduler

    model = tiny_sd3
    model.pipeline.scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=native_time_scale
    )
    forwards = []
    decodes = []
    hook = model.transformer.register_forward_pre_hook(
        lambda module, args: forwards.append(torch.is_grad_enabled())
    )
    decode_hook = model.pipeline.vae.decoder.register_forward_pre_hook(
        lambda module, args: decodes.append(torch.is_grad_enabled())
    )
    try:
        output = model(
            forward_type=ForwardType.REFL,
            forward_inputs={"prompts": ["a fox", "a cat"]},
            xt_source=xt_source,
        )
    finally:
        hook.remove()
        decode_hook.remove()
    prefix_steps = model.config.num_steps - (xt_source == "rollout")
    assert forwards == [False] * prefix_steps + [True]
    assert decodes == [True]
    context = output["forward_inputs"]
    assert context["prompts"] == ["a fox", "a cat"]
    assert all(
        not value.requires_grad for value in context.values() if torch.is_tensor(value)
    )
    assert output["samples"].shape[:2] == (2, 3)
    output["samples"].square().mean().backward()
    grads = [p.grad for p in model.transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in model.pipeline.vae.parameters())


@pytest.mark.parametrize("solver", ["ode", "sde", "dpm1", "dpm2"])
def test_refl_batch_states_match_full_sampling_prefix(tiny_sd3, solver):
    from rlinf.models.diffusion.sd3.utils import denoise_with_logprob

    model = tiny_sd3
    embeds, pooled = model.encode_prompts(["a fox", "a cat"])
    latents = torch.randn(2, 16, 8, 8)
    kwargs = {
        "pipeline": model.pipeline,
        "transformer": model.transformer,
        "prompt_embeds": embeds,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": None,
        "negative_pooled_prompt_embeds": None,
        "cfg_enabled": False,
        "guidance_scale": 1.0,
        "noise_level": 0.7 if solver == "sde" else 0.0,
        "num_steps": 4,
        "resolution": 16,
        "output_type": "latent",
        "solver": solver,
        "latents": latents.clone(),
    }
    torch.manual_seed(123)
    _, chain, _ = denoise_with_logprob(**kwargs, return_trajectory=True)
    steps = torch.tensor([2, 3])
    torch.manual_seed(123)
    selected, trajectory, log_probs = denoise_with_logprob(
        **kwargs, return_trajectory=False, sample_steps=steps
    )
    torch.testing.assert_close(selected, torch.stack([chain[2][0], chain[3][1]]))
    assert trajectory is None and log_probs is None
    with pytest.raises(ValueError, match="one sampling step"):
        denoise_with_logprob(
            **kwargs, return_trajectory=False, sample_steps=steps.float() + 0.5
        )


def test_sd3_unit_guidance_does_not_compute_unconditional_branch(tiny_sd3):
    model = tiny_sd3
    model.config.cfg = True
    model.config.guidance_scale = 1.0
    batch_sizes = []
    hook = model.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: batch_sizes.append(
            kwargs["hidden_states"].shape[0]
        ),
        with_kwargs=True,
    )
    try:
        model.predict_action_batch({"task_descriptions": ["a fox"]})
    finally:
        hook.remove()
    assert batch_sizes == [1] * model.config.num_steps


def test_refl_config_validates_training_batches():
    from rlinf.config import validate_refl_cfg

    cfg = OmegaConf.create(
        {
            "algorithm": {"update_epoch": 1, "group_size": 1},
            "actor": {
                "global_batch_size": 64,
                "refl_samples_per_step": 512,
                "refl_data": {"type": "visual.prompt_dataset"},
            },
            "runner": {},
            "env": {"train": None},
        }
    )
    validate_refl_cfg(cfg)
    cfg.actor.refl_samples_per_step = 511
    with pytest.raises(ValueError, match="positive multiple"):
        validate_refl_cfg(cfg)
    cfg.actor.refl_samples_per_step = 512
    cfg.algorithm.update_epoch = 2
    with pytest.raises(ValueError, match="update_epoch=1"):
        validate_refl_cfg(cfg)
    cfg.algorithm.update_epoch = 1
    cfg.actor.refl_data.type = "visual.image_text_dataset"
    with pytest.raises(ValueError, match="refl_data.type=visual.prompt_dataset"):
        validate_refl_cfg(cfg)


def test_refl_rejects_conflicting_inputs(tiny_sd3):
    with pytest.raises(ValueError, match="not both"):
        tiny_sd3(
            forward_type=ForwardType.REFL,
            forward_inputs={"prompts": ["a fox"]},
            refl_inputs={"x_t": torch.zeros(1), "timesteps": torch.zeros(1)},
        )


def test_dataset_factory_import_does_not_require_gym():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['gym'] = None; "
            "from rlinf.envs.sim.diffusion import build_reward_dataset; "
            "assert callable(build_reward_dataset)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sequential_evaluation_repeats_the_same_prompt_batches(
    tmp_path, image_reward_package
):
    pytest.importorskip("gym")
    from rlinf.envs.sim.diffusion.generation_env import GenerationEnv

    reward_cfg, _ = image_reward_package
    prompts = tmp_path / "eval.json"
    prompts.write_text(json.dumps([{"text": f"prompt {i}"} for i in range(5)]))
    cfg = {
        "dataset": {"type": "visual.prompt_dataset", "path": str(prompts)},
        "reward": {
            **reward_cfg,
            "type": "single",
            "model": "visual.image_reward",
            "use_gpu": False,
        },
        "sequential_eval": True,
        "is_eval": True,
        "rollout_epoch": 3,
    }
    envs = [GenerationEnv(cfg, 1, rank, 2) for rank in range(2)]
    rounds = []
    for _ in range(2):
        batches = []
        for _ in range(3):
            for env in envs:
                batches.append(env.reset()[0]["task_descriptions"])
                env.update_reset_state_ids()
        rounds.append(batches)
    assert rounds[0] == [[f"prompt {i % 5}"] for i in range(6)]
    assert rounds[1] == rounds[0]


@pytest.mark.parametrize("pre_weight", [0.0, 0.1, 0.01])
def test_refl_actor_updates_current_weights_and_respects_sample_budget(
    tmp_path, monkeypatch, pre_weight
):
    from contextlib import nullcontext

    from rlinf.scheduler import Worker
    from rlinf.scheduler.hardware import AcceleratorType
    from rlinf.workers.actor.fsdp_refl_policy_worker import EmbodiedReFLFSDPPolicy

    events = []

    class ToyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.sampled_weights = []

        def forward(self, forward_type, forward_inputs=None, data=None, **kwargs):
            if forward_type == ForwardType.SFT:
                events.append("pre")
                return (self.weight - 1).square()
            assert forward_type == ForwardType.REFL
            events.append("reward")
            self.sampled_weights.append(self.weight.detach().item())
            return {"samples": self.weight.expand(len(forward_inputs["prompts"]), 1)}

    class LocalActor(EmbodiedReFLFSDPPolicy):
        def __init__(self, cfg):
            self.cfg = cfg
            self._rank, self._world_size = 0, 1
            self._accelerator_type = AcceleratorType.NO_ACCEL
            self._timer_metrics = {}
            self.device = torch.device("cpu")
            self.amp_context = nullcontext()
            self.is_weight_offloaded = self.is_optimizer_offloaded = False
            self.refl_pre_loss_weight = pre_weight
            self.refl_margin, self.refl_grad_scale = 2.0, 0.1
            self.refl_xt_source = "rollout"
            self.gradient_accumulation = 2
            self.model = ToyPolicy()
            self.reward_scorer = lambda samples, context: -samples.square().sum(-1)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=1
            )
            self.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
            if pre_weight > 0:
                self._build_pre_data_loader()
            self._build_prompt_data_loader()

        def before_micro_batch(self, model, is_last_micro_batch):
            return nullcontext()

        def optimizer_step(self):
            events.append("step")
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return norm.item(), [self.optimizer.param_groups[0]["lr"]]

    # The CPU policy runs without an accelerator or cross-process collectives.
    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(
            current_device=lambda: torch.device("cpu"),
            empty_cache=lambda: None,
            synchronize=lambda: None,
            ipc_collect=lambda: None,
        ),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)
    prompts = tmp_path / "train.json"
    prompts.write_text(json.dumps([{"text": f"prompt {i}"} for i in range(4)]))
    images = tmp_path / "pairs"
    images.mkdir()
    for i in range(4):
        Image.new("RGB", (8, 8)).save(images / f"{i}.png")
        (images / f"{i}.txt").write_text(f"prompt {i}")
    cfg = OmegaConf.create(
        {
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "global_batch_size": 2,
                "refl_samples_per_step": 4,
                "refl_data": {"type": "visual.prompt_dataset", "path": str(prompts)},
                "refl_pre_data": {
                    "type": "visual.image_text_dataset",
                    "path": str(images),
                    "resolution": 8,
                    "num_workers": 0,
                },
            },
        }
    )
    actor = LocalActor(cfg)
    metrics = actor.run_training()
    update_events = ["pre", "pre", "step"] if pre_weight else []
    assert events == (update_events + ["reward", "reward", "step"]) * 2
    assert metrics["actor/refl_samples"] == 4
    assert metrics["actor/refl_reward_updates"] == 2
    assert metrics["actor/refl_pre_updates"] == (2 if pre_weight else 0)
    assert actor.lr_scheduler.last_epoch == 1
    sampled = actor.model.sampled_weights
    assert sampled[0] == sampled[1] and sampled[2] == sampled[3]
    assert sampled[0] == pytest.approx(0.5 + 0.1 * pre_weight)
    assert actor.model.weight.item() < sampled[-1]
    assert actor.model.weight.grad is None


def test_refl_runner_evaluates_saves_and_resumes_without_training_rollout(tmp_path):
    import queue
    import threading

    from rlinf.runners.refl_runner import ReFLRunner
    from rlinf.utils.distributed import ScopedTimer
    from rlinf.utils.logging import get_logger
    from rlinf.utils.metric_logger import MetricLogger

    class Result:
        def __init__(self, value=None):
            self.value = value

        def wait(self):
            return self.value

        def consume_durations(self, return_per_rank=False):
            return {}, [{}]

    class ActorGroup:
        worker_group_name = "ActorGroup"

        def __init__(self):
            self.weight = 0
            self.steps = []
            self.saved = []

        def init_worker(self):
            return Result()

        def set_global_step(self, step):
            self.step = step
            return Result()

        def run_training(self):
            self.steps.append(self.step)
            self.weight += 1
            return Result([{"actor/refl_samples": 4.0}])

        def sync_model_to_rollout(self):
            rollout.weight = self.weight
            return Result()

        def save_checkpoint(self, path, step):
            self.saved.append(step)
            torch.save(self.weight, Path(path) / "weight.pt")
            return Result()

        def load_checkpoint(self, path):
            self.weight = torch.load(Path(path) / "weight.pt", weights_only=True)
            return Result()

    class EvalGroup:
        def __init__(self):
            self.weight = 0
            self.evaluated = []

        def init_worker(self):
            return Result()

        def set_global_step(self, step):
            return Result()

        def sync_model_from_actor(self):
            return Result()

        def evaluate(self, **kwargs):
            self.evaluated.append(rollout.weight)
            return Result([{"image_reward": torch.tensor([float(rollout.weight)])}])

    def make_runner(actor, max_steps, resume_dir=None):
        # Worker groups are the RPC edge; the runner's loop and lifecycle are real.
        runner = ReFLRunner.__new__(ReFLRunner)
        runner.cfg = OmegaConf.create(
            {
                "runner": {
                    "max_epochs": 4,
                    "max_steps": max_steps,
                    "val_check_interval": 2,
                    "save_interval": 2,
                    "resume_dir": resume_dir,
                    "logger": {
                        "log_path": str(tmp_path),
                        "experiment_name": "runner",
                        "logger_backends": [],
                    },
                },
            }
        )
        runner.actor, runner.rollout, runner.env = actor, rollout, env
        runner.reward = None
        runner.env_channel, runner.rollout_channel = object(), object()
        runner.global_step = 0
        runner.set_max_steps()
        runner.timer = ScopedTimer(sync_cuda=False)
        runner.logger = get_logger()
        runner.metric_logger = MetricLogger(runner.cfg)
        runner.enable_per_worker_metric_log = False
        runner._profile_all_steps, runner._profile_steps = False, None
        runner.log_queue = queue.Queue()
        runner.log_thread = threading.Thread(target=runner._log_worker, daemon=True)
        runner.log_thread.start()
        runner.init_workers()
        return runner

    rollout, env = EvalGroup(), EvalGroup()
    first_actor = ActorGroup()
    first = make_runner(first_actor, 2)
    first.run()
    checkpoint = tmp_path / "runner/checkpoints/global_step_2/actor"
    assert torch.load(checkpoint / "weight.pt", weights_only=True) == 2
    assert first_actor.steps == [0, 1] and first_actor.saved == [2]
    assert not first.log_thread.is_alive()

    resumed_actor = ActorGroup()
    resumed = make_runner(resumed_actor, 4, str(checkpoint.parent))
    assert resumed.global_step == 2 and resumed_actor.weight == 2
    resumed.run()
    assert resumed_actor.steps == [2, 3] and resumed_actor.saved == [4]
    assert env.evaluated == rollout.evaluated == [2, 4]
    assert resumed.global_step == 4
    assert not resumed.log_thread.is_alive()


def test_refl_actor_seeds_model_initialization_and_rank_sampling(tmp_path):
    from rlinf.workers.actor.fsdp_refl_policy_worker import EmbodiedReFLFSDPPolicy

    class LocalActor(EmbodiedReFLFSDPPolicy):
        def __init__(self, cfg, rank):
            self.cfg = cfg
            self._rank, self._world_size = rank, 2
            self.enable_offload = False
            self.refl_pre_loss_weight = 0.0

        def setup_model_and_optimizer(self):
            self.model = torch.nn.Linear(2, 2)

    register_refl_scorer("test_seeded_reward", lambda cfg, device: lambda x, ctx: x)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"text": f"prompt {i}"} for i in range(4)]))
    cfg = OmegaConf.create(
        {
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "refl_reward_model": {"type": "test_seeded_reward"},
                "refl_data": {"type": "visual.prompt_dataset", "path": str(prompts)},
            },
        }
    )
    weights, noises = [], []
    for rank, ambient_seed in ((0, 123), (0, 456), (1, 789)):
        torch.manual_seed(ambient_seed)
        actor = LocalActor(cfg, rank)
        actor.device = torch.device("cpu")
        actor.init_worker()
        weights.append(actor.model.weight.detach().clone())
        noises.append(torch.randn(8))
    torch.testing.assert_close(weights[0], weights[1])
    torch.testing.assert_close(weights[0], weights[2])
    torch.testing.assert_close(noises[0], noises[1])
    assert not torch.equal(noises[0], noises[2])


@pytest.mark.parametrize("pre_weight,num_workers", [(0.0, 0), (0.1, 0), (0.1, 2)])
@pytest.mark.parametrize("rank", [0, 1])
def test_refl_checkpoint_restores_data_and_rng(
    tmp_path, monkeypatch, pre_weight, num_workers, rank
):
    import random

    from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy
    from rlinf.scheduler import Worker
    from rlinf.utils.logging import get_logger
    from rlinf.utils.utils import seed_everything
    from rlinf.workers.actor.fsdp_refl_policy_worker import EmbodiedReFLFSDPPolicy

    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            synchronize=lambda: None,
            ipc_collect=lambda: None,
        ),
    )
    prompts = tmp_path / "train.json"
    prompts.write_text(json.dumps([{"text": f"prompt {i}"} for i in range(12)]))
    images = tmp_path / "pairs"
    images.mkdir()
    for i in range(8):
        pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3) + i
        Image.fromarray(pixels).save(images / f"{i}.png")
        (images / f"{i}.txt").write_text(f"caption {i}")
    cfg = OmegaConf.create(
        {
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "global_batch_size": 2,
                "refl_samples_per_step": 2,
                "refl_data": {"type": "visual.prompt_dataset", "path": str(prompts)},
                "refl_pre_data": {
                    "type": "visual.image_text_dataset",
                    "path": str(images),
                    "resolution": 8,
                    "num_workers": num_workers,
                },
                "fsdp_config": {"save_full_model_weights": False},
            }
        }
    )

    def make_actor():
        actor = EmbodiedReFLFSDPPolicy.__new__(EmbodiedReFLFSDPPolicy)
        actor.cfg, actor._cfg = cfg, cfg.actor
        actor._rank, actor._world_size = rank, 2
        actor._logger, actor._stacklevel = get_logger(), 1
        actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
        actor.refl_pre_loss_weight = pre_weight
        actor.gradient_accumulation = 1
        actor.model = torch.nn.Linear(4, 1)
        actor.optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.01)
        actor.lr_scheduler = torch.optim.lr_scheduler.StepLR(actor.optimizer, 1)
        actor._strategy = FSDP2Strategy(cfg.actor, world_size=1)
        actor._build_prompt_data_loader()
        if pre_weight:
            actor._build_pre_data_loader()
        return actor

    def advance(actor):
        batch = next(actor._training_batches())[0]
        prompt = batch["forward_inputs"]["prompts"]
        pre = actor._next_pre_micro_batches()[0] if pre_weight else None
        noise = torch.randn(1, 4)
        target = random.random() + np.random.rand()
        actor.optimizer.zero_grad()
        loss = (actor.model(noise) - target).square().mean()
        if pre is not None:
            loss = (
                loss
                + pre_weight * actor.model(pre["images"].flatten()[:4]).square().mean()
            )
        loss.backward()
        actor.optimizer.step()
        actor.lr_scheduler.step()
        return prompt, pre, noise, loss.detach()

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/gloo", rank=0, world_size=1
    )
    try:
        seed_everything(42 + rank)
        actor = make_actor()
        for _ in range(5):
            advance(actor)
        checkpoint = tmp_path / "checkpoint"
        actor.save_checkpoint(str(checkpoint), step=5)
        assert (checkpoint / f"refl_data_rank_{rank}.pt").is_file()
        expected = [advance(actor) for _ in range(8)]
        expected_weights = {
            name: value.clone() for name, value in actor.model.state_dict().items()
        }
        expected_lr = actor.lr_scheduler.state_dict()

        seed_everything(987)
        resumed = make_actor()
        resumed.load_checkpoint(str(checkpoint))
        actual = [advance(resumed) for _ in range(8)]
        for (prompt_a, pre_a, noise_a, loss_a), (
            prompt_b,
            pre_b,
            noise_b,
            loss_b,
        ) in zip(expected, actual):
            assert prompt_a == prompt_b
            if pre_weight:
                assert pre_a["prompts"] == pre_b["prompts"]
                torch.testing.assert_close(
                    pre_a["images"], pre_b["images"], rtol=0, atol=0
                )
            torch.testing.assert_close(noise_a, noise_b, rtol=0, atol=0)
            torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
        for name, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
        assert resumed.lr_scheduler.state_dict() == expected_lr

        (checkpoint / f"refl_data_rank_{rank}.pt").unlink()
        legacy = make_actor()
        legacy.load_checkpoint(str(checkpoint))
        assert legacy.prompt_data_iter.epoch == 0
    finally:
        torch.distributed.destroy_process_group()
