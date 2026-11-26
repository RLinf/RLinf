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

"""Distillation values, gradients, masking and distributed normalization."""

import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rlinf.algorithms.distillation.config import (
    DistillationLossConfig,
    DistillationPlan,
    build_distillation_plan,
)
from rlinf.algorithms.distillation.fsdp import compute_distribution_loss
from rlinf.algorithms.distillation.losses import (
    DistillationForwardOutput,
    DistillationReductionContext,
    combine_distillation_and_task_loss,
    compute_distillation_loss,
    compute_distillation_pg_tokens,
    compute_distillation_tokens,
)
from rlinf.algorithms.distillation.megatron import (
    compute_vocab_parallel_distribution_loss,
    select_vocab_parallel_topk,
)
from rlinf.algorithms.losses import (
    compute_ppo_actor_loss,
    compute_ppo_actor_loss_tokens,
)

MODES = (
    "forward_kl_topk",
    "forward_kl_topk_tail",
    "reverse_kl_topk",
    "reverse_kl_topk_tail",
)


def reference(logits, teacher, ids, mode, eps=1e-6):
    """Explicit probability distributions independent of the production kernels."""
    logq = logits.log_softmax(-1)
    logp = teacher.detach().log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    if ids is not None:
        outside = torch.ones_like(p, dtype=torch.bool).scatter(-1, ids, False)
        pt = p.masked_fill(~outside, 0).sum(-1)
        qt = q.masked_fill(~outside, 0).sum(-1)
        p, q = p.gather(-1, ids), q.gather(-1, ids)
        logp, logq = logp.gather(-1, ids), logq.gather(-1, ids)
    reverse = mode.startswith("reverse")
    loss = (q * (logq - logp) if reverse else p * (logp - logq)).sum(-1)
    if ids is not None and ids.shape[-1] < logits.shape[-1] and mode.endswith("_tail"):
        loss = loss + (
            qt * (qt.log() - pt.clamp_min(eps).log())
            if reverse
            else pt * (pt.log() - qt.clamp_min(eps).log())
        )
    return loss.clamp_min(0) if mode == "forward_kl_topk" else loss


def _tp_process(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(29)
        original = torch.randn(2, 3, 7)
        for tied in (False, True):
            scores = torch.ones_like(original) if tied else original
            shard = torch.nn.functional.pad(scores, (0, 1), value=100)[
                ..., rank * 4 : (rank + 1) * 4
            ]
            for k in (1, 5, 7):
                selected = select_vocab_parallel_topk(shard, k, 7, dist.group.WORLD)
                expected_ids = scores.argsort(dim=-1, descending=True, stable=True)[
                    ..., :k
                ]
                torch.testing.assert_close(selected, expected_ids)
        teacher = torch.randn_like(original)
        mask = torch.tensor([[True, False, True], [True, True, False]])
        from rlinf.algorithms.distillation.losses import compute_mc_pg_tokens

        mc = DistillationLossConfig(loss_mode="k1", mc_k=3)
        ids = torch.tensor([0, 4, 4]).expand(2, 3, 3)
        t = teacher.log_softmax(-1).gather(-1, ids)
        behavior = original.log_softmax(-1).gather(-1, ids) - 0.1
        dense = original.clone().requires_grad_()
        expected = compute_mc_pg_tokens(
            dense.log_softmax(-1).gather(-1, ids), behavior, t, mask, mc
        )
        shard = (
            torch.nn.functional.pad(original, (0, 1), value=100)[
                ..., rank * 4 : (rank + 1) * 4
            ]
            .clone()
            .requires_grad_()
        )
        actual = compute_vocab_parallel_distribution_loss(
            shard,
            ids,
            t,
            vocab_start=rank * 4,
            vocab_end=(rank + 1) * 4,
            vocab_size=7,
            tp_group=dist.group.WORLD,
            config=mc,
            valid_mask=mask,
            behavior_logprobs=behavior,
        ).distribution_losses
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(
            shard.grad,
            torch.nn.functional.pad(dense.grad, (0, 1))[..., rank * 4 : (rank + 1) * 4],
        )
        for mode in MODES:
            for k in (3, -1):
                config = DistillationLossConfig(
                    loss_mode=mode,
                    use_policy_gradient=False,
                    topk_source="student",
                    topk=k,
                )
                ids = original.argsort(-1, descending=True)[..., :k] if k > 0 else None
                expected_z = original.clone().requires_grad_()
                expected = torch.where(
                    mask, reference(expected_z, teacher, ids, mode), 0
                )
                expected.sum().backward()
                local_z = (
                    torch.nn.functional.pad(original, (0, 1), value=100)[
                        ..., rank * 4 : (rank + 1) * 4
                    ]
                    .clone()
                    .requires_grad_()
                )
                logp = teacher.log_softmax(-1)
                targets = (
                    logp.gather(-1, ids)
                    if ids is not None
                    else torch.nn.functional.pad(logp, (0, 1))[
                        ..., rank * 4 : (rank + 1) * 4
                    ]
                )
                actual = compute_vocab_parallel_distribution_loss(
                    local_z,
                    ids,
                    targets,
                    vocab_start=rank * 4,
                    vocab_end=(rank + 1) * 4,
                    vocab_size=7,
                    tp_group=dist.group.WORLD,
                    config=config,
                    valid_mask=mask,
                )
                actual.distribution_losses.sum().backward()
                torch.testing.assert_close(
                    actual.distribution_losses, expected, atol=2e-6, rtol=2e-5
                )
                expected_grad = torch.nn.functional.pad(expected_z.grad, (0, 1))[
                    ..., rank * 4 : (rank + 1) * 4
                ]
                torch.testing.assert_close(
                    local_z.grad, expected_grad, atol=2e-6, rtol=2e-5
                )
    finally:
        dist.destroy_process_group()


def _empty_rank_process(rank, path):
    from unittest.mock import patch

    from rlinf.data.schema.distillation import TeacherScoreRequest
    from rlinf.data.schema.reasoning_results import RolloutResult
    from rlinf.scheduler import Worker
    from rlinf.utils.down_sampling import down_sample_batch
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        actor = object.__new__(FSDPActor)
        for selection in (False, True):
            actor.distillation_plan = DistillationPlan(
                enabled=True, need_rollout_topk=selection
            )
            for empty_ranks in ((0,), (1,), (0, 1)):
                result = RolloutResult(
                    num_sequence=2,
                    group_size=2,
                    prompt_lengths=[1, 1],
                    prompt_ids=[[1], [1]],
                    response_lengths=[1, 1],
                    response_ids=[[2], [2]],
                    is_end=[True, True],
                    rewards=[0.0, 0.0] if rank in empty_ranks else [0.0, 1.0],
                    response_texts=["answer", "answer"],
                    teacher_requests=[
                        TeacherScoreRequest(
                            str(i),
                            str(i),
                            "g",
                            "default",
                            (1, 2),
                            (False, True),
                            1,
                            "tok",
                            student_policy_version="0",
                        )
                        for i in range(2)
                    ],
                )
                result = down_sample_batch(result, {"reject_equal_reward": True})
                assert result.num_sequence == 2
                with (
                    patch.object(Worker, "torch_device_type", "cpu"),
                    patch.object(torch.Tensor, "cuda", lambda tensor: tensor),
                ):
                    batch = RolloutResult.merge_batches(
                        [result.to_actor_batch(1, 2, 0)]
                    )
                    assert actor.prepare_distillation_targets(batch) is None
                # No rank may have started selection or consumed the Teacher schema.
                assert "teacher_requests" in batch
        actor.distillation_plan = DistillationPlan(enabled=True, need_rollout_topk=True)
        with patch.object(Worker, "torch_device_type", "cpu"):
            batch = {"input_ids": torch.ones(2, 2, dtype=torch.long)}
            assert actor.prepare_distillation_targets(batch) is batch
    finally:
        dist.destroy_process_group()


class TestDistillation(unittest.TestCase):
    def test_mixed_teacher_gradients_use_global_denominator(self):
        from rlinf.data.schema.distillation import (
            SampleContext,
            TeacherTargets,
            teacher_targets_to_batch,
        )

        keys = ("code", "math", "math")
        mask = torch.tensor(
            [[False, True, False], [False, True, True], [False, True, True]]
        )
        targets = [
            TeacherTargets(
                str(i),
                key,
                key + "-model",
                "sampled",
                None,
                torch.ones(3, 1, dtype=torch.long),
                torch.full((3, 1), -0.8 if key == "code" else -1.5),
                mask[i],
                SampleContext(str(i), str(i), key, int(key == "math")),
            )
            for i, key in enumerate(keys)
        ]
        batch = teacher_targets_to_batch(
            targets, [1] * 3, [2] * 3, 3, response_only=False
        )
        batch["response_mask"] = mask
        current = (
            torch.tensor([0.3, 0.8, -0.2]).expand(3, 3, 3).clone().requires_grad_()
        )
        teacher_logits = torch.tensor(
            [[0.2, 0.7, 0.1], [0.7, 0.1, 0.2], [0.7, 0.1, 0.2]]
        )
        teacher_logprobs = teacher_logits.log().unsqueeze(1).expand_as(current)
        sampled = current.log_softmax(-1)[..., 1]
        batch["rollout_logprobs"] = sampled.detach()
        for mode in ("k1", "k2", "k3", "abs", *MODES):
            for pg in (True,) if mode == "k1" else (False, True):
                for topk in (2, -1) if mode in MODES else (1,):
                    with self.subTest(mode=mode, pg=pg, topk=topk):
                        cfg = DistillationLossConfig(
                            loss_mode=mode,
                            use_policy_gradient=pg,
                            topk=topk,
                            topk_source="student"
                            if mode.startswith("reverse")
                            else "teacher",
                        )
                        plan = DistillationPlan(
                            enabled=True,
                            teacher_keys=("code", "math"),
                            use_policy_gradient=pg,
                            target_format="full"
                            if topk == -1
                            else "sparse"
                            if mode in MODES
                            else "sampled",
                        )
                        reduction = DistillationReductionContext(5)

                        def loss_for(rows):
                            forward = DistillationForwardOutput(logprobs=sampled[rows])
                            if mode in MODES:
                                scores = teacher_logprobs[rows]
                                ids = None
                                if topk > 0:
                                    source = (
                                        current[rows]
                                        if cfg.topk_source == "student"
                                        else scores
                                    )
                                    ids = source.topk(topk, dim=-1).indices
                                    scores = scores.gather(-1, ids)
                                distribution = compute_distribution_loss(
                                    current[rows],
                                    ids,
                                    scores,
                                    cfg,
                                    valid_mask=mask[rows],
                                )
                                forward.distribution_losses = (
                                    distribution.distribution_losses
                                )
                                forward.teacher_mass = distribution.teacher_mass
                            return compute_distillation_loss(
                                forward,
                                {k: v[rows] for k, v in batch.items()},
                                plan=plan,
                                config=cfg,
                                reduction=reduction,
                            )

                        mixed, metrics = loss_for(slice(None))
                        self.assertEqual(
                            metrics["distillation/math/valid_tokens"].item(), 4
                        )
                        blocks = loss_for(slice(0, 1))[0] + loss_for(slice(1, None))[0]
                        torch.testing.assert_close(mixed, blocks)
                        torch.testing.assert_close(
                            torch.autograd.grad(mixed, current, retain_graph=True)[0],
                            torch.autograd.grad(blocks, current, retain_graph=True)[0],
                        )

    def test_mc_k_actor_loss_and_gradients_use_each_behavior_sample(self):
        from types import SimpleNamespace

        from rlinf.workers.actor.distillation import fsdp_distillation_forward

        config = DistillationLossConfig(loss_mode="k1", mc_k=3)
        tokens = torch.tensor([[0, 1, 2, 1]])
        ids = torch.tensor([[[0, 0, 0], [1, 1, 0], [2, 1, 2], [1, 2, 2]]])
        mask = torch.tensor([[False, True, False, True]])
        teacher = torch.tensor(
            [[[-1.0] * 3, [-0.7, -0.7, -1.5], [-1.0] * 3, [-0.1, -3.0, -3.0]]]
        )
        behavior = torch.tensor(
            [[[-1.0] * 3, [-1.1, -1.1, -0.8], [-1.0] * 3, [-2.0, -0.5, -0.5]]]
        )
        batch = {
            "input_ids": tokens,
            "response_mask": mask,
            "teacher_valid_mask": mask,
            "teacher_logprobs": teacher,
            "teacher_token_ids": ids,
            "distillation_behavior_logprobs": behavior,
        }
        for separate in (False, True):
            plan = DistillationPlan(
                enabled=True,
                separate_async=separate,
                canonical_loss_mode="k1",
                target_format="mc",
                use_policy_gradient=True,
                need_task_rewards=False,
                mc_k=3,
            )
            actor = SimpleNamespace(
                distillation_plan=plan,
                distillation_loss_config=config,
                compute_logprobs=lambda z, label: (
                    z.log_softmax(-1).gather(-1, label[:, None]).squeeze(-1)
                ),
            )
            z = torch.tensor(
                [[[0.1, 0.3, 0.5], [0.2, 0.1, 0.5], [0.6, -0.1, 0.5], [0.0, 0.0, 0.0]]],
                requires_grad=True,
            )
            output = fsdp_distillation_forward(
                actor, z, batch, packed=False, starts=[0], ends=[4], response_only=False
            )
            loss, metrics = compute_distillation_loss(
                output,
                batch,
                plan=plan,
                config=config,
                reduction=DistillationReductionContext(2),
            )
            current = z[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:])
            advantage = (teacher[:, 1:] - current).detach()
            ratio = (current - behavior[:, 1:]).exp()
            expected = -torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.28) * advantage
            )
            expected = expected[mask[:, 1:]].mean()
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(
                torch.autograd.grad(loss, z)[0], torch.autograd.grad(expected, z)[0]
            )
            self.assertEqual(metrics["distillation/token_count"].item(), 2)

    def test_separate_async_derives_behavior_and_support(self):
        cfg = {
            "runner": {"execution_mode": "separate_async"},
            "distillation": {
                "enabled": True,
                "teacher_models": {"x": {"model_path": "teacher"}},
            },
        }
        cfg["distillation"]["distillation_loss"] = loss = {"loss_mode": "k1", "mc_k": 4}
        plan = build_distillation_plan(cfg)
        self.assertTrue(plan.separate_async and plan.need_old_logprobs)
        self.assertEqual((plan.target_format, plan.mc_k), ("mc", 4))
        cfg["algorithm"] = {"recompute_logprobs": True}
        with self.assertRaisesRegex(ValueError, "behavior"):
            build_distillation_plan(cfg)
        cfg.pop("algorithm")
        loss.update(loss_mode="reverse_kl_topk", mc_k=1, topk_source="student")
        with self.assertRaisesRegex(ValueError, "Student top-k"):
            build_distillation_plan(cfg)
        cfg["runner"]["execution_mode"] = "sync"
        self.assertTrue(build_distillation_plan(cfg).need_rollout_topk)
        cfg["runner"]["async_mode"] = True
        with self.assertRaisesRegex(ValueError, "disagrees"):
            build_distillation_plan(cfg)

    def test_policy_gradient_rejects_greedy_sampling(self):
        cfg = {
            "distillation": {
                "enabled": True,
                "teacher_models": {"default": {"model_path": "teacher"}},
            },
            "algorithm": {
                "sampling_params": {
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "do_sample=True"):
            build_distillation_plan(cfg)
        cfg["algorithm"]["sampling_params"]["do_sample"] = True
        self.assertTrue(build_distillation_plan(cfg).use_policy_gradient)
        cfg["algorithm"]["sampling_params"]["do_sample"] = False
        cfg["distillation"]["distillation_loss"] = {"use_policy_gradient": False}
        self.assertFalse(build_distillation_plan(cfg).use_policy_gradient)

    def test_policy_gradient_validates_effective_sglang_sampling(self):
        import copy

        sampling = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": -1}
        cfg = {
            "distillation": {
                "enabled": True,
                "teacher_models": {"default": {"model_path": "teacher"}},
            },
            "algorithm": {"sampling_params": sampling},
            "rollout": {"rollout_backend": "sglang"},
        }
        for override in (
            {"do_sample": False},
            {"temperature": 0.7},
            {"top_p": 0.9},
            {"top_k": 10},
            {"repetition_penalty": 1.1},
            {"min_p": 0.1},
            {"presence_penalty": 0.1},
            {"logit_bias": {"3": 1}},
        ):
            with self.subTest(override=override):
                cfg["rollout"]["sampling_params"] = {**sampling, **override}
                before = copy.deepcopy(cfg)
                with self.assertRaisesRegex(ValueError, "Distillation PG"):
                    build_distillation_plan(cfg)
                self.assertEqual(cfg, before)
                cfg["rollout"]["rollout_backend"] = "vllm"
                self.assertTrue(build_distillation_plan(cfg).use_policy_gradient)
                cfg["rollout"]["rollout_backend"] = "sglang"
        cfg["rollout"]["sampling_params"] = None
        self.assertTrue(build_distillation_plan(cfg).use_policy_gradient)
        cfg["algorithm"]["sampling_params"] = {**sampling, "do_sample": False}
        with self.assertRaises(ValueError):
            build_distillation_plan(cfg)
        cfg["rollout"]["sampling_params"] = sampling
        self.assertTrue(build_distillation_plan(cfg).use_policy_gradient)

    def test_megatron_distribution_uses_model_vocabulary(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from omegaconf import OmegaConf

        topology = SimpleNamespace(
            get_data_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_rank=lambda: 0,
            get_tensor_model_parallel_group=lambda: dist.group.WORLD,
        )
        with patch.dict(
            sys.modules, {"megatron.core": SimpleNamespace(parallel_state=topology)}
        ):
            from rlinf.workers.actor.megatron_distillation import distribution_processor

        config = DistillationLossConfig(
            loss_mode="forward_kl_topk_tail", topk=2, use_policy_gradient=False
        )
        actor = SimpleNamespace(
            tokenizer=range(5),
            cfg=OmegaConf.create({"actor": {"model": {"padded_vocab_size": 8}}}),
            distillation_loss_config=config,
        )
        logits = torch.arange(16.0).reshape(1, 2, 8).requires_grad_()
        ids = torch.tensor([[[7, 6], [5, 0]]])
        teacher = torch.zeros_like(logits).log_softmax(-1).gather(-1, ids)
        mask = torch.ones(1, 2, dtype=torch.bool)
        expected = compute_distribution_loss(logits, ids, teacher, config)
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo", init_method=f"file://{directory}/init", rank=0, world_size=1
            )
            try:
                actual = distribution_processor(actor, logits, teacher, mask, ids)
                torch.testing.assert_close(
                    actual["distribution_losses"], expected.distribution_losses
                )
                torch.testing.assert_close(
                    torch.autograd.grad(
                        actual["distribution_losses"].sum(), logits, retain_graph=True
                    )[0],
                    torch.autograd.grad(expected.distribution_losses.sum(), logits)[0],
                )
            finally:
                dist.destroy_process_group()

    def test_megatron_joint_loss_preserves_token_weighting(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from omegaconf import OmegaConf

        # Only the external Megatron topology API is needed for this CPU test.
        topology = SimpleNamespace(get_data_parallel_world_size=lambda: 1)
        with patch.dict(
            sys.modules, {"megatron.core": SimpleNamespace(parallel_state=topology)}
        ):
            from rlinf.workers.actor.megatron_distillation import (
                combine_actor_distillation,
            )

        actor = SimpleNamespace(
            response_len=3,
            distillation_plan=DistillationPlan(
                enabled=True,
                canonical_loss_mode="forward_kl_topk",
                target_format="sparse",
                need_task_rewards=True,
            ),
            distillation_loss_config=DistillationLossConfig(
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                use_task_rewards=True,
                distillation_loss_coef=0.7,
            ),
            cfg=OmegaConf.create(
                {
                    "algorithm": {
                        "loss_agg_func": "token-mean",
                        "use_valid_token_scale": True,
                    }
                }
            ),
            global_valid_token=4,
            num_microbatches=2,
            _distillation_denominator=3,
        )
        current = torch.zeros(2, 3, requires_grad=True)
        distributions = torch.tensor(
            [[2.0, 0.0, 0.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        masks = torch.tensor([[True, False, False], [True, True, True]])
        teacher_masks = torch.tensor([[True, False, False], [False, True, True]])
        advantages = torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
        entropy = torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]], requires_grad=True)
        ref_kl = torch.tensor([[4.0, 0.0, 0.0], [2.0, 2.0, 2.0]], requires_grad=True)
        combined = []
        for i in range(2):
            task_loss, _ = compute_ppo_actor_loss(
                current[i : i + 1],
                torch.zeros(1, 3),
                0.2,
                0.2,
                advantages[i : i + 1],
                loss_mask=masks[i : i + 1],
            )
            output = {
                "distribution_losses": torch.nn.functional.pad(
                    distributions[i : i + 1], (0, 1)
                )
            }
            batch = {
                "response_mask": torch.nn.functional.pad(masks[i : i + 1], (1, 0)),
                "teacher_valid_mask": teacher_masks[i : i + 1],
                "teacher_logprobs": torch.zeros(1, 3, 1),
            }
            loss, _ = combine_actor_distillation(
                actor,
                output,
                batch,
                task_loss,
                dynamic=False,
                regularization_loss=-0.2 * entropy[i : i + 1][masks[i : i + 1]].mean()
                + 0.3 * ref_kl[i : i + 1][masks[i : i + 1]].mean(),
            )
            combined.append(loss)
        actual = torch.stack(combined).mean()
        expected = -(current.exp() * advantages)[masks].sum() / 4
        expected = expected + 0.7 * distributions[teacher_masks].sum() / 3
        expected = expected - 0.2 * entropy[masks].mean() + 0.3 * ref_kl[masks].mean()
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(
            actual, (current, distributions, entropy, ref_kl), retain_graph=True
        )
        expected_grad = torch.autograd.grad(
            expected, (current, distributions, entropy, ref_kl)
        )
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value)

    def test_chunked_qwen_head_matches_dense_gradients(self):
        from contextlib import nullcontext
        from copy import deepcopy
        from types import SimpleNamespace

        from torch.overrides import TorchFunctionMode
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from rlinf.workers.actor.distillation import install_distillation_head
        from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor

        class RecordProjections(TorchFunctionMode):
            def __init__(self):
                self.rows = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is torch.nn.functional.linear and args[1].shape == (71, 16):
                    self.rows.append(args[0].numel() // 16)
                return func(*args, **(kwargs or {}))

        torch.manual_seed(17)
        cfg = Qwen3Config(
            vocab_size=71,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
            attn_implementation="eager",
        )
        dense = Qwen3ForCausalLM(cfg)
        actor = object.__new__(MAFSDPActor)
        actor.model = deepcopy(dense)
        actor.distillation_head = install_distillation_head(actor.model)
        self.assertIs(
            actor.model.get_output_embeddings().weight,
            actor.model.get_input_embeddings().weight,
        )
        self.assertEqual(actor.model.state_dict().keys(), dense.state_dict().keys())
        actor.distillation_chunk_size = 2
        actor.amp_context = nullcontext()
        actor.cfg = SimpleNamespace(
            algorithm=SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.8))
        )
        actor.enable_dynamic_batch_size = True
        actor.max_tokens_per_mbs = 16
        actor.tokenizer = SimpleNamespace(eos_token_id=0)
        actor.kl_beta = 0.0
        actor.entropy_op_type = "torch"
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 0]])
        positions = torch.arange(6).expand(2, -1)
        batch = {
            "input_ids": tokens,
            "position_ids": positions,
            "attention_mask": tokens.ne(0),
            "prompt_lengths": torch.tensor([2, 1]),
            "response_lengths": torch.tensor([4, 4]),
            "response_mask": torch.tensor(
                [[0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0]], dtype=torch.bool
            ),
            "teacher_valid_mask": torch.tensor(
                [[0, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 0]], dtype=torch.bool
            ),
        }
        teacher = torch.randn(2, 6, 71).log_softmax(-1)
        packed_ids = torch.cat([tokens[0], tokens[1, :5]]).unsqueeze(0)
        packed_pos = torch.cat([positions[0], positions[1, :5]]).unsqueeze(0)
        for mode, support in [
            ("k1", "sampled"),
            ("forward_kl_topk", "teacher"),
            ("forward_kl_topk", "student"),
        ]:
            with self.subTest(mode=mode, support=support):
                actor.model.zero_grad()
                dense.zero_grad()
                dense_z = (
                    dense(
                        input_ids=packed_ids, position_ids=packed_pos, use_cache=False
                    ).logits[0]
                    / 0.8
                )
                task = support == "student"
                mask = batch["response_mask"] & batch["teacher_valid_mask"]
                rows, cols = (batch["response_mask"] if task else mask).nonzero(
                    as_tuple=True
                )
                # The second packed sequence begins at index six.
                z = dense_z[rows * 6 + cols - 1]
                logp = z.log_softmax(-1).gather(1, tokens[rows, cols, None]).squeeze(1)
                actor.distillation_plan = DistillationPlan(
                    canonical_loss_mode=mode,
                    need_task_rewards=task,
                    need_entropy=task,
                    need_current_logprobs=mode == "k1" or task,
                )
                config = DistillationLossConfig(
                    loss_mode=mode,
                    topk=64,
                    topk_source="teacher" if support == "sampled" else support,
                    use_policy_gradient=mode == "k1",
                )
                actor.distillation_loss_config = config
                if mode == "k1":
                    batch["teacher_logprobs"] = teacher.gather(-1, tokens.unsqueeze(-1))
                    reference = logp
                else:
                    ids = teacher.topk(64).indices
                    if support == "student":
                        ids[rows, cols] = z.detach().topk(64).indices
                    batch["teacher_token_ids"] = ids
                    batch["teacher_logprobs"] = teacher.gather(-1, ids)
                    output = compute_distribution_loss(
                        z,
                        ids[rows, cols],
                        batch["teacher_logprobs"][rows, cols],
                        config,
                        valid_mask=mask[rows, cols],
                    )
                    reference = output.distribution_losses
                    if task:
                        reference = (
                            reference
                            + logp
                            - (z.softmax(-1) * z.log_softmax(-1)).sum(-1)
                        )
                saved_shapes = []

                def record_saved(tensor):
                    saved_shapes.append(tensor.shape)
                    return tensor

                with (
                    RecordProjections() as calls,
                    torch.autograd.graph.saved_tensors_hooks(
                        record_saved, lambda tensor: tensor
                    ),
                ):
                    actual = actor.forward_training_batch(batch)
                    self.assertEqual(
                        calls.rows,
                        [2] * (len(rows) // 2) + ([1] if len(rows) % 2 else []),
                    )
                    values = (
                        actual.logprobs if mode == "k1" else actual.distribution_losses
                    )
                    if task:
                        values = values + actual.logprobs + actual.entropy
                    torch.testing.assert_close(values[rows, cols], reference)
                self.assertFalse(
                    any(shape and shape[-1] == cfg.vocab_size for shape in saved_shapes)
                )
                values.sum().backward()
                reference.sum().backward()
                for (name, parameter), (_, expected) in zip(
                    actor.model.named_parameters(),
                    dense.named_parameters(),
                    strict=True,
                ):
                    torch.testing.assert_close(
                        parameter.grad, expected.grad, atol=2e-5, rtol=2e-4, msg=name
                    )
        batch["response_mask"].zero_()
        actor.model.zero_grad()
        actor.forward_training_batch(batch).distribution_losses.sum().backward()
        for parameter in actor.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.count_nonzero().item(), 0)

    def test_fsdp_sampled_logprobs_keep_model_vocabulary(self):
        from rlinf.workers.actor.distillation import fsdp_distillation_forward
        from rlinf.workers.actor.fsdp_actor_worker import FSDPActor

        actor = object.__new__(FSDPActor)
        actor.distillation_plan = DistillationPlan(need_entropy=True)
        actor.distillation_loss_config = DistillationLossConfig()
        actor.tokenizer = range(5)
        actor.response_len = 2
        actor.entropy_op_type = "torch"
        for packed in (False, True):
            for response_only in (False, True):
                with self.subTest(packed=packed, response_only=response_only):
                    z = torch.zeros(1, 3, 8, requires_grad=True)
                    batch = {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "response_mask": torch.tensor([[False, True, True]]),
                        "teacher_valid_mask": torch.ones(
                            1, 2 if response_only else 3, dtype=torch.bool
                        ),
                    }
                    out = fsdp_distillation_forward(
                        actor,
                        z,
                        batch,
                        packed=packed,
                        starts=[0],
                        ends=[3],
                        response_only=response_only,
                    )
                    expected = actor.compute_logprobs(
                        z[:, :2], batch["input_ids"][:, 1:]
                    )
                    actual = out.logprobs[:, -2:]
                    torch.testing.assert_close(
                        (actual - expected).exp(), torch.ones_like(actual)
                    )
                    torch.testing.assert_close(
                        out.entropy[:, -2:],
                        torch.full_like(actual, torch.tensor(8.0).log()),
                    )
                    torch.testing.assert_close(
                        torch.autograd.grad(actual.sum(), z, retain_graph=True)[0],
                        torch.autograd.grad(expected.sum(), z)[0],
                    )

    def test_fsdp_distribution_uses_complete_model_vocabulary(self):
        from rlinf.workers.actor.distillation import fsdp_distillation_forward
        from rlinf.workers.actor.fsdp_actor_worker import FSDPActor

        actor = object.__new__(FSDPActor)
        actor.distillation_plan = DistillationPlan(
            canonical_loss_mode="forward_kl_topk", target_format="sparse"
        )
        actor.distillation_loss_config = DistillationLossConfig(
            loss_mode="forward_kl_topk", topk=2, use_policy_gradient=False
        )
        actor.tokenizer = range(5)
        actor.response_len = 1
        actor.entropy_op_type = "torch"
        z = torch.zeros(1, 2, 8)
        z[..., 5:] = 10
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "response_mask": torch.tensor([[False, True]]),
            "teacher_valid_mask": torch.tensor([[True]]),
            "teacher_token_ids": torch.tensor([[[7, 6]]]),
            "teacher_logprobs": z[:, :1].log_softmax(-1)[..., [7, 6]],
        }
        kwargs = {"packed": False, "starts": [0], "ends": [2], "response_only": True}
        out = fsdp_distillation_forward(actor, z, batch, **kwargs)
        torch.testing.assert_close(out.distribution_losses, torch.zeros(1, 1))
        expected_mass = z[:, :1].softmax(-1)[..., [7, 6]].sum(-1)
        torch.testing.assert_close(out.student_mass, expected_mass)
        torch.testing.assert_close(out.teacher_mass, expected_mass)
        torch.testing.assert_close(
            out.logprobs, actor.compute_logprobs(z[:, :1], batch["input_ids"][:, 1:])
        )

    def test_config_aliases_and_invalid_combinations(self):
        for alias, canonical in [("kl", "k1"), ("mse", "k2"), ("low_var_kl", "k3")]:
            self.assertEqual(
                DistillationLossConfig(loss_mode=alias).canonical_loss_mode, canonical
            )
        for mode in ("kl", "k1"):
            with self.assertRaises(ValueError):
                DistillationLossConfig(loss_mode=mode, use_policy_gradient=False)
        for k in (0, -2, True):
            with self.assertRaises(ValueError):
                DistillationLossConfig(loss_mode="forward_kl_topk", topk=k)
        for mode in ("reverse_kl_topk", "reverse_kl_topk_tail"):
            with self.assertRaises(ValueError):
                DistillationLossConfig(loss_mode=mode, topk_source="teacher")
            DistillationLossConfig(loss_mode=mode, topk=-1)
        with self.assertRaises(ValueError):
            DistillationLossConfig(
                loss_mode="reverse_kl_topk_tail",
                topk_source="student",
                log_prob_min_clamp=-10,
            )

    def test_plan_dependencies_and_no_config_mutation(self):
        import copy

        cfg = {
            "distillation": {
                "enabled": True,
                "teacher_models": {"default": {"model_path": "teacher"}},
                "distillation_loss": {
                    "loss_mode": "reverse_kl_topk_tail",
                    "topk_source": "student",
                    "use_policy_gradient": False,
                },
            }
        }
        before = copy.deepcopy(cfg)
        plan = build_distillation_plan(cfg)
        self.assertFalse(
            plan.need_old_logprobs
            or plan.need_task_advantages
            or plan.need_current_logprobs
        )
        self.assertTrue(plan.need_rollout_topk)
        self.assertEqual(cfg, before)

    def test_sampled_estimators_and_gradients(self):
        for mode in ("k1", "k2", "abs", "k3", "kl", "mse", "low_var_kl"):
            with self.subTest(mode=mode):
                s = torch.tensor([-0.2, -2.0, -25.0], requires_grad=True)
                t = torch.tensor([-1.0, -1.0, -0.1], requires_grad=True)
                config = DistillationLossConfig(loss_mode=mode)
                result = compute_distillation_tokens(s, t, config=config)
                d = s - t.detach()
                canonical = config.canonical_loss_mode
                expected = {
                    "k1": lambda: d,
                    "k2": lambda: d.square() / 2,
                    "abs": lambda: d.abs(),
                    "k3": lambda: (
                        (-d).clamp(-20, 20).exp() - (-d).clamp(-20, 20) - 1
                    ).clamp(-10, 10),
                }[canonical]()
                torch.testing.assert_close(result, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(result.sum(), s)[0],
                    torch.autograd.grad(expected.sum(), s)[0],
                )
                self.assertIsNone(t.grad)

    def test_distribution_values_and_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for mode in MODES:
                for k in (3, 7, -1):
                    with self.subTest(device=device, mode=mode, k=k):
                        torch.manual_seed(7)
                        z = torch.randn(2, 4, 7, device=device, requires_grad=True)
                        teacher = torch.randn_like(z, requires_grad=True)
                        ids = (
                            z.detach().argsort(-1, descending=True)[..., :k]
                            if k > 0
                            else None
                        )
                        scores = teacher.log_softmax(-1)
                        if ids is not None:
                            scores = scores.gather(-1, ids)
                        config = DistillationLossConfig(
                            loss_mode=mode,
                            topk=k,
                            topk_source="student",
                            use_policy_gradient=False,
                        )
                        actual = compute_distribution_loss(
                            z, ids, scores, config
                        ).distribution_losses
                        expected = reference(z, teacher, ids, mode)
                        torch.testing.assert_close(
                            actual, expected, atol=2e-6, rtol=2e-5
                        )
                        actual_grad, teacher_grad = torch.autograd.grad(
                            actual.sum(),
                            (z, teacher),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        self.assertIsNone(teacher_grad)
                        expected_grad = torch.autograd.grad(expected.sum(), z)[0]
                        torch.testing.assert_close(
                            actual_grad, expected_grad, atol=2e-6, rtol=2e-5
                        )
                        self.assertGreater(actual_grad.abs().sum().item(), 0)
                        self.assertIsNone(teacher.grad)

    def test_masked_nan_and_zero_microbatch(self):
        for mode in MODES:
            z = torch.full((2, 3, 7), float("nan"), requires_grad=True)
            ids = torch.full((2, 3, 3), -99, dtype=torch.long)
            teacher = torch.full((2, 3, 3), float("nan"))
            config = DistillationLossConfig(
                loss_mode=mode, topk=3, topk_source="student", use_policy_gradient=False
            )
            loss = compute_distribution_loss(
                z, ids, teacher, config, valid_mask=torch.zeros(2, 3, dtype=torch.bool)
            ).distribution_losses.sum()
            loss.backward()
            self.assertEqual(loss.item(), 0)
            torch.testing.assert_close(z.grad, torch.zeros_like(z))

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            compute_distribution_loss(
                torch.zeros(1, 5),
                torch.tensor([[1, 1]]),
                torch.full((1, 2), -2.0),
                DistillationLossConfig(loss_mode="forward_kl_topk", topk=2),
            )

    def test_reverse_tail_teacher_zero_mass(self):
        z = torch.tensor([[0.3, -0.7, -1.2]], requires_grad=True)
        teacher = torch.tensor([[0.0]])
        config = DistillationLossConfig(
            loss_mode="reverse_kl_topk_tail",
            topk=1,
            topk_source="student",
            use_policy_gradient=False,
        )
        output = compute_distribution_loss(z, torch.tensor([[0]]), teacher, config)
        self.assertGreater(output.tail_loss.item(), 0)
        output.distribution_losses.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_pg_clipping_order_and_old_policy_detached(self):
        s = torch.tensor([[0.0, 2.0, -2.0, 30.0]], requires_grad=True)
        old = torch.zeros_like(s, requires_grad=True)
        d = torch.tensor([[1.0, -2.0, 2.0, 0.5]], requires_grad=True)
        weights = torch.tensor([[0.1, 2.0, 3.0, 0.4]])
        cfg = DistillationLossConfig(loss_mode="k1")
        actual, _ = compute_distillation_pg_tokens(
            s, old, d, config=cfg, rollout_is_weights=weights
        )
        ratio = s.clamp(-20, 20).exp()
        advantage = -d.detach()
        base = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.28))
        expected = (
            torch.where(advantage < 0, torch.minimum(base, -advantage * 3), base)
            * weights
        )
        torch.testing.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected.sum(), s, retain_graph=True)[0]
        actual.sum().backward()
        torch.testing.assert_close(s.grad, expected_grad)
        self.assertIsNone(old.grad)
        self.assertIsNone(d.grad)

    def test_original_ppo_reduction_uses_same_token_contract(self):
        s = torch.tensor([[0.0, 0.8, -1.0]], requires_grad=True)
        old = torch.zeros_like(s)
        advantage = torch.tensor([[1.0, -2.0, 0.3]])
        mask = torch.tensor([[True, True, False]])
        tokens, _ = compute_ppo_actor_loss_tokens(
            s,
            old,
            advantage,
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3,
        )
        loss, _ = compute_ppo_actor_loss(
            s, old, 0.2, 0.2, advantage, loss_mask=mask, clip_ratio_c=3
        )
        torch.testing.assert_close(loss, tokens[mask].mean())

    def test_global_reduction_invariant_to_microbatch_and_dp_partition(self):
        mask = torch.tensor(
            [[True, True, False], [True, False, False], [True, True, True]]
        )
        x = torch.arange(9.0).reshape(3, 3).requires_grad_()
        expected = x.square()[mask].mean()
        context = DistillationReductionContext(
            mask.sum().item(), dp_size=2, gradient_divisor=3
        )
        actual = (
            sum(
                context.reduce(x[i : i + 1].square(), mask[i : i + 1]) / 3
                for i in range(3)
            )
            / 2
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0]
        )
        self.assertEqual(
            combine_distillation_and_task_loss(
                None, x.sum(), use_task_rewards=False, coefficient=10
            ).item(),
            x.sum().item(),
        )

    def test_sequence_reductions_and_empty_update(self):
        mask = torch.tensor(
            [[True, True, False], [True, False, False], [False, False, False]]
        )
        x = torch.arange(9.0).reshape(3, 3).requires_grad_()
        for mode in ("seq-mean-token-mean", "seq-mean-token-sum"):
            context = DistillationReductionContext(2, mode=mode)
            actual = sum(
                context.reduce(x[i : i + 1], mask[i : i + 1]) for i in range(3)
            )
            expected = (
                (x[0, :2].mean() + x[1, 0]) / 2
                if mode.endswith("token-mean")
                else (x[0, :2].sum() + x[1, 0]) / 2
            )
            torch.testing.assert_close(actual, expected)
        zero = DistillationReductionContext(0).reduce(
            x.square(), torch.zeros_like(mask)
        )
        zero.backward()
        self.assertEqual(zero.item(), 0)
        torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_distillation_metrics_support_microbatch_aggregation(self):
        response_mask = torch.tensor([[True, True, False], [True, True, True]])
        teacher_valid_mask = torch.tensor([[True, False, True], [True, True, False]])
        for mode in ("k1", "forward_kl_topk"):
            with self.subTest(mode=mode):
                config = DistillationLossConfig(
                    loss_mode=mode, use_policy_gradient=mode == "k1"
                )
                plan = build_distillation_plan(
                    {
                        "distillation": {
                            "enabled": True,
                            "teacher_models": {"default": {"model_path": "teacher"}},
                            "distillation_loss": {
                                "loss_mode": mode,
                                "use_policy_gradient": config.use_policy_gradient,
                            },
                        }
                    }
                )
                microbatch_metrics = []
                for i in range(2):
                    _, metrics = compute_distillation_loss(
                        DistillationForwardOutput(
                            logprobs=torch.full((1, 3), -1.0, requires_grad=True),
                            distribution_losses=torch.ones(1, 3, requires_grad=True),
                        ),
                        {
                            "response_mask": response_mask[i : i + 1],
                            "teacher_valid_mask": teacher_valid_mask[i : i + 1],
                            "teacher_logprobs": torch.full((1, 3, 1), -1.2),
                            "rollout_logprobs": torch.full((1, 3), -1.0),
                        },
                        plan=plan,
                        config=config,
                        reduction=DistillationReductionContext(3),
                    )
                    microbatch_metrics.append(metrics)
                means = {
                    key: torch.stack([m[key] for m in microbatch_metrics]).mean()
                    for key in microbatch_metrics[0]
                }
                self.assertEqual(means["distillation/token_count"].item(), 1.5)
                self.assertTrue(all(torch.isfinite(value) for value in means.values()))

    def test_pure_distillation_metrics_without_task_fields(self):
        from rlinf.utils.distributed import (
            compute_rollout_metrics,
            compute_rollout_metrics_dynamic,
        )

        batch = {
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "prompt_lengths": torch.tensor([1, 2]),
            "response_lengths": torch.tensor([3, 2]),
        }
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo", init_method=f"file://{directory}/init", rank=0, world_size=1
            )
            try:
                for function in (
                    compute_rollout_metrics,
                    compute_rollout_metrics_dynamic,
                ):
                    metrics, prompts, responses = function(batch, 2, 3)
                    self.assertEqual(metrics["prompt_length/mean"].item(), 1.5)
                    self.assertEqual(metrics["response_length/mean"].item(), 2.5)
                    self.assertNotIn("rewards/mean", metrics)
                    self.assertEqual(prompts.tolist(), [1, 2])
                    self.assertEqual(responses.tolist(), [3, 2])
            finally:
                dist.destroy_process_group()

    def test_two_rank_vocab_parallel_values_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_tp_process, args=(directory + "/init",), nprocs=2, join=True)

    def test_two_rank_filtered_batches_skip_before_target_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _empty_rank_process, args=(directory + "/init",), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
