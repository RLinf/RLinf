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

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from rlinf.config import validate_fsdp_cfg
from rlinf.hybrid_engines.fsdp.mixed_precision import (
    build_mixed_precision,
    get_mixed_precision_wrap_policy,
)


@pytest.mark.parametrize("model_type", ["qwen2.5", "openpi"])
@pytest.mark.parametrize("root_cast", [True, False])
def test_strategy_passes_wrapper_overrides_and_root_cast_options(
    monkeypatch, root_cast, model_type
):
    from rlinf.hybrid_engines.fsdp.strategy import fsdp as strategy_module

    fsdp_cfg = config(
        mixed_precision={
            "param_dtype": "bf16",
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
            "cast_root_forward_inputs": root_cast,
        },
        wrap_policy={"module_classes_to_wrap": ["Linear"]},
        sharding_strategy="no_shard",
        use_orig_params=True,
    )
    actor = validate_fsdp_cfg(
        OmegaConf.create(
            {
                "fsdp_config": fsdp_cfg,
                "model": {"model_type": model_type, "is_lora": False},
            }
        )
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(strategy_module, "FSDP", lambda **kwargs: kwargs)
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    kwargs = strategy_module.FSDPStrategy(actor, 1).wrap_model(model, object())
    assert kwargs["mixed_precision"].cast_root_forward_inputs is root_cast
    assert kwargs["mixed_precision"].cast_forward_inputs
    plan = kwargs["auto_wrap_policy"]._run_policy(model, set(), kwargs)
    assert plan[model[0]]["mixed_precision"].param_dtype == torch.bfloat16
    assert plan[model[1]]["mixed_precision"].param_dtype == torch.float32


def config(selector="module_names", targets=None, wrap_mode="individual", **kwargs):
    return OmegaConf.create(
        {
            "mixed_precision_policies": {
                "sensitive": {
                    "param_dtype": "fp32",
                    "reduce_dtype": "fp32",
                    "buffer_dtype": "fp32",
                    "cast_forward_inputs": True,
                }
            },
            "wrappers": {
                "sensitive": {selector: targets or ["1"], "wrap_mode": wrap_mode}
            },
            "mixed_precision_rules": {"sensitive": "sensitive"},
            **kwargs,
        }
    )


@pytest.mark.parametrize(
    ("selector", "targets"),
    [
        ("module_names", ["1"]),
        ("module_classes", ["torch.nn.modules.linear.Linear"]),
        ("wrap_names", ["sensitive"]),
    ],
)
def test_selectors_create_wrapper_overrides(selector, targets):
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 2))
    model[1]._fsdp_wrap_name = "sensitive"
    default = MixedPrecision(param_dtype=torch.bfloat16)
    policy = get_mixed_precision_wrap_policy(
        model, config(selector, targets), None, default
    )
    resolved = policy._run_policy(model, set(), {"mixed_precision": default})
    assert set(resolved) == {model[1]}
    assert resolved[model[1]]["mixed_precision"].param_dtype == torch.float32
    assert resolved[model[1]]["mixed_precision"].cast_forward_inputs


def test_nested_wrappers_do_not_inherit_parent_override():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    default = MixedPrecision(param_dtype=torch.bfloat16)

    def base_policy(module, recurse, nonwrapped_numel):
        return lambda_auto_wrap_policy(
            module, recurse, nonwrapped_numel, lambda m: isinstance(m, nn.Linear)
        )

    policy = get_mixed_precision_wrap_policy(
        model, config(targets=["0"]), base_policy, default
    )
    resolved = policy._run_policy(model, set(), {"mixed_precision": default})
    assert resolved[model[0]]["mixed_precision"].param_dtype == torch.float32
    assert resolved[model[0][0]]["mixed_precision"] is default


def test_no_rules_preserves_original_policy():
    sentinel = object()
    assert (
        get_mixed_precision_wrap_policy(
            nn.Linear(2, 2), OmegaConf.create({}), sentinel, MixedPrecision()
        )
        is sentinel
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"wrappers": {"sensitive": {"module_names": ["absent"]}}},
            "matched no",
        ),
        (
            {"mixed_precision_rules": {"sensitive": "typo"}},
            "unknown policy",
        ),
        ({"strategy": "fsdp2"}, "FSDP1"),
        ({"disable": True}, "disable=true"),
        ({"amp_autocast": {"enabled": True}}, "amp_autocast"),
    ],
)
def test_invalid_rules_fail_before_wrapping(changes, message):
    with pytest.raises(ValueError, match=message):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.ReLU(), nn.Linear(2, 2)),
            config(**changes),
            None,
            MixedPrecision(),
        )


def test_overlapping_rules_are_rejected():
    cfg = config()
    cfg.wrappers.overlap = {"module_names": ["*"]}
    with pytest.raises(ValueError, match="Multiple"):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.ReLU(), nn.Linear(2, 2)), cfg, None, MixedPrecision()
        )


def test_batchnorm_override_cannot_silently_lose_to_pytorch():
    with pytest.raises(ValueError, match="BatchNorm"):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.ReLU(), nn.BatchNorm1d(2)),
            config(),
            None,
            MixedPrecision(),
        )


def test_tied_parameters_cannot_cross_wrapper_boundaries():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[1].weight = model[0].weight
    with pytest.raises(ValueError, match="Shared parameter"):
        get_mixed_precision_wrap_policy(model, config(), None, MixedPrecision())


def test_explicit_null_keeps_original_dtype_and_torch_defaults():
    mp = build_mixed_precision(OmegaConf.create({"param_dtype": None}))
    assert mp.param_dtype is None
    assert mp.cast_forward_inputs == MixedPrecision().cast_forward_inputs


def test_unknown_runtime_options_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        build_mixed_precision(OmegaConf.create({"output_dtype": "fp32"}))


def test_yaml_validation_rejects_fsdp2_overrides():
    cfg = config(strategy="fsdp2", mixed_precision={})
    with pytest.raises(ValueError, match="FSDP1"):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"mixed_precision_policies": {"sensitive": {"param_dtype": "fp32"}}},
            "missing dtype",
        ),
        (
            {"wrappers": {"sensitive": {"module_names": "1"}}},
            "string list",
        ),
        (
            {
                "wrappers": {
                    "sensitive": {"module_names": ["1"], "wrap_names": ["sensitive"]}
                }
            },
            "exactly one",
        ),
        (
            {
                "mixed_precision_policies": {
                    "sensitive": {
                        "param_dtype": "fp32",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                        "output_dtype": "fp32",
                    }
                }
            },
            "Unsupported",
        ),
    ],
)
def test_invalid_policy_schema_fails_during_yaml_validation(changes, message):
    cfg = config(mixed_precision={}, **changes)
    with pytest.raises(ValueError, match=message):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


def test_actual_fsdp_wrappers_forward_backward_and_buffers(tmp_path):
    """Exercise native FSDP1 hooks rather than only inspecting policy kwargs."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if dist.is_initialized():
        pytest.skip("Test requires its own single-rank process group")
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )

    class LinearWithBuffer(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.register_buffer("offset", torch.ones(2))
            self.seen_dtype = None

        def forward(self, x):
            self.seen_dtype = (x.dtype, self.weight.dtype, self.offset.dtype)
            return super().forward(x) + self.offset

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(LinearWithBuffer(), LinearWithBuffer()).to(device)
        default = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
        cfg = config()

        def base(module, recurse, nonwrapped_numel):
            return recurse or isinstance(module, LinearWithBuffer)

        policy = get_mixed_precision_wrap_policy(model, cfg, base, default)
        wrapped = FSDP(
            model, auto_wrap_policy=policy, mixed_precision=default, device_id=device
        )
        loss = wrapped(torch.randn(2, 2, device=device)).sum()
        loss.backward()
        assert wrapped[0].module.seen_dtype == (torch.bfloat16,) * 3
        assert wrapped[1].module.seen_dtype == (torch.float32,) * 3
        assert all(p.grad.dtype == torch.float32 for p in wrapped.parameters())
    finally:
        dist.destroy_process_group()


def linear_policy(module, recurse, nonwrapped_numel):
    return recurse or isinstance(module, nn.Linear)


@pytest.mark.parametrize("targets", [["0.0", "0.1"], ["0.*"], ["0"]])
def test_group_creates_one_boundary_and_suppresses_inner_wrappers(targets):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), nn.Linear(2, 2)
    )
    default = MixedPrecision(param_dtype=torch.bfloat16)
    policy = get_mixed_precision_wrap_policy(
        model, config(targets=targets, wrap_mode="combined"), linear_policy, default
    )
    plan = policy._run_policy(model, set(), {"mixed_precision": default})
    assert set(plan) == {model[0], model[1]}
    assert plan[model[0]]["mixed_precision"].param_dtype == torch.float32
    assert plan[model[1]]["mixed_precision"] is default


def test_selector_without_group_keeps_separate_wrappers():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)))
    policy = get_mixed_precision_wrap_policy(
        model, config(targets=["0.*"]), None, MixedPrecision()
    )
    assert set(policy._run_policy(model, set(), {})) == {model[0][0], model[0][1]}


@pytest.mark.parametrize(
    "kind, message",
    [
        ("coverage", "unselected states"),
        ("dtype", "uniform original"),
        ("grad", "requires_grad"),
        ("nested", "inner wrappers"),
        ("root", "model root"),
    ],
)
def test_invalid_groups_fail_before_wrapping(kind, message):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)),
        nn.Linear(2, 2),
    )
    cfg = config(targets=["0.*"], wrap_mode="combined")
    if kind == "coverage":
        cfg.wrappers.sensitive.module_names = ["0.0", "0.1"]
    elif kind == "dtype":
        model[0][1].bfloat16()
    elif kind == "grad":
        model[0][1].requires_grad_(False)
    elif kind == "nested":
        cfg.wrappers.inner = {"module_names": ["0.0"]}
    else:
        cfg.wrappers.sensitive.module_names = ["0", "1"]
    with pytest.raises(ValueError, match=message):
        get_mixed_precision_wrap_policy(model, cfg, linear_policy, MixedPrecision())


def test_group_allows_mixed_requires_grad_with_orig_params():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)))
    model[0][0].requires_grad_(False)
    policy = get_mixed_precision_wrap_policy(
        model,
        config(targets=["0.*"], wrap_mode="combined", use_orig_params=True),
        linear_policy,
        MixedPrecision(),
    )
    assert set(policy._run_policy(model, set(), {})) == {model[0]}


def test_unbound_wrapper_uses_default_precision():
    model = nn.Sequential(nn.Linear(2, 2))
    default = MixedPrecision(param_dtype=torch.bfloat16)
    cfg = config(targets=["0"], mixed_precision_rules={})
    policy = get_mixed_precision_wrap_policy(model, cfg, None, default)
    assert (
        policy._run_policy(model, set(), {"mixed_precision": default})[model[0]][
            "mixed_precision"
        ]
        is default
    )


def test_every_selector_must_match():
    with pytest.raises(ValueError, match="matched no"):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.Linear(2, 2)),
            config(targets=["0", "typo.*"]),
            None,
            MixedPrecision(),
        )


@pytest.mark.parametrize("wrap_mode", ["combined", "subtree"])
def test_actual_group_owns_multiple_modules(tmp_path, wrap_mode):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if dist.is_initialized():
        pytest.skip("Requires its own process group")
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group-init'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
            nn.Linear(2, 2),
        ).to(device)
        seen = []
        for child in list(model[0]) + list(model[1]):
            child.register_forward_pre_hook(
                lambda module, args: seen.append((args[0].dtype, module.weight.dtype))
            )
        default = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
        policy = get_mixed_precision_wrap_policy(
            model,
            config(
                targets=["0.*"] if wrap_mode == "combined" else ["[01]"],
                wrap_mode=wrap_mode,
            ),
            linear_policy,
            default,
        )
        wrapped = FSDP(
            model, auto_wrap_policy=policy, mixed_precision=default, device_id=device
        )
        assert isinstance(wrapped[0], FSDP)
        assert all(isinstance(child, nn.Linear) for child in wrapped[0].module)
        assert wrapped[0]._flat_param.numel() == 12
        if wrap_mode == "subtree":
            assert isinstance(wrapped[1], FSDP)
            assert all(isinstance(child, nn.Linear) for child in wrapped[1].module)
            assert wrapped[1]._flat_param.numel() == 12
        output = wrapped(torch.randn(2, 2, device=device))
        output.float().square().sum().backward()
        second_dtype = torch.float32 if wrap_mode == "subtree" else torch.bfloat16
        assert (
            seen
            == [(torch.float32, torch.float32)] * 2 + [(second_dtype, second_dtype)] * 2
        )
        assert output.dtype == torch.bfloat16
        assert all(p.grad is not None for p in wrapped.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("container", [nn.ModuleList, nn.ModuleDict])
def test_group_requires_executable_parent(container):
    children = [nn.Linear(2, 2), nn.Linear(2, 2)]
    block = container(
        children if container is nn.ModuleList else dict(zip(["a", "b"], children))
    )
    model = nn.Sequential(block)
    with pytest.raises(ValueError, match="forward execution boundary"):
        get_mixed_precision_wrap_policy(
            model, config(targets=["0.*"], wrap_mode="combined"), None, MixedPrecision()
        )


def test_unknown_wrapper_binding_fails_validation():
    with pytest.raises(ValueError, match="Unknown wrapper"):
        validate_fsdp_cfg(
            OmegaConf.create(
                {
                    "fsdp_config": config(
                        mixed_precision={}, mixed_precision_rules={"typo": "sensitive"}
                    )
                }
            )
        )


@pytest.mark.parametrize("select_parent", [False, True])
def test_pruned_policy_cannot_hide_explicit_child_wrapper(select_parent):
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    cfg = config(targets=["0.0"])
    if select_parent:
        cfg.wrappers.parent = {"module_names": ["0"]}

    def base(module, recurse, nonwrapped_numel):
        return module is model

    with pytest.raises(ValueError, match="non-recursing policy"):
        get_mixed_precision_wrap_policy(model, cfg, base, MixedPrecision())


@pytest.mark.parametrize(
    "field", ["wrappers", "mixed_precision_policies", "mixed_precision_rules"]
)
def test_empty_invalid_config_mapping_is_rejected(field):
    cfg = config(mixed_precision={}, **{field: []})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


@pytest.mark.parametrize("wrap_mode", ["individual", "subtree"])
@pytest.mark.parametrize("wrap_child", [False, True])
def test_wrapper_dtype_validation_excludes_separately_wrapped_children(
    wrap_child, wrap_mode
):
    class MixedDtypeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.ones(2))
            self.child = nn.Linear(2, 2).bfloat16()

        def forward(self, x):
            return self.child(x) + self.offset

    model = nn.Sequential(MixedDtypeModule())
    cfg = config(targets=["0"], wrap_mode=wrap_mode)
    base = linear_policy if wrap_child else None
    if not wrap_child or wrap_mode == "subtree":
        with pytest.raises(ValueError, match="uniform original parameter dtype"):
            get_mixed_precision_wrap_policy(model, cfg, base, MixedPrecision())
    else:
        policy = get_mixed_precision_wrap_policy(model, cfg, base, MixedPrecision())
        plan = policy._run_policy(model, set(), {})
        assert set(plan) == {model[0], model[0].child}


@pytest.mark.parametrize(
    "selector, targets",
    [
        ("module_names", ["[01]"]),
        ("module_classes", ["torch.nn.modules.container.Sequential"]),
        ("wrap_names", ["mlp"]),
    ],
)
def test_subtree_creates_one_complete_wrapper_per_match(selector, targets):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
        nn.Linear(2, 2),
    )
    model[0]._fsdp_wrap_name = model[1]._fsdp_wrap_name = "mlp"
    default = MixedPrecision(param_dtype=torch.bfloat16)
    cfg = config(selector, targets, wrap_mode="subtree")
    policy = get_mixed_precision_wrap_policy(model, cfg, linear_policy, default)
    plan = policy._run_policy(model, set(), {"mixed_precision": default})
    assert set(plan) == {model[0], model[1], model[2]}
    assert plan[model[0]]["mixed_precision"].param_dtype == torch.float32
    assert plan[model[1]]["mixed_precision"].param_dtype == torch.float32
    assert plan[model[2]]["mixed_precision"] is default


@pytest.mark.parametrize("mode", [True, False, None, "each"])
def test_invalid_wrap_modes_fail_config_validation(mode):
    with pytest.raises(ValueError, match="wrap_mode must be"):
        validate_fsdp_cfg(
            OmegaConf.create(
                {"fsdp_config": config(wrap_mode=mode, mixed_precision={})}
            )
        )


def test_old_group_option_is_rejected():
    cfg = config(mixed_precision={})
    cfg.wrappers.sensitive.group = True
    with pytest.raises(ValueError, match="unsupported options.*group"):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


def test_subtree_cannot_select_nested_boundaries():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    with pytest.raises(ValueError, match="inner wrappers"):
        get_mixed_precision_wrap_policy(
            model,
            config(targets=["0", "0.0"], wrap_mode="subtree"),
            linear_policy,
            MixedPrecision(),
        )


@pytest.mark.parametrize("root_keep,child_keep", [(True, False), (False, True)])
def test_adamw_rejects_incompatible_root_or_child_policy(
    tmp_path, root_keep, child_keep
):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from rlinf.hybrid_engines.fsdp.optim import validate_adamw_mixed_precision

    if dist.is_initialized():
        pytest.skip("Test requires its own single-rank process group")
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(device)

        def policy(keep):
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16 if keep else torch.float32,
                keep_low_precision_grads=keep,
                cast_forward_inputs=True,
            )

        model[1] = FSDP(
            model[1],
            device_id=device,
            use_orig_params=True,
            mixed_precision=policy(child_keep),
        )
        wrapped = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            mixed_precision=policy(root_keep),
        )
        with pytest.raises(ValueError, match="actor.optim.use_fp32_master_params"):
            validate_adamw_mixed_precision(wrapped, use_fp32_master_params=False)
    finally:
        dist.destroy_process_group()
