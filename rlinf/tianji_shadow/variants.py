"""The r5b progress-only checkpoint, paired with the published 1116v1 H."""
import copy
from pathlib import Path

from .settings import BASE_VERSION, FT_VERSION, defaults

PROGRESS_SHA256 = "8f5622b0eba5d5eabcd9a405b849104a8a9a3881ce9da73a0f9f7f481380d9cd"


def progress_variant(predictor, assets):
    import torch

    from .inference import sha

    if predictor.identity.get("version") != BASE_VERSION:
        raise ValueError("The progress variant requires the published 1116v1 parent")
    path = Path(assets).parent / "shadow_sub16ft_h16v1" / "hover.pt"
    if sha(path) != PROGRESS_SHA256:
        raise ValueError("Sub-16ft_H-16v1 weight checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    original = predictor.hover.state_dict()
    if state.keys() != original.keys() or any(
        state[k].dtype != original[k].dtype or state[k].shape != original[k].shape
        or not torch.isfinite(state[k]).all()
        for k in original
    ):
        raise ValueError("Sub-16ft_H-16v1 weight schema mismatch")
    if any(not torch.equal(original[k], state[k]) for k in original if not k.startswith("progress.")):
        raise ValueError("Sub-16ft_H-16v1 must retain 1116v1 adapter and H head")
    if all(torch.equal(original[k], state[k]) for k in original if k.startswith("progress.")):
        raise ValueError("Sub-16ft_H-16v1 progress head did not change")
    variant = copy.copy(predictor)
    variant.hover = copy.deepcopy(predictor.hover)
    variant.hover.load_state_dict(state, strict=True)
    variant.hover.eval()
    variant.identity = dict(predictor.identity, version=FT_VERSION, profiles=defaults(FT_VERSION),
                            parent_version=BASE_VERSION, risk_version=BASE_VERSION,
                            progress_checkpoint="progress_140.pt", progress_step=140,
                            progress=defaults(FT_VERSION)["two"]["thresholds"],
                            assets=dict(predictor.identity["assets"]) | {"hover.pt": PROGRESS_SHA256})
    variant.identity["parent_stage_rules_sha256"] = variant.identity.pop("stage_rules_sha256")
    return variant
