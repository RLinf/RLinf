"""CPU preflight: strict-load the checkpoint into the resolved deployment model."""
import inspect
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def main():
    rlinf_root = Path(os.environ.get("RLINF_ROOT", Path(__file__).resolve().parents[2]))
    with initialize_config_dir(
        config_dir=str(rlinf_root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="tianji2_rlt_stage2_infer", overrides=sys.argv[2:])
    m = OmegaConf.to_container(cfg.actor.model, resolve=True)
    assert m["action_dim"] == 16 and cfg.env.eval.override_cfg.action_space == "joint"
    assert cfg.rollout.joint_actor_eval and cfg.rollout.default_actor_switch
    assert cfg.rollout.rlt_feature_model.openpi.config_name == "tianji_stage1_joint"
    params = inspect.signature(RLTMLPPolicy).parameters
    torch.set_num_threads(1)
    model = RLTMLPPolicy(**{k: v for k, v in m.items() if k in params})
    model.load_state_dict(
        torch.load(sys.argv[1], map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    obs = {
        "z_rl": torch.zeros(4, m["z_dim"]),
        "proprio": torch.zeros(4, m["proprio_dim"]),
        "ref_chunk": torch.zeros(4, m["ref_num_action_chunks"], 16),
    }
    obs["proprio"][:, -m["stage_onehot_dim"]:][:, :4] = torch.eye(4)
    if m.get("tactile_enabled", False):
        from rlinf.utils.xense_tactile import empty_tactile

        obs.update(
            {
                key: torch.from_numpy(value)
                .unsqueeze(0)
                .repeat((4,) + (1,) * value.ndim)
                for key, value in empty_tactile().items()
            }
        )
    with torch.no_grad():
        actions, _ = model.predict_action_batch(
            env_obs=obs, mode="eval", return_obs=True
        )
    actions = torch.as_tensor(actions)
    assert actions.shape == (4, m["num_action_chunks"], 16) and torch.isfinite(actions).all()
    tactile = "true" if m.get("tactile_enabled", False) else "false"
    print(
        "STAGE2_JOINT_INFER_CHECKPOINT_OK",
        sys.argv[1],
        "tactile=" + tactile,
        flush=True,
    )


if __name__ == "__main__":
    main()
