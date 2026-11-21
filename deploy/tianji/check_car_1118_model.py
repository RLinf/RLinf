"""Full car policy smoke test. It never connects to LoopInfra."""

import json
import os
from pathlib import Path
import time

import torch
from hydra import compose, initialize_config_dir

from car_1118_assets import check_layout
from rlinf.envs.realworld.tianji.tianji_contract import contract_from_env, identity_state
from rlinf.models.embodiment.openpi_rlinf import get_model
from rlinf.workers.rollout.hf.tianji_vla_eval_worker import joint_reference_actions


def main():
    prompts = check_layout(Path(os.environ["TIANJI_RLT_STAGE1_MODEL_PATH"]))
    config_dir = str(Path(os.environ["EMBODIED_PATH"]) / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        config = compose(config_name="tianji2_joint_stage1")
    contract = contract_from_env()
    assert contract.control_hz == 30
    assert torch.cuda.is_available()
    model = (
        get_model(config.rollout.rlt_feature_model)
        .eval()
        .requires_grad_(False)
        .to("cuda:0")
    )
    obs = {
        "states": torch.as_tensor(identity_state()[None]),
        "joint_states": torch.zeros((1, 16)),
        "rlt_qvel": torch.zeros((1, 14)),
        "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        "extra_view_images": torch.zeros((1, 2, 224, 224, 3), dtype=torch.uint8),
    }
    seconds = []
    chunk = int(os.environ["TIANJI_STAGE2_CHUNK"])
    offset = int(os.environ["RLT_REF_OFFSET"])
    with torch.inference_mode():
        for prompt in prompts:
            before = time.monotonic()
            obs["task_descriptions"] = [prompt]
            features = model.extract_rlt_obs(obs)
            actions = joint_reference_actions(features, chunk, offset)
            assert features["z_rl"].shape == (1, 2048)
            assert torch.isfinite(features["z_rl"]).all()
            assert actions.shape == (1, chunk, 16)
            torch.cuda.synchronize()
            seconds.append(round(time.monotonic() - before, 4))
    print(
        json.dumps(
            {
                "stages": len(seconds),
                "control_hz": contract.control_hz,
                "action_dim": 16,
                "inference_seconds": seconds,
                "device": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
