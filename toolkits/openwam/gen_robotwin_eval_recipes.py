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
"""Generate OpenWAM RoboTwin evaluation recipes for every RoboTwin 2.0 task.

OpenWAM's RoboTwin checkpoints are multi-task (50 aloha-agilex tasks) and act in
the dataset's 20-D absolute EEF space, so every task shares one environment
preset (``env/robotwin_openwam_aloha.yaml``) and differs only in ``task_name``
and the step budget. The budget is RoboTwin's ``task_config/_eval_step_limit.yml``
rounded up to a multiple of the 32-step action chunk. Tasks that have RLinf eval
seeds use them; the others fall back to RoboTwinEnv's random seeds.

Run from the repository root::

    python toolkits/openwam/gen_robotwin_eval_recipes.py
"""

from __future__ import annotations

import json
from pathlib import Path

CHUNK = 32
CHECKPOINT = "/path/to/openwam-robotwin-checkpoint"

# RoboTwin 2.0 ``task_config/_eval_step_limit.yml`` (RLinf_support branch).
STEP_LIMITS = {
    "adjust_bottle": 400,
    "beat_block_hammer": 400,
    "blocks_ranking_rgb": 1200,
    "blocks_ranking_size": 1200,
    "click_alarmclock": 400,
    "click_bell": 400,
    "dump_bin_bigbin": 600,
    "grab_roller": 400,
    "handover_block": 800,
    "handover_mic": 600,
    "hanging_mug": 900,
    "lift_pot": 400,
    "move_can_pot": 400,
    "move_pillbottle_pad": 400,
    "move_playingcard_away": 400,
    "move_stapler_pad": 400,
    "open_laptop": 700,
    "open_microwave": 1500,
    "pick_diverse_bottles": 400,
    "pick_dual_bottles": 400,
    "place_a2b_left": 400,
    "place_a2b_right": 400,
    "place_bread_basket": 700,
    "place_bread_skillet": 500,
    "place_burger_fries": 500,
    "place_can_basket": 700,
    "place_cans_plasticbox": 800,
    "place_container_plate": 400,
    "place_dual_shoes": 600,
    "place_empty_cup": 500,
    "place_fan": 400,
    "place_mouse_pad": 400,
    "place_object_basket": 700,
    "place_object_scale": 400,
    "place_object_stand": 400,
    "place_phone_stand": 400,
    "place_shoe": 500,
    "press_stapler": 400,
    "put_bottles_dustbin": 1700,
    "put_object_cabinet": 700,
    "rotate_qrcode": 400,
    "scan_object": 500,
    "shake_bottle": 700,
    "shake_bottle_horizontally": 700,
    "stack_blocks_three": 1200,
    "stack_blocks_two": 800,
    "stack_bowls_three": 1200,
    "stack_bowls_two": 900,
    "stamp_seal": 400,
    "turn_switch": 400,
}

PRESET = """# Shared RoboTwin 2.0 preset for OpenWAM's multi-task aloha-agilex checkpoints.
# Task-specific recipes override task_config.task_name and the step budget; see
# toolkits/openwam/gen_robotwin_eval_recipes.py.
env_type: robotwin
total_num_envs: null

auto_reset: False
ignore_terminations: False

reward_coef: 1.0
use_custom_reward: True
use_rel_reward: True
center_crop: false

seed: 0
group_size: 1
use_fixed_reset_state_ids: False
max_steps_per_rollout_epoch: 416
max_episode_steps: 416

is_eval: False

assets_path: "/path/to/robotwin_assets"
seeds_path: null

# The checkpoint predicts 20-D absolute dual-arm EEF poses; RoboTwinEnv runs the
# ``ee`` controller and publishes a matching 20-D native_proprio.
openwam_action_representation: absolute_eef20
# OpenWAM's RoboTwin protocol samples instructions from the unseen pool;
# RoboTwin's own default is seen.
instruction_type: unseen

video_cfg:
  save_video: False
  info_on_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: False

task_config:
  task_name: click_bell
  step_lim: 416
  planner_backend: mplib
  render_freq: 0
  episode_num: 100
  use_seed: false
  save_freq: 15
  embodiment: [aloha-agilex]
  language_num: 100
  domain_randomization:
    random_background: true
    cluttered_table: true
    clean_background_rate: 0.02
    random_head_camera_dis: 0
    random_table_height: 0.03
    random_light: true
    crazy_random_light_rate: 0.02
  camera:
    head_camera_type: D435
    wrist_camera_type: D435
    collect_head_camera: true
    collect_wrist_camera: true   # head on top, left/right wrist below in the 384x320 canvas
  data_type:
    rgb: true
    third_view: false
    depth: false
    pointcloud: false
    observer: false
    endpose: true                # RoboTwin publishes the end-effector poses
    qpos: true
    mesh_segmentation: false
    actor_segmentation: false
  pcd_down_sample_num: 1024
  pcd_crop: true
  save_path: ./data
  clear_cache_freq: 8
  collect_data: true
  eval_video_log: true
"""

RECIPE = """# OpenWAM (RoboTwin study checkpoint, dual-system joint self-attention) on RoboTwin {task}.
# Generated by toolkits/openwam/gen_robotwin_eval_recipes.py; edit the generator, not this file.
defaults:
  - env/robotwin_openwam_aloha@env.eval
  - override hydra/job_logging: stdout
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${{oc.env:EMBODIED_PATH}}/config/

cluster:
  num_nodes: 1
  # Keep the SAPIEN environments off the OpenWAM rollout GPU.
  component_placement:
    env: 0
    rollout: 1

runner:
  task_type: embodied_eval
  logger:
    log_path: "../results"
    project_name: rlinf
    experiment_name: "robotwin_{task}_openwam_eval"
    logger_backends: ["tensorboard"]

  max_epochs: 1
  max_steps: -1

  only_eval: True
  val_check_interval: -1
  save_interval: -1

  resume_dir: null
  ckpt_path: null

env:
  group_name: "EnvGroup"

  # Override the default values in env/robotwin_openwam_aloha
  eval:
    rollout_epoch: 1
    # The sub-environments of one env worker plan their ee targets under a
    # shared lock (about 25 to 30 s per 32-step chunk each), so more envs per
    # worker only serialize; scale trajectories with rollout_epoch or more env
    # GPUs instead. robotwin_step_timeout_s (preset) bounds each chunk wait.
    total_num_envs: 2
    use_custom_reward: False
    use_rel_reward: True
    auto_reset: True
    ignore_terminations: True
    # RoboTwin's step limit for {task} is {limit}; OpenWAM executes whole
    # 32-action chunks, so the budget is rounded up to {steps}.
    max_episode_steps: {steps}
    max_steps_per_rollout_epoch: {steps}
    reward_coef: 1.0
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True
    assets_path: "/path/to/robotwin_assets"
    seeds_path: {seeds}
    video_cfg:
      save_video: False
      video_base_dir: ${{runner.logger.log_path}}/video/eval
    task_config:
      task_name: {task}
      step_lim: {steps}

rollout:
  group_name: "RolloutGroup"
  generation_backend: "huggingface"
  enable_offload: False
  pipeline_stage_num: 1
  model:
    model_type: "openwam"
    precision: bf16
    is_lora: false
    model_path: {checkpoint}
    device: cuda
    load_to_device: true
    ckpt_name: null
    num_frames: 33
    num_action_chunks: 32   # OpenWAM's RoboTwin protocol executes the full 32-step chunk
    action_dim: 20
    height: 384
    width: 320
    denoise_steps: 10
    openwam:
      inference_horizon: null
"""


def rounded_steps(limit: int, chunk: int = CHUNK) -> int:
    return ((limit + chunk - 1) // chunk) * chunk


def render_recipes(repo: Path) -> dict[Path, str]:
    """Return the env preset and the 50 task recipes keyed by their repo path."""
    files = {
        repo / "examples/embodiment/config/env/robotwin_openwam_aloha.yaml": PRESET
    }
    seeds_file = repo / "rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
    seeded = set(json.loads(seeds_file.read_text()).keys())
    out_dir = repo / "evaluations/robotwin"
    for task, limit in sorted(STEP_LIMITS.items()):
        seeds = (
            "${oc.env:REPO_PATH}/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
            if task in seeded
            else "null   # no RLinf eval seeds for this task yet: random RoboTwin seeds"
        )
        files[out_dir / f"robotwin_{task}_openwam_eval.yaml"] = RECIPE.format(
            task=task,
            limit=limit,
            steps=rounded_steps(limit),
            seeds=seeds,
            checkpoint=CHECKPOINT,
        )
    return files


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    files = render_recipes(repo)
    for path, text in files.items():
        path.write_text(text)
    print(f"wrote {len(files) - 1} recipes and the env preset")


if __name__ == "__main__":
    main()
