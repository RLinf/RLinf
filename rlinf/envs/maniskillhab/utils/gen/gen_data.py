import os
import random
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path

from tqdm import tqdm

from gymnasium import spaces

import numpy as np
import torch

# ManiSkill specific imports
import mani_skill.envs
from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose

from mshab.agents.ppo import Agent as PPOAgent
from mshab.agents.sac.agent import Agent as SACAgent
from mshab.envs.make import EnvConfig, make_env
from mshab.envs.wrappers.record import RecordEpisode
from mshab.utils.array import recursive_slice, to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.label_dataset import get_episode_label_and_events
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


NUM_ENVS = 252
SEED = 2024
MAX_TRAJECTORIES = 1000

SAVE_TRAJECTORIES = True
SAVE_GRASP_POSE = False

RECORD_VIDEO = False
DEBUG_VIDEO_GEN = False

SUBTASK_TO_EPISODE_LABELS = dict(
    pick=["straightforward_success"],
    place=["placed_in_goal_success"],
    navigate=["navigation_success"],
    open=["open_success"],
    close=dict(
        fridge=["closed_success", "success_then_excessive_collisions"],
        kitchen_counter=["closed_success"],
    ),
)


def eval(
    task="tidy_house",
    subtask="pick",
    obj_name="all",
    override_cfg_path=None,
    override_ckpt_path=None,
):
    # timer
    timer = NonOverlappingTimeProfiler()

    # seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # NOTE (arth): maybe require cuda since we only allow gpu sim anyways
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------------------------------
    # ENVS
    # -------------------------------------------------------------------------------------------------

    split = "train" if SAVE_TRAJECTORIES or SAVE_GRASP_POSE else "val"
    if override_cfg_path is not None:
        _override_task_plan_fp_path = Path(
            parse_cfg(override_cfg_path).env.task_plan_fp
        )
        task = _override_task_plan_fp_path.parent.parent.parent.stem
        subtask = _override_task_plan_fp_path.parent.parent.stem
        obj_name = _override_task_plan_fp_path.stem
    if SAVE_GRASP_POSE:
        assert subtask == "pick", f"{subtask} should be pick when {SAVE_GRASP_POSE=}"
    SUBTASKS = [subtask]

    print("Generating data with the following args...")
    print(task, subtask, split, obj_name)
    print("override_cfg_path", override_cfg_path)
    print("override_ckpt_path", override_ckpt_path, flush=True)

    env_cfg = EnvConfig(
        # env
        env_id=f"{subtask.capitalize()}SubtaskTrain-v0",
        obs_mode="rgbd",
        num_envs=NUM_ENVS,
        max_episode_steps=500 if subtask == "navigate" else 200,
        # misc
        record_video=RECORD_VIDEO or DEBUG_VIDEO_GEN,
        info_on_video=False,
        debug_video=DEBUG_VIDEO_GEN,
        debug_video_gen=DEBUG_VIDEO_GEN,
        continuous_task=True,
        cat_state=True,
        cat_pixels=False,
        task_plan_fp=(
            ASSET_DIR
            / f"scene_datasets/replica_cad_dataset/rearrange/task_plans/{task}/{subtask}/{split}/{obj_name}.json"
        ),
        spawn_data_fp=(
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange/spawn_data"
            / task
            / subtask
            / split
            / "spawn_data.pt"
        ),
        extra_stat_keys=[],
        env_kwargs=dict(
            require_build_configs_repeated_equally_across_envs=False,
            add_event_tracker_info=True,
            robot_force_mult=0.001,
            robot_force_penalty_min=0.2,
            target_randomization=False,
        ),
    )
    logger_cfg = LoggerConfig(
        workspace="mshab_exps",
        exp_name=(
            (
                f"gen_data_save_trajectories/{task}/{subtask}/{split}/{obj_name}"
                if SAVE_TRAJECTORIES
                else (
                    f"gen_data_gen_grasp_poses/{task}/{subtask}/{split}/{obj_name}"
                    if SAVE_GRASP_POSE
                    else f"gen_data/{task}/{subtask}/{split}/{obj_name}"
                )
            )
            if override_cfg_path is None
            else f"EVAL--{'/'.join(str(override_cfg_path).split('/')[1:-1])}"
        ),
        clear_out=False,
        tensorboard=False,
        wandb=False,
        exp_cfg=dict(env_cfg=asdict(env_cfg)),
    )

    logger = Logger(
        logger_cfg=logger_cfg,
        save_fn=None,
    )
    wrappers = []
    valid_episode_labels = SUBTASK_TO_EPISODE_LABELS[subtask]
    if isinstance(valid_episode_labels, dict):
        valid_episode_labels = valid_episode_labels[obj_name]
    if SAVE_TRAJECTORIES:
        wrappers = [
            partial(
                RecordEpisode,
                output_dir=logger.exp_path,
                save_trajectory=True,
                save_video=False,
                info_on_video=False,
                save_on_reset=True,
                save_video_trigger=None,
                max_steps_per_video=None,
                clean_on_close=True,
                record_reward=True,
                source_type="RL",
                source_desc=f"RL policy trained exclusively to {subtask} {obj_name} in the {task} task.",
                record_env_state=False,
                label_episode=True,
                valid_episode_labels=valid_episode_labels,
                max_trajectories=MAX_TRAJECTORIES,
            )
        ]
    eval_envs = make_env(
        env_cfg,
        video_path=logger.eval_video_path,
        wrappers=wrappers,
    )
    eval_obs, _ = eval_envs.reset()

    # -------------------------------------------------------------------------------------------------
    # SPACES
    # -------------------------------------------------------------------------------------------------

    obs_space = eval_envs.unwrapped.single_observation_space
    pixels_obs_space: spaces.Dict = obs_space["pixels"]
    state_obs_space: spaces.Box = obs_space["state"]
    act_space = eval_envs.unwrapped.single_action_space
    model_pixel_obs_space = dict()
    for k, space in pixels_obs_space.items():
        shape, low, high, dtype = (
            space.shape,
            space.low,
            space.high,
            space.dtype,
        )
        if len(shape) == 4:
            shape = (shape[0] * shape[1], shape[-2], shape[-1])
            low = low.reshape((-1, *low.shape[-2:]))
            high = high.reshape((-1, *high.shape[-2:]))
        model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
    model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------

    mshab_ckpt_dir = ASSET_DIR / "mshab_checkpoints"
    if not mshab_ckpt_dir.exists():
        mshab_ckpt_dir = Path("mshab_checkpoints")

    policies = dict()
    for subtask_name in SUBTASKS:
        cfg_path = (
            Path(override_cfg_path)
            if override_cfg_path is not None
            else mshab_ckpt_dir / "rl" / task / subtask_name / obj_name / "config.yml"
        )
        ckpt_path = (
            Path(override_ckpt_path)
            if override_ckpt_path is not None
            else mshab_ckpt_dir / "rl" / task / subtask_name / obj_name / "policy.pt"
        )

        algo_cfg = parse_cfg(default_cfg_path=cfg_path).algo
        if algo_cfg.name == "ppo":
            policy = PPOAgent(eval_obs, act_space.shape)
            policy.eval()
            policy.load_state_dict(torch.load(ckpt_path, map_location=device)["agent"])
            policy.to(device)
            policy_act_fn = lambda obs: policy.get_action(obs, deterministic=True)
        elif algo_cfg.name == "sac":
            policy = SACAgent(
                model_pixel_obs_space,
                state_obs_space.shape,
                act_space.shape,
                actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
                critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
                critic_layer_norm=algo_cfg.critic_layer_norm,
                critic_dropout=algo_cfg.critic_dropout,
                encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
                encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
                cnn_features=list(algo_cfg.cnn_features),
                cnn_filters=list(algo_cfg.cnn_filters),
                cnn_strides=list(algo_cfg.cnn_strides),
                cnn_padding=algo_cfg.cnn_padding,
                log_std_min=algo_cfg.actor_log_std_min,
                log_std_max=algo_cfg.actor_log_std_max,
                device=device,
            )
            policy.eval()
            policy.to(device)
            policy.load_state_dict(torch.load(ckpt_path, map_location=device)["agent"])
            policy_act_fn = lambda obs: policy.actor(
                obs["pixels"],
                obs["state"],
                compute_pi=False,
                compute_log_pi=False,
            )[0]
        else:
            raise NotImplementedError(f"algo {algo_cfg.name} not supported")
        policies[subtask_name] = policy_act_fn

    def act(obs, subtask_type):
        with torch.no_grad():
            with torch.device(device):
                action = torch.zeros(eval_envs.num_envs, *act_space.shape)

                pick_tasks = subtask_type == 0
                place_tasks = subtask_type == 1
                navigate_tasks = subtask_type == 2
                open_tasks = subtask_type == 3
                close_tasks = subtask_type == 4

                if torch.any(pick_tasks):
                    action[pick_tasks] = policies["pick"](
                        recursive_slice(obs, pick_tasks)
                    )
                if torch.any(place_tasks):
                    action[place_tasks] = policies["place"](
                        recursive_slice(obs, place_tasks)
                    )
                if torch.any(navigate_tasks):
                    action[navigate_tasks] = policies["navigate"](
                        recursive_slice(obs, navigate_tasks)
                    )
                if torch.any(open_tasks):
                    action[open_tasks] = policies["open"](
                        recursive_slice(obs, open_tasks)
                    )
                if torch.any(close_tasks):
                    action[close_tasks] = policies["close"](
                        recursive_slice(obs, close_tasks)
                    )

                return action

    # -------------------------------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------------------------------

    get_subtask_type = lambda: eval_envs.unwrapped.task_ids[
        eval_envs.unwrapped.subtask_pointer
    ]

    eval_obs = to_tensor(eval_envs.reset(seed=SEED)[0], device=device, dtype="float")
    subtask_type = get_subtask_type()
    last_subtask_type = subtask_type.clone()
    pbar = tqdm(range(MAX_TRAJECTORIES), total=MAX_TRAJECTORIES)
    articulation_types_by_label = dict()
    rcumulative_forces_by_label = defaultdict(list)

    if SAVE_GRASP_POSE:
        grasp_pose_fp = (
            ASSET_DIR
            / "scene_datasets/replica_cad_dataset/rearrange/grasp_poses"
            / task
            / obj_name
            / "grasp_poses.pt"
        )
        success_qposes = []
        success_obj_raw_poses_wrt_tcp = []

    def check_done():
        if SAVE_TRAJECTORIES:
            # NOTE (arth): eval_envs.env._env is bad, fix in wrappers instead (prob with get_attr func)
            return eval_envs.env._env.reached_max_trajectories
        return len(eval_envs.return_queue) >= MAX_TRAJECTORIES

    def after_step(done, info: dict):
        if SAVE_GRASP_POSE:
            if torch.any(info["success"]):
                qps: torch.Tensor = eval_envs.unwrapped.agent.robot.qpos
                qps[..., :3] = 0
                for q in qps[info["success"]].cpu():
                    success_qposes.append(q)
                obj_poses_wrt_tcp: Pose = (
                    eval_envs.unwrapped.agent.tcp.pose.inv()
                    * eval_envs.unwrapped.subtask_objs[0].pose
                )
                for raw_pose in obj_poses_wrt_tcp.raw_pose[info["success"]].cpu():
                    success_obj_raw_poses_wrt_tcp.append(raw_pose)

        update_pbar()
        if torch.any(done) and len(env_cfg.extra_stat_keys) > 0:
            torch.save(
                eval_envs.extra_stats,
                logger.exp_path / "eval_extra_stat_keys.pt",
            )
            for env_num, extra_stats_idx in zip(
                torch.where(done)[0],
                -torch.arange(1, torch.sum(done.int() + 1)),
            ):
                episode_extra_stats = recursive_slice(
                    eval_envs.extra_stats, extra_stats_idx
                )
                label, _, _ = get_episode_label_and_events(
                    eval_envs.unwrapped.task_cfgs,
                    episode_extra_stats["success"],
                    episode_extra_stats,
                )
                if label not in articulation_types_by_label:
                    articulation_types_by_label[label] = defaultdict(int)
                base_subtask = eval_envs.unwrapped.base_task_plans[
                    (eval_envs.unwrapped.task_plan[0].composite_subtask_uids[env_num],)
                ].subtasks[0]
                if getattr(base_subtask, "articulation_config", None) is None:
                    articulation_type = None
                else:
                    articulation_type = (
                        base_subtask.articulation_config.articulation_type
                    )
                articulation_types_by_label[label][articulation_type] += 1
                rcumulative_forces_by_label[label].append(
                    torch.max(episode_extra_stats["robot_cumulative_force"]).item()
                )

    step_num = 0

    def update_pbar():
        if SAVE_TRAJECTORIES:
            diff = eval_envs.env._env.num_saved_trajectories - pbar.last_print_n
        else:
            diff = len(eval_envs.return_queue) - pbar.last_print_n

        if diff > 0:
            pbar.update(diff)

        pbar.set_description(f"{step_num=}")

    while not check_done():
        still_running = subtask_type <= 4
        last_subtask_type[still_running] = subtask_type[still_running]
        timer.end("other")
        action = act(eval_obs, last_subtask_type)
        timer.end("sample")
        eval_obs, _, term, trunc, info = eval_envs.step(action)
        timer.end("sim_sample")
        eval_obs = to_tensor(
            eval_obs,
            device=device,
            dtype="float",
        )
        subtask_type = get_subtask_type()
        after_step(term | trunc, info)
        step_num += 1

    if SAVE_GRASP_POSE:
        success_qposes = to_tensor(success_qposes)
        success_obj_raw_poses_wrt_tcp = to_tensor(success_obj_raw_poses_wrt_tcp)
        os.makedirs(grasp_pose_fp.parent, exist_ok=True)
        with open(grasp_pose_fp, "wb") as f:
            torch.save(
                dict(
                    success_qpos=success_qposes,
                    success_obj_raw_pose_wrt_tcp=success_obj_raw_poses_wrt_tcp,
                ),
                f,
            )

    # -------------------------------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------------------------------
    if SAVE_GRASP_POSE:
        print(f"{len(success_qposes)=}, {len(success_obj_raw_poses_wrt_tcp)=}")

    results_logs = dict(
        num_trajs=len(eval_envs.return_queue),
        return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
        .float()
        .mean()
        / eval_envs.max_episode_steps,
        success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
        .float()
        .mean(),
        success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device)
        .float()
        .mean(),
        len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
    )
    time_logs = timer.get_time_logs(pbar.last_print_n * env_cfg.max_episode_steps)
    print(
        "results",
        results_logs,
    )
    print("time", time_logs)

    with open(logger.exp_path / "output.txt", "w") as f:
        f.write("results\n" + str(results_logs) + "\n")
        f.write("time\n" + str(time_logs) + "\n")

    # -------------------------------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------------------------------

    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    import sys

    eval(task=sys.argv[1], subtask=sys.argv[2], obj_name=sys.argv[3])
