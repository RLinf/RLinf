#! /usr/bin/env python

"""
DROID 上评估 DreamZero-DROID 策略的脚本（客户端 eval）。

用法（示例）：

1. 按 DreamZero 官方仓库说明启动 WebSocket policy server，例如：

   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
     socket_test_optimized_AR.py \
     --port 6000 \
     --enable-dit-cache \
     --model-path <path/to/DreamZero-DROID-checkpoint>

2. 在本仓库根目录（包含 dreamzero/ 与 RLinf/）下运行本脚本，例如：

   python -m toolkits.eval_scripts_dreamzero.droid_eval \
     --episodes 10 \
     --scene 1 \
     --host localhost \
     --port 6000 \
     --headless True
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import numpy as np
import torch
from isaaclab.app import AppLauncher
from tqdm import tqdm

# 需要在 app 启动后再导入这两个模块
#   - sim_evals.environments: 注册 DROID 环境
#   - isaaclab_tasks.utils.parse_env_cfg: 解析 DROID 配置

from toolkits.eval_scripts_dreamzero import setup_logger, setup_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DreamZero-DROID policy on DROID sim environment."
    )
    # 日志相关
    parser.add_argument(
        "--log_dir", type=str, default="logs/droid_eval_dreamzero", help="日志与结果目录"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="droid_dreamzero_eval",
        help="实验名称（用于日志文件命名）",
    )

    # 评估设置
    parser.add_argument(
        "--episodes", type=int, default=10, help="评估的 episode 数"
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=1,
        help="DROID 场景 id（1/2/3），决定指令与场景配置",
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=True,
        help="是否无图形界面（headless）运行 IsaacLab",
    )

    # DreamZero-DROID server 地址
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="DreamZero-DROID WebSocket server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6000,
        help="DreamZero-DROID WebSocket server port",
    )

    # action chunk / open-loop horizon
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=8,
        help="每次从策略获取的 open-loop 关节 action chunk 长度",
    )

    # 视频保存设置
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="是否保存每个 episode 的视频（默认不保存）",
    )
    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="runs",
        help="保存视频的根目录（若启用 --save_video）",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=15,
        help="保存视频时的帧率",
    )

    args, _ = parser.parse_known_args()
    return args


def get_instruction_for_scene(scene: int) -> str:
    """根据 DROID 场景 id 返回自然语言任务描述。"""
    if scene == 1:
        return "put the cube in the bowl"
    if scene == 2:
        return "pick up the can and put it in the mug"
    if scene == 3:
        return "put the banana in the bin"
    raise ValueError(f"Scene {scene} not supported")


def main() -> None:
    args = parse_args()

    # 设置 logger
    logger = setup_logger(args.exp_name, args.log_dir)
    logger.info("Starting DROID eval for DreamZero-DROID policy.")
    logger.info(f"Args: {args}")

    # 启动 IsaacLab App（和 run_sim_eval.py 一致的流程）
    parser = argparse.ArgumentParser(description="DROID DreamZero eval app launcher.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 重要：App 启动后才能导入 IsaacLab / sim_evals 相关模块
    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # 构造 DROID 环境配置
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = get_instruction_for_scene(args.scene)
    env_cfg.set_scene(args.scene)

    # 创建环境
    env = gym.make("DROID", cfg=env_cfg)

    # IsaacLab 中，第一次 reset 有时材质/纹理未完全加载，官方脚本通常会 reset 两次
    obs, _ = env.reset()
    obs, _ = env.reset()

    # 构造策略客户端（WebSocket -> DreamZero-DROID）
    policy_client = setup_policy(args)
    logger.info(
        f"Using DreamZero-DROID client with host={args.host}, port={args.port}, "
        f"action_chunk={args.action_chunk}"
    )

    # 视频保存目录
    video_dir: Path | None = None
    if args.save_video:
        video_dir = (
            Path(args.video_base_dir)
            / datetime.now().strftime("%Y-%m-%d")
            / datetime.now().strftime("%H-%M-%S")
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Video will be saved to: {video_dir}")

    max_steps = env.env.max_episode_length
    logger.info(f"Max steps per episode from env: {max_steps}")

    successes = 0
    total_episodes = 0

    with torch.no_grad():
        for ep in range(args.episodes):
            logger.info(f"Starting episode {ep + 1}/{args.episodes}")
            video_frames: list[np.ndarray] = []
            policy_client.reset()

            # 某些任务定义下，成功/失败标志可能体现在 env.info 中，
            # 这里先用 term/trunc 作为 episode 结束条件，成功统计需要按具体任务自定义。
            episode_success = False

            for _ in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{args.episodes}"):
                ret = policy_client.infer(obs, instruction)
                action = torch.tensor(ret["action"])[None]

                if not args.headless:
                    cv2.imshow(
                        "DROID Cameras", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR)
                    )
                    cv2.waitKey(1)

                if args.save_video:
                    video_frames.append(ret["viz"])

                obs, _, term, trunc, info = env.step(action)

                # 如需更精细的 success 判定，可以在 info 中寻找特定 key
                if term or trunc:
                    # 这里简单地将 term 视作成功，可按需要修改。
                    episode_success = bool(term)
                    break

            total_episodes += 1
            if episode_success:
                successes += 1

            logger.info(
                f"Episode {ep}: "
                f"terminated={'yes' if episode_success else 'no'}, "
                f"term_or_trunc={'yes' if (term or trunc) else 'no'}"
            )
            logger.info(
                f"Current success: {successes}/{total_episodes} "
                f"({successes / total_episodes * 100:.1f}%)"
            )

            # 保存视频
            if args.save_video and video_dir is not None and len(video_frames) > 0:
                out_path = video_dir / f"episode_{ep}.mp4"
                mediapy.write_video(
                    out_path,
                    video_frames,
                    fps=args.video_fps,
                )
                logger.info(f"Saved episode video to {out_path}")

    logger.info(
        f"Final success rate: {successes}/{total_episodes} "
        f"({successes / total_episodes * 100:.1f}%)"
    )

    # 清理资源
    env.close()
    simulation_app.close()
    logger.info("DROID DreamZero eval finished.")


if __name__ == "__main__":
    main()