import sys
import os
import datetime

TARGET_LIBERO_PATH = '/opt/libero'
CONFLICT_LIBERO_PATH = '/mnt/mnt/public/hyx/LIBERO-plus/libero'

if CONFLICT_LIBERO_PATH in sys.path:
    sys.path.remove(CONFLICT_LIBERO_PATH)
if os.path.dirname(CONFLICT_LIBERO_PATH) in sys.path:
     sys.path.remove(os.path.dirname(CONFLICT_LIBERO_PATH))

if TARGET_LIBERO_PATH not in sys.path:
    sys.path.insert(1, TARGET_LIBERO_PATH)


import numpy as np
import imageio.v2 as imageio 
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from liberoplus.liberoplus.envs import OffScreenRenderEnv 
    from liberoplus.liberoplus import get_libero_path 
except ImportError as e:
    print(f"依赖错误: 请确保 LIBERO已安装 (pip install -e.) 且所有依赖项满足: {e}")
    exit()

EPISODE_LENGTH = 200 
FPS = 20 
RESOLUTION = (512, 512) 
OUTPUT_DIR = Path("./custom_video_output")

DEFAULT_BDDL_PATH = "/mnt/mnt/public/hyx/LIBERO-plus/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate copy 2.bddl"

def setup_environment(bddl_file_path: Path, resolution: tuple):
    """
    初始化 LIBERO 环境，直接使用提供的 BDDL 文件路径。
    """
    if not bddl_file_path.is_file():
        raise FileNotFoundError(
            f"错误: 提供的 BDDL 文件未找到或不是有效文件: {bddl_file_path}"
        )
        
    print(f"正在从 BDDL 文件加载环境: {bddl_file_path}")

    bddl_path_str = str(bddl_file_path)
    
    camera_width, camera_height = resolution
    
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path_str, 
        
        camera_names=["agentview", "robot0_eye_in_hand"], 
        
        has_renderer=False,
        has_offscreen_renderer=True, 
        control_freq=FPS, 
        render_camera="agentview", 
        
        camera_heights=camera_height, 
        camera_widths=camera_width, 
    )
    
    env.seed(np.random.randint(0, 1000))
    return env, f"Custom Task: {bddl_file_path.name}"

def run_random_episode(env, episode_length, output_path):
    """在环境中运行固定长度的随机动作剧集，并保存视频。"""
    print(f"\n--- 启动随机仿真 ({episode_length} 步) ---")
    
    # 获取动作空间上下限
    try:
        action_spec = env.env.action_spec
    except AttributeError:
        action_spec = env.action_spec
    low = action_spec[0]
    high = action_spec[1]
    
    frames = [] 
    
    obs = env.reset()

    initial_frame = obs.get('agentview_image')
    if initial_frame is not None:
        frames.append(np.flip(initial_frame, axis=0))
    
    for t in tqdm(range(episode_length), desc="正在仿真步骤"):
        action = np.random.uniform(low=low, high=high)
        obs, reward, done, info = env.step(action)
        terminated = done
        truncated = False
        if 'TimeLimit.truncated' in info:
             truncated = info['TimeLimit.truncated']
             
        frame = obs.get('agentview_image') 
        
        if frame is not None:
            flipped_frame = np.flip(frame, axis=0) 
            frames.append(flipped_frame)
        
        if terminated or truncated:
            print(f"\n剧集提前结束于步骤 {t+1}。")
            break

  
    output_dir_path = Path(output_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    task_name = getattr(env, 'task_name', None)
    if task_name is None and hasattr(env, 'env') and hasattr(env.env, 'task_name'):
         task_name = env.env.task_name
         
    if task_name:
         task_name = task_name.split('/')[-1].split('.')[0]
         task_name = task_name.replace(' ', '_').replace('/', '_')
    else:
         task_name = "random_episode" 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{task_name}_{timestamp}.mp4"
    final_video_path = output_dir_path / video_filename
    
    print(f"\n正在编码 {len(frames)} 帧视频...")
    imageio.mimsave(str(final_video_path), frames, fps=FPS, quality=9) 
    print(f"\n✅ 视频已成功保存到: {final_video_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="LIBERO-plus 定制 BDDL 文件视频生成器（环境健全性检查）。")
    parser.add_argument("--bddl_path", type=str, default=DEFAULT_BDDL_PATH, help="BDDL 文件的完整绝对路径。")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH, help="仿真步数。")
    args = parser.parse_args()
    
    env = None
    bddl_path = Path(args.bddl_path)
    VIDEO_OUTPUT_DIR = Path("/mnt/mnt/public/hyx/LIBERO-plus/libero_output")
    
    try:
        env, description = setup_environment(bddl_path, RESOLUTION)
        print(f"环境描述: {description}")
        run_random_episode(env, args.steps, str(VIDEO_OUTPUT_DIR))
        
    except FileNotFoundError as e:
        print(f"\n[启动故障] BDDL 文件加载失败。错误详情: {e}")
    except Exception as e:
        print(f"\n[运行时异常] 仿真过程中发生意外错误: {e}")
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()