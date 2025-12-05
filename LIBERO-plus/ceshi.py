import sys
import os
import argparse
import numpy as np
import imageio.v2 as imageio
import torch
from tqdm import tqdm
from pathlib import Path
import datetime

# --- 1. 路径与环境配置 ---
CURRENT_ROOT = "/mnt/mnt/public/hyx/LIBERO-plus"

# 清理 sys.path 防止指向 /opt
new_sys_path = [p for p in sys.path if "/opt/libero" not in p]
sys.path = new_sys_path

if CURRENT_ROOT not in sys.path:
    sys.path.insert(0, CURRENT_ROOT)

print("🔍 [Path Patch] Check sys.path[0]:", sys.path[0])

paths = {
    "assets": os.path.join(CURRENT_ROOT, "libero/libero/assets"),
    "bddl_files": os.path.join(CURRENT_ROOT, "libero/libero/bddl_files"),
    "init_states": os.path.join(CURRENT_ROOT, "libero/libero/init_files"), 
    "benchmark_root": os.path.join(CURRENT_ROOT, "libero/libero"),
}

os.environ["LIBERO_ASSET_ROOT"] = paths["assets"]
os.environ["LIBERO_BDDL_PATH"] = paths["bddl_files"]
os.environ["LIBERO_INIT_STATES_PATH"] = paths["init_states"]

print(f"🔧 [Env Vars] BDDL Path set to: {paths['bddl_files']}")

# --- 2. Monkey Patch (劫持路径) ---
try:
    import liberoplus.liberoplus

    def force_local_path(path_name):
        if path_name in paths:
            return paths[path_name]
        return os.path.join(CURRENT_ROOT, "libero/libero", path_name)

    # 强制覆盖 get_libero_path
    libero.libero.get_libero_path = force_local_path
    print("✅ [Monkey Patch] Successfully hijacked get_libero_path")

    from liberoplus.liberoplus import benchmark, get_libero_path
    from liberoplus.liberoplus.envs import OffScreenRenderEnv
    
    print(f"✅ LIBERO Loaded. Current BDDL Path: {get_libero_path('bddl_files')}")
        
except ImportError as e:
    print(f"❌ Dependency Error: {e}")
    sys.exit(1)

# --- 3. 辅助函数 ---
def get_benchmark_instance(suite_name):
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name.lower() not in benchmark_dict:
        print(f"Benchmark '{suite_name}' not found. Available: {list(benchmark_dict.keys())}")
        sys.exit(1)
    return benchmark_dict[suite_name.lower()]()

def evaluate_suite(args):
    print(f"🚀 Loading Benchmark Suite: {args.suite_name}")
    benchmark_suite = get_benchmark_instance(args.suite_name)
    
    num_tasks = benchmark_suite.get_num_tasks()
    print(f"📊 Total tasks in suite: {num_tasks}")

    if args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = range(num_tasks)

    output_dir = Path(args.output) / args.suite_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_runs = 0

    for i in tqdm(task_ids, desc="Evaluating Tasks"):
        task = benchmark_suite.get_task(i)
        
        # 获取初始状态
        init_states = benchmark_suite.get_task_init_states(i) 
        if isinstance(init_states, torch.Tensor):
            init_states = init_states.cpu().numpy()
        init_state = init_states[0] 

        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        env_args = {
            "bddl_file_name": bddl_path,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "render_gpu_device_id": 0,
            "control_freq": 20,
            "camera_heights": 512,
            "camera_widths": 512,
        }
        
        try:
            env = OffScreenRenderEnv(**env_args)
        except Exception as e:
            print(f"Task {i} Env Creation Failed: {e}")
            continue

        env.seed(args.seed)
        env.reset()
        
        # 获取初始 observation
        obs = env.set_init_state(init_state)
        if obs is None:
            obs = env.env._get_observations()

        done = False
        steps = 0
        frames = []
        
        while steps < args.max_steps:
            # --- 🛠️ 修复核心：手动采样随机动作 ---
            # Robosuite 的 action_spec 返回的是 (low, high) 元组
            # 我们需要利用这个范围生成随机数
            low, high = env.env.action_spec
            action = np.random.uniform(low, high)
            # ----------------------------------
            
            obs, reward, done, info = env.step(action)
            
            if args.save_video:
                img = obs['agentview_image']
                if img is not None:
                    if img.dtype != np.uint8:
                         img = (img * 255).astype(np.uint8)
                    # Robosuite 图像通常需要上下翻转
                    frames.append(np.flip(img, axis=0))
            
            steps += 1
            if done:
                break
        
        env.close()

        is_success = done 
        if is_success:
            success_count += 1
        total_runs += 1

        print(f"Task {i} ({task.name}): {'✅ Success' if is_success else '❌ Fail'}")

        if args.save_video and len(frames) > 0:
            video_name = f"task_{i:04d}_{'success' if is_success else 'fail'}.mp4"
            imageio.mimsave(str(output_dir / video_name), frames, fps=20, quality=8)

    print(f"\nEvaluation Finished. Success Rate: {success_count}/{total_runs} = {success_count/total_runs:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_name", type=str, default="libero_spatial", help="Suite name (e.g., libero_spatial)")
    parser.add_argument("--task_id", type=int, default=None, help="Specify task ID")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=str, default="./libero_plus_results")
    parser.add_argument("--save_video", action='store_true', help="Save video")
    
    args = parser.parse_args()
    evaluate_suite(args)