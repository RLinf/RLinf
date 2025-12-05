#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_init_states.py
- 从指定 bddl_base_dir 读取所有 .bddl
- 为每个任务生成 num_inits 个初始状态
- 保存为 .pruned_init 压缩文件到 output_dir

用法示例：
    python generate_init_states.py \
        --bddl_base_dir /path/to/bddl_dir \
        --output_dir /path/to/output_dir \
        --num_inits 50 \
        --height 128 \
        --width 128
"""

import os
import zipfile
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from liberopro.liberopro.envs import OffScreenRenderEnv


def generate_init_states(
    bddl_base_dir: str,
    output_dir: str,
    num_inits: int = 50,
    height: int = 128,
    width: int = 128,
):
    bddl_base_dir = Path(bddl_base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .bddl 文件
    bddl_files = list(bddl_base_dir.glob("*.bddl"))
    print(f"找到 {len(bddl_files)} 个 BDDL 文件")

    for bddl_file in tqdm(bddl_files, desc="处理 BDDL 文件"):
        task_base_name = bddl_file.stem
        print(f"\n开始处理任务: {task_base_name}")

        all_initial_states = []

        for i in tqdm(range(num_inits), desc=f"生成 {task_base_name} 的初始状态"):
            env = None
            try:
                env_args = {
                    "bddl_file_name": str(bddl_file),
                    "camera_heights": height,
                    "camera_widths": width,
                }
                env = OffScreenRenderEnv(**env_args)

                initial_state = env.get_sim_state()
                all_initial_states.append(initial_state)

            except Exception as e:
                print(f"  生成第 {i+1} 个状态时出错: {e}")

            finally:
                if env is not None and hasattr(env, 'close'):
                    env.close()

        output_filename = f"{task_base_name}.pruned_init"
        output_filepath = output_dir / output_filename

        try:
            with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                all_initial_states = np.array(all_initial_states)
                pickled_states_list = pickle.dumps(all_initial_states)
                zipf.writestr("archive/data.pkl", pickled_states_list)
                zipf.writestr("archive/version", b"1")

            print(f"成功保存 {len(all_initial_states)} 个状态到: {output_filepath}")

        except Exception as e:
            print(f"保存状态列表时出错: {e}")

    print("\n所有任务处理完成！")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate init states for LIBERO BDDL tasks.")
    parser.add_argument("--bddl_base_dir", type=str, required=True, help="Directory containing BDDL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pruned_init files.")
    parser.add_argument("--num_inits", type=int, default=50, help="Number of init states to generate per task.")
    parser.add_argument("--height", type=int, default=128, help="Camera height.")
    parser.add_argument("--width", type=int, default=128, help="Camera width.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_init_states(
        bddl_base_dir=args.bddl_base_dir,
        output_dir=args.output_dir,
        num_inits=args.num_inits,
        height=args.height,
        width=args.width,
    )