# /workspace/RLinf/rlinf/models/embodiment/gr00t_1_6/auto_converter.py
import os
import glob
import torch
import shutil

def auto_convert_latest_checkpoint(base_model_path="/workspace/RLinf/GR00T-N1.6-3B", logs_dir="/workspace/RLinf/logs"):
    # 🚨 终极防冲突锁：在分布式训练中，只允许全局主节点 (Rank 0) 执行转换！
    # 防止多卡并行时，8个进程同时读写硬盘导致崩溃。
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    if rank != "0" or local_rank != "0":
        return

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("⚠️ 未安装 safetensors，跳过模型自动转换。")
        return

    print("\n" + "="*60)
    print("🚀 [Auto-MLOps] 侦测到 Python 进程即将结束，触发自动化模型收尾钩子...")
    
    # 自动寻找最新生成的 full_weights.pt
    pt_files = glob.glob(os.path.join(logs_dir, "**", "full_weights.pt"), recursive=True)
    if not pt_files:
        print("⚠️ 找不到 full_weights.pt，可能训练尚未保存权重。退出转换。")
        return
        
    latest_pt_path = max(pt_files, key=os.path.getmtime)
    print(f"🔍 自动锁定最新 Checkpoint:\n   {latest_pt_path}")
    
    # 提取时间戳
    try:
        exp_timestamp = latest_pt_path.split("logs/")[1].split("/")[0]
        hf_output_path = f"{base_model_path}-SFT-HF-{exp_timestamp}"
    except Exception:
        hf_output_path = f"{base_model_path}-SFT-HF-Latest"

    # 如果已经转换过，则跳过
    if os.path.exists(os.path.join(hf_output_path, "model.safetensors")):
        print("✅ 检测到最新权重已转换，完美退出。")
        print("="*60 + "\n")
        return

    os.makedirs(hf_output_path, exist_ok=True)

    print(f"📋 [1/3] 从基座模型拷贝 config 及元数据...")
    for filename in os.listdir(base_model_path):
        if not filename.endswith((".safetensors", ".pt", ".bin", ".index.json")):
            src = os.path.join(base_model_path, filename)
            dst = os.path.join(hf_output_path, filename)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    print("🔄 [2/3] 提取并净化 SFT 权重 (剥离 FSDP 前缀)...")
    state_dict = torch.load(latest_pt_path, map_location="cpu", weights_only=True)
    clean_state_dict = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in state_dict.items()}

    print("💾 [3/3] 打包为 Safetensors 格式...")
    save_file(clean_state_dict, os.path.join(hf_output_path, "model.safetensors"))

    print(f"🎉 自动化转换圆满成功！\n👉 下一步 RL 请配置:\n   {hf_output_path}")
    print("="*60 + "\n")