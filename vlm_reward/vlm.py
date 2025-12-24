import os
import sys
import subprocess
import time
import requests

def launch_vllm_server(
    # 模型路径
    model_path="/mnt/pfs/pg4hw0/yibin_workspace/model/Qwen2.5-VL-7B-Instruct",
    port=8000,
    
    gpu_ids="0,1", 
    tensor_parallel_size=2, 
    
    # 显存策略：
    # 如果这两张卡是【专门】给 vLLM 用的（PPO跑在卡2,3...），设为 0.9
    # 如果 PPO 训练也要挤在这两张卡上，建议设为 0.4 或更低
    gpu_memory_utilization=0.9, 
    
    max_model_len=8192, 
):
    """
    在 GPU 0,1 上启动 vLLM Server (TP=2)
    """
    # 1. 设置环境变量，让 vLLM 只能看到这两张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    print("="*50)
    print(f" Starting vLLM Distributed Inference")
    print(f"   GPUs: {gpu_ids}")
    print(f"   TP Size: {tensor_parallel_size}")
    print(f"   Model: {model_path}")
    print("="*50)

    # 2. 检查模型路径是否存在，防止报错
    if not os.path.exists(model_path):
        print(f" Error: Model path not found: {model_path}")
        sys.exit(1)

    # 3. 构建启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--trust-remote-code",      # Qwen 系列必须
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        
        # 【关键】分布式并行设置
        "--tensor-parallel-size", str(tensor_parallel_size),
        
        "--max-model-len", str(max_model_len),
        "--limit-mm-per-prompt", "image=8",
        "--dtype", "bfloat16",      # 推荐 bf16
        "--disable-log-requests"
    ]

    # 4. 启动子进程
    # 注意：vLLM 多卡启动可能需要 Ray，它会自动处理，确保环境里装了 ray (pip install ray)
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )

    api_url = f"http://localhost:{port}/v1/models"
    
    start_time = time.time()
    server_ready = False
    
    while True:
        try:
            resp = requests.get(api_url)
            if resp.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        
        # 检查进程是否意外退出
        if process.poll() is not None:
            print("\n vLLM process exited unexpectedly! Check logs above.")
            break
            
        time.sleep(5)
        
        if time.time() - start_time > 600: # 10分钟超时
            print("\n Timeout waiting for server startup.")
            process.terminate()
            break

    if server_ready:
        print(f"\n vLLM Server is READY at http://localhost:{port}")
        print("   You can now start your RL training script.")
        
        # 阻塞主线程，直到手动 Ctrl+C
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n Shutting down vLLM...")
            process.terminate()
            process.wait()

if __name__ == "__main__":
    launch_vllm_server()