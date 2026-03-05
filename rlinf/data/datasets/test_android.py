from omegaconf import DictConfig
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

project_root = Path("/mnt/project_rlinf/yuanqwang/mobile-agent")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

rlinf_path = project_root / "RLinf"
if str(rlinf_path) not in sys.path:
    sys.path.insert(0, str(rlinf_path))

import os

current_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = []
if current_pythonpath:
    pythonpath_parts.append(current_pythonpath)
pythonpath_parts.append(str(project_root))
pythonpath_parts.append(str(rlinf_path))
os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)

from omegaconf import OmegaConf
from rlinf.scheduler import Cluster
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.env.agent_worker import AndroidAgentWorker
from rlinf.workers.env.reward_worker import RewardWorker
from rlinf.scheduler import Channel


from rlinf.data.datasets.android import AndroidWorldDataset
from rlinf.data.tokenizers import hf_tokenizer
def create_test_config():
    """创建测试配置：两个 device、两个 worker."""
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH",
        "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
    )

    config_dict = {
        "cluster": {
            "num_nodes": 1,
            "component_placement": {
                # ModelParallelComponentPlacement 需要 actor/rollout/reward（GPU 放置）
                "actor": {"node_group": "gpu", "placement": "0"},
                "rollout": {"node_group": "gpu", "placement": "0"},
                "reward": {"node_group": "gpu", "placement": "all"},
                # Android 相关 worker（节点放置）
                "agent_worker": {
                    "node_group": "android_world",
                    "placement": "0-1",
                },
                "reward_worker": {
                    "node_group": "android_world",
                    "placement": "0-1",
                },
            },
            "node_groups": [
                # GPU 节点组：不指定 hardware 时自动检测 GPU
                {
                    "label": "gpu",
                    "node_ranks": "0",
                },
                {
                    "label": "android_world",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "python_interpreter_path": "/opt/venv/reason/bin/python3",
                        }
                    ],
                    "hardware": {
                        "type": "ADB",
                        "configs": [
                            {"device_id": "localhost:5555", "adb_path": "adb", "node_rank": 0},
                            {"device_id": "localhost:5557", "adb_path": "adb", "node_rank": 0},
                        ],
                    },
                }
            ],
        },
        "actor": {
            "group_name": "ActorGroup",
            "training_backend": "fsdp",
            "model": {
                "tensor_model_parallel_size": 1,
                "model_path": "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
                "model_type": "qwen3_vl",
            },
        },
        "algorithm": {
            "sampling_params": {
                "do_sample": True,
                "temperature": 0.6,
                "top_k": 1000000,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "max_new_tokens": 2048,
            },
        },
        "rollout": {
            "rollout_backend": "sglang",
            "model": {
                "model_path": "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
                "model_type": "qwen3_vl",
                "precision": "bf16",
            },
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "max_running_requests": 4,
            "gpu_memory_utilization": 0.90,
            "cuda_graph_max_bs": 128,
            "validate_weight": False,
            "validate_save_dir": None,
            "sglang": {
                "attention_backend": "triton",
                "decode_log_interval": 500000,
                "use_torch_compile": False,
                "torch_compile_max_bs": 16,
            },
            "detokenize": True,
            "return_logprobs": False,
        },
        "data": {
            "type": "android",
            "task_family": "android_world",
            "n_instances_per_task": 1,
            "seed": 1234,
            "max_prompt_length": 8192,
            "filter_prompt_by_length": True,
            "tokenizer": {
                "tokenizer_model": tokenizer_path,
                "use_fast": False,
                "trust_remote_code": True,
                "padding_side": "right",
            },
        },
        "reward": {
            "reward_type": "android",
            "reward_scale": 1.0,
            "device_id": "localhost:5555",
            "grpc_port": 8554,
            "adb_path": "adb",
        },
    }
    return OmegaConf.create(config_dict)

def get_test_dataset(cfg):
    tokenizer = hf_tokenizer(cfg.data.tokenizer.tokenizer_model)
    dataset = AndroidWorldDataset(
        config=cfg,
        tokenizer=tokenizer,
        seed=cfg.data.get("seed", 42),
    )
    return dataset

if __name__ == "__main__":
    cfg = create_test_config()
    dataset = get_test_dataset(cfg)
    for idx, item in enumerate(dataset):
        print(idx, item.meta["task_name"])