# Step 5: Debug One-Iter (Pi06 Full Loop)

完整 pi06 训练循环的一次迭代测试，验证 actor + 离线数据加载 + advantage 标签的端到端正确性。
使用 rollout 数据集和 pi05_base_pytorch 基础模型（不依赖 Step 4 ckpt）。

## 可变参数

```bash
# Rollout 数据集路径（需要已有 advantages_test.parquet，由 Step 3 产出）
ROLLOUT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/collected_data_40_eps"

# 基础模型路径（需要包含 physical-intelligence/libero/norm_stats.json）
BASE_MODEL_PATH="/mnt/project_rlinf/liuzhihao/vla_lib_fork_gaofeng/pretrained_models/pi05_base_pytorch"

# 配置名称
CONFIG_NAME="one_iter_debug_libero10_test"

# Checkpoint 转换脚本
CONVERT_SCRIPT="rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh"
```

## 测试命令

```bash
source switch_env openpi

bash examples/embodiment/run_debug_one_iter.sh ${CONFIG_NAME}
```

## Checkpoint 转换

```bash
# 找到 checkpoint 路径
LATEST_LOG=$(ls -td logs/debug_${CONFIG_NAME}-* 2>/dev/null | head -1)
CKPT_DIR="${LATEST_LOG}/debug_pi06_test/checkpoints/global_step_5/actor/model_state_dict"

# 用 symlink 避免路径中冒号问题
ln -sfn "${CKPT_DIR}" /tmp/step5_ckpt

# 转换
bash ${CONVERT_SCRIPT} fsdp_model_convertor \
    convertor.ckpt_path=/tmp/step5_ckpt/full_weights.pt \
    convertor.save_path=/tmp/step5_ckpt \
    model.model_path=${BASE_MODEL_PATH} \
    model.openpi.config_name=pi05_libero
```

## 验证方法

```bash
# 检查训练完成
LATEST_LOG=$(ls -td logs/debug_${CONFIG_NAME}-* 2>/dev/null | head -1)
echo "Latest log dir: ${LATEST_LOG}"
find "${LATEST_LOG}" -name "full_weights.pt" | head -1

# 验证 SafeTensors 转换
ls -la /tmp/step5_ckpt/*.safetensors 2>/dev/null && echo "PASS: SafeTensors generated" || echo "FAIL: No SafeTensors found"
```

## 对应配置文件

`examples/embodiment/config/one_iter_debug_libero10_test.yaml`

## 关键代码路径

| 文件 | 说明 |
|------|------|
| `examples/embodiment/train_debug_one_iter.py` | 入口脚本 |
| `rlinf/runners/debug_pi06_runner.py` | DebugPi06Runner (训练循环) |
| `rlinf/workers/actor/debug_fsdp_actor_worker_cfg.py` | DebugCFGFSDPActor |
| `rlinf/workers/cfg/fsdp_cfg_worker.py` | FSDPCfgWorker (底层CFG worker) |
| `rlinf/workers/cfg/utils.py` | DatasetWithAdvantage |

## 测试配置要点

- `offline_training_step: 5` — 仅5步离线训练
- `val_check_interval: -1` — 不做环境评估（加速测试）
- `save_interval: 5` — 第5步保存checkpoint
- `component_placement: 0-0` — 单GPU运行
- `micro_batch_size: 2, global_batch_size: 2` — 单GPU最小batch
- `advantage_tag: "test"` — 加载 Step 3 生成的 advantages_test.parquet
- `model_type: "cfg_model"` — CFG模型

## 与 Step 4 (cfg_sft) 的区别

| 方面 | Step 4 (cfg_sft) | Step 5 (debug_one_iter) |
|------|-----------------|----------------------|
| Runner | SFTRunner | DebugPi06Runner |
| Worker | FSDPCfgWorker | DebugCFGFSDPActor |
| GPU | 8 GPUs | 1 GPU (component_placement: 0-0) |
| 数据集 | SFT + rollout | Rollout only |
| 架构 | 纯SFT训练 | 完整RL循环（actor + optional env/rollout） |

## 预期运行时间

约 2-5 分钟（模型加载 + Ray集群初始化 + 5步离线训练 + ckpt转换）
