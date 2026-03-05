# Step 2: Train Value Model (VLA Lib SFT)

训练 value model (critic head)，学习预测每个状态的 V(s)。训练完成后自动转换 checkpoint 为 HF SafeTensors 格式。

## 可变参数

```bash
# 数据集路径（需要已有 return 列，由 Step 1 产出）
SFT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot"
ROLLOUT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/collected_data_40_eps"

# 基础模型路径
BASE_MODEL_PATH="/mnt/project_rlinf/liuzhihao/vla_lib_fork_gaofeng/pretrained_models/pi05_base_pytorch"

# 配置名称
CONFIG_NAME="libero_value_model_test"

# Checkpoint 转换脚本
CONVERT_SCRIPT="rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf_vla_lib.sh"
```

## 测试命令

```bash
source switch_env openpi

bash examples/vla_lib_sft/run_vla_lib_sft.sh ${CONFIG_NAME}
```

## Checkpoint 转换

训练完成后，将 FSDP checkpoint 转换为 HF SafeTensors:

```bash
# 找到 checkpoint 路径
LATEST_LOG=$(ls -td logs/vla_lib_sft/${CONFIG_NAME}-* 2>/dev/null | head -1)
CKPT_DIR="${LATEST_LOG}/vla_lib_value_sft/checkpoints/global_step_5/actor/model_state_dict"

# 用 symlink 避免路径中冒号问题
ln -sfn "${CKPT_DIR}" /tmp/step2_ckpt

# 转换
bash ${CONVERT_SCRIPT} \
    convertor.ckpt_path=/tmp/step2_ckpt/full_weights.pt \
    convertor.save_path=/tmp/step2_ckpt
```

## 验证方法

```bash
# 检查 checkpoint 是否生成
LATEST_LOG=$(ls -td logs/vla_lib_sft/${CONFIG_NAME}-* 2>/dev/null | head -1)
echo "Latest log dir: ${LATEST_LOG}"

# 查找训练 checkpoint
find "${LATEST_LOG}" -name "full_weights.pt" | head -1

# 验证 SafeTensors 转换
ls -la /tmp/step2_ckpt/*.safetensors 2>/dev/null && echo "PASS: SafeTensors generated" || echo "FAIL: No SafeTensors found"
```

## 对应配置文件

`examples/vla_lib_sft/config/libero_value_model_test.yaml`

## 关键代码路径

| 文件 | 说明 |
|------|------|
| `examples/vla_lib_sft/train_vla_lib_sft.py` | 入口脚本 |
| `rlinf/runners/vla_lib_sft_runner.py` | VlaLibSFTRunner (训练循环) |
| `rlinf/workers/vla_lib_sft/fsdp_value_sft_worker.py` | FSDPValueSftWorker (FSDP训练) |
| `rlinf/models/embodiment/vla_lib_value_model/modeling_critic.py` | Critic head实现 |
| `rlinf/datasets/vla_lib/value_mixture_dataset.py` | ValueMixtureDataset |

## 测试配置要点

- `max_steps: 5` — 仅训练5步，快速验证
- `freeze_vlm: True` — 冻结VLM加速测试
- `micro_batch_size: 2, global_batch_size: 16` — 8 GPUs × 2 = 16
- `save_interval: 5` — 第5步保存checkpoint
- `critic_forward_mode: "expert"` — 使用expert head (Gemma 100m)
- `expert_loss_type: "categorical"` — 分类损失 (201 bins)

## 预期运行时间

约 2-5 分钟（模型加载 + 5步训练 + ckpt转换）
