# Step 4: CFG SFT Training

使用 advantage 标记的数据集训练 CFG (Classifier-Free Guidance) 模型。训练完成后转换 checkpoint。

## 可变参数

```bash
# SFT 数据集路径（需要已有 advantages_test.parquet，由 Step 3 产出）
SFT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot"

# Rollout 数据集路径
ROLLOUT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/collected_data_40_eps"

# 基础模型路径（需要包含 physical-intelligence/libero/norm_stats.json）
BASE_MODEL_PATH="/mnt/project_rlinf/liuzhihao/vla_lib_fork_gaofeng/pretrained_models/pi05_base_pytorch"

# 配置名称
CONFIG_NAME="libero_cfg_sft_test"

# Checkpoint 转换脚本
CONVERT_SCRIPT="rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh"
```

## 前置检查

```bash
# 确认 norm_stats.json 存在
ls ${BASE_MODEL_PATH}/physical-intelligence/libero/norm_stats.json

# 确认 paligemma tokenizer 已缓存
ls /root/.cache/openpi/big_vision/paligemma_tokenizer.model
```

## 测试命令

```bash
source switch_env openpi

bash examples/cfg/run_cfg_sft.sh ${CONFIG_NAME}
```

## Checkpoint 转换

```bash
# 找到 checkpoint 路径
LATEST_LOG=$(ls -td logs/cfg_sft/${CONFIG_NAME}-* 2>/dev/null | head -1)
CKPT_DIR="${LATEST_LOG}/cfg_sft_test/checkpoints/global_step_5/actor/model_state_dict"

# 用 symlink 避免路径中冒号问题
ln -sfn "${CKPT_DIR}" /tmp/step4_ckpt

# 转换
bash ${CONVERT_SCRIPT} fsdp_model_convertor \
    convertor.ckpt_path=/tmp/step4_ckpt/full_weights.pt \
    convertor.save_path=/tmp/step4_ckpt \
    model.model_path=${BASE_MODEL_PATH} \
    model.openpi.config_name=pi05_libero
```

## 验证方法

```bash
# 检查训练完成
LATEST_LOG=$(ls -td logs/cfg_sft/${CONFIG_NAME}-* 2>/dev/null | head -1)
echo "Latest log dir: ${LATEST_LOG}"
find "${LATEST_LOG}" -name "full_weights.pt" | head -1

# 验证 SafeTensors 转换
ls -la /tmp/step4_ckpt/*.safetensors 2>/dev/null && echo "PASS: SafeTensors generated" || echo "FAIL: No SafeTensors found"
```

## 对应配置文件

`examples/cfg/config/libero_cfg_sft_test.yaml`

## 关键代码路径

| 文件 | 说明 |
|------|------|
| `examples/cfg/train_cfg_sft.py` | 入口脚本 |
| `rlinf/runners/sft_runner.py` | SFTRunner (训练循环) |
| `rlinf/workers/cfg/fsdp_cfg_worker.py` | FSDPCfgWorker (FSDP训练) |
| `rlinf/workers/cfg/utils.py` | DatasetWithAdvantage (advantage数据集包装) |
| `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py` | OpenPI CFG模型 |

## 测试配置要点

- `max_steps: 5` — 仅训练5步
- `micro_batch_size: 2, global_batch_size: 16` — 8 GPUs × 2 = 16
- `save_interval: 5` — 第5步保存checkpoint
- `advantage_tag: "test"` — 加载 Step 3 生成的 advantages_test.parquet
- `model_type: "cfg_model"` — CFG模型
- `unconditional_prob: 0.3` — 30%概率使用无条件输入（CFG核心）
- `guidance_type: "positive"` — 正向引导

## 预期运行时间

约 2-5 分钟（模型加载 + 5步训练 + ckpt转换）
