# Step 3: Compute Advantages

使用训练好的 value model，计算 TD(N) advantages，生成 is_success (True/False) 标签。需要处理 SFT 和 rollout 两个数据集。

## 可变参数

```bash
# 数据集路径（需要已有 return 列，由 Step 1 产出）
SFT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot"
ROLLOUT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/collected_data_40_eps"

# Value model checkpoint 路径（由 Step 2 转换后产出）
# 应指向包含 SafeTensors 的目录
VALUE_CHECKPOINT="/tmp/step2_ckpt"

# 配置名称
CONFIG_NAME="compute_advantages_test"

# GPU数量（测试用1块）
NPROC=1
```

## 测试命令

```bash
source switch_env openpi

bash examples/process/run_compute_advantages.sh ${CONFIG_NAME} --nproc ${NPROC} \
    advantage.value_checkpoint=${VALUE_CHECKPOINT}
```

## 验证方法

```bash
# 检查两个数据集的 advantages parquet 是否生成
for DATASET_PATH in "${SFT_DATASET_PATH}" "${ROLLOUT_DATASET_PATH}"; do
    echo "=== Checking: ${DATASET_PATH} ==="
    ADVANTAGE_FILE="${DATASET_PATH}/meta/advantages_test.parquet"
    python3 -c "
import pandas as pd
df = pd.read_parquet('${ADVANTAGE_FILE}')
assert 'advantage' in df.columns, 'advantage column missing'
assert 'is_success' in df.columns, 'is_success column missing'
n_pos = df['is_success'].sum()
n_total = len(df)
print(f'Total samples: {n_total}')
print(f'Positive (is_success=True): {n_pos} ({100*n_pos/n_total:.1f}%)')
print(f'Advantage range: [{df[\"advantage\"].min():.4f}, {df[\"advantage\"].max():.4f}]')
print('PASS')
"
done
```

## 对应配置文件

`examples/process/config/compute_advantages_test.yaml`

## 关键代码路径

| 文件 | 说明 |
|------|------|
| `examples/process/compute_advantages.py` | 主脚本，支持torchrun |
| `examples/process/run_compute_advantages.sh` | Shell启动器 |

## Advantage 计算公式

```
A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t)
is_success = (advantage >= threshold)  # threshold由 positive_quantile=0.3 确定 (top 30%)
```

当 `global_threshold: true` 时，阈值在所有数据集上统一计算。

## 测试配置要点

- `batch_size: 4` — 小batch便于测试
- `pipeline_batch_size: 128` — 小流水线batch
- `num_dataloader_workers: 2` — 少量worker
- `tag: "test"` — 生成 advantages_test.parquet（不覆盖已有的 advantages）
- `distributed.enabled: false` — 单GPU测试
- `global_threshold: true` — 两个数据集使用统一阈值

## 预期运行时间

约 5-10 分钟（模型加载 + ~24000 frames 推理）
