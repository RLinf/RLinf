# Step 1: Compute Returns

对 LeRobot 数据集计算 return, reward, prompt 列。需要处理 SFT 和 rollout 两个数据集。

## 可变参数

```bash
# SFT 数据集路径（LeRobot格式）
SFT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot"

# Rollout 数据集路径（LeRobot格式）
ROLLOUT_DATASET_PATH="/mnt/project_rlinf_hs/liuzhihao/data_to_use/collected_data_40_eps"

# 配置名称
CONFIG_NAME="compute_returns_test"
```

## 测试命令

```bash
source switch_env openpi

bash examples/process/run_compute_returns.sh ${CONFIG_NAME}
```

## 验证方法

```bash
# 检查两个数据集的 stats.json 是否都包含 return 和 reward 字段
for DATASET_PATH in "${SFT_DATASET_PATH}" "${ROLLOUT_DATASET_PATH}"; do
    echo "=== Checking: ${DATASET_PATH} ==="
    python3 -c "
import json
with open('${DATASET_PATH}/meta/stats.json') as f:
    stats = json.load(f)
assert 'return' in stats, 'return not in stats'
assert 'reward' in stats, 'reward not in stats'
print(f'Return range: [{stats[\"return\"][\"min\"]}, {stats[\"return\"][\"max\"]}]')
print(f'Reward range: [{stats[\"reward\"][\"min\"]}, {stats[\"reward\"][\"max\"]}]')
print('PASS')
"
done
```

## 对应配置文件

`examples/process/config/compute_returns_test.yaml`

## 关键代码路径

| 文件 | 说明 |
|------|------|
| `examples/process/compute_returns.py` | 主脚本，Hydra入口 |
| `examples/process/run_compute_returns.sh` | Shell启动器 |

## 逻辑核查 (vs 原始repo)

- `failure_reward` 改为必须显式配置（原先默认-300.0）— **改进，非错误**
- 返回值计算公式不变: `G_t = r_t + gamma * G_{t+1}`
- SFT数据集: reward=-1/step, 最后一步=0
- Rollout数据集: reward=-1/step, 最后一步=0(成功) 或 failure_reward(失败)

## 预期运行时间

约 30 秒（两个数据集共 ~24000 frames）
