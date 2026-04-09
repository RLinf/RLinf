# placement_test（中文说明）

本目录用于做 **placement 吞吐/时延测量**：通过不同的 `cluster.component_placement` 与 `pipeline_stage_num` 组合，批量生成测试 YAML，并用脚本连续跑完一组实验。

## 1. 批量生成 placement_test YAML

使用 `placement_test/generate_yaml_batch.py`（或 `.sh` 包装脚本）从一个 **基 YAML** 批量生成一组 YAML 到 `placement_test/`。

### 1.1 生成器会修改哪些字段？哪些保持不变？

生成器 **只修改** 下面 3 个关键字段：

- `env.train.total_num_envs`
- `rollout.pipeline_stage_num`
- `cluster.component_placement`

除此之外（环境/任务/模型/算法/优化器/采样参数等）都会从基 YAML **原样拷贝**，不会改动。

### 1.2 每个 envnum 会生成哪些配置？

对每个 `total_num_envs`（envnum），会生成：

- **collocated**：`pipeline_stage_num = 1`，命名标签 `env07-rollout07`（三组件都在 `0-7`）
- **hybrid**：`pipeline_stage_num = 2`，命名标签 `env03-rollout47`（env 在 `0-3`，rollout 在 `4-7`，actor 在 `0-7`）
- **额外 hybrid（可选）**：当 `envnum % 3 == 0` 时，再生成一份 `env01-rollout27`（`pipeline_stage_num = 2`）

这些标签/placement 与当前 `placement_test/*.yaml` 里已有的命名与 placement 规则保持一致。

### 1.3 命名前缀：`--prefix auto` 与手动 `--prefix`

默认 `--prefix auto` 会从基 YAML 的 Hydra `defaults` 推断一个短前缀 `环境_模型`：

- **环境**：来自 `defaults` 中的 `env/<...>@env.train`，取下划线 `_` 之前的“族名”
  - 例如 `env/maniskill_put_on_plate_in_scene_25_main@env.train` → `maniskill`
  - 例如 `env/robotwin_beat_block_hammer@env.train` → `robotwin`
- **模型**：来自 `defaults` 中的 `model/<...>@actor.model`，取 `model/` 与 `@` 之间的**完整**名字（不按 `_` 截断）
  - 例如 `model/openvla@actor.model` → `openvla`
  - 例如 `model/openvla_oft@actor.model` → `openvla_oft`

因此自动前缀可能是：`maniskill_openvla`、`robotwin_openvla_oft` 等。

如果你希望把任务名/算法也写入文件名前缀（例如 `robotwin_beat_block_hammer_grpo_openvlaoft`），建议直接手动指定：

- `--prefix robotwin_beat_block_hammer_grpo_openvlaoft`

### 1.4 Dry-run（只打印将要生成的文件名）

```bash
python3 placement_test/generate_yaml_batch.py \
  --base-yaml examples/embodiment/config/aaa_maniskill_ppo_openvla.yaml \
  --envnum-start 96 --envnum-end 128 --envnum-step 16 \
  --include-env01-rollout27-when-divisible-by-3 \
  --dry-run
```

### 1.5 实际写文件（推荐使用 wrapper）

```bash
bash placement_test/generate_yaml_batch.sh \
  --base-yaml examples/embodiment/config/aaa_maniskill_ppo_openvla.yaml \
  --envnum-start 96 --envnum-end 512 --envnum-step 16 \
  --include-env01-rollout27-when-divisible-by-3
```

### 1.6 示例：RobotWin + OpenVLA-OFT，自定义前缀

```bash
bash placement_test/generate_yaml_batch.sh \
  --base-yaml examples/embodiment/config/robotwin_beat_block_hammer_grpo_openvlaoft.yaml \
  --prefix robotwin_beat_block_hammer_grpo_openvlaoft \
  --envnum-start 96 --envnum-end 512 --envnum-step 16 \
  --include-env01-rollout27-when-divisible-by-3
```

## 2. 使用 run_test_batch.sh 连续跑一组 YAML

批量连续跑由 `placement_test/run_test_batch.sh` 驱动。

### 2.1 在 TASKS 里填写要跑的 YAML 列表

打开 `placement_test/run_test_batch.sh`，编辑其中的 `TASKS=( ... )` 数组。

**注意**：数组里填写的是 YAML 文件名的 basename（不带 `.yaml`），例如：

- `maniskill_openvla-envnum96-env03-rollout47-actor07-pipelinestage2`
- `robotwin_beat_block_hammer_grpo_openvlaoft-envnum384-env07-rollout07-actor07-pipelinestage1`

### 2.2 单机运行

```bash
bash placement_test/run_test_batch.sh
```

脚本会为每个任务循环执行：

- 强清理残留进程（`ray stop --force`、`pkill python/ray`、清 `/dev/shm/ray/*`）
- 启动 Ray head（`ray_utils/start_ray.sh`）
- 调用 `placement_test/run_placement_test.sh <yaml_basename>` 执行单次训练/测量
- 任务间再次清理，避免互相污染

### 2.3 多机运行（rank 0 + worker）

同一个脚本支持 worker 节点，通过 `RANK` 环境变量区分：

- head 节点：

```bash
export RANK=0
bash placement_test/run_test_batch.sh
```

- 每个 worker 节点：

```bash
export RANK=1
bash placement_test/run_test_batch.sh
```

`run_test_batch.sh` 会用仓库根目录下的 `task_sync.txt` 作为信号文件：rank 0 写入当前正在跑的 YAML 名称，worker 检测到后加入 Ray 集群并同步运行时环境。

## 3. 备注

- 如果你要测试不同模型/不同环境，核心原则是：**先写好基 YAML**（包含你想要的 env/model/algorithm），然后用生成器只改 envnum、pipeline_stage_num 与 component placement。
- 如果你要扩展更多 placement pattern（不止 `env07/03/01`、`rollout07/47/27`），可以在 `placement_test/generate_yaml_batch.py` 的 `PLACEMENTS` 字典里新增一个映射。

