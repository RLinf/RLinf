# LeRobot Norm Stats 快速计算规格

## 1. 背景

当前 `toolkits/lerobot/calculate_norm_stats.py` 通过 OpenPI 通用
`LeRobotDataset` 和 PyTorch DataLoader 遍历数据。在图像直接存储在
Parquet 中的数据集上，这条路径会读取并解码每帧图像，但最终归一化
统计只使用变换后的 `state` 和 `actions`。

当前目标数据集 `data/lerobot-data_mixed_8_v30` 约为 273 GB，包含
423,922 帧和三路图像。只投影读取 `observation.state`、`action`
和 `episode_index` 的全量实测约读取 56 MB，用时约 26 秒。

## 2. 目标

新增 `toolkits/lerobot/calculate_norm_stats_fast.py`，为 RLinf 中的 OpenPI
LeRobot 数据配置提供通用的快速 norm stats 计算路径。

必须满足：

- 不读取或解码真实图像、视频数据。
- 保留 OpenPI dataconfig 定义的 repack 和 data transform 语义。
- 支持单个或多个 `action_sequence_keys`。
- 保留 LeRobot 的 action horizon 和 episode 边界 padding 语义。
- 统计全部帧，包括不足一个 batch 的最后一批。
- 默认写入独立的 `norm_stats_fast.json`，不与旧脚本正在生成的
  `norm_stats.json` 冲突。
- 输出 JSON 结构与现有 OpenPI norm stats 格式兼容。
- 兼容 LeRobot v2/v3 本地数据集及已缓存的 Hugging Face repo id。

## 3. 非目标

- 不替换或删除现有 `calculate_norm_stats.py`。
- 不支持 RLDS 数据集。
- 不保证支持通过真实像素内容计算 `state` 或 `actions` 的自定义
  transform。
- 不引入抽样或近似统计；默认始终扫描全量数据。
- 本规格阶段不修改任何代码、脚本或文档。

## 4. 命令行接口

快速脚本保持与现有脚本一致的必需参数：

```bash
python toolkits/lerobot/calculate_norm_stats_fast.py \
    --config-name pi05_aloha_robotwin \
    --repo-id /path/to/lerobot_dataset
```

- `--config-name`：传给 `get_openpi_config()` 的 OpenPI 配置名。
- `--repo-id`：本地 LeRobot 数据集目录，或位于 `HF_LEROBOT_HOME`
  下的 repo id。
- `--output-path`：可选的输出文件路径。默认为解析后数据集根目录下的
  `norm_stats_fast.json`。
- `--overwrite`：可选开关，默认为 `False`。仅显式指定时才允许覆盖
  `--output-path` 指向的已有文件。
- worker 数量默认沿用训练配置中的 `num_workers`，不新增必需参数。
- 使用 `normalize.serialize_json()` 产生与 `normalize.save()` 相同的 JSON
  结构，但允许写入自定义文件名。
- 输出先写入同目录临时文件，完整序列化成功后再原子更名，避免中断后
  留下不完整 JSON。

快速脚本默认不会覆盖旧脚本的结果。OpenPI 默认仍只会自动加载正式名称
`norm_stats.json`。快速结果完成数值对照后，应由用户显式更新训练配置的
`norm_stats_path`，或手动将通过验证的文件提升为 `norm_stats.json`。快速脚本
不自动进行该替换。

## 5. 设计

### 5.1 配置和数据集解析

1. 通过 `resolve_lerobot_dataset_root()` 解析本地路径或 repo id。
2. 检查 `meta/info.json` 和 `data/` 存在，否则返回包含实际解析路径的
   `FileNotFoundError`。
3. 通过 `get_openpi_config(config_name, repo_id=repo_id)` 构建真实
   `DataConfig`，获取 action horizon、batch size、worker 数、
   `action_sequence_keys`、repack transforms 和 data transforms。
4. 如果配置指向 RLDS，立即返回明确的不支持错误。

### 5.2 Parquet 列投影

1. 通过 `meta/info.json` 特征定义识别 image/video 特征。
2. 扫描 `data/**/*.parquet`，只投影：
   - 全部非视觉特征；
   - `index`、`episode_index`、`task_index` 等定位和 prompt 构造所需特征；
   - `action_sequence_keys` 引用的所有动作列。
3. 根据全局 `index` 恢复确定性顺序，不依赖文件系统 glob 顺序。
4. 检查全局 index 唯一且连续，episode 内帧连续，同一 episode
   不能分裂成多个不相邻区间。

投影阶段不得将 image/video 列传给 PyArrow 读取器。

### 5.3 Action horizon 与 episode padding

对每个 anchor frame 和每个 action key：

1. 构造 `[0, 1, ..., action_horizon - 1]` 偏移。
2. 将偏移后的索引限制在当前 episode 内。
3. 超过 episode 末尾的位置重复最后一帧动作，不允许跨 episode
   取数。
4. 多个 action key 分别构造 horizon，然后交给现有 transform 合并或重排。

实现应使用 NumPy 索引批量构造 chunk，不应对 horizon 内每个
step 进行 Python 循环。

### 5.4 复用现有 transforms

1. 从投影后的非视觉列构造与 LeRobot sample 键名一致的输入。
2. prompt 优先使用数据集原始字段；若 `prompt_from_task=True`，则使用
   `task_index` 和数据集 task metadata 还原。
3. 为 repack transform 要求的视觉键提供形状兼容的共享占位数组，
   不读取原始像素。占位数组仅用于让现有 transform 完成键名和
   形状处理，不进入统计。
4. 按旧脚本相同顺序应用：
   - `data_config.repack_transforms.inputs`；
   - `data_config.data_transforms.inputs`；
   - 删除字符串输出。
5. 只提取变换后的 `state` 和 `actions`。

若 transform 缺少非视觉输入、产生不稳定维度，或数值输出依赖真实
像素，快速脚本应终止并给出可操作错误，提示用户改用
`calculate_norm_stats.py`。不得静默计算可能错误的 stats。

### 5.5 统计聚合

1. 使用 `openpi.shared.normalize.RunningStats`。
2. 按配置 batch size 将变换后的 sample 聚合为 batch，保持确定性顺序。
3. `state` 每个 anchor frame 计数一次。
4. `actions` 保留 `[batch, horizon, action_dim]` 形状交给
   `RunningStats.update()`，由其按旧语义展平 horizon。
5. 最后一个不完整 batch 也必须更新 stats。这与旧脚本的
   `drop_last=True` 不同，是有意修正。
6. 调用 `get_statistics()` 生成 `mean`、`std`、`q01`和 `q99`。

## 6. 性能要求

- 性能收益必须主要来自减少 I/O 和图像解码，不得以抽样换取速度。
- 对目标数据集，Parquet 读取量应保持在非视觉列的量级，不应随
  273 GB 图像数据规模线性增长。
- 数据扫描和变换应有独立的进度信息，至少显示已处理帧数、总帧数
  和阶段名称。
- 避免构造整个 `[num_frames, horizon, action_dim]` 数组；动作
  chunk 应分批或按 episode 构造，限制峰值内存。

## 7. 错误处理

以下情况必须在写出前失败：

- 数据集路径或 `meta/info.json` 不存在。
- 未发现 Parquet 数据文件。
- dataconfig 没有 repo id 或指向 RLDS。
- 缺少 action key、state 所需原始特征、episode/index 字段。
- 存在重复/缺失全局 index，或 episode 帧不连续。
- transform 后缺少 `state` 或 `actions`，或各 sample 维度不一致。
- 数据集少于两个可统计向量。
- 输出文件已存在且未显式传入 `--overwrite`。

错误应包含 config 名、repo id/解析路径、缺失键或失败 transform 的类名。

## 8. 代码接入

实现完成后：

- 保留 `toolkits/lerobot/calculate_norm_stats.py` 不变，用于兼容特殊 transform
  和作为数值参考。
- 将 `tmp/merge_data.sh` 的调用切换为
  `toolkits/lerobot/calculate_norm_stats_fast.py`，其余参数不变。
- 在英文和中文 `sft_openpi.rst` 归一化统计章节中同步推荐快速脚本，
  说明它的全量统计、列投影、默认输出名和兼容边界。

## 9. 测试与验收

### 9.1 单元测试

- 单 episode 的 horizon 构造和末尾重复 padding。
- 多 episode 时不跨边界取动作。
- 多 `action_sequence_keys` 维持各自维度和顺序。
- 多 Parquet 文件顺序打乱后按 `index` 恢复正确帧序。
- 最后一个不完整 batch 被统计。
- 图像特征不出现在 PyArrow 投影列表中。
- 默认生成 `norm_stats_fast.json`，不改动已有 `norm_stats.json`。
- 已有输出在没有 `--overwrite` 时被拒绝，显式覆盖时使用原子写入。
- 缺列、不连续 index、不支持 transform 产生明确错误。

### 9.2 数值对照

- 构造小型合成 LeRobot 数据集，对照快速路径与旧 DataLoader 路径
  变换后的 `state` 和 `actions`。
- 对于帧数可被 batch size 整除的数据，比较 mean/std/q01/q99，
  差异必须仅限于 NumPy 浮点聚合误差。
- 对于不可整除的数据，单独验证快速路径的结果包含尾批，不要求与
  旧脚本相等。

### 9.3 性能验收

- 在 `data/lerobot-data_mixed_8_v30` 上记录扫描用时、总用时、处理帧数和
  峰值内存。
- 确认无图像解码调用，且投影读取规模与非视觉列大小一致。
- 快速脚本应显著快于旧脚本；如果变换或 histogram 成为新瓶颈，
  应在不改变统计语义的前提下分批或并行化。

### 9.4 质量门禁

- 运行 Ruff lint/format 检查和相关 pytest。
- 若更新 EN/ZH 文档，运行两种语言的 Sphinx build，并运行
  `.codex/skills/docs-check/check_rst_markup.py`。
- 文档中的脚本路径、CLI 参数和 config 名必须与实际代码一致。

## 10. 完成标准

以下条件全部满足时视为实现完成：

1. 快速脚本可在目标 ALOHA 数据集上生成 OpenPI 可加载的
   `norm_stats_fast.json`，且不修改已有 `norm_stats.json`。
2. 标准 RLinf OpenPI LeRobot dataconfig 可复用各自的非像素依赖
   transforms，无需在快速脚本中写死 ALOHA 变换公式。
3. action horizon、episode padding、多 action key 和尾批行为通过测试。
4. Parquet 扫描不读取图像/视频列。
5. 性能相比旧路径有明显改善，且无抽样导致的统计偏差。
6. `tmp/merge_data.sh` 和 EN/ZH 文档已同步，所有相关检查通过。
