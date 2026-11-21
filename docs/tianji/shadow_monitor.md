# 后台 Monitor（`rlinf.tianji_shadow`）使用与配置说明

`rlinf/tianji_shadow` 是随 Stage2 Joint 联合运行的可选后台监控侧车模块，在独立 GPU（默认 `CUDA_VISIBLE_DEVICES=1`）上复用当前 Policy 的 `2048` 维 RLT 特征与 `16` 维关节状态，实时输出子阶段完成候选、风险 H 接管候选与超时告警，并在轨迹结束后生成人工 vs 模型对照复核日志。

> [!NOTE]
> 原始的 `tianji2` 工作站运维记录保留在 [rlinf/tianji_shadow/README_zh.md](../../rlinf/tianji_shadow/README_zh.md)（其中包含特定机器 `/home/xys/...` 与 `/data1/rqh/...` 历史路径）。新机器部署请统一使用下方的环境变量配置，勿硬编码用户目录。

---

## 核心环境变量（去硬编码模板）

| 环境变量 | 默认值 / 说明 |
| --- | --- |
| `TIANJI_STAGE2_SHADOW_VARIANT` | 启用侧车监控的入口标识名称（由启动器传入 `python -m rlinf.tianji_shadow supervise <VARIANT> ...`） |
| `TIANJI_SHADOW_ASSETS` | Monitor 权重、归一化统计与 `identity` 所在目录（代码默认 `$HOME/tianji_monitor_assets/shadow1113`，新机请显式导出） |
| `TIANJI_SHADOW_GPU` | 运行 Monitor 编码器的 GPU 序号（默认 `1`） |
| `TIANJI_SHADOW_VERSION` | 规则版本（如 `1116v1`，影响单点/双点确认阈值与默认开启状态） |
| `TIANJI_SHADOW_DEFAULT` | 设为 `on` 时每条轨迹默认开启 Monitor；否则可在 `8879` 面板等待开始时按 `Y` / `N` 切换 |
| `TIANJI_SHADOW_COLLECTOR` | 采集器 HTTP 地址（默认 `http://127.0.0.1:8780`），用于拉取只读 Xense 触觉帧 |
| `TIANJI_RLT_OUTPUT` | 当前运行输出根目录，Monitor 日志写入 `$TIANJI_RLT_OUTPUT/monitor_shadow/` 与 `monitor_shadow_runtime.log` |

---

## 离线重算命令（通用模板）

使用部署的 `RLinf` 根目录与 GPU Python 环境执行离线重算（无需依赖硬编码路径）：

```bash
export PYTHONPATH="/path/to/RLinf-tianji-hil"
export RUN_DIR="/path/to/runs/<run_id>"

# 1113 规则重算
CUDA_VISIBLE_DEVICES=1 /path/to/gpu-python -B \
  -m rlinf.tianji_shadow replay "$RUN_DIR/monitor_shadow" \
  --device cuda:0 > /tmp/monitor_replay.jsonl

# 1116v1 规则重算（支持 --thresholds 试算自定义阈值 JSON）
CUDA_VISIBLE_DEVICES=1 /path/to/gpu-python -B \
  -m rlinf.tianji_shadow replay1116 \
  --output /tmp/shadow1116-replay \
  "$RUN_DIR/monitor_shadow/<episode_dir>"
```

## 轨迹输出目录结构

每条开启监控的轨迹保存在 `$TIANJI_RLT_OUTPUT/monitor_shadow/episode_<id>_<monotonic_ns>/`：
- `identity.json`：Monitor 源权重校验、规则阈值快照（`rule_profiles`）与当次 Policy 路径。
- `human.jsonl`：HIL 面板确认的子阶段切分（含回退重入）、`H`/`P`、暂停/恢复与结果。
- `observations.jsonl` + `obs_*.npz`：每步 RLT 特征、16D 关节、5 点因果窗口、触觉掩码与拒绝原因。
- `predictions.jsonl` & `comparison.json`：独立/对照阶段推进预测、模型 H 候选与人工锚点对比摘要。
