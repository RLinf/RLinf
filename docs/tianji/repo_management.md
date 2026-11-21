# 项目管理与分支治理指南（`RLinf-tianji-hil` + `LoopInfra`）

本文档记录 `RLinf-tianji-hil` 与 `LoopInfra` 的协同版本管理策略、现场部署目录（`/home/ubuntu/Stage2-30HZ-*`）现状分析，以及当前 `tianji_hil` 分支（`b68acff`）经只读代码审计发现的**已知缺陷与修复行动清单**。

---

## 1. 现场部署现状分析（`Stage2-30HZ-*` vs Git 仓库）

在 `tianji1` 上同时存在四个相关目录：
- `/home/ubuntu/RLinf-tianji-hil`：Git 仓库（`origin: git@github.com:louisreided793-hub/RLinf.git`，分支 `tianji_hil` @ `b68acff`）。
- `/home/ubuntu/LoopInfra`：Git 仓库（`origin: git@github.com:louisreided793-hub/LoopInfra.git`，分支 `tianji_hil` @ `61f4b58`）。
- `/home/ubuntu/Stage2-30HZ-1118v1` 与 `/home/ubuntu/Stage2-30HZ-sub6-1118v1`：**当前正在使用的实机部署目录**（**非废弃目录**），桌面快捷方式 `/home/ubuntu/Desktop/Stage2-30HZ-*.desktop` 直接指向其中的 `launcher/stage2_30hz_1118v1.sh --run`。

### 现状痛点
1. **无版本快照拷贝**：`Stage2-30HZ-1118v1/` 内包含无 `.git` 的裁剪版 `RLinf/` 与 `LoopInfra_rtj/`（其 `third_party` 软链指向 `/home/ubuntu/LoopInfra/third_party`），且 `Stage2-30HZ-sub6-1118v1/site.env` 跨目录引用了 `~/Stage2-30HZ-1118v1/LoopInfra_rtj` 与 `.runtime/`。
2. **启动脚本未入库**：`RLinf-tianji-hil` 仓库内没有任何 Tianji 启动脚本；所有实际运行的脚本（约 50 个）全部留在 `Stage2-30HZ-1118v1/launcher/`，其中仅约 5 个服务于当前小车 `1118v1` 部署，其余约 45 个为历史 `tianji2` 工作站遗留脚本（`r4e`, `r4p`, `r5a`, `r5b`, `v8h` 等）。
3. **硬编码与路径漂移**：
   - [rlinf/tianji_shadow/README_zh.md](../../rlinf/tianji_shadow/README_zh.md)、[settings.py](../../rlinf/tianji_shadow/settings.py) 与 [__main__.py](../../rlinf/tianji_shadow/__main__.py) 中残留 `tianji2` 工作站的 `/home/xys/rlt_test_rtj/...`、`/home/xys/tianji_monitor_assets/...`、`/data1/rqh/...` 路径。
   - [tianji_joint.py](../../rlinf/envs/realworld/tianji/tianji_joint.py) 默认 `LOOPINFRA_ROOT=~/LoopInfra_joint`，而 [tianji_backend.py](../../rlinf/envs/realworld/tianji/tianji_backend.py) 与 [tianji_episode_archive.py](../../rlinf/envs/realworld/tianji/tianji_episode_archive.py) 默认 `LOOPINFRA_ROOT=~/LoopInfra_rtj`。

---

## 2. 已知分支缺陷与代码修复清单（Known Branch Defects & Action Plan）

在 `2026-11-21` 的只读静态与 AST 导入审计中，发现 `/home/ubuntu/RLinf-tianji-hil` 的 `tianji_hil` 分支（单次压缩提交 `b68acff`，基于 `main` `dcf00fd`）存在以下阻塞性问题，**必须完成修复后才能替代 `Stage2-30HZ-*` 运行**：

### 缺陷 A：`b68acff` 遗漏的必需模块（`Stage2-30HZ-1118v1/RLinf` 中存在但 `tianji_hil` Git 树缺失）
以下文件被 `tianji_hil` 修改过的代码直接导入，但在当前 Git 提交树中不存在（`git ls-tree` 返回缺失）：
- `rlinf/utils/tianji_startup.py`（被 `rlinf/runners/async_embodied_runner.py` 顶层强制导入，`embodied_runner.py` 运行时导入）
- `rlinf/utils/xense_tactile.py`（被 `tianji_env.py`、`tianji_backend.py`、`rlinf/algorithms/rlt/rollout.py` 导入）
- `rlinf/utils/rtc.py`（被 `rlinf/envs/realworld/tianji/stage1/async_runner.py` 导入）
- `rlinf/models/embodiment/openpi/rtc_guidance.py`（被 `openpi_rlinf/eval_action_model.py`、`stage1/predictor.py`、`stage1/rtc_sampler.py` 导入）
- `rlinf/algorithms/rlt/action_grid.py`、`direct_human_bc.py`、`human_bc.py`（被 `rlinf/config.py`、`fsdp_rlt_ac_policy_worker.py`、`env_worker.py` 导入）
- `rlinf/data/schema/embodied_trajectory_builder.py`（被 `env_worker.py` 导入）
- `rlinf/models/embodiment/mlp_policy/tactile_encoder.py` 与 `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` 的 Stage2 扩展（当前仓库中的 `rlt_mlp_policy.py` 停留在 `main` 版本，缺少 `gaussian_smoother`、`residual_time_operator`、`TactileEncoder`、`critic_penalty_head`、`critic_block_encoders` 等配置字段支持）
- `rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py`（被 `openpi_rlinf/utils/rlt_utils.py` 导入）

### 缺陷 B：部分工作区因 `core.sparsecheckout=true` 未检出文件
- `/home/ubuntu/RLinf-tianji-hil` 使用了 `--filter=blob:none` 与稀疏检出（`.git/info/sparse-checkout`）。
- 导致已存在于 Git 树中的关键文件未落盘，包括：`examples/embodiment/train_async.py`、`rlinf/algorithms/rlt/{route,transition}.py`、`rlinf/data/schema/embodied_types.py`、`rlinf/models/embodiment/modules/{rlt_token_transformer,value_head}.py`、`rlinf/envs/wrappers/`、`rlinf/workers/env/rtc_env_worker.py`、`rlinf/runners/embodied_eval_runner.py`。

### 缺陷 C：压缩提交回退了上游 `main` 的目录重构
- `b68acff` 覆盖了 [rlinf/envs/__init__.py](../../rlinf/envs/__init__.py)（删除了 `rlinf/envs/sim/` 前缀并把 `REAL` 改回 `REALWORLD`）、[rlinf/config.py](../../rlinf/config.py)、[evaluations/eval_embodied_agent.py](../../evaluations/eval_embodied_agent.py)、[rlinf/runners/async_embodied_runner.py](../../rlinf/runners/async_embodied_runner.py)（移除 `AsyncWeightSyncMixin`）、[rlinf/models/embodiment/openpi/dataconfig/__init__.py](../../rlinf/models/embodiment/openpi/dataconfig/__init__.py) 等文件，导致上游仿真环境与部分工作流导入损坏。

---

## 3. 推荐的后续治理路线图

1. **阶段 1（代码修复与完整性校验）**：
   - 在获得确认后，从 `Stage2-30HZ-1118v1/RLinf` 补齐上述 10+ 缺失模块，并将 Tianji 改动干净地迁移/变基到 `main` 之上，避免破坏 `rlinf/envs/sim/` 与 `AsyncWeightSyncMixin`。
   - 对实际部署使用的 `RLinf-tianji-hil` 检出关闭 `sparse-checkout`（`git sparse-checkout disable`），确保 `examples/embodiment/train_async.py` 与 `rlinf/` 全量可用。
   - 在 CI / 本地预检中加入最小导入冒烟检查（验证四大工作流入口均无 `ModuleNotFoundError`）。
2. **阶段 2（双仓库协同与去快照化）**：
   - 将 `Stage2-30HZ-1118v1/launcher/` 中仍有效的通用启动脚本（`stage2_30hz_1118v1.sh`、`tianji_schedule_env.py`、`run_tianji_wuji_collector.sh` 等）整理迁入仓库版本控制，清理约 45 个 `tianji2` 遗留脚本。
   - 机器差异（Python 路径、GPU 分配、模型缓存路径、复位位姿 ID、`LOOPINFRA_ROOT`）统一收口到未跟踪的 `site.env`（模板见 [site.env.example](site.env.example)）。
   - 统一 `LOOPINFRA_ROOT` 默认行为并消除 `tianji_joint.py` 的 `sys.path` 硬导入（改为 HTTP FK 接口或独立轻量化包）。
   - 新增专用的 DAgger 数据收集配置（例如 `tianji_dagger_collect.yaml`），使 `episode_archive=true` 与 `deploy_actor_updates=false` 成为显式的一等模式。
3. **阶段 3（桌面入口切换）**：
   - 当上述修复与无动作/真机自检全部通过后，再将桌面 `.desktop` 快捷方式切换至统一版本化工作区，归档 `/home/ubuntu/Stage2-30HZ-*` 临时目录。
