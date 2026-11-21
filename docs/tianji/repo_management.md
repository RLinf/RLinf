# 项目管理与分支治理指南（`RLinf@tianji_hil` + `LoopInfra@tianji_hil`）

本文档记录 `RLinf` 与 `LoopInfra` 在 `tianji_hil` 分支上的协同版本管理策略、现场部署目录（`/home/ubuntu/Stage2-30HZ-*`）现状，以及 `tianji_hil` 分支的修复完成状态与后续切换规范。

---

## 1. 双仓库协同与 `Stage2-30HZ-*` 现场现状

在实机工作站 `tianji1` 上同时存在以下目录：
- `/home/ubuntu/RLinf-tianji-hil`：Git 仓库（`origin: git@github.com:louisreided793-hub/RLinf.git`，分支 `tianji_hil`）。
- `/home/ubuntu/LoopInfra`：Git 仓库（`origin: git@github.com:louisreided793-hub/LoopInfra.git`，分支 `tianji_hil`）。
- `/home/ubuntu/Stage2-30HZ-1118v1` 与 `/home/ubuntu/Stage2-30HZ-sub6-1118v1`：**当前实机桌面的现有运行目录**，桌面快捷方式 `/home/ubuntu/Desktop/Stage2-30HZ-*.desktop` 指向其中的 `launcher/stage2_30hz_1118v1.sh --run`。

> [!IMPORTANT]
> `tianji1` 上现有的 `/home/ubuntu/Stage2-30HZ-1118v1` 与 `/home/ubuntu/Stage2-30HZ-sub6-1118v1` 桌面目录**保持原样未修改**，直到操作员完成实机验收并决定正式切换。

---

## 2. `RLinf@tianji_hil` 修复状态汇总

此前针对压缩提交 `b68acff` 审计出的缺口已完成全面修复：

1. **缺失运行模块已全量恢复**：
   - 补齐 `rlinf/utils/{tianji_startup.py, xense_tactile.py, rtc.py}`、`rlinf/models/embodiment/openpi/rtc_guidance.py`、`rlinf/models/embodiment/mlp_policy/tactile_encoder.py`、`rlinf/algorithms/rlt/{action_grid.py, direct_human_bc.py, human_bc.py}`、`rlinf/data/schema/embodied_trajectory_builder.py`、`rlinf/utils/ckpt_convertor/openpi/*` 等核心模块。
   - 同步补齐 `rlinf/__init__.py`、`rlinf/algorithms/rlt/{route.py, transition.py}`、`rlinf/data/schema/embodied_types.py`、`rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py`、`rlinf/workers/actor/fsdp_sac_policy_worker.py` 等 Stage2 扩展。
2. **上游 `origin/main` 回归已正向合并**：
   - 基于 `origin/main` 正向合入 Tianji HIL / Stage1 / Stage2 / DAgger 钩子，完整保留 `rlinf.envs.sim.*`、`SupportedEnvType.REAL` + `REALWORLD`/`TIANJI` 路由、`PSI0`/`EnvLayout`/校验器、`apxinf`/`phyai`/`sglang` 后端、`AsyncWeightSyncMixin`、`PolicyInput`/`EnvTransition` 及全部上游 OpenPI dataconfig。
3. **DAgger 一等支持与配置独立**：
   - `tianji_hil.py` 与 `tianji_env.py` 解耦支持 `hil_mode="dagger"`：在 `dagger` 模式下启用 `episode_archive`、`tactile_raw_archive` 与 `Discard`，同时关闭训练专用的里程碑奖励、`-1` 步失败惩罚与 `tianji_shadow`。
   - 新增专用配置 [examples/embodiment/config/tianji_dagger_collect.yaml](../../examples/embodiment/config/tianji_dagger_collect.yaml)（显式设置 `hil_mode: dagger`、`deploy_actor_updates: false`、关闭 RL 更新与 replay 持久化）。
4. **`LOOPINFRA_ROOT` 与 6/15 子阶段参数化**：
   - 在 `rlinf/envs/realworld/tianji/` 中通过统一的 `default_loopinfra_root()` 将 `LOOPINFRA_ROOT` 默认值收口为 `~/LoopInfra`（消除此前 `~/LoopInfra_joint` 与 `~/LoopInfra_rtj` 漂移）。
   - 将 `Stage2-30HZ-1118v1`（15 阶段）与 `Stage2-30HZ-sub6-1118v1`（6 阶段）的硬编码差异收口为 `TIANJI_CAR_NUM_SUBTASKS` 与 `TIANJI_CAR_BANNER`。
5. **版本化启动器归档至 `deploy/tianji/`**：
   - 将有效通用启动脚本与预检工具清理去硬编码后迁入 [deploy/tianji/](../../deploy/tianji/README.md)，丢弃约 45 个历史 `tianji2` 消融遗留脚本。
   - 新增本地 CPU 冒烟测试 [tests/unit_tests/test_tianji_hil_smoke.py](../../tests/unit_tests/test_tianji_hil_smoke.py)。

---

## 3. 实机部署与切换原则

1. **禁用稀疏检出**：实机部署检出 `RLinf` 时应使用完整工作树（确保 `core.sparsecheckout` 关闭），保证 `examples/embodiment/train_async.py` 与 `deploy/tianji/` 完整可用。
2. **机器差异收口到未跟踪的 `site.env`**：参见 [deploy_template.md](deploy_template.md) 与 [site.env.example](site.env.example)，勿将实机路径、Token 或硬件 SHA ID 提交进 Git。
3. **平滑切换桌面入口**：在完成 `bash deploy/tianji/check_tianji_local.sh --mode offline` 及实机 `--verify` 后，再通过 `stage2_30hz_1118v1.sh --install` 将桌面图标切换至版本化工作区，并对既有 `Stage2-30HZ-*` 临时目录进行归档。
