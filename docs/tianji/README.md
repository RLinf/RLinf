# Tianji HIL 文档总览（`RLinf@tianji_hil`）

本目录汇总 `RLinf`（分支 `tianji_hil`）围绕 Tianji Marvin Pro 双臂系统的工程结构、与 `LoopInfra`（分支 `tianji_hil`）的职责边界，以及四大核心工作流的操作与维护指南。

## 职责边界：`RLinf` vs `LoopInfra`

| 维度 | `RLinf` | `LoopInfra` |
| --- | --- | --- |
| 核心职责 | VLA/RLT 推理、Online RL/BC 训练、HIL 控制面板（`:8879`）、分阶段提示词与轨迹归档（`episodes/`） | 机械臂/夹爪 SDK 控制、相机与 Xense 触觉同步采集、PICO 遥操接管仲裁、`/api/policy_schedule` 执行队列 |
| 通信方式 | HTTP 客户端（[tianji_backend.py](../../rlinf/envs/realworld/tianji/tianji_backend.py)）+ 本机面板服务（[tianji_hil.py](../../rlinf/envs/realworld/tianji/tianji_hil.py)） | HTTP 采集器服务（`127.0.0.1:8780`）+ 数据质检服务（`127.0.0.1:8790`） |
| 运行环境 | 训练/推理环境（`RLINF_PYTHON`，通过 `site.env` 配置） | 采集器独立环境（`LOOPINFRA_PYTHON` / `$LOOPINFRA_ROOT/.loopinfra-python`，启动时 `env -u PYTHONPATH` 隔离） |
| 协同分支 | `RLinf@tianji_hil` | `LoopInfra@tianji_hil` |

> 详细协议见 [interface_contract.md](interface_contract.md)；版本化启动器见 [deploy/tianji/README.md](../../deploy/tianji/README.md)。

---

## 四大核心工作流速查矩阵

| 工作流 | 启动入口（代码 / 脚本） | 核心配置（`examples/embodiment/config/`） | `hil_mode` 与关键开关 | `LoopInfra` 交互端点 |
| --- | --- | --- | --- | --- |
| **1. 模型推理** | [eval_embodied_agent.py](../../evaluations/eval_embodied_agent.py) / [run_tianji_stage1_local.sh](../../deploy/tianji/run_tianji_stage1_local.sh) / [run_tianji_stage2_infer_local.sh](../../deploy/tianji/run_tianji_stage2_infer_local.sh) / [stage2_30hz_1118v1.sh](../../deploy/tianji/stage2_30hz_1118v1.sh) (`--run`) | [tianji2_joint_stage1.yaml](../../examples/embodiment/config/tianji2_joint_stage1.yaml) / [tianji2_rlt_stage2_infer.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_infer.yaml) | `hil_mode=inference`（或 `stage1`），`only_eval=true`，`deploy_actor_updates=false` | `GET /api/packet?tianji_stage1=1`、`POST /api/policy_schedule`、`POST /api/robot` |
| **2. HIL 数据收集** | `TianjiEnv` + LoopInfra collector UI（`:8780/collector`）或 RLinf 归档模式 | [realworld_tianji.yaml](../../examples/embodiment/config/env/realworld_tianji.yaml) | `hil_mode=training` 或 `dagger`，`episode_archive=true`，PICO 接管由 LoopInfra 原生处理 | `GET /api/packet?after=<id>`、`POST /api/hil-recording`、`GET /api/tianji/tactile*` |
| **3. DAgger 数据收集** | [run_tianji_dagger_local.sh](../../deploy/tianji/run_tianji_dagger_local.sh) / [stage2_30hz_1118v1.sh](../../deploy/tianji/stage2_30hz_1118v1.sh) (`--dagger`) / [train_async.py](../../examples/embodiment/train_async.py) | [tianji_dagger_collect.yaml](../../examples/embodiment/config/tianji_dagger_collect.yaml) | **`hil_mode=dagger`**，`episode_archive=true`，`tactile_raw_archive=true`，`discard_episodes=true`，`runner.deploy_actor_updates=false`，RL 更新与 replay 落盘关闭 | `/api/packet`、`/api/policy_schedule`、`/api/tianji/execution`、`/api/hil-recording` |
| **4. Online RL** | [start_tianji_stage2_async.sh](../../deploy/tianji/start_tianji_stage2_async.sh) / [resume_tianji_stage2_async.sh](../../deploy/tianji/resume_tianji_stage2_async.sh) / [train_async.py](../../examples/embodiment/train_async.py) | [tianji2_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_async.yaml) / [tianji2_rlt_stage2_tactile.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_tactile.yaml) | `hil_mode=training`，`only_eval=false`，`deploy_actor_updates=true`，`stream_chunks=true`，`rtc_guidance=true` | `/api/policy_schedule` + `/api/policy_schedule/row`（实测执行标签）、`/api/robot` |

> [!NOTE]
> `RLinf@tianji_hil` 已完成缺失模块恢复、与上游 `main` 的兼容性合并、`LOOPINFRA_ROOT`（统一默认 `~/LoopInfra`）、`hil_mode="dagger"` 解耦、6/15 子阶段参数化以及 `deploy/tianji/` 归档。`tianji1` 现场既有的 `~/Stage2-30HZ-*` 桌面目录仍保持原状，待操作员验证后按需切换。

---

## 文档导航

1. [interface_contract.md](interface_contract.md)：`RLinf` 与 `LoopInfra` 的 HTTP API、数据 Schema、内部 IPC、环境变量与耦合点。
2. [workflows.md](workflows.md)：模型推理、HIL 数据收集、DAgger 数据收集、Online RL 四大流程操作手册、`:8879` 面板状态机与 `episodes/` 归档格式。
3. [repo_management.md](repo_management.md)：双仓库协同版本管理、`Stage2-30HZ-*` 现状、已完成修复总结与切换建议。
4. [shadow_monitor.md](shadow_monitor.md)：`rlinf/tianji_shadow` 后台 Monitor 与离线复算（`replay` / `replay1116`）使用说明。
5. [deploy_template.md](deploy_template.md) & [site.env.example](site.env.example)：去硬编码的部署环境模板与变量说明（与 [deploy/tianji/site.env.example](../../deploy/tianji/site.env.example) 保持同步）。
