# Tianji HIL 文档总览（`RLinf-tianji-hil`）

本目录汇总 `RLinf-tianji-hil`（分支 `tianji_hil`）围绕 Tianji Marvin Pro 双臂系统的工程结构、与 `LoopInfra`（分支 `tianji_hil`）的职责边界，以及四大核心工作流的操作与维护指南。

## 职责边界：`RLinf-tianji-hil` vs `LoopInfra`

| 维度 | `RLinf-tianji-hil` | `LoopInfra` |
| --- | --- | --- |
| 核心职责 | VLA/RLT 推理、Online RL/BC 训练、HIL 控制面板（`:8879`）、分阶段提示词与轨迹归档（`episodes/`） | 机械臂/夹爪 SDK 控制、相机与 Xense 触觉同步采集、PICO 遥操接管仲裁、`/api/policy_schedule` 执行队列 |
| 通信方式 | HTTP 客户端（[tianji_backend.py](../../rlinf/envs/realworld/tianji/tianji_backend.py)）+ 本机面板服务（[tianji_hil.py](../../rlinf/envs/realworld/tianji/tianji_hil.py)） | HTTP 采集器服务（`127.0.0.1:8780`）+ 数据质检服务（`127.0.0.1:8790`） |
| 运行环境 | 训练/推理环境（如 `miniconda3/envs/train` + `.venv-gpu`） | 采集器独立环境（`miniconda3/envs/loopinfra`，启动时需 `env -u PYTHONPATH` 隔离） |
| 协同分支 | `RLinf-tianji-hil@tianji_hil` | `LoopInfra@tianji_hil` |

> 详细协议见 [interface_contract.md](interface_contract.md)。

---

## 四大核心工作流速查矩阵

| 工作流 | 启动入口（代码 / 脚本） | 核心配置（`examples/embodiment/config/`） | `hil_mode` 与关键开关 | `LoopInfra` 交互端点 |
| --- | --- | --- | --- | --- |
| **1. 模型推理** | [eval_embodied_agent.py](../../evaluations/eval_embodied_agent.py)（或 `python -m rlinf.envs.realworld.tianji.stage1`） | [tianji2_joint_stage1.yaml](../../examples/embodiment/config/tianji2_joint_stage1.yaml) / [tianji2_rlt_stage2_infer.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_infer.yaml) | `hil_mode=inference`（或 `stage1`），`only_eval=true`，`deploy_actor_updates=false` | `GET /api/packet?tianji_stage1=1`、`POST /api/policy_schedule`、`POST /api/robot` |
| **2. HIL 数据收集** | `TianjiEnv` + LoopInfra collector UI（`:8780/collector`）或 RLinf 归档模式 | [realworld_tianji.yaml](../../examples/embodiment/config/env/realworld_tianji.yaml) | `hil_mode=training`，`episode_archive=true`，PICO 接管由 LoopInfra 原生处理 | `GET /api/packet?after=<id>`、`POST /api/hil-recording`、`GET /api/tianji/tactile*` |
| **3. DAgger 数据收集** | `examples/embodiment/train_async.py`（当前无独立入口；需按安全参数运行） | [tianji2_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_async.yaml) | `hil_mode=training`，`episode_archive=true`，**`runner.deploy_actor_updates=false`**（保持执行 VLA，仅采集 H/P 修正轨迹） | `/api/packet`、`/api/policy_schedule`、`/api/tianji/execution`、`/api/hil-recording` |
| **4. Online RL** | `examples/embodiment/train_async.py`（`AsyncEmbodiedRunner` 2-GPU 异步） | [tianji2_rlt_stage2_async.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_async.yaml) / [tianji2_rlt_stage2_tactile.yaml](../../examples/embodiment/config/tianji2_rlt_stage2_tactile.yaml) | `hil_mode=training`，`only_eval=false`，`deploy_actor_updates=true`，`stream_chunks=true`，`rtc_guidance=true` | `/api/policy_schedule` + `/api/policy_schedule/row`（实测执行标签）、`/api/robot` |

> [!IMPORTANT]
> 当前 `RLinf-tianji-hil` 的 `tianji_hil` 提交（`b68acff`）存在 **缺失运行模块** 与 **`sparse-checkout` 未检出 `examples/embodiment/train_async.py` 等文件** 的问题；在修复这些分支缺陷前，请勿直接用本仓库替代运行中的 `/home/ubuntu/Stage2-30HZ-*` 现场部署。完整问题清单与修复路径见 [repo_management.md](repo_management.md)。

---

## 文档导航

1. [interface_contract.md](interface_contract.md)：`RLinf-tianji-hil` 与 `LoopInfra` 的 HTTP API、数据 Schema、内部 IPC、环境变量与紧耦合点。
2. [workflows.md](workflows.md)：模型推理、HIL 数据收集、DAgger 数据收集、Online RL 四大流程操作手册、`:8879` 面板状态机与 `episodes/` 归档格式。
3. [repo_management.md](repo_management.md)：双仓库协同版本管理、`Stage2-30HZ-*` 现状分析、已知 `b68acff` 分支缺陷与代码修复清单。
4. [shadow_monitor.md](shadow_monitor.md)：`rlinf/tianji_shadow` 后台 Monitor 与离线复算（`replay` / `replay1116`）使用说明。
5. [deploy_template.md](deploy_template.md) & [site.env.example](site.env.example)：去硬编码的部署环境模板与变量说明。
