# Tianji HIL Integration

本分支（`tianji_hil`）承载 Tianji Marvin Pro 双臂实机部署与训练集成，覆盖 **模型推理、HIL 数据收集、DAgger 数据收集、Online RL** 四大核心流程，并与独立仓库 `LoopInfra`（同样位于 `tianji_hil` 分支）通过 HTTP/二进制协议协同运行。

- 完整文档入口与四流程总览：[docs/tianji/README.md](tianji/README.md)
- `RLinf-tianji-hil` ↔ `LoopInfra` 接口契约：[docs/tianji/interface_contract.md](tianji/interface_contract.md)
- 四大流程运行手册与 HIL 面板说明：[docs/tianji/workflows.md](tianji/workflows.md)
- 仓库协同管理与已知分支缺陷清单：[docs/tianji/repo_management.md](tianji/repo_management.md)
- 后台 Monitor（`tianji_shadow`）指南：[docs/tianji/shadow_monitor.md](tianji/shadow_monitor.md)
- 部署模板与环境变量示例：[docs/tianji/deploy_template.md](tianji/deploy_template.md) / [docs/tianji/site.env.example](tianji/site.env.example)

> [!NOTE]
> `RLinf-tianji-hil` 负责策略推理、RL/BC 训练、HIL 网页控制面板（`:8879`）与轨迹归档；`LoopInfra` 负责底层机器人 SDK、相机/触觉同步采集、PICO 遥操接管仲裁与运动队列执行。当前 `tianji_hil` 分支存在部分待修复的缺失模块与 `sparse-checkout` 状态，详见 [repo_management.md](tianji/repo_management.md)。
