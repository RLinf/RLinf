# Tianji HIL 部署配置模板说明

本模板用于替代早期分散在 `Stage2-30HZ-*` 快照中的手写配置，将**版本化代码/配置模板**与**机器私有路径/ID**彻底分离。

> 说明：由于当前 `/home/ubuntu/RLinf-tianji-hil` 启用了 `core.sparsecheckout=true` 且顶层目录被 cone 规则过滤，本模板暂存于 `docs/tianji/`（`site.env.example`）；后续取消 `sparse-checkout` 后可迁移至顶层 `deploy/tianji/`。

## 使用原则

1. **禁止修改仓库内的模板文件写入实机密码、Token 或固定机器路径**。
2. 在工作站上复制 [site.env.example](site.env.example) 为未跟踪的 `site.env`（或置于运行目录之外的 `/etc/tianji/site.env` / `~/.config/tianji/site.env`），按本机环境填写：
   - `TIANJI_CPU_PYTHON` / `TIANJI_GPU_PYTHON` / 可选追加 `PYTHONPATH`
   - `LOOPINFRA_ROOT` 与 `LOOPINFRA_URL`
   - `TIANJI_RLT_STAGE1_MODEL_PATH`、`TIANJI_SUBTASK_PROMPTS`、`OPENPI_DATA_HOME`
   - 机器校验 ID（如 `TIANJI_CAR_POSE_CONFIG_ID`、`TIANJI_CAR_RESET_POSE_ID`）
3. 在同一会话内始终保持 `RLinf-tianji-hil` 与 `LoopInfra` 处于配套的 `tianji_hil` 分支提交，并在运行目录保存 `deployment.json` 记录双仓库 commit SHA、模型权重 SHA-256 与关键调度参数。
