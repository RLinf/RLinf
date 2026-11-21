# Tianji HIL 部署配置模板说明

本模板用于替代早期分散在 `Stage2-30HZ-*` 快照中的手写配置，将**版本化启动脚本**（[deploy/tianji/](../../deploy/tianji/README.md)）与**机器私有路径/ID**（未跟踪的 `site.env`）彻底分离。

## 使用原则

1. **禁止修改仓库内的模板文件写入实机密码、Token 或固定机器路径**。
2. 在工作站上复制 [site.env.example](site.env.example)（与 [deploy/tianji/site.env.example](../../deploy/tianji/site.env.example) 保持同步）为未跟踪的 `site.env`（启动器按 `$TIANJI_SITE_ENV` → `deploy/tianji/site.env` → `$RLINF_ROOT/site.env` → `~/.config/tianji/site.env` 顺序加载），按本机环境填写：
   - `RLINF_ROOT`（通过 `deploy/tianji/` 启动时可自动识别）与 `LOOPINFRA_ROOT`（默认 `$HOME/LoopInfra`）
   - `RLINF_PYTHON`（或拆分的 `TIANJI_CPU_PYTHON` / `TIANJI_GPU_PYTHON`）与可选 `LOOPINFRA_PYTHON`
   - `TIANJI_MODEL_DIR`（`TIANJI_RLT_STAGE1_MODEL_PATH`）、`TIANJI_RUNS_DIR`、`TIANJI_SUBTASK_PROMPTS`、`OPENPI_DATA_HOME`
   - 小车部署参数与校验 ID：`TIANJI_CAR_NUM_SUBTASKS`（`15` 或 `6`）、`TIANJI_CAR_BANNER`、`TIANJI_CAR_POSE_CONFIG_ID`、`TIANJI_CAR_RESET_POSE_ID`
3. 在同一会话内始终保持 `RLinf` 与 `LoopInfra` 处于配套的 `tianji_hil` 分支提交，并在运行目录保存 `deployment.json` 记录双仓库配置、模型权重 SHA-256 与关键调度参数。
