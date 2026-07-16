概览
====

RLinf 提供统一的具身智能评测入口，支持在仿真或真机环境中并行 rollout，并输出任务级成功率等指标。

目录结构
--------

所有评测相关代码与配置位于仓库根目录的 ``evaluations/`` 下：

.. code-block:: text

   evaluations/
   ├── eval_embodied_agent.py   # 评测主程序
   ├── run_eval.sh              # 一键启动脚本
   ├── libero/                  # LIBERO 评测配置
   ├── robotwin/                # RoboTwin 评测配置
   ├── behavior/                # BEHAVIOR-1K 评测配置
   ├── maniskill/               # ManiSkill OOD 评测配置
   ├── realworld/               # 真机评测配置
   └── polaris/                 # PolaRiS 评测配置

评测架构
--------

评测流程由 ``EmbodiedEvalRunner`` 驱动：Env Worker 与 Rollout Worker 通过 Channel 交互，在 ``env.eval`` 配置下完成并行评测。终端与日志中会输出 ``eval/success_once``、``eval/return`` 等指标。

典型数据流：

1. **配置加载** — Hydra 从 ``evaluations/<benchmark>/`` 读取 YAML，并通过 ``defaults`` 引用 ``examples/embodiment/config/`` 下的环境与模型 preset。
2. **Worker 启动** — 根据 ``cluster.component_placement`` 在 GPU 上启动 Env Worker 与 Rollout Worker。
3. **并行 Rollout** — Env Worker 重置环境并返回观测；Rollout Worker 根据模型生成动作；循环直至 episode 结束。
4. **指标汇总** — 统计 ``success_once``、``return`` 等任务级指标并写入日志。

下一步
------

- 安装环境：:doc:`installation`
- 5 分钟快速体验：:doc:`quick_tour`
- 按 benchmark 深入：:doc:`../guides/index`
