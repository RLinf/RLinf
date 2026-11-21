概览
====

RLinf 提供统一的具身智能评测入口，支持在仿真或真机环境中并行 rollout，并输出任务级成功率等指标。

目录结构
--------

所有评测入口脚本与 benchmark 配置都位于仓库根目录的 ``evaluations/`` 下：

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 路径
     - 说明
   * - ``evaluations/eval_embodied_agent.py``
     - 具身评测主程序。
   * - ``evaluations/run_eval.sh``
     - 统一启动脚本，负责推断 benchmark、准备日志目录与初始化仿真环境。
   * - ``evaluations/behavior/``
     - BEHAVIOR-1K 家居任务评测配置。
   * - ``evaluations/geniesim/``
     - GenieSim 服务化仿真评测配置。
   * - ``evaluations/libero/``
     - LIBERO 评测配置（Spatial、Object、Goal、10、Mem、RTC、ApxInf、SGLang）与自定义初始状态。
   * - ``evaluations/maniskill/``
     - ManiSkill 分布外（OOD）泛化评测配置。
   * - ``evaluations/polaris/``
     - PolaRiS 桌面操作评测配置。
   * - ``evaluations/realworld/``
     - Franka 单臂、Dual Franka RTC 与 Dual YAM 真机评测配置。
   * - ``evaluations/robocasa365/``
     - RoboCasa365 pretrain 任务切片评测配置。
   * - ``evaluations/robodojo/``
     - RoboDojo Isaac Sim 评测配置。
   * - ``evaluations/robolab/``
     - RoboLab 120 任务 DROID joint-position 评测配置。
   * - ``evaluations/robotwin/``
     - RoboTwin 双臂操作评测配置。
   * - ``evaluations/simple/``
     - 面向 Psi0 的 SIMPLE 人形机器人全身评测配置。

评测架构
--------

评测 runner 通过 channel 协调 env worker 与 rollout worker，按 ``env.eval`` 配置执行并行评测，并将 ``eval/success_once``、``eval/return`` 等指标打印到终端并写入日志。

1. **加载配置：** Hydra 读取 ``evaluations/<benchmark>/`` 下的 YAML，并通过 ``defaults`` 组合 ``examples/embodiment/config/`` 中的环境与模型预设。
2. **启动 worker：** 按 ``cluster.component_placement`` 在硬件上启动 env worker 与 rollout worker。
3. **并行 rollout：** env worker 重置环境并发送观测，rollout worker 返回动作块，循环执行直到回合结束。
4. **汇总指标：** 计算并记录 ``eval/success_once``、``eval/return`` 等任务级指标。

下一步
------

- 安装依赖：:doc:`installation`
- 5 分钟快速体验：:doc:`quick_tour`
- 查看各 benchmark 指南：:doc:`../guides/index`
