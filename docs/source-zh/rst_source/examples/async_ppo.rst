Async PPO 强化学习训练
===============================================

本示例展示 RLinf 框架使用 **Async PPO (Asynchronous Proximal Policy Optimization)** 算法训练具身智能（Embodied AI）策略网络的完整流程。
该算法通过分布式架构将环境交互（Env）、策略推理（Rollout）和模型训练（Actor）解耦，支持大规模并行采样与高效训练，适用于 ManiSkill 等仿真环境。

主要目标是让模型具备以下能力：

1. **分布式训练**：利用多节点/多GPU资源加速训练。
2. **高效采样**：解耦环境与推理，提高数据吞吐量。
3. **策略优化**：使用 PPO 算法稳定地优化策略网络（如 OpenVLA）。

环境
----

**ManiSkill3 环境 (仿真)**

-  **Environment**：ManiSkill3 仿真平台
-  **Task**：控制机械臂完成任务，例如 ``PutOnPlateInScene-v1``
-  **Observation**：机器人状态、物体位置、视觉图像等
-  **Action Space**：连续动作空间（位置、旋转、夹爪）

算法
-----------------------------------------



**算法特性**

-   **Staleness Control**：控制异步训练中的策略滞后问题 (`staleness_threshold`)。
-   **Behavior Cloning Regularization**：支持结合离线数据或演示数据进行辅助训练。
-   **Advantage Normalization**：使用优势归一化稳定训练。

运行脚本
--------

**1. 配置文件**

RLinf 提供了针对 ManiSkill 环境的 Async PPO 默认配置文件：

-   **配置文件**: ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml``

**2. 关键参数配置**

**2.1 集群配置 (Cluster)**

通过 ``cluster`` 参数配置各个组件的资源分配：

.. code:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-7    # Actor Worker 占用 GPU 0-7
       env: 0-3      # Env Worker 运行在 GPU 0-3 (或对应CPU资源)
       rollout: 4-7  # Rollout Worker 运行在 GPU 4-7

**2.2 算法参数 (Algorithm)**

.. code:: yaml

   algorithm:
     staleness_threshold: 2       # 允许的最大策略版本滞后
     normalize_advantages: True   # 是否归一化优势函数
     reward_type: action_level    # 奖励类型

**3. 启动命令**

使用提供的脚本启动训练：

.. code:: bash

   # 默认运行 ManiSkill + Async PPO + OpenVLA
   bash examples/embodiment/run_async_ppo.sh

或者指定配置文件：

.. code:: bash

   bash examples/embodiment/run_async_ppo.sh maniskill_async_ppo_openvla


