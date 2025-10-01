RL与RoboTwin仿真器
=====================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档提供了在RLinf框架内设置和使用RoboTwin环境进行强化学习实验的综合指南，
专注于微调视觉-语言-动作模型（VLAs）用于机器人操作任务。

主要目标是通过以下方式开发能够执行机器人操作的模型：

1. **视觉理解**：处理来自多个机器人摄像头的RGB图像。
2. **语言理解**：解释自然语言任务描述。
3. **动作生成**：为双臂操作产生精确的机器人动作。
4. **强化学习**：通过PPO/GRPO与环境反馈优化策略。

环境设置
-----------------------

**RoboTwin环境**

- **环境**：基于*robosuite*（MuJoCo）构建的RoboTwin仿真平台。
- **任务**：控制双臂7自由度机器人执行家庭操作技能（抓取放置、堆叠、空间重排）。
- **观察**：来自头部、左侧和右侧摄像头的多摄像头RGB图像（分辨率240×320，调整为224×224）。
- **动作空间**：14维连续动作（每臂7自由度）
  - 左臂：3D位置 + 3D旋转 + 夹爪控制
  - 右臂：3D位置 + 3D旋转 + 夹爪控制

**安装步骤**

1. **克隆所需仓库**

   .. code-block:: bash

      # 克隆RoboTwin仓库
      git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/robotwin

2. **下载资源**

   .. code-block:: bash

      cd third_party/robotwin/assets
      bash _download.py

3. **构建并设置环境变量**

   .. code-block:: bash

      cd third_party/robotwin
      bash script/_install.sh
      export PYTHONPATH="/path/to/third_party/robotwin":$PYTHONPATH

4. **配置资源路径**

   使用正确的资源路径更新配置文件：

   **文件：`assets/embodiments/aloha-agilex/curobo_left.yml`**
   .. code-block:: yaml

      # 替换为适当的路径
      urdf_path: path/to/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
      collision_spheres: path/to/assets/embodiments/aloha-agilex/collision_aloha_left.yml

   **文件：`assets/embodiments/aloha-agilex/curobo_right.yml`**
   .. code-block:: yaml

      # 替换为适当的路径
      urdf_path: path/to/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
      collision_spheres: path/to/assets/embodiments/aloha-agilex/collision_aloha_right.yml

5. **测试环境**

   .. code-block:: bash

      python robotwin_test.py

**数据结构**

- **图像**：多摄像头RGB张量 ``[batch_size, num_cameras, 3, 224, 224]``
- **任务描述**：操作任务的自然语言指令
- **动作**：双臂控制的14维连续值
- **奖励**：基于任务完成和操作成功的步骤级奖励

算法
-----------------------------------------

**核心算法组件**

1. **PPO（近端策略优化）**
   - 使用GAE（广义优势估计）进行优势估计
   - 带比率限制的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **GRPO（组相对策略优化）**
   - 对于每个状态/提示，策略生成*G*个独立动作
   - 通过减去组平均奖励计算每个动作的优势

3. **视觉-语言-动作模型**
   - 具有多模态融合的OpenVLA-OFT架构
   - 动作标记化和去标记化
   - 用于批评函数的价值头

运行脚本
-------------------

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

您可以灵活配置env、rollout和actor组件的GPU数量。
使用上述配置，您可以实现env和rollout之间的管道重叠，以及与actor的共享。
此外，通过在配置中设置`pipeline_stage_num = 2`，您可以实现rollout和actor之间的管道重叠，提高rollout效率。

.. code-block:: yaml
   
   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

您也可以重新配置放置以实现完全共享，其中env、rollout和actor组件共享所有GPU。

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

您也可以重新配置放置以实现完全分离，其中env、rollout和actor组件各自使用自己的GPU而不相互干扰，消除了卸载功能的需要。

**2. 配置文件**

我们支持**OpenVLA-OFT**模型与**PPO**和**GRPO**算法。
相应的配置文件为：

- **OpenVLA-OFT + PPO**：``examples/embodiment/config/robotwin_ppo_openvlaoft.yaml``
- **OpenVLA-OFT + GRPO**：``examples/embodiment/config/robotwin_grpo_openvlaoft.yaml``

**3. 启动命令**

要使用选定的配置开始训练，请运行以下命令：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在RoboTwin环境中使用PPO算法训练OpenVLA模型，请运行：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_ppo_openvlaoft

可视化和结果
-------------------------

**1. TensorBoard日志记录**

.. code-block:: bash

   # 启动TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 跟踪的关键指标**

- **训练指标**：
  - ``actor/loss``：PPO策略损失
  - ``actor/value_loss``：价值函数损失
  - ``actor/entropy``：策略熵
  - ``actor/grad_norm``：梯度范数
  - ``actor/lr``：学习率

- **Rollout指标**：
  - ``rollout/reward_mean``：平均回合奖励
  - ``rollout/reward_std``：奖励标准差
  - ``rollout/episode_length``：平均回合长度
  - ``rollout/success_rate``：任务完成率

- **环境指标**：
  - ``env/success_rate``：跨环境的成功率
  - ``env/step_reward``：逐步奖励

**3. 视频生成**

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ./logs/video/train

**4. WandB集成**

.. code-block:: yaml

   trainer:
     logger:
       wandb:
         enable: True
         project_name: "RLinf"
         experiment_name: "openvla-robotwin"

获取帮助
------------

如果您遇到本指南未涵盖的问题，请：

1. 查看`CuRobo文档 <https://github.com/NVlabs/curobo>`_
2. 查看`RoboTwin文档 <https://github.com/RoboTwin-Platform/RoboTwin>`_
3. 在RLinf中创建问题

许可证
-------

请参考CuRobo和RoboTwin组件的各个仓库许可证。
