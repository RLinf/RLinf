真机评测
========

RLinf 支持在 Franka 机械臂上评测与部署 VLA 策略，涵盖 Bin-relocation（pick-and-place）任务，以及通过 YAML 配置的通用真机环境（``FrankaEnv-v1``，用于自定义任务）。

SO101 机械臂评测参见 :doc:`so101_eval`\ 。

相关训练文档：:doc:`../../examples/embodied/franka_pi0_sft_deploy`、:doc:`../../examples/embodied/franka`、:doc:`../../examples/embodied/sft_dreamzero`

环境准备
--------

**硬件**

- Franka Emika Panda 机械臂 + Intel RealSense 相机（默认）
- 一台 **GPU 节点** （训练 / rollout）与一台 **机器人控制节点** （直连 Franka 与相机，无需 GPU）
- 所有节点处于同一局域网；机械臂仅需与控制节点互通

使用 ZED 相机或 Robotiq 夹爪时，请参阅 :doc:`../../examples/embodied/franka_zed_robotiq`。

**依赖安装**

控制节点与 GPU 节点需分别安装依赖：

.. code-block:: bash

   # 机器人控制节点（Franka + 相机 + ROS）
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate
   source <your_catkin_ws>/devel/setup.bash

.. code-block:: bash

   # GPU / rollout 节点（π₀ 评测）
   bash requirements/install.sh embodied --model openpi --env franka
   source .venv/bin/activate

DreamZero 真机评测还需在 GPU 节点安装 DreamZero 依赖，详见 :doc:`../../examples/embodied/sft_dreamzero`。

**节点拓扑**

真机评测通常采用 **「1 个 GPU 节点 + 1 个 Franka 控制节点」** 的异构布局：

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``RLINF_NODE_RANK``
     - 角色
     - 说明
   * - rank 0
     - GPU / head
     - 运行 ``rollout``；在此节点提交 ``run_eval.sh``
   * - rank 1
     - 机器人控制
     - 运行 ``env`` worker，直连 Franka 与相机

``realworld_pnp_eval.yaml`` 与 ``realworld_pnp_eval_dreamzero.yaml`` 使用上述双节点布局；``realworld_eval.yaml`` （自定义任务）为 **单机** 布局，``env`` 与 ``rollout`` 均部署在同一 Franka 节点。

Ray 集群的完整搭建、固件版本与 libfranka 兼容性见 :doc:`../../examples/embodied/franka` 与 :doc:`../../guides/realworld_robot`。

示例配置
--------

``evaluations/realworld/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - 配置文件
     - 任务
     - 模型
   * - ``realworld_pnp_eval.yaml``
     - Bin-relocation (PnP)
     - π₀
   * - ``realworld_pnp_eval_dreamzero.yaml``
     - Bin-relocation (PnP)
     - DreamZero
   * - ``realworld_eval.yaml``
     - 自定义任务（``FrankaEnv-v1``）
     - π₀

若 ``evaluations/realworld/<config>.yaml`` 不存在，``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下同名配置（需设置 ``runner.task_type: embodied_eval`` 与 ``runner.only_eval: True``）。详见 :doc:`../reference/cli`。
Dual Franka 部署目前通过该回退路径使用 ``realworld_eval_dual_franka``。

启动前检查
----------

在正式评测前，建议依次完成以下检查：

1. **相机连接** （控制节点）：

   .. code-block:: bash

      python -m toolkits.realworld_check.test_franka_camera

   记录输出的相机序列号，填入 ``env.eval.override_cfg.camera_serials``。

2. **目标位姿** （PnP 任务，控制节点）：

   .. code-block:: bash

      export FRANKA_ROBOT_IP=<robot_ip>
      python -m toolkits.realworld_check.test_franka_controller
      # 输入 getpos_euler 获取目标末端位姿

   将结果填入 ``env.eval.override_cfg.target_ee_pose``。PnP 任务中该位姿表示运动空间中间的最低点，并用于成功判定与工作空间截断；详见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`。

3. **Ray 集群连通** （各节点）：

   .. code-block:: bash

      ray status

   应同时看到 GPU 节点与 Franka 节点在线。

4. **Dummy 模式（可选）**：参考 ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml``，在 ``override_cfg`` 中设置 ``is_dummy: True`` 验证集群配置，无需真实机器人运动。

.. warning::

   真机评测前务必确认工作空间限位（``ee_pose_limit_min`` / ``ee_pose_limit_max``）与急停功能正常；首次评测建议将 ``env.eval.rollout_epoch`` 设为较小值。

Ray 集群启动
------------

在 **每个节点**、执行 ``ray start`` **之前**，先统一环境变量（可使用 ``ray_utils/realworld/setup_before_ray.sh``）：

.. code-block:: bash

   source ray_utils/realworld/setup_before_ray.sh
   export RLINF_NODE_RANK=<0|1>          # 集群内唯一
   # 多网卡时指定对外可达网卡：
   # export RLINF_COMM_NET_DEVICES=<network_interface>

控制节点还需 source ROS / franka 工作空间（若未写入 setup 脚本）：

.. code-block:: bash

   source <your_catkin_ws>/devel/setup.bash

然后启动 Ray（记 head 节点 IP 为 ``<head_ip>``）：

.. code-block:: bash

   # GPU 节点（rank 0，head）
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

   # Franka 控制节点（rank 1）
   export RLINF_NODE_RANK=1
   ray start --address=<head_ip>:6379

.. important::

   ``ray start`` 会冻结当时的 Python 解释器与环境变量；请在各节点完成 venv、ROS、``PYTHONPATH`` 等配置后再启动 Ray。

完整评测流程（PnP / π₀）
------------------------

**Step 1：配置 Ray 集群**

按上文在各节点启动 Ray，确保 ``ray status`` 显示双节点在线。

**Step 2：准备模型**

π₀ PnP 评测需要：

- ``rollout.model.model_path``：Pi0 基座模型目录，且须包含 ``<repo_id>/norm_stats.json`` （SFT 前置步骤生成，见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`）
- ``runner.ckpt_path``：SFT 导出的 ``full_weights.pt``

**Step 3：编辑配置**

修改 ``evaluations/realworld/realworld_pnp_eval.yaml``，至少更新以下字段：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: "4090"
         node_ranks: 0          # GPU 节点 rank
       - label: franka
         node_ranks: 1          # Franka 控制节点 rank
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   runner:
     ckpt_path: /path/to/full_weights.pt

   env:
     eval:
       rollout_epoch: 20
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
         task_description: "pick up the object and place it into the container"

   rollout:
     model:
       model_path: /path/to/pi0-model
       openpi:
         config_name: "pi0_realworld"

``node_ranks`` 与 ``component_placement`` 须与集群中实际的 ``RLINF_NODE_RANK`` 一致。

**Step 4：启动评测**

在 **GPU / head 节点** 执行：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval

也可通过 Hydra 覆盖参数，无需修改 YAML 文件：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval \
     rollout.model.model_path=/path/to/pi0-model \
     runner.ckpt_path=/path/to/full_weights.pt

**Step 5：查看结果**

策略以 ``runner.only_eval: True`` 模式运行；终端输出任务指标，日志见 :doc:`../reference/results`。

.. _realworld-eval-config:

评测配置详解
------------

以下说明适用于 ``realworld_pnp_eval.yaml``；自定义任务评测见下文，DreamZero 见 :doc:`../../examples/embodied/sft_dreamzero`。

必须修改的字段
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 字段
     - 位置
     - 说明
   * - ``robot_ip``
     - ``cluster.node_groups[].hardware.configs``
     - Franka 机器人 IP
   * - ``node_ranks`` / ``component_placement``
     - ``cluster``
     - 与 ``RLINF_NODE_RANK`` 对齐；env 在 Franka 节点，rollout 在 GPU 节点
   * - ``target_ee_pose``
     - ``env.eval.override_cfg``
     - PnP 目标末端位姿 ``[x,y,z,rx,ry,rz]``，影响成功判定与运动截断
   * - ``camera_serials``
     - ``env.eval.override_cfg``
     - RealSense 序列号列表（**非** ``node_groups`` 字段）
   * - ``task_description``
     - ``env.eval.override_cfg``
     - 语言指令，须与 SFT 训练一致
   * - ``model_path``
     - ``rollout.model``
     - Pi0 基座模型目录（含 ``norm_stats.json``）
   * - ``ckpt_path``
     - ``runner``
     - SFT checkpoint（``full_weights.pt``）
   * - ``config_name``
     - ``rollout.model.openpi``
     - PnP 任务为 ``"pi0_realworld"``

关键 ``env.eval`` 字段
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 字段
     - 说明
   * - ``rollout_epoch``
     - 评测轮数，每轮重复完整 rollout；默认 20
   * - ``max_episode_steps``
     - 单条轨迹最大步数，默认 200
   * - ``max_steps_per_rollout_epoch``
     - 每轮 rollout 步数上限；**必须能被** ``rollout.model.num_action_chunks`` **整除** （PnP 默认 ``num_action_chunks=10``）
   * - ``total_num_envs``
     - 并行环境数，真机通常为 1
   * - ``use_spacemouse``
     - 是否启用空间鼠标人工干预，评测时通常为 ``False``

``run_eval.sh`` 行为
~~~~~~~~~~~~~~~~~~~~

- ``realworld`` benchmark **不** 调用 ``setup_sim_env``，无需仿真相关环境变量
- 日志写入 ``logs/<时间戳>-<config_name>/eval_embodiment.log``
- 评测入口脚本须在 **head（GPU）节点** 提交

DreamZero 真机评测（``realworld_pnp_eval_dreamzero.yaml``）完整流程见 :doc:`../../examples/embodied/sft_dreamzero`。

自定义真机任务评测
------------------

本节面向 **你自己定义的任务**，与上文内置的 Bin-relocation（PnP）任务不同。

RLinf 提供通用真机环境 ``FrankaEnv-v1``：在 YAML 中配置 ``task_description``、目标/复位位姿、工作空间限位等即可，**无需编写新的 Python 环境类**。典型用法是：采集示教数据 → 监督微调（SFT）训练策略 → 在真机上部署评测（流程见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`）。环境模板为 ``examples/embodiment/config/env/realworld_franka_sft_env.yaml``；评测配置为 ``evaluations/realworld/realworld_eval.yaml``。

**节点拓扑**

``realworld_eval.yaml`` 为 **单机** 配置：``env`` 与 ``rollout`` 均部署在 Franka 节点（``num_nodes: 1``），适用于 GPU 与控制合署一机的场景。

关键 ``override_cfg`` 字段
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # 目标位姿
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # 复位位姿（应高于目标）
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
     action_scale: [1.0, 1.0, 1.0]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

**启动评测**

替换 ``ROBOT_IP`` 与 ``MODEL_PATH`` 后执行：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval

自定义任务的采集、训练与部署流程见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`。

Dual Franka 部署
----------------

Dual Franka SFT 部署复用统一评测启动器，并通过配置回退使用
``examples/embodiment/config/realworld_eval_dual_franka.yaml``。
将 ``rollout.model.model_path`` 设为已同步的 checkpoint 目录，将
``actor.model.openpi_data.repo_id`` 设为包含 ``norm_stats.json`` 的 repo id。

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval_dual_franka \
       rollout.model.model_path=/path/to/deploy/global_step_<N> \
       actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
       env.eval.override_cfg.task_description="handover the object"

完整采集、SFT、checkpoint 同步与脚踏按键流程见 :doc:`../../examples/embodied/dual_franka`。

查看结果
--------

- **终端指标：** 任务成功率、回报等（具体字段因环境而异）
- **日志：** ``logs/<时间戳>-<config_name>/eval_embodiment.log``
- **视频：** 若 ``env.eval.video_cfg.save_video: True``，视频保存在 ``video_base_dir`` 或 ``<log_path>/video/eval/``

详见 :doc:`../reference/results`。

常见问题
--------

- **安全：** 评测前确认工作空间限位与急停功能正常；首次评测降低 ``rollout_epoch``。
- **节点拓扑：** ``env`` worker 必须部署在可直连 Franka 的节点；``node_ranks`` 须与 ``RLINF_NODE_RANK`` 一致。PnP 与 Dual Franka 为双节点，自定义任务评测为单机。
- **相机未找到：** 在控制节点运行 ``python -m toolkits.realworld_check.test_franka_camera``，核对 ``camera_serials``。
- **动作异常：** 检查 ``norm_stats.json`` 是否位于 ``model_path/<repo_id>/``，以及 ``openpi.config_name`` 是否与训练一致。
- **Ray 只见一个节点：** 检查防火墙、``RLINF_COMM_NET_DEVICES`` 与 head IP 是否可被其他节点访问。
- **步数报错：** 确保 ``max_steps_per_rollout_epoch`` 能被 ``num_action_chunks`` 整除。
- **ZED / Robotiq：** 参阅 :doc:`../../examples/embodied/franka_zed_robotiq`。
