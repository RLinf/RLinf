Franka真机Pi0监督微调与部署全流程
=====================================

本文档介绍如何在 RLinf 框架中完成 **Bin-relocation** 任务的真机全流程演示，
该任务会将目标物体从起点位置搬运到终点位置（放入盘子视为成功完成任务）。
涵盖从真实世界采集专家数据、SFT 训练 Pi0，到策略真机部署的完整步骤。

主要流程如下：

1. **数据采集**：使用空间鼠标遥操作采集成功示范数据（LeRobot 格式）。
2. **SFT 训练**：基于 Pi0 模型在全量参数模式下进行监督微调。
3. **真机部署**：将训练好的策略在真实机器人上评估运行。

硬件与软件环境准备
---------------------

硬件要求
~~~~~~~~~~~~

- **机械臂**：Franka Emika Panda 机械臂。
- **相机**：Intel RealSense 相机（腕部相机用于观测）。
- **计算节点**：一台带有 GPU 的计算机，用于 SFT 训练与 rollout。
- **机器人控制节点**：一台与机械臂处于同一局域网的小型计算机（不需要 GPU），用于控制 Franka 机械臂。
- **空间鼠标（可选）**：用于远程操控进行数据采集。

.. note::

   关于硬件环境搭建的详细说明（包括 ROS Noetic、libfranka、serl_franka_controllers 等依赖），
   请参考 :doc:`franka` 中的「硬件环境搭建」与「依赖安装」章节。

软件依赖
~~~~~~~~~~~~

**控制节点** （数据采集）需要安装 Franka 控制相关依赖，可参考 :doc:`franka` 依赖安装部分。

**训练 / Rollout 节点** （SFT 训练 + 部署）需要安装 OpenPI 模型相关依赖：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

.. note::

   **训练 / Rollout 节点依赖安装注意事项** 
   注意在 '--model' 参数中指定 'openpi' 而不是 'openvla'。

环境配置
~~~~~~~~~~~~

数据采集、训练和部署均依赖 Franka 真机环境配置模板
``examples/embodiment/config/env/realworld_bin_relocation.yaml``。
该配置定义了 Bin-relocation 任务的关键参数，包括末端位姿限制、成功判定阈值等。
你可以在此基础上根据实际任务调整 ``override_cfg`` 中的字段。

完整流程
---------------------

以下各步骤对应 Bin-relocation 任务的完整 pipeline。

第一步：获取目标位姿
----------------------

在正式采集数据之前，需要先确定任务的目标末端位姿（target_ee_pose）。

需要注意的是，在 Bin-relocation 任务中，目标末端位姿的实际含义被定义为表示运动空间的中间的最低点。
特别的，为了避免franka末端撞击盘子边缘，会基于目标末端位姿将一定空间范围截去，用于限制机械臂的运动范围。
详细参考 ``rlinf/envs/realworld/franka/tasks/franka_bin_relocation.py`` 中的定义。

参考 :doc:`franka` 中的「获取任务的目标位姿」章节，
使用脚本 ``toolkits.realworld_check.test_franka_controller`` 获取目标位姿。
记录此位姿，后续步骤中将替换到配置文件中。

第二步：采集专家数据
----------------------

参考 :doc:`franka` 中的「数据采集」章节，在控制节点上采集专家数据。

特别的，除了原有配置外，还需要针对 Bin-relocation 任务，做出以下修改：

1. 对于抓放任务，需要将环境从 peg insertion 切换为 bin relocation：

.. code-block:: yaml

  defaults:
    - env/realworld_bin_relocation@env.eval
    - override hydra/job_logging: stdout

2. 对于抓放任务，需要使用夹爪的自由度：

.. code:: yaml

  env:
    eval:
      no_gripper: False

3. 使用键盘在数据采集时进行标注，在采集时按下 ``c`` 本条轨迹会被记录为成功，并重置机械臂位姿：

.. code:: yaml

  env:
    eval:
      keyboard_reward_wrapper: single_stage

4. 修改 ``task_description`` 为当前任务的描述：

.. code:: yaml

   env:
     eval:
       override_cfg:
         task_description: "pick up the object and place it into the container"

5. 设置导出 LeRobot 格式的数据：

.. code:: yaml

  env:
    data_collection:
      enabled: True
      export_format: "lerobot"


在采集过程中，使用空间鼠标操作机械臂进行任务。
在每条轨迹采集完成后，按下 ``c`` 键，本条轨迹会被记录为成功，并重置机械臂位姿。
此时，注意还原目标物体到起点位置。

采集脚本默认在收集 20 个 episode 后结束（可通过配置中的 ``num_data_episodes`` 字段修改），
采集到的 LeRobot 格式数据会保存在
``logs/<running-timestamp>/collected_data`` 路径下。

第三步：SFT 训练 Pi0
----------------------

创建真机数据集格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本步骤参考 :doc:`sft_openpi` 中的「支持的数据集」章节。针对真机Franka环境，
可以创建出 ``pi0_realworld`` 数据格式，其定义在以下文件：

1. ``rlinf/models/embodiment/openpi/__init__.py``
2. ``rlinf/models/embodiment/openpi/dataconfig/realworld_dataconfig.py``

为了统一真机和各仿真环境对策略的调用接口，创建 
3. ``rlinf/models/embodiment/openpi/policies/realworld_policy.py``。

计算归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据 :doc:`sft_openpi` 中的「新 LeRobot 数据集的归一化统计」章节，
需要为刚采集得到的 LeRobot 数据集计算归一化统计量，
这是启动 SFT 训练的前提条件。

首先，将控制节点采集到的数据上传到训练节点的数据目录中，
例如 ``/path/to/lerobot_data``。文件结构应按照如下：

.. code::

   /path/to/lerobot_data
   |-- realworld_franka_bin_relocation
      |-- data
      |-- meta
    |-- franka_dagger （其他repo_id同理）
        |-- data
        |-- meta
    |-- ...

这里 ``realworld_franka_bin_relocation`` 对应在``rlinf/models/embodiment/openpi/__init__.py``中定义的 TrainConfig 字段中的 ``repo_id``。
  
然后，在训练节点上运行：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/replay_buffer/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

注意事项：

- 运行脚本前必须先设置 ``HF_LEROBOT_HOME``，指向 LeRobot 数据集的根目录。
- ``config_name`` 必须与训练时使用的 OpenPI dataconfig 一致。
- ``repo_id`` 必须与你的 LeRobot 格式数据集名称一致。

该脚本会将生成的统计信息写入
``<assets_dir>/<exp_name>/<repo_id>/norm_stats.json``。

OpenPI 加载器会在运行时从``<model_path>/<repo_id>``读取归一化统计信息。

运行 OpenPI SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用``pi0_realworld``数据格式，需要修改SFT训练配置文件``examples/sft/config/realworld_sft_openpi.yaml``：

.. code:: yaml

   data:
     train_data_paths: "/path/to/lerobot_data"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_realworld"

将归一化统计信息放置在模型路径下，OpenPI 加载器会在运行时从``<model_path>/<repo_id>``读取。
文件结构应按照如下：

.. code::

  /path/to/pi0-base-model
  |-- config.json
  |-- model.safetensors
  |-- realworld_franka_bin_relocation
    |-- norm_stats.json
  |-- franka_dagger
    |-- norm_stats.json
  |-- ...

运行 SFT 训练脚本：

.. code:: bash

   bash examples/sft/run_vla_sft.sh realworld_sft_openpi

SFT 导出的 checkpoint 会在后续章节中部署使用。
更多 OpenPI 数据集及 SFT 训练说明可参考 :doc:`sft_openpi`。

第五步：真机部署
----------------------

修改 ``examples/embodiment/config/realworld_pnp_eval.yaml``，
使其与你的集群、相机、目标位姿一致：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: ROBOT_IP

   env:
     eval:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
         task_description: "pick up the object and place it into the container"

SFT 训练完成后，将模型检查点路径也更新到部署配置文件中：

.. code:: yaml

   runner:
     ckpt_path: "/path/to/your/sft/checkpoint/full_weights.pt"

   rollout:
     model:
       model_path: "/path/to/pi0-model"

在 Ray 集群启动后（参考 :doc:`franka` 中的「集群配置」章节），
在 head 节点上运行部署脚本：

.. code:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_pnp_eval

该脚本以 **纯评估模式** （``runner.only_eval: True``）运行，
策略将根据输入的观测自主控制机器人完成 Bin-relocation 任务。

可以通过修改 ``eval_rollout_epoch`` 参数来控制评估的轮数。

.. code:: yaml

   runner:
     eval_rollout_epoch: 20

通用真机 SFT 环境与部署
---------------------------

除了上述Bin relocation任务，RLinf 还提供了一个 **通用 SFT 环境** （``FrankaEnv-v1``），允许你完全通过 YAML
配置来定义新的真机任务，无需编写自定义环境类。适用于：

- 在新任务上采集 SFT 示教数据
- 在真机上部署（评估）已训练的策略

通用 SFT 环境
~~~~~~~~~~~~~~~~~~~~~

环境配置模板位于
``examples/embodiment/config/env/realworld_franka_sft_env.yaml``。
你需要根据自己的任务修改以下关键字段：

.. code:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # 目标位姿 [x,y,z,rx,ry,rz]
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # 复位位姿（应高于目标）
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]  # 成功判定容差
     action_scale: [1.0, 1.0, 1.0]                         # [xyz, rpy, 夹爪]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

底层实现上，``FrankaEnv`` 现在接受 ``override_cfg`` 字典，并使用类变量
``CONFIG_CLS`` 来实例化数据类配置（默认为 ``FrankaRobotConfig``）。
``PegInsertionEnv`` 和 ``BottleEnv`` 等子类通过覆盖 ``CONFIG_CLS``
来使用各自的数据类，同时共享相同的构造函数。

真机评估 / 部署
~~~~~~~~~~~~~~~~~~~~~

完整的评估配置位于
``examples/embodiment/config/realworld_eval.yaml``，它将通用 SFT
环境与 Pi0 actor 以 **纯评估模式** （``runner.only_eval: True``）组合使用。

运行前请替换以下占位符：

- ``ROBOT_IP`` — 你的 Franka 机器人 IP 地址。
- ``MODEL_PATH`` — 已训练的模型检查点路径。

然后执行：

.. code:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval
