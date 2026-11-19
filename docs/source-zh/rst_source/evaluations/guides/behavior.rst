BEHAVIOR-1K 评测
================

BEHAVIOR-1K 是基于 OmniGibson 与 Isaac Sim 的大规模家居场景仿真基准，控制双臂 R1 Pro 机器人完成抓取、放置、整理等家庭操作任务。RLinf 支持在 BEHAVIOR 环境中并行评测 OpenPI 等 VLA 策略，并输出 ``eval/success_once`` 等指标。

相关训练文档：:doc:`../../examples/embodied/behavior`

环境准备
--------

**安装依赖**

.. code-block:: bash

   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

当前 ``evaluations/behavior/`` 仅提供 OpenPI π₀.₅ 示例；训练侧还支持 OpenVLA-OFT，可从 ``examples/embodiment/config/`` 中的训练配置派生评测 YAML（见 :doc:`../reference/configuration`）。

**硬件与 Isaac Sim**

评测使用 OmniGibson 3.9.3 和 Isaac Sim 5.1。

- 使用配有 RT Core 的 NVIDIA GPU。NVIDIA 为 Isaac Sim 5.1 列出的最低显存为 16 GB。A100、H100 缺少 RT Core，不在支持范围内。
- NVIDIA 列出的 Linux 测试驱动版本为 580.65.06。支持的驱动版本和操作系统见 `Isaac Sim 5.1 系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_\ 。

包含 OmniGibson 3.9.3 的 RLinf BEHAVIOR Docker 镜像待发布。

**环境变量**

在启动评测的终端中设置 BEHAVIOR 资产的根目录。

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

OmniGibson 从根目录下的 ``behavior-1k-assets/`` 读取场景和物体资产，从 ``omnigibson-robot-assets/`` 读取机器人资产，并使用 ``omnigibson.key`` 解密资产。将 2026 任务的缓存布局和统计文件放在同一根目录下的 ``2026-challenge-task-instances/`` 中。

上述安装命令将 Isaac Sim 5.1 安装为 Python 包，导入包时会自动设置 ``ISAAC_PATH``\ 。运行安装命令前，请清除终端中已有的 ``ISAAC_PATH``、``EXP_PATH`` 和 ``CARB_APP_PATH``\ 。

资产下载和许可设置命令见 :doc:`../../examples/embodied/behavior`\ 。

示例配置
--------

``evaluations/behavior/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 配置文件
     - 环境 preset
     - 模型
   * - ``behavior_openpi_pi05_eval.yaml``
     - ``behavior_r1pro``
     - π₀.₅（``openpi``）

若 ``evaluations/behavior/<config>.yaml`` 不存在，``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下同名配置（例如 ``behavior_ppo_openpi_pi05``）。回退配置包含 ``actor`` / ``algorithm`` 等训练段，但设置 ``runner.only_eval: True`` 后仍可正常评测。

完整评测流程
------------

**Step 1：激活环境并设置路径**

.. code-block:: bash

   source .venv/bin/activate
   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

**Step 2：准备模型**

推荐预训练权重：`RLinf/RLinf-Pi0-Behavior <https://huggingface.co/RLinf/RLinf-Pi0-Behavior>`_（下载命令见训练文档）。若使用第三方 OpenPI 权重（如 OpenPI-Comet），需先转换为 PyTorch 格式后再填入 ``rollout.model.model_path``。

**Step 3：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path``。通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`；BEHAVIOR 特有字段与评测协议见下文 :ref:`behavior-eval-config`。

示例 ``behavior_openpi_pi05_eval.yaml`` 中还需保持与训练一致的 OpenPI 参数（``action_dim: 23``、``num_action_chunks: 32``、``openpi.config_name: pi05_behavior`` 等）。``num_action_chunks`` 是环境实际执行的 chunk；网络 horizon 来自 ``pi05_behavior`` 官方 ``TrainConfig.model.action_horizon``\ （**32**），除非另外设置 ``openpi.action_horizon``。

**Step 4：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

上面的命令按默认配置在基础布局 0 上评测“打开收音机”任务。评测 2026 年公开测试集的首个实例时，需指定该任务的缓存目录，并将实例编号设为 301。公开测试索引 0 对应布局 301。布局决定物体的初始状态和机器人的初始位姿。

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     "env.eval.omni_config.task.activity_instance_dir=${OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances" \
     env.eval.omni_config.task.activity_instance_id=301

**Step 5：查看结果**

终端输出 ``eval/success_once``；日志与视频见 :doc:`../reference/results`。

.. _behavior-eval-config:

评测配置详解
------------

BEHAVIOR 评测 **每次运行对应单个任务** （由 ``omni_config.task.activity_name`` 指定），不会在一次启动中自动遍历全部 100 个任务。以下字段共同决定并行规模、单条轨迹长度与初始场景实例。

评测协议概述
~~~~~~~~~~~~

2026 挑战共有 100 个任务。RLinf 提供全部 100 个任务的指令。``behavior_r1pro`` preset 默认任务为 ``turning_on_radio``\ ，场景为 ``house_double_floor_lower``\ 。

每条评测轨迹由以下配置唯一确定：

- ``omni_config.task.activity_name``：任务名称，决定语言指令与 BDDL 定义；
- ``omni_config.task.activity_definition_id``：任务定义变体编号（通常为 ``0``）；
- ``omni_config.task.activity_instance_id`` 与 ``instance_resample_mode``：初始物体布局与机器人位姿。

``instance_resample_mode`` 支持三种模式：

- ``disabled`` （默认）：每次 reset 加载 ``activity_instance_id`` 对应的固定实例；若设置了 ``activity_instance_dir``，则从该目录读取对应 JSON。
- ``offline``\ ：每次重置时，从 ``activity_instance_dir`` 指向的任务目录中随机选取一个缓存布局。
- ``online``：reset 时在线重采样物体布局（需 ``online_object_sampling: True`` 且 ``use_presampled_robot_pose: False``，启动较慢）。

.. note::

   单次启动不会自动遍历全部任务或 init state。要评测多个任务，需分别修改 ``activity_name`` 并多次运行，或编写批量脚本；要评测多个 instance，请使用 ``instance_resample_mode: offline`` 并配合 ``rollout_epoch`` 取平均。

``env.eval`` 通用字段
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - BEHAVIOR 设置建议
   * - ``total_num_envs``
     - 所有环境 worker 的环境总数。π₀.₅ 评测默认用一个流水线阶段跑 4 个环境。如果只用一个环境 worker，4 个环境共用一个仿真进程。
   * - ``rollout_epoch``
     - 评测轮数。π₀.₅ 示例默认运行 4 轮。
   * - ``max_episode_steps``
     - π₀.₅ 评测留空时采用该任务的官方时限。SDK 从 0 开始计数，达到人类示范平均步数的 1.5 倍（向下取整）时判定超时。也可以直接填写本回合允许执行的动作数。
   * - ``max_steps_per_rollout_epoch``
     - 留空时按整组动作计算足够覆盖回合时限的采样预算。默认配置下，最后一组只执行回合剩余的动作。手动填写的预算必须能被 ``rollout.model.num_action_chunks`` 整除。
   * - ``num_env_subprocess``
     - 每个环境 worker 的仿真子进程数，默认为 1。进程数少于流水线阶段数时，多个阶段共用进程。只有当前阶段的环境接收动作并累计回合步数。进程处理动作时，会推进其中所有场景的物理仿真。每个 worker 的环境数须能分别被进程数和阶段数整除。
   * - ``skip_intermediate_renders``
     - 设为 ``True`` 时跳过 chunk 内中间 observation，显著加速环境步进；保存的视频仅含 chunk 边界帧。

``omni_config`` 关键字段
~~~~~~~~~~~~~~~~~~~~~~~~

以下字段位于 ``env.eval.omni_config`` 下（继承自 ``examples/embodiment/config/env/behavior_r1pro.yaml``，可在评测 YAML 中覆盖）：

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null          # 指向缓存 instance JSON 目录
           instance_file_format: tro_state        # template | tro_state
           instance_resample_mode: disabled       # disabled | offline | online
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true               # 仅加载任务相关房间，缩短启动时间

更完整的字段说明（``partial_scene_load``、``instance_generator.py`` 用法等）见 :doc:`../../examples/embodied/behavior` 中的「behavior_r1pro.yaml 关键配置说明」。

GPU 与集群布局
~~~~~~~~~~~~~~

BEHAVIOR 环境步进较慢，通常建议给 env 分配足够 GPU，并与 rollout 共享或分离 placement：

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # env 与 rollout 共享全部 GPU（示例默认）

也可将 env 与 rollout 分到不同 GPU 以缓解显存压力，详见训练文档「关键集群配置」一节。

进阶用法
--------

**切换评测任务**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**使用离线 instance 随机采样**

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
           instance_file_format: tro_state
           instance_resample_mode: offline
       rollout_epoch: 5

**调整并行规模**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.total_num_envs=4 \
     env.eval.num_env_subprocess=2

**从训练配置评测**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_ppo_openpi_pi05 \
     rollout.model.model_path=/path/to/model

常见问题
--------

- **数据下载：** BEHAVIOR 资源体积较大，请按 :doc:`../../examples/embodied/behavior` 完成 Isaac Sim、OmniGibson 资产与 license 配置后再评测。
- **Headless 模式：** ``run_eval.sh`` 默认设置 ``OMNIGIBSON_HEADLESS=1``。
- **显存不足：** 减少每张 GPU 上的并行环境数或仿真进程数。每启动一个仿真进程都会占用额外显存。
- **渲染马赛克/模糊：** 当前 GPU 无 Ray Tracing 能力，建议换用 RTX 30/40 系列或更高。
- **启动极慢：** 首次加载大场景耗时较长；保持 ``partial_scene_load: true`` 可只加载任务相关房间。
- **视频帧数少于预期：** ``skip_intermediate_renders: True`` 会跳过 chunk 内中间帧，仅保留策略实际消费的 observation 对应帧。
- **instance 加载失败：** ``activity_instance_dir`` 中的 JSON 文件名须与 ``activity_name``、``activity_definition_id``、``scene_model`` 匹配；详见 ``rlinf/envs/sim/behavior/instance_loader.py``。
- **步数校验失败：** ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除。
