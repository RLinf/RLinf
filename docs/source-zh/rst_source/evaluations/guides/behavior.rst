BEHAVIOR-1K 评测
================

BEHAVIOR-1K 是基于 OmniGibson 与 Isaac Sim 的大规模家居场景仿真基准，控制双臂 R1 Pro 机器人完成抓取、放置、整理等家庭操作任务。RLinf 支持在 BEHAVIOR 环境中并行评测 OpenPI 等 VLA 策略，并输出 ``eval/success_once`` 等指标。

相关训练文档：:doc:`../../examples/embodied/behavior`

环境准备
--------

**安装依赖**

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

当前 ``evaluations/behavior/`` 仅提供 OpenPI π₀.₅ 示例；训练侧还支持 OpenVLA-OFT，可从 ``examples/embodiment/config/`` 中的训练配置派生评测 YAML（见 :doc:`../reference/configuration`）。

**硬件与 Isaac Sim**

RLinf 的安装脚本使用基于 BEHAVIOR-1K v3.9.4 的仿真环境和 Isaac Sim 5.1。BEHAVIOR 对 GPU 与驱动有额外要求，详见训练文档中的 `Isaac Sim 依赖说明 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_\ 。要点如下：

- 需要配备支持 Ray Tracing、显存至少 16 GB 的 NVIDIA GPU。A100、H100 等无 RT Core 的卡不受支持。
- Linux 需要 NVIDIA 驱动 580.65.06 或更高版本。

**环境变量**

每次评测前须设置 OmniGibson 数据路径。Isaac Sim 5.1 安装在 Python 环境中。如果当前终端配置过旧版 Isaac Sim，执行 ``unset ISAAC_PATH EXP_PATH CARB_APP_PATH`` 清除旧路径。

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

BEHAVIOR 资源体积超过 30 GB，2026 资产下载与 license 配置步骤见 :ref:`资源下载 <behavior-sim-assets>`\ 。

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
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

**Step 2：准备模型**

推荐预训练权重：`RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_\ （已在收音机任务上验证）。若使用第三方 OpenPI 权重（如 OpenPI-Comet），需先转换为 PyTorch 格式后再填入 ``rollout.model.model_path``。在仓库根目录下载权重。

.. code-block:: bash

   MODEL_DIR="$PWD/checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 --local-dir "$MODEL_DIR"

**Step 3：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path``。通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`；BEHAVIOR 特有字段与评测协议见下文 :ref:`behavior-eval-config`。

示例 ``behavior_openpi_pi05_eval.yaml`` 中还需保持与训练一致的 OpenPI 参数（``action_dim: 23``、``num_action_chunks: 32``、``openpi.config_name: pi05_behavior`` 等）。``num_action_chunks`` 是环境实际执行的 chunk；网络 horizon 来自 ``pi05_behavior`` 官方 ``TrainConfig.model.action_horizon``\ （**32**），除非另外设置 ``openpi.action_horizon``。

**Step 4：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path="$MODEL_DIR"

默认运行 ``turning_on_radio`` 的实例 0。运行公开实例 301 时，按 :ref:`behavior-public-instances` 修改评测 YAML 后重新运行上面的命令。

**Step 5：查看结果**

终端输出 ``eval/success_once``；日志与视频见 :doc:`../reference/results`。

.. _behavior-eval-config:

评测配置详解
------------

BEHAVIOR 评测 **每次运行对应单个任务** （由 ``omni_config.task.activity_name`` 指定），不会在一次启动中自动遍历全部 100 个任务。以下字段共同决定并行规模、单条轨迹长度与初始场景实例。

评测协议概述
~~~~~~~~~~~~

2026 BEHAVIOR 挑战共 100 个 household 任务（任务名列表见 ``rlinf/envs/sim/behavior/behavior_task.jsonl``）。``behavior_r1pro`` preset 默认任务为 ``turning_on_radio``，场景为 ``house_double_floor_lower``。

每条评测轨迹由以下配置唯一确定：

- ``omni_config.task.activity_name``：任务名称，决定语言指令与 BDDL 定义；
- ``omni_config.task.activity_definition_id``：任务定义变体编号（通常为 ``0``）；
- ``omni_config.task.activity_instance_id`` 与 ``instance_resample_mode``：初始物体布局与机器人位姿。

``instance_resample_mode`` 支持三种模式：

- ``disabled`` （默认）：每次 reset 加载 ``activity_instance_id`` 对应的固定实例；若设置了 ``activity_instance_dir``，则从该目录读取对应 JSON。
- ``offline``：每次 reset 从 ``activity_instance_dir`` 中 **随机** 选取一个缓存实例（需先下载官方 ``2026-challenge-task-instances``）。
- ``online``：reset 时在线重采样物体布局（需 ``online_object_sampling: True``、``use_presampled_robot_pose: False``、``activity_instance_dir: null``，且每个仿真进程只运行一个环境）。

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
     - 全局并行环境数，请按 GPU 显存酌情设置，示例默认 ``4``。
   * - ``rollout_epoch``
     - 评测轮次，相同配置下重复运行并取指标平均；示例默认 ``4``。
   * - ``max_episode_steps``
     - 单条轨迹步数上限。π₀.₅ 示例留 ``null`` 即采用所选任务的官方超时，也可填写动作数上限。
   * - ``max_steps_per_rollout_epoch``
     - 每轮 rollout 的总交互步数。留 ``null`` 时按整组动作计算足够覆盖回合时限的预算，手动填写的预算 **必须能被** ``rollout.model.num_action_chunks`` **整除**。固定回合的最后一组只执行剩余动作。
   * - ``num_env_subprocess``
     - 单个 env worker 内划分的 Isaac 仿真子进程数（默认 ``1``）。适当增大可缓解步进瓶颈，但会成倍增加显存与进程开销；每个 worker 的环境数须能分别被 ``num_env_subprocess`` 和 ``rollout.pipeline_stage_num`` 整除。
   * - ``skip_intermediate_renders``
     - 设为 ``True`` 时跳过 chunk 内中间渲染和 observation，加快环境步进，保存的视频仅含 chunk 边界帧。设为 ``False`` 可录下每一步。

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

更完整的字段说明（``partial_scene_load``、实例设置等）见 :doc:`../../examples/embodied/behavior` 中的「behavior_r1pro.yaml 关键配置说明」。

.. _behavior-public-instances:

固定评测公开实例 301
~~~~~~~~~~~~~~~~~~~~~~~~

收音机的公开实例 301 记录了机器人和任务物体的初始位姿。仿真器先用 0 号完整模板创建场景，再从实例目录读取 301 的布局。将下列字段合入评测 YAML 中同名的配置项，保留原有的其他设置。

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_id: 301
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances
         scene:
           scene_instance: house_double_floor_lower_task_turning_on_radio_0_0_template

运行同一任务的实例 302，只需将 ``activity_instance_id`` 改为 ``302``\ 。更换任务时，更新任务名和场景名，并选用对应的完整模板和实例目录。

GPU 与集群布局
~~~~~~~~~~~~~~

BEHAVIOR 环境步进较慢，通常建议给 env 分配足够 GPU，并与 rollout 共享或分离 placement：

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # env 与 rollout 共享全部 GPU（示例默认）

也可将 env 与 rollout 分到不同 GPU 以缓解显存压力，详见训练文档「关键集群配置」一节。

默认使用一个流水线阶段（\ ``rollout.pipeline_stage_num: 1``），同一仿真进程中的环境一起接收动作。多个阶段共享进程时，所有场景的物理时间都会推进，等待新动作的机器人可能继续按上一条指令运动。

进阶用法
--------

**切换评测任务**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**使用离线 instance 随机采样**

沿用 :ref:`behavior-public-instances` 中的场景模板和实例目录。每次重置时，每个环境各自从目录中随机选择一个布局，同一布局可能重复出现。

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
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
- **Isaac Sim 启动失败：** 执行 ``unset ISAAC_PATH EXP_PATH CARB_APP_PATH`` 清除旧版 Isaac Sim 路径，再激活已安装的环境。
- **Headless 模式：** ``run_eval.sh`` 默认设置 ``OMNIGIBSON_HEADLESS=1``。
- **显存不足：** 降低 ``total_num_envs`` 或 ``num_env_subprocess``。
- **渲染马赛克/模糊：** 当前 GPU 无 Ray Tracing 能力，建议换用 RTX 30/40 系列或更高。
- **启动极慢：** 首次加载大场景耗时较长；保持 ``partial_scene_load: true`` 可只加载任务相关房间。
- **视频帧数少于预期：** ``skip_intermediate_renders: True`` 会跳过 chunk 内中间帧，仅保留策略实际消费的 observation 对应帧。
- **instance 加载失败：** ``activity_instance_dir`` 中的 JSON 文件名须与 ``activity_name``、``activity_definition_id``、``scene_model`` 匹配；详见 ``rlinf/envs/sim/behavior/instance_loader.py``。
- **步数校验失败：** ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除。
