BEHAVIOR-1K 评测
============================

用 OpenPI π₀.₅ 控制 R1 Pro 机器人完成“打开收音机”任务。本例从默认场景开始，也可以选择 2026 挑战的公开实例，测量模型在指定初始布局上的成功率。

安装环境并准备资产
------------------

RLinf 的安装脚本使用基于 BEHAVIOR-1K v3.9.4 的仿真环境和 Isaac Sim 5.1。运行仿真需要配有 RT Core、显存至少 16 GB 的 NVIDIA GPU。驱动和操作系统要求见 `Isaac Sim 5.1 系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_\ 。

在 RLinf 仓库根目录运行安装脚本。如果当前终端配置过其他 Isaac Sim 版本，先清除旧路径。

.. code-block:: bash

   unset ISAAC_PATH EXP_PATH CARB_APP_PATH
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

将 ``OMNIGIBSON_DATA_PATH`` 指向存放资产和密钥的目录。首次安装请按 :ref:`下载 BEHAVIOR 资产 <behavior-sim-assets>` 获取场景、机器人模型、资产密钥和 2026 公开实例。

.. code-block:: bash

   export OMNIGIBSON_DATA_PATH=/path/to/behavior-2026-data

下载模型并运行收音机任务
------------------------

本例使用已在收音机任务上验证的 `π₀.₅ PT50 权重 <https://huggingface.co/RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32>`_\ 。在仓库根目录下载权重，并将模型目录传给评测脚本。

.. code-block:: bash

   MODEL_DIR="$PWD/checkpoints/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32"
   hf download RLinf/RLinf-Pi05-BEHAVIOR-1K-PT50-CS32 --local-dir "$MODEL_DIR"

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path="$MODEL_DIR"

默认运行 ``turning_on_radio`` 的实例 0。终端中的 ``eval/success_once`` 表示至少成功过一次的轨迹所占的比例，日志和录像位置见 :doc:`评测结果 <../reference/results>`\ 。若要运行公开实例 301，按 :ref:`behavior-public-instances` 修改评测 YAML 后重新运行上面的命令。

.. _behavior-eval-config:

配置评测
--------

默认任务与场景
~~~~~~~~~~~~~~

评测入口 ``evaluations/behavior/behavior_openpi_pi05_eval.yaml`` 从 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 读取以下默认值。\ ``activity_instance_dir: null`` 表示使用场景自带的布局，\ ``partial_scene_load: true`` 只加载任务需要的房间。

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null
           instance_file_format: tro_state
           instance_resample_mode: disabled
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true

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

从公开实例中随机采样
~~~~~~~~~~~~~~~~~~~~

保留 301 示例的场景模板和实例目录，用下面的命令评测 5 轮。每次重置时，每个环境各自从目录中随机选择一个实例，同一实例可能重复出现。

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path="$MODEL_DIR" \
     env.eval.omni_config.task.instance_resample_mode=offline \
     env.eval.rollout_epoch=5

若要在重置时现场生成布局，在 ``env.eval.omni_config.task`` 下将 ``instance_resample_mode`` 设为 ``online``，启用 ``online_object_sampling``，并关闭 ``use_presampled_robot_pose``\ 。此模式要求 ``activity_instance_dir`` 为 ``null``，每个仿真进程只运行一个环境。

并行环境、步数与录像
~~~~~~~~~~~~~~~~~~~~

在评测 YAML 的 ``env.eval`` 下设置以下参数。其他评测选项见 :ref:`env-eval-fields`\ 。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - BEHAVIOR 设置建议
   * - ``total_num_envs``
     - 所有环境 worker 的环境总数。此例默认运行 4 个环境。
   * - ``rollout_epoch``
     - 评测轮数。π₀.₅ 示例默认运行 4 轮。
   * - ``max_episode_steps``
     - 留空时采用所选任务的官方步数上限，也可填写每个回合最多执行的动作数。
   * - ``max_steps_per_rollout_epoch``
     - 留空时按整组动作计算足够覆盖回合时限的采样预算。默认配置下，最后一组只执行回合剩余的动作。手动填写的预算必须能被 ``rollout.model.num_action_chunks`` 整除。
   * - ``num_env_subprocess``
     - 每个环境 worker 的仿真进程数，默认为 1。每个 worker 的环境数须能分别被进程数和流水线阶段数整除。
   * - ``skip_intermediate_renders``
     - 设为 ``True`` 时，一个完整的动作块执行 32 步，只在第 32 步渲染并采集观测，录像每个动作块保留一帧。若要录下每一步，将它设为 ``False`` 并开启录像。

默认使用一个流水线阶段（\ ``rollout.pipeline_stage_num: 1``），同一仿真进程里的环境一起接收动作。如果两个阶段共用一个进程，给阶段 A 发动作也会推进阶段 B 的物理时间。阶段 B 的机器人在等待新动作期间可能继续按上一条指令运动。

模型推理和仿真也可以分配到不同的 GPU。\ ``cluster.component_placement`` 的写法见 :doc:`组件放置 <../../concepts/placement>`\ 。

**其他用法**

若要用已有的训练配置做评测，可将启动命令中的配置名换成 ``behavior_ppo_openpi_pi05``\ 。评测配置目录中没有同名文件时，启动脚本会从 ``examples/embodiment/config/`` 加载训练配置。
