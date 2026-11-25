SIMPLE 评测
===========

概览
----

在 SIMPLE 的八个 G1 人形全身 Teleop 任务上评测 Psi0，每个任务使用对应的 checkpoint、reset states 与 HSSD 场景。八个任务均运行在 ``mujoco_isaac``\ （50 Hz 渲染、200 Hz 物理与 30/24/6 RTC 协议），并输出 ``eval/success``。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      SIMPLE（8 个 G1 人形全身 Teleop 任务）

   .. grid-item-card:: 支持模型
      :text-align: center

      Psi0（任务专属与 PPO 微调 checkpoint）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/simple/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`Psi0 指南 </rst_source/examples/embodied/psi0>`

.. note::

   **适用范围：** 覆盖固定 Psi0 发布中带 task-specific checkpoint 的八个 Teleop 任务。``G1WholebodyBendPickMP-v0`` 与 ``G1WholebodyTabletopGraspMP-v0`` 使用不同的 state、action 与 System-0 协议，暂不支持。

.. _simple-reset-data:

安装与运行
----------

先按 :doc:`Psi0 指南 </rst_source/examples/embodied/psi0>` 完成共用准备：:ref:`环境安装 <psi0-setup-env>`、:ref:`下载 Qwen3-VL 元数据 <psi0-qwen-metadata>`\ （通过 ``HF_HOME`` 缓存，不使用 ``--local-dir``）、将 :ref:`SIMPLE 仿真资源 <simple-sim-assets>` 下载到 ``SIMPLE_ASSETS_DIR``，并 :ref:`软链资源 <simple-link-assets>` 到 ``${VIRTUAL_ENV}/SIMPLE/data``。

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: 单条轨迹快测（DR0）

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

      **2. 下载权重与 Reset 数据**

      下载 OpenOven checkpoint 与 reset 数据（依赖 Psi0 指南中导出的 ``HF_HOME``）：

      .. code-block:: bash

         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" "${PSI0_RUN_REL}/run_config.json" "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 --local-dir ./psi0-simple-artifacts/psi-model
         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 --local-dir ./psi0-simple-artifacts/psi-data
         mkdir -p "${SIMPLE_RESET_DIR}" && unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" -d "${SIMPLE_RESET_DIR}"

      **3. 下载 HSSD 场景**\ （依赖 Psi0 指南中导出的 ``SIMPLE_ASSETS_DIR``）

      .. code-block:: bash

         export SIMPLE_SCENE=102344280
         hf download USC-PSI-Lab/SIMPLE "scenes_hssd_${SIMPLE_SCENE}.zip" \
           --repo-type dataset --local-dir "${SIMPLE_ASSETS_DIR}"
         unzip -o -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip"

      **4. 启动评测**

      .. code-block:: bash

         bash evaluations/run_eval.sh simple simple_psi0_eval \
           rollout.model.model_path="${PSI0_RUN_DIR}" \
           env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

   .. tab-item:: 完整三档 DR（30 条轨迹）

      在三个域随机化（DR）档位各运行 10 条轨迹，并分别报告三个 level 的成功率。

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

      **2. 下载权重与 Reset 数据**

      .. code-block:: bash

         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" "${PSI0_RUN_REL}/run_config.json" "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 --local-dir ./psi0-simple-artifacts/psi-model
         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 --local-dir ./psi0-simple-artifacts/psi-data
         mkdir -p "${SIMPLE_RESET_DIR}" && unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" -d "${SIMPLE_RESET_DIR}"

      **3. 下载 HSSD 场景**

      .. code-block:: bash

         export SIMPLE_SCENE=102344280
         hf download USC-PSI-Lab/SIMPLE "scenes_hssd_${SIMPLE_SCENE}.zip" \
           --repo-type dataset --local-dir "${SIMPLE_ASSETS_DIR}"
         unzip -o -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip"

      **4. 启动评测**

      .. code-block:: bash

         for SIMPLE_DR_LEVEL in 0 1 2; do
           bash evaluations/run_eval.sh simple simple_psi0_eval \
             rollout.model.model_path="${PSI0_RUN_DIR}" \
             env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
             env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
             env.eval.reset_dataset.num_episodes=10 \
             env.eval.rollout_epoch=10
         done

``evaluations/simple/simple_psi0_eval.yaml`` 默认运行 OpenOven（1 条轨迹，DR0）。所有任务共用 ``PSI0_ARTIFACT_ROOT``；切换任务时按 :ref:`任务列表 <simple-tasks>` 同一行同步替换 ``SIMPLE_TASK``、``PSI0_RUN_REL`` 与 ``SIMPLE_SCENE``。

.. _simple-tasks:

.. dropdown:: 支持的任务与切换方式

   `固定 Psi0 checkpoint 发布 <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_ 中的八个 Teleop 任务（``官方结果`` 为 `SIMPLE 固定版本结果表 <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_ 在 level 0/1/2 各 10 条轨迹上的成功次数）：

   .. list-table::
      :header-rows: 1
      :widths: 22 40 16 10 12

      * - ``SIMPLE_TASK``
        - run 名
        - ``SIMPLE_SCENE``
        - horizon
        - 官方结果
      * - ``G1WholebodyXMovePickTeleop-v0``
        - ``g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205``
        - ``107734119_175999932``
        - 800
        - 10/10/6
      * - ``G1WholebodyHandoverTeleop-v0``
        - ``g1wholebodyhandover-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604071507``
        - ``107734119_175999932``
        - 800
        - 7/7/10
      * - ``G1WholebodyLocomotionPickBetweenTablesTeleop-v0``
        - ``g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126``
        - ``102344022``
        - 1200
        - 7/5/6
      * - ``G1WholebodyXMoveBendPickTeleop-v0``
        - ``g1wholebodyxmovebendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b112.gpus7.2604100422``
        - ``107734119_175999932``
        - 800
        - 10/9/9
      * - ``G1WholebodyCloseDoorTeleop-v0``
        - ``g1wholebodyclosedoorteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605070100``
        - ``107734119_175999932``
        - 1000
        - 10/10/10
      * - ``G1WholebodyOpenOvenTeleop-v0``
        - ``g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604``
        - ``102344280``
        - 1000
        - 7/5/4
      * - ``G1WholebodyOpenFaucetTeleop-v0``
        - ``g1wholebodyopenfaucetteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605081439``
        - ``107733960_175999701``
        - 1000
        - 3/3/4
      * - ``G1WholebodyPickAndPlaceAndHugContainerTeleop-v0``
        - ``g1wholebodypickandplaceandhugcontainerteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604280201``
        - ``102344307``
        - 1000
        - 7/6/3

   **切换其他任务**

   同步更新以下五个变量（已导出的路径不会自动更新），若资产尚未下载则重跑下载命令，再覆盖 ``task_id`` 与步数上限：

   .. code-block:: bash

      export SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0
      export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
      export SIMPLE_SCENE=107734119_175999932
      export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
      export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

      bash evaluations/run_eval.sh simple simple_psi0_eval \
        rollout.model.model_path="${PSI0_RUN_DIR}" \
        env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
        env.eval.init_params.task_id="simple/${SIMPLE_TASK}" \
        env.eval.max_episode_steps=800 \
        env.eval.max_steps_per_rollout_epoch=816

   ``max_steps_per_rollout_epoch`` 须为不小于任务 horizon 的最小 24 的倍数：

   .. _simple-task-horizons:

   .. list-table::
      :header-rows: 1
      :widths: 45 25 30

      * - 任务
        - ``max_episode_steps``
        - ``max_steps_per_rollout_epoch``
      * - XMovePick、Handover、XMoveBendPick
        - 800
        - 816
      * - CloseDoor、OpenOven、OpenFaucet、PickAndPlaceAndHugContainer
        - 1000
        - 1008
      * - LocomotionPickBetweenTables
        - 1200
        - 1200

.. _simple-trained-checkpoint:

.. dropdown:: 评测训练后的 Checkpoint

   使用原始 run 目录构建模型与 processor，通过 ``runner.ckpt_path`` 加载保存的 PPO 权重，并设置 ``rollout.model.add_value_head=true`` 恢复 value-head 参数：

   .. code-block:: bash

      export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
      export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
      export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
      export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
      export PSI0_PPO_CHECKPOINT=./logs/psi0_ppo/checkpoints/global_step_20

      bash evaluations/run_eval.sh simple simple_psi0_eval \
        rollout.model.model_path="${PSI0_RUN_DIR}" \
        rollout.model.add_value_head=true \
        runner.ckpt_path="${PSI0_PPO_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
        env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
        env.eval.reset_dataset.num_episodes=10 \
        env.eval.rollout_epoch=10

结果与常见问题
--------------

评测输出 ``eval/success``，日志与视频保存在 ``logs/<timestamp>-simple_psi0_eval/``\ （详见 :doc:`../reference/results`）。八个任务均采用 ``mujoco_isaac``、50 Hz 渲染、200 Hz 物理、最多 300 步稳定以及 30/24/6 RTC。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - Qwen metadata 加载失败
     - Psi0 需要 Hugging Face 原始 snapshot 结构；下载时设置 ``HF_HOME``\ （不使用 ``--local-dir``），评测前导出同一个 ``HF_HOME``。
   * - 任务不匹配或缺少 HSSD 场景
     - 同步更新 ``SIMPLE_TASK``、``PSI0_RUN_REL``、``SIMPLE_SCENE``、``PSI0_RUN_DIR`` 与 ``SIMPLE_RESET_DIR``，并传入 ``env.eval.init_params.task_id="simple/${SIMPLE_TASK}"``。
   * - 步数校验报错
     - 将 ``env.eval.max_steps_per_rollout_epoch`` 设为不小于 ``max_episode_steps`` 的最小 24 的倍数（800 对应 ``816``，1000 对应 ``1008``，1200 对应 ``1200``）。
   * - PPO checkpoint 缺少 value head 权重
     - 传入 ``runner.ckpt_path`` 时同时设置 ``rollout.model.add_value_head=true``。
