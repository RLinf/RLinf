SIMPLE 评测
===========

本页介绍如何通过 RLinf 在 SIMPLE 的八个 Teleop 任务上评测 Psi0。每个任务使用对应的 checkpoint 和 reset states；PPO 后训练见 :doc:`Psi0 指南 </rst_source/examples/embodied/psi0>`。

适用范围
--------

当前范围是固定 Psi0 发布中具有 task-specific checkpoint 的八个 Teleop 任务。``G1WholebodyBendPickMP-v0`` 和 ``G1WholebodyTabletopGraspMP-v0`` 使用另一套 state、action 和 System-0 协议，暂不支持。

.. _simple-reset-data:

一键上手
--------

八个任务共用 :doc:`Psi0 指南 </rst_source/examples/embodied/psi0>` 的第 1–4 步。先完成以下共用准备（如需使用 Docker，见 :ref:`第 1 步 <psi0-setup-env>`）：

- :ref:`下载 Qwen Metadata <psi0-qwen-metadata>`。
- :ref:`下载仿真资源 <simple-sim-assets>` 到 ``SIMPLE_ASSETS_DIR``，或复用已经下载的资源。
- 将资源 :ref:`链接 <simple-link-assets>` 到 ``${VIRTUAL_ENV}/SIMPLE/data``。

.. note::

   Psi0 loader 需要 Qwen metadata 的 Hugging Face 原始 snapshot。下载时请设置 ``HF_HOME``，不要使用 ``--local-dir``；评测前也要导出同一个 ``HF_HOME``。

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

下方命令完成安装、下载 OpenOven 的 checkpoint、reset states 与 HSSD 场景（解压到 ``SIMPLE_ASSETS_DIR``），并启动 ``evaluations/simple/simple_psi0_eval.yaml``\ （默认运行 OpenOven、DR0、1 条轨迹）。所有任务共用 ``PSI0_ARTIFACT_ROOT``；切换任务时，按 :ref:`任务列表 <simple-tasks>` 同一行同步替换 ``SIMPLE_TASK``、``PSI0_RUN_REL`` 与 ``SIMPLE_SCENE``。

.. tab-set::

   .. tab-item:: Psi0 + SIMPLE（默认 OpenOven）

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

         # 2. 下载任务 checkpoint、reset 数据与 HSSD 场景
         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
         export SIMPLE_SCENE=102344280

         mkdir -p "${PSI0_ARTIFACT_ROOT}"
         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" \
           "${PSI0_RUN_REL}/run_config.json" \
           "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 \
           --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset \
           --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
           --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
         mkdir -p "${SIMPLE_RESET_DIR}"
         unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" \
           -d "${SIMPLE_RESET_DIR}"

         hf download USC-PSI-Lab/SIMPLE "scenes_hssd_${SIMPLE_SCENE}.zip" \
           --repo-type dataset \
           --local-dir "${SIMPLE_ASSETS_DIR}"
         unzip -o -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_${SIMPLE_SCENE}.zip"

         # 3. 启动评测（1 条轨迹，DR0）
         bash evaluations/run_eval.sh simple simple_psi0_eval \
           rollout.model.model_path="${PSI0_RUN_DIR}" \
           env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

评测输出 ``eval/success``，日志和视频保存在 ``logs/<timestamp>-simple_psi0_eval/`` 下。

完整的三个 DR level 各运行 10 条轨迹：

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_psi0_eval \
       rollout.model.model_path="${PSI0_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10
   done

.. _simple-tasks:

支持的任务
----------

下表给出 `固定 Psi0 checkpoint 发布 <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_ 中的八个可替换 Teleop 任务。``官方结果`` 是 `SIMPLE 固定版本结果表 <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_ 中 Psi0 在 level 0/1/2 各 10 条轨迹上的成功次数。

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

切换其他任务
~~~~~~~~~~~~

其他 Teleop 任务沿用同一配置。以 XMovePick 为例，先更新下面五个变量；已导出的路径不会随任务或 run 名自动更新：

.. code-block:: bash

   export SIMPLE_TASK=G1WholebodyXMovePickTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
   export SIMPLE_SCENE=107734119_175999932
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"

资产尚未准备时，用新变量重新执行模型、reset 数据和场景下载命令，然后运行评测：

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.init_params.task_id="simple/${SIMPLE_TASK}" \
     env.eval.max_episode_steps=800 \
     env.eval.max_steps_per_rollout_epoch=816

checkpoint、reset 数据与 ``task_id`` 必须属于同一任务。三档评测循环也需补上 ``env.eval.init_params.task_id="simple/${SIMPLE_TASK}"``，并保留两项资产路径和步数覆盖。``max_steps_per_rollout_epoch`` 是不小于任务 horizon 的最小 24 的倍数。

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

所有八个任务都使用 ``mujoco_isaac``、50 Hz render、200 Hz physics、最多 300 步 stabilization，以及 30/24/6 RTC。官方结果每档只有 10 条轨迹，应分别报告三个 level；单条 smoke 或总成功数不能作为严格复现结论。

.. _simple-trained-checkpoint:

评测训练后的 Checkpoint
------------------------

使用原始 run 目录构建模型和 processor，通过 ``runner.ckpt_path`` 加载保存的 PPO 权重。设置 ``rollout.model.add_value_head=true`` 恢复一同保存的 value-head 参数。

评测 OpenOven checkpoint 时，先选择对应资产，再按官方 1000 步 horizon 运行：

.. code-block:: bash

   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   export PSI0_PPO_CHECKPOINT=/absolute/path/to/global_step_20

   bash evaluations/run_eval.sh simple simple_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     rollout.model.add_value_head=true \
     runner.ckpt_path="${PSI0_PPO_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
     env.eval.reset_dataset.num_episodes=10 \
     env.eval.rollout_epoch=10
