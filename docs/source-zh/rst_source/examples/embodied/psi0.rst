Psi0 强化学习
=============

本页介绍如何从官方 OpenOven checkpoint 开始，在 SIMPLE 人形机器人任务上对 Psi0 运行 PPO 后训练。各任务的 checkpoint 和评测命令见 :doc:`SIMPLE 评测指南 </rst_source/evaluations/guides/simple>`。

`Psi0 <https://github.com/physical-superintelligence-lab/Psi0>`__ 由 Qwen3-VL-2B 视觉语言模型（System-2）、flow-matching action expert（System-1）和全身控制器（System-0）组成。RLinf 在 PPO 中冻结 System-2，更新 action expert 与新增的 value head。

概览
----

使用 PPO 微调官方 OpenOven checkpoint。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      SIMPLE

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      G1 OpenOven Teleop 任务

   .. grid-item-card:: 硬件
      :text-align: center

      4 GPU

观测与动作
~~~~~~~~~~

adapter 将头部相机图像、机器人状态和任务指令输入模型，再将预测的动作交给 SIMPLE 全身控制器执行。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - ``main_images`` 中的头部相机 RGB 图像与 ``states`` 中的 32 维本体状态，由 checkpoint 的 transform 完成状态 padding 和归一化。
   * - 动作
     - 36 维指令，包含上身关节、腰部、基座高度和导航命令。每次预测 30 步，执行 24 步，保留 6 步用于下一轮 RTC 条件。
   * - 奖励
     - SIMPLE 任务奖励与成功信号。
   * - 指令
     - ``task_descriptions`` 中的自然语言任务指令。

.. _psi0-setup-env:
.. _psi0-qwen-metadata:
.. _simple-sim-assets:
.. _simple-link-assets:
.. _psi0-model-download:
.. _psi0-ppo-training:

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: SIMPLE

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model psi0 --env simple && source .venv/bin/activate

      **2a. 下载 Qwen3-VL 元数据与 SIMPLE 共享资源**

      下载 Qwen3-VL 元数据 snapshot（按模型 ID 从 ``HF_HOME`` 读取，不要传入 ``--local-dir``）与 `USC-PSI-Lab/SIMPLE <https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE>`_ 共享仿真资源，并软链到 ``${VIRTUAL_ENV}/SIMPLE/data``：

      .. code-block:: bash

         export HF_HOME=./hf-cache
         export SIMPLE_ASSETS_DIR=./simple_assets

         hf download Qwen/Qwen3-VL-2B-Instruct \
           chat_template.json config.json generation_config.json merges.txt \
           preprocessor_config.json tokenizer.json tokenizer_config.json \
           video_preprocessor_config.json vocab.json

         bash requirements/embodied/download_assets.sh --assets simple --dir "${SIMPLE_ASSETS_DIR}"
         rm -rf "${VIRTUAL_ENV}/SIMPLE/data"
         ln -s "$(realpath "${SIMPLE_ASSETS_DIR}")" "${VIRTUAL_ENV}/SIMPLE/data"

      **2b. 下载 OpenOven checkpoint 与 reset 数据**

      下载官方 OpenOven checkpoint 与任务 reset 数据：

      .. code-block:: bash

         export PSI0_ARTIFACT_ROOT=./psi0-simple-artifacts
         export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
         export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

         hf download USC-PSI-Lab/psi-model \
           "${PSI0_RUN_REL}/argv.txt" "${PSI0_RUN_REL}/run_config.json" \
           "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
           --revision d34a91932d25c45ef211582315b9224c7dc8ace9 --local-dir ./psi0-simple-artifacts/psi-model
         export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

         hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
           --repo-type dataset --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
           --local-dir ./psi0-simple-artifacts/psi-data
         export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
         mkdir -p "${SIMPLE_RESET_DIR}"
         unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" -d "${SIMPLE_RESET_DIR}"

      **2c. 下载 HSSD 场景**

      下载并解压 HSSD 场景到 ``SIMPLE_ASSETS_DIR``：

      .. code-block:: bash

         export SIMPLE_ASSETS_DIR=./simple_assets
         hf download USC-PSI-Lab/SIMPLE scenes_hssd_102344280.zip \
           --repo-type dataset --local-dir ./simple_assets
         unzip -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip" -d "${SIMPLE_ASSETS_DIR}"
         rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip"

      **3. 启动训练**

      .. list-table::
         :header-rows: 1
         :widths: 24 50 26

         * - 配方
           - 配置
           - 命令后缀
         * - Psi0 + PPO（OpenOven）
           - ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml``
           - ``simple_openoven_ppo_psi0``

      在 ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml`` 中，将 ``actor.model.model_path`` 设为 ``PSI0_RUN_DIR``\ （rollout 会自动沿用同一路径），将 ``env.train.reset_dataset.path`` 设为 ``SIMPLE_RESET_DIR``：

      .. code-block:: yaml

         env:
           train:
             reset_dataset:
               path: ./psi0-simple-artifacts/reset-states/G1WholebodyOpenOvenTeleop-v0
         actor:
           model:
             model_path: ./psi0-simple-artifacts/psi-model/psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

      启动 OpenOven 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

.. note::

   训练或评测前保持 ``HF_HOME`` 指向同一个缓存目录。``VIRTUAL_ENV`` 默认位于 ``.venv``\ （预构建容器中位于 ``/opt/venv/psi0``）。若已下载过共享 SIMPLE 资源，将 ``SIMPLE_ASSETS_DIR`` 设为该目录并跳过 ``download_assets.sh``。

配置使用 stochastic-transition 采样运行 PPO，冻结 System-2，训练 action expert 和 value head。默认四卡四环境、DR0、600 步 horizon。每轮通过 8 轮 rollout 采集 32 条轨迹，``global_batch_size=200``、``update_epoch=1``，执行 4 次优化器更新。

.. dropdown:: 进阶配置（可选）

   请保持以下 RTC 与 pipeline 设置不变。RTC 参数的默认值位于 ``examples/embodiment/config/model/psi0.yaml``；每个 rollout worker 只运行一个环境，因此 ``rollout.pipeline_stage_num`` 需保持为 ``1``。

   .. code-block:: yaml

      actor:
        model:
          num_action_chunks: 24
          psi0:
            plan_horizon: 30
            overlap_horizon: 6
      rollout:
        pipeline_stage_num: 1

可视化与结果
------------

通过 TensorBoard 和终端日志查看训练进度：

.. code-block:: text

   logs/<timestamp>-simple_openoven_ppo_psi0/run_embodiment.log

指标含义见 :doc:`训练指标 </rst_source/reference/metrics>`。保存的权重可按 :ref:`评测训练后的 Checkpoint <simple-trained-checkpoint>` 运行评测。
