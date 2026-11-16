Psi0
====

本页介绍如何从官方 OpenOven checkpoint 开始，在 SIMPLE 人形机器人任务上对 Psi0
进行 PPO 后训练。各任务的 checkpoint 和评测命令见
:doc:`SIMPLE 评测指南 </rst_source/evaluations/guides/simple>`。

`Psi0 <https://github.com/physical-superintelligence-lab/Psi0>`__ 由 Qwen3-VL-2B
视觉语言模型（System-2）、flow-matching action expert（System-1）和全身控制器
（System-0）组成。RLinf 在 PPO 中冻结 System-2，更新 action expert 与新增的 value head。

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
     - ``main_images`` 中的头部相机 RGB 图像与 ``states`` 中的 32 维本体状态，
       由 checkpoint 的 transform 完成状态 padding 和归一化。
   * - 动作
     - 36 维指令，包含上身关节、腰部、基座高度和导航命令。
       每次预测 30 步，执行 24 步，保留 6 步用于下一轮 RTC 条件。
   * - 奖励
     - SIMPLE 任务奖励与成功信号。
   * - 指令
     - ``task_descriptions`` 中的自然语言任务指令。

安装
----

Psi0 + SIMPLE 的准备分为五步。第 1–4 步由所有 SIMPLE 任务共用，
第 5 步下载单个任务的资源。

.. _psi0-setup-env:

1. 安装运行环境
~~~~~~~~~~~~~~~

在 RLinf 仓库根目录安装并激活 Psi0 + SIMPLE 运行环境：

.. code-block:: bash

   # 中国大陆用户可以添加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model psi0 --env simple
   source .venv/bin/activate

也可以构建 Docker 镜像，并切换到其中的 Psi0 环境：

.. code-block:: bash

   docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-simple -t rlinf:embodied-simple .
   # Inside the container:
   source switch_env psi0

.. _psi0-qwen-metadata:

2. 下载 Qwen Metadata
~~~~~~~~~~~~~~~~~~~~~

联合 checkpoint 已包含 Qwen3-VL 权重，但 loader 仍会按模型 ID 读取 Qwen 的配置、
tokenizer 和 processor：

.. code-block:: bash

   # 中国大陆用户可以设置 HF_ENDPOINT=https://hf-mirror.com，使用代理加速下载。
   export HF_HOME=/path/to/hf-cache
   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json

.. note::

   loader 需要这些文件的 Hugging Face 原始 snapshot。下载时请设置 ``HF_HOME``，不要使用 ``--local-dir``；训练或评测前也要导出同一个 ``HF_HOME``。

.. _simple-sim-assets:

3. 下载仿真资源
~~~~~~~~~~~~~~~

从 `USC-PSI-Lab/SIMPLE <https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE>`_
下载各 SIMPLE 任务共用的资源（G1 机器人、GraspNet 物体、铰接物体和 vMaterials）。
如果已经下载过，将 ``SIMPLE_ASSETS_DIR`` 设为该目录并跳过下载：

.. code-block:: bash

   export SIMPLE_ASSETS_DIR=/path/to/simple_assets
   # 中国大陆用户可以添加 --use-mirror，使用代理加速下载。
   bash requirements/embodied/download_assets.sh --assets simple --dir "${SIMPLE_ASSETS_DIR}"

.. _simple-link-assets:

4. 链接仿真资源
~~~~~~~~~~~~~~~

SIMPLE 从其代码目录下的 ``data/`` 读取资源，安装脚本把代码放在
``${VIRTUAL_ENV}/SIMPLE``。代码中的 ``data/`` 只有 SIMPLE 的 README 和下载脚本，
将它替换为指向资源目录的软链：

.. code-block:: bash

   rm -rf "${VIRTUAL_ENV}/SIMPLE/data"
   ln -s "${SIMPLE_ASSETS_DIR}" "${VIRTUAL_ENV}/SIMPLE/data"

.. note::

   ``VIRTUAL_ENV`` 是第 1 步激活的虚拟环境，默认位于 ``/path/to/rlinf/.venv``；使用 Docker 时位于 ``/opt/venv/psi0``。

.. _psi0-model-download:

5. 下载任务资源
~~~~~~~~~~~~~~~

下载官方 OpenOven checkpoint 及其 run 配置、OpenOven 的 reset 数据和 HSSD 场景：

.. code-block:: bash

   export PSI0_ARTIFACT_ROOT=/path/to/psi0-simple-artifacts
   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision d34a91932d25c45ef211582315b9224c7dc8ace9 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"
   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

   hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
     --repo-type dataset \
     --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" \
     -d "${SIMPLE_RESET_DIR}"

   hf download USC-PSI-Lab/SIMPLE scenes_hssd_102344280.zip \
     --repo-type dataset \
     --local-dir "${SIMPLE_ASSETS_DIR}"
   unzip -q "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip" -d "${SIMPLE_ASSETS_DIR}"
   rm "${SIMPLE_ASSETS_DIR}/scenes_hssd_102344280.zip"

.. _psi0-ppo-training:

运行
----

在 ``examples/embodiment/config/simple_openoven_ppo_psi0.yaml`` 中，将
``rollout.model.model_path`` 设为第 5 步的 ``PSI0_RUN_DIR``，将
``env.train.reset_dataset.path`` 设为 ``SIMPLE_RESET_DIR``。actor 使用同一个模型路径：

.. code-block:: yaml

   env:
     train:
       reset_dataset:
         path: /path/to/psi0-simple-artifacts/reset-states/G1WholebodyOpenOvenTeleop-v0
   rollout:
     model:
       model_path: /path/to/psi0-simple-artifacts/psi-model/psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

然后启动 OpenOven 训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

配置使用 stochastic-transition 采样运行 PPO，冻结 System-2，训练 action expert 和 value head。
默认四卡四环境、DR0、600 步 horizon。每轮通过 8 轮 rollout 采集 32 条轨迹，
``global_batch_size=200``、``update_epoch=1``，执行 4 次优化器更新。

.. note::

   请保持以下设置不变。RTC 参数的默认值位于 ``examples/embodiment/config/model/psi0.yaml``；每个 rollout worker 只运行一个环境，因此 ``rollout.pipeline_stage_num`` 需保持为 ``1``。

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

指标含义见 :doc:`训练指标 </rst_source/reference/metrics>`。
保存的权重可按 :ref:`评测训练后的 Checkpoint <simple-trained-checkpoint>` 运行评测。
