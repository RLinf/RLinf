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

      8 个人形机器人 Teleop 任务

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

在 RLinf 仓库根目录安装并激活 Psi0 + SIMPLE 运行环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model psi0 --env simple
   source .venv/bin/activate

安装脚本固定了
`Psi0 <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
和
`SIMPLE <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_
代码版本。系统依赖见 :doc:`安装指南 </rst_source/start/installation>`。

同一条安装命令会下载 OpenOven 的仿真资源，并解压到 ``.venv/SIMPLE/data``。
对应目录已经存在时跳过该包。安装前设置 ``SIMPLE_ASSETS_PATH`` 后，资源放在虚拟环境外面，
``.venv/SIMPLE/data`` 只保留指向它的软链，重复安装不会重新下载。
详见 :ref:`仿真资源 <simple-sim-assets>`。

.. _psi0-model-download:

下载模型
--------

下载官方 OpenOven checkpoint 及其 run 配置：

.. code-block:: bash

   export PSI0_ARTIFACT_ROOT=/path/to/psi0-simple-artifacts
   export HF_HOME="${PSI0_ARTIFACT_ROOT}/hf-cache"
   export SIMPLE_TASK=G1WholebodyOpenOvenTeleop-v0
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenoventeleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605120604

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision d34a91932d25c45ef211582315b9224c7dc8ace9 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

将 ``model_path`` 指向这个 run 目录。目录内的配置决定模型结构和预处理 transform，
联合 checkpoint 包含 System-2 和 System-1 权重。

loader 还会按模型 ID 读取 Qwen 配置、tokenizer 和 processor，
在同一 ``HF_HOME`` 中缓存：

.. code-block:: bash

   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json

保留 Hugging Face 标准缓存，供 loader 离线复用。Qwen metadata 默认解析 ``main``；
严格锁定版本时，loader 也需使用对应 revision。

启动 SIMPLE 前，下载并解压同一任务的 reset 数据：

.. code-block:: bash

   hf download USC-PSI-Lab/psi-data "simple-eval/${SIMPLE_TASK}.zip" \
     --repo-type dataset \
     --revision 6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203 \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/${SIMPLE_TASK}"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${PSI0_ARTIFACT_ROOT}/psi-data/simple-eval/${SIMPLE_TASK}.zip" \
     -d "${SIMPLE_RESET_DIR}"

训练配置通过 ``PSI0_RUN_DIR`` 读取 actor 和 rollout 的模型，
通过 ``SIMPLE_RESET_DIR`` 读取 reset 数据。

.. _psi0-ppo-training:

运行
----

导出资产路径后，启动 OpenOven 训练配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh simple_openoven_ppo_psi0

配置使用 stochastic-transition 采样运行 PPO，冻结 System-2，训练 action expert 和 value head。
默认四卡四环境、DR0、600 步 horizon。每轮通过 8 轮 rollout 采集 32 条轨迹，
``global_batch_size=200``、``update_epoch=1``，执行 4 次优化器更新。

切换其他 Teleop 任务时，按 :ref:`SIMPLE 任务表 <simple-tasks>` 将 ``PSI0_RUN_DIR``
和 ``SIMPLE_RESET_DIR`` 指向对应资产。在
``examples/embodiment/config/simple_openoven_ppo_psi0.yaml`` 中修改
``env.train.init_params.task_id``、``env.train.max_episode_steps`` 和
``env.train.max_steps_per_rollout_epoch``，并让 ``actor.global_batch_size``
整除每轮采样的 chunk 数。训练脚本从 YAML 读取这些设置。

.. admonition:: 进一步配置
   :class: note

   - 模型设置 → ``examples/embodiment/config/model/psi0.yaml``，
     保持 plan、execution 和 overlap horizon 为 30/24/6。
   - 放置 → 每个 rollout worker 使用一个环境，保持 ``rollout.pipeline_stage_num: 1``。

可视化与结果
------------

通过 TensorBoard 和终端日志查看训练进度：

.. code-block:: text

   logs/<timestamp>-simple_openoven_ppo_psi0/run_embodiment.log

指标含义见 :doc:`训练指标 </rst_source/reference/metrics>`。
保存的权重可按 :ref:`评测训练后的 Checkpoint <simple-trained-checkpoint>` 运行评测。
