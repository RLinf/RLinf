LingbotVLA 2.0 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 框架（图片来源：`Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__）。

在 RLinf 中对 LingbotVLA 2.0 运行 RoboTwin 监督微调（SFT）。先将 RoboTwin HDF5 示范转换为 LeRobot 格式，再启动 SFT 配方、查看训练 checkpoint，最后使用 RoboTwin 评测入口评测 SFT checkpoint。

概览
----------------------------------------

需要从官方 RoboTwin checkpoint 出发训练动作专家时，使用本页面的 SFT 流程。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      LingbotVLA 2.0

   .. grid-item-card:: 方法
      :text-align: center

      动作专家 SFT

   .. grid-item-card:: 数据
      :text-align: center

      RoboTwin 2.0 → LeRobot

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPU

| **你将完成：** 安装模型环境 → 转换 RoboTwin 示范 → 运行 ``robotwin_sft_lingbotvlav2`` → 查看 loss 与 checkpoint → 评测 checkpoint。
| **前置条件：** :doc:`RLinf 安装 </rst_source/start/installation>` · RoboTwin RLinf_support 仓库与资产 · LingbotVLA 2.0 RoboTwin checkpoint · Qwen3-VL tokenizer checkpoint。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - RoboTwin
     - Click Bell
     - ``robotwin_sft_lingbotvlav2``
     - 从官方 RoboTwin checkpoint 出发，对动作专家进行 SFT。

观测与动作
~~~~~~~~~~

SFT 数据沿用发布模型使用的 RoboTwin 观测与动作约定。

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 头部相机、左右腕部相机，以及 14 维关节状态。
   * - Action
     - 由 50 步、每步 55 维的 latent 动作块解码得到 14 维关节指令。
   * - Prompt
     - RoboTwin episode 中的自然语言任务指令。

准备数据集
----------------------------------------

启动 SFT 前，先把解压后的 RoboTwin 数据转换为一个 LeRobot 数据集。输入目录必须包含 ``data/episode<i>.hdf5`` 和 ``instructions/episode<i>.json``。

设置输出目录并转换 Click Bell 数据：

.. code-block:: bash

    export LINGBOT_VLA_SFT_DATASET=/path/to/robotwin2_lerobot/click_bell_clean_50
    python toolkits/lerobot/convert_robotwin_to_lerobot.py \
      --raw-root /path/to/click_bell/aloha-agilex_clean_50 \
      --output-root "$LINGBOT_VLA_SFT_DATASET"

转换脚本会写出包含三个 RoboTwin 相机流、14 维 state、下一时刻 action 和 episode 指令的 LeRobot v3 数据集。

下载模型
----------------------------------------

下载发布的 RoboTwin checkpoint 与 Qwen3-VL tokenizer checkpoint，然后设置 SFT 模型模板读取的路径：

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin \
      --local-dir /path/to/lingbot-vla-v2-6b-robotwin
    hf download Qwen/Qwen3-VL-4B-Instruct \
      --local-dir /path/to/Qwen3-VL-4B-Instruct

    export LINGBOT_VLA_V2_PATH=/path/to/.venv/lingbot-vla-v2
    export LINGBOT_VLA_V2_MODEL_ROOT=/path/to/lingbot-vla-v2-6b-robotwin
    export LINGBOT_VLA_V2_CHECKPOINT=$LINGBOT_VLA_V2_MODEL_ROOT/checkpoints/global_step_50000/hf_ckpt
    export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct

模型模板会从 ``LINGBOT_VLA_V2_MODEL_ROOT`` 读取 ``lingbotvla_cli.yaml``，并从 ``LINGBOT_VLA_V2_PATH`` 下的官方 robot 配置和归一化统计路径读取其他设置。

安装
----------------------------------------

运行 SFT 脚本前，先准备 LingbotVLA 2.0 RoboTwin 环境：

.. code-block:: bash

    bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin
    source .venv/bin/activate

Docker 选项和 RoboTwin 仓库准备方式见 :doc:`LingbotVLA 2.0 强化学习 <lingbotvlav2>` 与 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>`。

运行
----------------------------------------

完成数据集、模型路径和环境准备后，启动 SFT 配方：

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

该配方使用 ``examples/sft/config/model/lingbotvlav2.yaml``，训练 30,000 steps，每 1,000 steps 保存 checkpoint，并在 SFT 模型配置中使用 ``flex_cached``。启动脚本会把日志写入 ``logs/<timestamp>-robotwin_sft_lingbotvlav2``。

评测 checkpoint 前，设置评测配置使用的 RoboTwin 环境变量：

.. code-block:: bash

    export ROBOT_PLATFORM=ALOHA
    export ROBOTWIN_PATH=/path/to/RoboTwin
    export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH

使用现有 RoboTwin 评测入口评测 SFT checkpoint。命令行覆盖会让评测使用与 SFT 相同的 ``flex_cached`` attention；共享 eval YAML 仍保留 ``eager`` 默认值。

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      rollout.model.lingbotvlav2.attention_implementation=flex_cached \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

评测协议和结果解释见 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>`。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard，并指向 SFT 日志目录：

.. code-block:: bash

    tensorboard --logdir ./logs --port 6006

查看训练 loss 和保存的 ``global_step_N`` checkpoint。RoboTwin 评测会报告 ``eval/success_once``；比较 base 与 SFT 时，使用同一个评测器和 seed 协议。
