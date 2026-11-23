LingbotVLA 1.0 强化学习
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png
   :align: center
   :width: 90%

   LingbotVLA 1.0 在 RoboTwin 上（图片来源：`RLinf <https://github.com/RLinf>`__）。

`LingbotVLA 1.0 <https://huggingface.co/robbyant/lingbot-vla-4b>`__ 是一个基于 Qwen2.5-VL 的视觉-语言-动作模型，以自回归方式生成连续动作块。RLinf 将其原生接入，嵌入 RLinf 的 Python 内存空间，支持零延迟的 Tensor 级交互，并在 RoboTwin 2.0 仿真器上支持全参数 SFT 与 GRPO 微调。

概览
----------------------------------------

先 SFT、再用 GRPO 微调 LingbotVLA 1.0，完成 RoboTwin 2.0 双臂操作任务。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: 算法
      :text-align: center

      SFT · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Click Bell · Place Shoe

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · 8–16 GPU

| **你将完成：** 原生安装 → 克隆 RoboTwin 与资产 → 下载检查点 → SFT / GRPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboTwin 仓库与资产 · LingbotVLA 1.0 与 Qwen 底座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据环境、任务族以及配置或权重工件选择对应的模型页面。

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_grpo_lingbotvla``
     - 在 RoboTwin 操作任务上使用 LingbotVLA 运行 GRPO。
   * - RoboTwin
     - Place Shoe
     - ``robotwin_place_shoe_grpo_lingbotvla``
     - 在第二个 RoboTwin 任务变体上运行 GRPO。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - LingbotVLA 所需的 RoboTwin 相机观测与机器人状态。
   * - Action
     - LingbotVLA 策略解码出的连续机器人动作。
   * - Reward
     - RoboTwin 任务成功信号或 shaped task reward。
   * - Prompt
     - RoboTwin episode 的自然语言任务指令。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: GRPO 训练

      **1. 安装环境**

      安装 LingbotVLA 1.0 环境、克隆 RoboTwin 仓库并下载 RoboTwin 资产：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
         (cd RoboTwin && bash script/_download_assets.sh)

      **2. 下载资源与模型**

      下载 LingbotVLA 1.0 基础权重、固定 revision 的 RoboTwin SFT 权重（请勿改用 ``main`` 分支最新版本）与 Qwen2.5-VL 底座：

      .. code-block:: bash

         hf download robbyant/lingbot-vla-4b --local-dir ./lingbot-vla-4b
         hf download robbyant/lingbot-vla-4b-posttrain-robotwin \
           --revision 3e0c7c476bde3daaac00f79f3741a292a299f60a \
           --local-dir ./lingbot-vla-4b-posttrain-robotwin
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      **3. 启动训练**

      ``examples/embodiment/config/`` 下内置的 GRPO 配置如下：

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - 任务
           - 配置文件
         * - Click Bell
           - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``
         * - Place Shoe
           - ``examples/embodiment/config/robotwin_place_shoe_grpo_lingbotvla.yaml``

      在配置中填写本地模型路径（rollout 会自动沿用 ``actor.model.model_path`` 与 ``tokenizer_path``），并将 ``env.train.assets_path`` 与 ``env.eval.assets_path`` 指向 RoboTwin 资产目录：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./lingbot-vla-4b-posttrain-robotwin
             tokenizer_path: ./Qwen2.5-VL-3B-Instruct
             lingbotvla:
               config_path: ./lingbot-vla-4b

      先导出相关仓库路径，再启动 GRPO 训练：

      .. code-block:: bash

         export ROBOTWIN_PATH=$PWD/RoboTwin
         export LINGBOT_VLA_PATH=$(python -c "import lingbotvla, os; print(os.path.dirname(lingbotvla.__path__[0]))")
         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

      .. dropdown:: 进阶配置（可选）

         - **统一归一化键**：LingbotVLA 1.0 使用 ``robotwin_50.json`` 中的全局归一化键（如 ``action.arm.position``），切换 RoboTwin 任务时无需覆写 ``unnorm_key``。
         - **主机内存回收（Linux/glibc）**：GRPO 配置默认启用 ``env.train.malloc_trim_reset_interval: 8``，每个环境实例每 8 次自动 reset 回收一次空闲 glibc 堆内存（仅影响主机 RSS，不影响显存）。设为 ``0`` 可关闭，评估环境未配置时保持关闭。
         - **独立评测**：按 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>` 运行独立评估，使用 ``robotwin_click_bell_lingbotvla_eval``、``robotwin_place_shoe_lingbotvla_eval`` 等配置。

   .. tab-item:: SFT 训练

      **1. 安装环境**

      安装 LingbotVLA 1.0 环境、克隆 RoboTwin 仓库并下载 RoboTwin 资产：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
         (cd RoboTwin && bash script/_download_assets.sh)

      **2. 下载资源与模型**

      下载 LingbotVLA 1.0 基础权重、固定 revision 的 RoboTwin SFT 权重（请勿改用 ``main`` 分支最新版本）与 Qwen2.5-VL 底座：

      .. code-block:: bash

         hf download robbyant/lingbot-vla-4b --local-dir ./lingbot-vla-4b
         hf download robbyant/lingbot-vla-4b-posttrain-robotwin \
           --revision 3e0c7c476bde3daaac00f79f3741a292a299f60a \
           --local-dir ./lingbot-vla-4b-posttrain-robotwin
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      **3. 启动训练**

      在 ``examples/sft/config/robotwin_sft_lingbotvla.yaml`` 中填写本地模型路径，并将 ``data.train_data_paths`` 指向转换后的 LeRobot 数据集目录：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./lingbot-vla-4b-posttrain-robotwin
             tokenizer_path: ./Qwen2.5-VL-3B-Instruct
             lingbotvla:
               config_path: ./lingbot-vla-4b

      先导出相关仓库路径，再启动 SFT 训练：

      .. code-block:: bash

         export ROBOTWIN_PATH=$PWD/RoboTwin
         export LINGBOT_VLA_PATH=$(python -c "import lingbotvla, os; print(os.path.dirname(lingbotvla.__path__[0]))")
         bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla

      .. dropdown:: 进阶配置（可选）

         - **统一归一化键**：LingbotVLA 1.0 使用 ``robotwin_50.json`` 中的全局归一化键（如 ``action.arm.position``），切换 RoboTwin 任务时无需覆写 ``unnorm_key``。
         - **独立评测**：按 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>` 运行独立评估，使用 ``robotwin_click_bell_lingbotvla_eval``、``robotwin_place_shoe_lingbotvla_eval`` 等配置。

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table:: LingbotVLA 1.0 在 RoboTwin 任务上的评估结果
   :header-rows: 1

   * - 任务
     - SFT
     - RLinf-GRPO
   * - ``click_bell``
     - |huggingface| `96.88% <https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin/tree/3e0c7c476bde3daaac00f79f3741a292a299f60a>`__
     - |huggingface| `98.75% <https://huggingface.co/RLinf/RLinf-lingbotvla-click-bell-grpo>`__
   * - ``place_shoe``
     - |huggingface| `93.75% <https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin/tree/3e0c7c476bde3daaac00f79f3741a292a299f60a>`__
     - |huggingface| `98.44% <https://huggingface.co/RLinf/RLinf-lingbotvla-place-shoe-grpo>`__
   * - Average
     - 95.31%
     - **98.60%**
   * - Δ Avg.
     - ---
     - **+3.29%**

.. note::

   LingbotVLA 1.0 结果使用 ``demo_randomized`` 设置。任务级仿真选项见 `RoboTwin configuration documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`__。
