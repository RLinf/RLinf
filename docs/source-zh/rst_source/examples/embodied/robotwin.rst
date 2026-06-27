基于 RoboTwin 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 双臂操作任务（图片来源：`RoboTwin <https://robotwin-platform.github.io>`__）。

`RoboTwin 2.0 <https://robotwin-platform.github.io>`__ 是包含大规模任务套件的双臂操作基准。
你将使用 RLinf 在 ``place_empty_cup``、``adjust_bottle`` 等 RoboTwin 任务上对 VLA 策略进行强化学习微调。

概览
----------------------------------------

在 RoboTwin 2.0 上微调 VLA；OpenVLA-OFT + GRPO 平均任务成功率提升约 +57%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅ · Lingbot-VLA

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO · DAgger

   .. grid-item-card:: 任务
      :text-align: center

      46 个支持任务 · 10 个配置任务

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · 8–16 GPUs

| **你将完成：** 安装 → 克隆 RoboTwin + 资产 → 下载 SFT 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboTwin 仓库与资产 · SFT 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RoboTwin 支持 46 个操作任务。RLinf 提供了以下 ready-to-run 环境配置：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 任务
     - 描述
   * - ``adjust_bottle``
     - 使用正确手臂将桌上的瓶子拾起并保持瓶口朝上。
   * - ``place_empty_cup``
     - 将空杯放到杯垫上。
   * - ``place_container_plate``
     - 将容器放到盘子上。
   * - ``pick_dual_bottles``
     - 用双臂分别抓取两个瓶子。
   * - ``move_can_pot``
     - 将易拉罐移动到锅旁。
   * - ``lift_pot``
     - 用双臂抬起锅。
   * - ``handover_block``
     - 左臂抓取红色积木并交接给右臂，随后放到蓝色垫子上。
   * - ``beat_block_hammer``
     - 抓起锤子敲击积木。
   * - ``click_bell``
     - 按下铃铛顶部中央。
   * - ``place_shoe``
     - 抓取鞋子并放到垫子上。

.. note::

   RLinf 目前尚未支持四个 RoboTwin 任务：``place_fan``、``open_laptop``、
   ``place_object_scale`` 和 ``put_object_cabinet``。Dense reward 仍在逐步扩展到更多任务。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 规格
   * - ``images``
     - 头部相机 RGB，``[B, 224, 224, 3]`` uint8；启用时会中心裁剪。
   * - ``wrist_images``
     - 可选左/右腕部相机 RGB，``[B, n, 224, 224, 3]`` uint8，或 ``None``。
   * - ``states``
     - 本体状态，``[B, 14]`` float32。
   * - ``task_descriptions``
     - 自然语言任务描述。
   * - ``actions``
     - VLA 相关的 ALOHA 风格双臂连续 action chunk。

安装
----------------------------------------

.. include:: _setup_common.rst

**Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-robotwin

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-robotwin

在镜像中切换到对应虚拟环境：

.. code:: bash

   # OpenVLA-OFT
   source switch_env openvla-oft

   # OpenPI π₀ / π₀.₅
   # source switch_env openpi

   # Lingbot-VLA，如镜像中可用
   # source switch_env lingbotvla

**自定义环境**

为你要运行的模型安装依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。

   # OpenVLA-OFT
   bash requirements/install.sh embodied --model openvla-oft --env robotwin

   # OpenPI π₀ / π₀.₅
   # bash requirements/install.sh embodied --model openpi --env robotwin

   # Lingbot-VLA
   # bash requirements/install.sh embodied --model lingbotvla --env robotwin

   source .venv/bin/activate

克隆 RoboTwin 并下载资产：

.. code:: bash

   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   cd RoboTwin
   bash script/_download_assets.sh

   export PYTHONPATH=/path/to/RoboTwin:$PYTHONPATH
   export ROBOT_PLATFORM=ALOHA

下载模型
----------------------------------------

下载与你的配置匹配的 SFT 检查点。示例：

**OpenVLA-OFT**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

**OpenPI π₀ / π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
   git clone https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir RLinf-Pi0-RoboTwin-SFT-adjust_bottle
   hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir RLinf-Pi05-RoboTwin-SFT-adjust_bottle

.. include:: _model_path.rst

对于 Lingbot-VLA 配方，请将 ``actor.model.model_path`` 和 ``rollout.model.model_path`` 指向你的 Lingbot-VLA SFT 检查点。

.. note::

   配置中的动作归一化键 ``unnorm_key`` 必须与训练该 SFT 检查点时所用的 ``unnorm_key``
   一致，例如 ``unnorm_key: "place_empty_cup"``，否则动作会被错误地反归一化。

运行
----------------------------------------

选择一个配方并启动训练：

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - 配方
     - 配置
     - 命令后缀
   * - OpenVLA-OFT + GRPO
     - ``examples/embodiment/config/robotwin_place_empty_cup_grpo_openvlaoft.yaml``
     - ``robotwin_place_empty_cup_grpo_openvlaoft``
   * - OpenVLA-OFT + PPO
     - ``examples/embodiment/config/robotwin_place_empty_cup_ppo_openvlaoft.yaml``
     - ``robotwin_place_empty_cup_ppo_openvlaoft``
   * - π₀ + PPO
     - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml``
     - ``robotwin_adjust_bottle_ppo_openpi``
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml``
     - ``robotwin_adjust_bottle_ppo_openpi_pi05``
   * - OpenPI + DAgger
     - ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``
     - ``robotwin_adjust_bottle_dagger_openpi``
   * - Lingbot-VLA + GRPO
     - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``
     - ``robotwin_click_bell_grpo_lingbotvla``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_place_empty_cup_grpo_openvlaoft
   bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_openpi_pi05

这条命令会：

1. 使用选定的 RoboTwin Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 RoboTwin env 组件创建 Ray worker。
3. 运行 rollout，计算任务奖励，并更新选定的 VLA 策略。

独立评估请走 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>`。
该指南负责 ``ROBOTWIN_PATH`` / ``assets_path`` 设置、可用评测配置
（如 ``robotwin_place_empty_cup_openvlaoft_eval`` 与 ``robotwin_adjust_bottle_openpi_pi05_eval``）
和结果解读。

.. note::

   请将配置中的 ``assets_path`` 替换为已下载的 RoboTwin assets 路径。提供的配置使用
   ``rlinf/envs/robotwin/seeds/`` 下的 train/eval seed 文件。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

视频通过 env video 配置保存：

.. code:: yaml

   video_cfg:
     save_video: True
     video_base_dir: ${runner.logger.log_path}/video/eval

.. list-table:: OpenVLA-OFT 在七个 RoboTwin 任务上的评估结果
   :header-rows: 1

   * - 任务
     - SFT
     - RLinf-GRPO
     - RLinf-PPO
   * - ``beat_block_hammer``
     - |huggingface| `10.15% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`__
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`__
     - ---
   * - ``pick_dual_bottles``
     - |huggingface| `20.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`__
     - |huggingface| `92.96% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`__
     - ---
   * - ``place_empty_cup``
     - |huggingface| `75.78% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`__
     - |huggingface| `94.53% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`__
     - |huggingface| `92.97% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-PPO-place_empty_cup>`__
   * - ``place_container_plate``
     - |huggingface| `54.69% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_container_plate>`__
     - |huggingface| `95.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_container_plate>`__
     - ---
   * - ``move_can_pot``
     - |huggingface| `9.37% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`__
     - |huggingface| `83.59% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`__
     - ---
   * - ``lift_pot``
     - |huggingface| `3.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`__
     - ---
   * - ``handover_block``
     - |huggingface| `28.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`__
     - ---
   * - Average
     - 28.79%
     - **86.16%**
     - ---
   * - Δ Avg.
     - ---
     - **+57.37%**
     - ---

.. list-table:: OpenPI 在 RoboTwin ``adjust_bottle`` 上的评估结果
   :header-rows: 1

   * - 方法
     - SFT
     - RLinf-PPO
   * - π₀
     - |huggingface| `76.56% <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle>`__
     - |huggingface| `98.44% <https://huggingface.co/RLinf/RLinf-Pi0-RoboTwin-PPO-adjust_bottle>`__
   * - π₀.₅
     - |huggingface| `85.94% <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle>`__
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-PPO-adjust_bottle>`__

.. note::

   OpenVLA-OFT 结果使用 ``demo_randomized`` 设置。OpenPI 结果使用 ``demo_clean``。
   任务级仿真选项见 `RoboTwin configuration documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`__。
