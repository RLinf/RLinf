RoboTwin 强化学习
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 双臂操作任务（图片来源：`RoboTwin <https://robotwin-platform.github.io>`__）。

`RoboTwin 2.0 <https://robotwin-platform.github.io>`__ 是包含大规模任务套件的双臂操作基准。你将使用 RLinf 在 ``place_empty_cup``、``adjust_bottle`` 等 RoboTwin 任务上用强化学习微调 VLA policy。

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

      NVIDIA CUDA · :ref:`AMD ROCm <robotwin-hardware>`\ （1–2 节点 · 8–16 GPUs）

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

   RLinf 目前尚未支持四个 RoboTwin 任务：``place_fan``、``open_laptop``、``place_object_scale`` 和 ``put_object_cabinet``。Dense reward 仍在逐步扩展到更多任务。

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

.. _robotwin-hardware:

安装与运行
----------

.. include:: _setup_common.rst

NVIDIA CUDA 按模型选择对应标签页；AMD GPU 请切换到 **AMD ROCm** 标签页。

.. tab-set::

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. 安装环境**

      安装 RLinf，在同级目录克隆 ``RLinf_support`` 分支的 RoboTwin 并下载资产：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA

      **2. 下载资源与模型**

      下载 OpenVLA-OFT SFT checkpoint（以 ``place_empty_cup`` 为例）：

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir ./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

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

      通过命令行覆盖 ``actor.model.model_path``\ （rollout 会自动沿用同一路径）以及 ``env.train.assets_path`` 和 ``env.eval.assets_path``\ （指向 ``RoboTwin`` 目录，即 ``assets/`` 的上一级目录），直接启动训练（也可以在所选 YAML 中填写相同的键）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_place_empty_cup_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
           env.train.assets_path=../RoboTwin \
           env.eval.assets_path=../RoboTwin

   .. tab-item:: OpenPI
      :sync: openpi

      **1. 安装环境**

      安装 RLinf，在同级目录克隆 ``RLinf_support`` 分支的 RoboTwin 并下载资产：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"

      **2. 下载资源与模型**

      下载 ``adjust_bottle`` 的 OpenPI π₀.₅ SFT checkpoint（使用 π₀ 时，改用 ``RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle``）：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 24 48 28

         * - 配方
           - 配置
           - 命令后缀
         * - π₀ + PPO
           - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml``
           - ``robotwin_adjust_bottle_ppo_openpi``
         * - π₀.₅ + PPO
           - ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml``
           - ``robotwin_adjust_bottle_ppo_openpi_pi05``
         * - OpenPI + DAgger
           - ``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``
           - ``robotwin_adjust_bottle_dagger_openpi``

      在所选 YAML 中设置 ``actor.model.model_path``\ （rollout 会自动沿用同一路径），并将 ``env.train.assets_path`` 与 ``env.eval.assets_path`` 设为 ``RoboTwin`` 目录：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle
         env:
           train:
             assets_path: ../RoboTwin
           eval:
             assets_path: ../RoboTwin

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_openpi_pi05

   .. tab-item:: LingBotVLA
      :sync: lingbotvla

      **1. 安装环境**

      安装 RLinf，在同级目录克隆 ``RLinf_support`` 分支的 RoboTwin 并下载资产：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"

      **2. 下载资源与模型**

      在本地准备好你的 Lingbot-VLA SFT 检查点（例如放在 ``./LingBotVLA-RoboTwin-SFT``）。

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 24 48 28

         * - 配方
           - 配置
           - 命令后缀
         * - Lingbot-VLA + GRPO
           - ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml``
           - ``robotwin_click_bell_grpo_lingbotvla``

      将 ``actor.model.model_path`` 指向 Lingbot-VLA SFT 检查点（rollout 会自动沿用同一路径），并将 ``env.train.assets_path`` 与 ``env.eval.assets_path`` 设为 ``RoboTwin`` 目录：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./LingBotVLA-RoboTwin-SFT
         env:
           train:
             assets_path: ../RoboTwin
           eval:
             assets_path: ../RoboTwin

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla

   .. tab-item:: AMD ROCm
      :sync: rocm

      RoboTwin 同样可以安装在 AMD ROCm 上，只需在 task config 中设置 ``planner_backend: mplib``。

      **1. 安装环境**

      在已安装 ROCm 的宿主机上安装，随后克隆 ``RLinf_support`` 分支的 RoboTwin 并下载资产（省略 ``--rocm`` 可自动检测已安装版本；``--model`` 也可替换为 ``openvla-oft`` 或 ``lingbotvla``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin
         (cd ../RoboTwin && bash script/_download_assets.sh)
         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"

      .. dropdown:: 构建并启动 ROCm Docker 镜像（可选）

         如需从当前代码构建 ROCm 版 RoboTwin 镜像，在宿主机运行以下命令，并将 ``ROCM_ARCHS`` 设置为 GPU 的 ``gfx`` 架构（Radeon PRO W7900 为 ``gfx1100``）：

         .. code-block:: bash

            DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
               --build-arg PLATFORM=amd \
               --build-arg ROCM_VER=7.2 \
               --build-arg ROCM_ARCHS=gfx1100 \
               --build-arg BUILD_TARGET=embodied-robotwin \
               -t rlinf:embodied-robotwin-rocm7.2 .

         使用 AMD 内核与渲染设备启动容器：

         .. code-block:: bash

            docker run -it --rm \
               --device=/dev/kfd --device=/dev/dri --group-add video \
               --ipc=host --shm-size 32g --network host \
               -v "$PWD":/workspace/RLinf -w /workspace/RLinf \
               rlinf:embodied-robotwin-rocm7.2 bash

         .. warning::

            ``ROCM_ARCHS`` 必须与目标 GPU 匹配。Docker 构建期间可能无法访问设备，``flash-attn`` 等扩展需要显式指定架构。在 ``gfx90a`` 与 ``gfx942`` 以外的 GPU 上，安装脚本还会将 ``flash-attn`` 切换到 Triton 后端。

      **2. 下载资源与模型**

      下载对应的 SFT checkpoint（以 ``adjust_bottle`` 上的 OpenPI π₀.₅ 为例）：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

      **3. 启动训练**

      ``pytorch3d``、``warp-lang`` 与 ``curobo`` 仅支持 CUDA，安装脚本会跳过它们。这三个包服务于 ``curobo`` 运动规划器，因此 RoboTwin 需要改用另一种规划器：在 task config 中设置 ``planner_backend: mplib``\ （默认的 ``curobo`` 会在构造环境时失败）。在 ``examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml`` 中填写 ``actor.model.model_path`` 与 ``assets_path`` 后，使用相同的命令启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_ppo_openpi_pi05

.. dropdown:: 进阶配置（可选）

   - **相机近裁剪面（``camera_near``）：** RLinf 以 0.01 m 的近裁剪面创建 RoboTwin 相机，而非 RoboTwin 默认的 0.1 m：SAPIEN 3.0.1 会裁掉比近裁剪面更近的物体，0.1 m 会让腕部相机看不到夹爪手指。在 RoboTwin 环境配置中将 ``camera_near`` 设为 ``null`` 即可沿用 RoboTwin 自身的取值。

   - **动作归一化键（``unnorm_key``）：** 配置中的 ``unnorm_key``\ （例如 ``unnorm_key: "place_empty_cup"``）必须与训练该 SFT 检查点时所用的键一致，否则动作会被错误地反归一化。

   - **种子文件：** 提供的配置使用 RLinf 内置的 RoboTwin train/eval seed 文件。

   - **独立评测：** 独立评测请走 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>`，其中包含 ``ROBOTWIN_PATH`` / ``assets_path`` 设置、``robotwin_place_empty_cup_openvlaoft_eval`` 与 ``robotwin_adjust_bottle_openpi_pi05_eval`` 等评测配置以及结果解读。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

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

   OpenVLA-OFT 结果使用 ``demo_randomized`` 设置。OpenPI 结果使用 ``demo_clean``。任务级仿真选项见 `RoboTwin configuration documentation <https://robotwin-platform.github.io/doc/usage/configurations.html>`__。
