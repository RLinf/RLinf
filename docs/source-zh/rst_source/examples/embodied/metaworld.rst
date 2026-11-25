MetaWorld 强化学习
==================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
   :align: center
   :width: 90%

   Meta-World 基准（图片来源：`Meta-World <https://metaworld.farama.org>`__）。

`Meta-World <https://metaworld.farama.org>`__ 是一个基于 MuJoCo 的多任务操作基准：一台 7 自由度机械臂完成 50 个多样的桌面任务。RLinf 借助它用强化学习微调视觉-语言-动作（VLA）policy，并评测分布外（OOD）泛化。

概览
----------------------------------------

在 Meta-World 的 50 个任务上用强化学习微调 VLA；π₀ + PPO 平均成功率约 78%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      MT50 · ML45（5 个 OOD）

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 张 GPU

| **你将完成：** 安装依赖 → 下载 SFT 模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SFT 检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - 套件
     - 任务数
     - 设置
   * - MT50
     - 50
     - 在全部 50 个任务上开展多任务训练与评测。
   * - ML45
     - 45 + 5
     - 在 45 个任务上训练；在 5 个留出（OOD）任务上评测。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 工作区周围离屏相机的 RGB 图像（480×480）。
   * - 动作 (Action)
     - 4 维连续动作：3D 末端执行器位置（x, y, z）+ 夹爪开合。
   * - 奖励 (Reward)
     - 稀疏奖励——基于任务完成。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀
      :sync: pi0

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env metaworld && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 OpenPI π₀ SFT checkpoint（也可以从 `ModelScope <https://www.modelscope.cn/models/RLinf/RLinf-Pi0-MetaWorld>`__ 下载）：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-MetaWorld-SFT --local-dir ./RLinf-Pi0-MetaWorld-SFT

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 25 25 50

         * - 设定
           - 算法
           - 配置
         * - MT50
           - PPO
           - ``metaworld_50_ppo_openpi.yaml``
         * - ML45
           - PPO
           - ``metaworld_45_ppo_openpi.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi \
           actor.model.model_path=./RLinf-Pi0-MetaWorld-SFT

   .. tab-item:: OpenPI π₀.₅
      :sync: pi05

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env metaworld && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 OpenPI π₀.₅ SFT checkpoint：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-MetaWorld-SFT --local-dir ./RLinf-Pi05-MetaWorld-SFT

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 25 25 50

         * - 设定
           - 算法
           - 配置
         * - MT50
           - PPO
           - ``metaworld_50_ppo_openpi_pi05.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi_pi05 \
           actor.model.model_path=./RLinf-Pi05-MetaWorld-SFT

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env metaworld && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 OpenVLA-OFT SFT checkpoint：

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-MetaWorld-SFT --local-dir ./RLinf-OpenVLAOFT-MetaWorld-SFT

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 25 25 50

         * - 设定
           - 算法
           - 配置
         * - MT50
           - GRPO
           - ``metaworld_50_grpo_openvlaoft.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh metaworld_50_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-MetaWorld-SFT

.. admonition:: 进一步配置
   :class: note

   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 全量配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`恢复训练 <../../guides/resume>`

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

MetaWorld 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下表 Diffusion Policy、TinyVLA 和 SmolVLA 的结果参考 `SmolVLA 论文 <https://arxiv.org/abs/2403.04880>`_。π\ :sub:`0`\ 与 π\ :sub:`0.5`\ 的 SFT 结果是通过 LeRobot 官方提供的 `数据集 <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ 重新训练所得。

.. list-table:: **MetaWorld-MT50 性能对比（Success Rate, %）**
   :widths: 15 10 10 10 10 10
   :header-rows: 1

   * - **Methods**
     - **Easy**
     - **Medium**
     - **Hard**
     - **Very Hard**
     - **Avg.**
   * - Diffusion Policy
     - 23.1
     - 10.7
     - 1.9
     - 6.1
     - 10.5
   * - TinyVLA
     - 77.6
     - 21.5
     - 11.4
     - 15.8
     - 31.6
   * - SmolVLA
     - 87.1
     - 51.8
     - 70.0
     - 64.0
     - 68.2
   * - π\ :sub:`0`\
     - 77.9
     - 51.8
     - 53.3
     - 20.0
     - 50.8
   * - π\ :sub:`0`\  + PPO
     - **92.1**
     - **74.6**
     - 61.7
     - **84.0**
     - **78.1**
   * - π\ :sub:`0.5`\
     - 68.2
     - 37.3
     - 41.7
     - 28.0
     - 43.8
   * - π\ :sub:`0.5`\  + PPO
     - 86.4
     - 55.5
     - **75.0**
     - 66.0
     - 70.7
