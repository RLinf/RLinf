Genesis 强化学习
================

.. figure:: https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png
   :align: center
   :width: 90%

   Genesis（图片来源：`Genesis <https://genesis-world.readthedocs.io/>`__）。

`Genesis <https://genesis-world.readthedocs.io/>`__ 是面向机器人任务的 GPU 加速多物理场仿真器。你将使用 RLinf 在 Franka cube-pick 任务上，通过 PPO 训练 MLP 或 CNN policy。

概览
----------------------------------------

训练 Franka Panda 策略在 Genesis 中抓取方块。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP · CNN

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      CubePick

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1 GPU

| **你将完成：** 安装 → 可选下载 ResNet → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 安装步骤中的 Genesis 依赖。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``cube_pick``
     - 控制 Franka Panda 机械臂抓取并抬起方块。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - MLP 使用 16 维状态；CNN 使用 256×256 RGB 加 16 维状态。
   * - 动作
     - 9 维连续动作：7 个 Franka 机械臂关节位置和 2 个夹爪位置。
   * - 奖励
     - 稠密 approach reward 和 grasp-success bonus。
   * - 提示词
     - 不使用；这是低维/CNN policy 控制配方。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: MLP / CNN

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env genesis && source .venv/bin/activate

      **2. 下载模型**

      MLP + PPO 配方可跳过此步。CNN + PPO 配方需要下载预训练 ResNet-10 检查点：

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./RLinf-ResNet10-pretrained

      **3. 启动训练**

      可用训练配置：

      .. list-table::
         :header-rows: 1
         :widths: 26 48 26

         * - 配方
           - 配置
           - 命令后缀
         * - MLP + PPO
           - ``examples/embodiment/config/genesis_cubepick_ppo_mlp.yaml``
           - ``genesis_cubepick_ppo_mlp``
         * - CNN + PPO
           - ``examples/embodiment/config/genesis_cubepick_ppo_cnn.yaml``
           - ``genesis_cubepick_ppo_cnn``

      .. include:: _model_path.rst

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         # 状态观测 MLP + PPO
         bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_mlp

         # 视觉 CNN + PPO
         bash examples/embodiment/run_embodiment.sh genesis_cubepick_ppo_cnn actor.model.model_path=$PWD/RLinf-ResNet10-pretrained

      .. dropdown:: 进阶配置（可选）

         两个配置默认都运行在 GPU ``0``。按硬件情况调整 ``cluster.component_placement``、``env.train.total_num_envs`` 和 batch size。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配方
     - 结果描述
   * - MLP + PPO
     - 使用默认 ``genesis_cubepick_ppo_mlp`` 参数时，``env/success_once`` 可达到约 80%。
