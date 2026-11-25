RoboCasa 强化学习
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
   :align: center
   :width: 90%

   RoboCasa（图片来源：`RoboCasa <https://robocasa.ai/>`__）。

`RoboCasa <https://robocasa.ai/>`__ 是基于 robosuite 的厨房操作基准，包含多样化布局、物体和原子任务。你将使用 RLinf 在 RoboCasa ``CloseDrawer`` 任务上，通过 PPO 微调 OpenPI π₀ 策略。

概览
----------------------------------------

在 RoboCasa 的移动操作厨房任务上微调 OpenPI π₀。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      CloseDrawer

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPU

| **你将完成：** 安装 → 下载厨房资产 + 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboCasa 厨房资产 · SFT 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``CloseDrawer``
     - 使用 PandaOmron 移动机械臂关闭厨房抽屉。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 默认两个 RGB 视角（224×224 的 ``base_image`` 和 ``wrist_image``）以及 25 维本体状态。
   * - 动作
     - 12 维连续动作：机械臂位置增量、机械臂旋转增量、夹爪、底座控制和模式选择。
   * - 奖励
     - 稀疏任务完成奖励。
   * - 提示词
     - RoboCasa 任务生成的自然语言指令。

.. note::

   RoboCasa 包含更多原子任务，公开的 RLinf 配方使用 ``examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml`` 训练 ``CloseDrawer``。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robocasa && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 RoboCasa 厨房资产（约 5 GB，只需下载一次）和 OpenPI π₀ 检查点：

      .. code-block:: bash

         python -m robocasa.scripts.download_kitchen_assets
         hf download RLinf/RLinf-Pi0-RoboCasa --local-dir ./RLinf-Pi0-RoboCasa

      **3. 启动训练**

      内置配置为 ``examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml``。

      .. include:: _model_path.rst

      使用 ``run_embodiment.sh`` 启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_ppo_openpi actor.model.model_path=$PWD/RLinf-Pi0-RoboCasa

      .. dropdown:: 进阶配置（可选）

         - **显存调优**：默认配置使用 ``actor,env,rollout: all`` 共置。按 GPU 显存预算调整 ``env.train.total_num_envs``、``env.eval.total_num_envs`` 和 ``actor.global_batch_size``。
         - **独立评测**：使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>`，通过配置回退机制复用相同后缀 ``robocasa_closedrawer_ppo_openpi``。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. note::

   本页面暂未发布固定的 RoboCasa 成功率表。使用 ``env/success_once`` 和评估视频比较运行结果。
