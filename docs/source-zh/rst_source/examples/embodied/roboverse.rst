RoboVerse 强化学习
==================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://roboverseorg.github.io/static/images/teaser.jpg
   :align: center
   :width: 90%

   RoboVerse（图片来源：`RoboVerse <https://roboverseorg.github.io/>`__）。

`RoboVerse <https://roboverseorg.github.io/>`__ 是面向机器人操作任务的仿真套件，支持多个后端。你将使用 RLinf 在 RoboVerse 厨房操作任务上，通过 PPO 微调 OpenPI π₀.₅ policy。

概览
----------------------------------------

在带有两个 RGB 视角和稀疏奖励的 RoboVerse 任务上微调 OpenPI π₀.₅。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      Bowl on cabinet

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPUs

| **你将完成：** 安装 → 下载资源 + 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboVerse 资源 · SFT 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - 任务
     - 描述
   * - ``libero_90.kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet``
     - 在厨房场景中将黑色碗放到柜子上方。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 224×224 主相机 RGB、腕部相机 RGB，以及 8 维本体状态。
   * - 动作
     - 7 维连续动作：3D 末端执行器位置、3D 旋转向量和夹爪。
   * - 奖励
     - 稀疏任务完成奖励。
   * - 提示词
     - RoboVerse 任务的自然语言指令。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀.₅
      :sync: openpi-roboverse

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env roboverse && source .venv/bin/activate

      **2. 下载资源与模型**

      在 RLinf 仓库根目录下载默认 RoboVerse 资源与 OpenPI π₀.₅ SFT 检查点：

      .. code-block:: bash

         # 下载 RoboVerse 资源
         hf download manity/roboverse_data --repo-type dataset --local-dir ./roboverse_data

         # 下载 OpenPI π₀.₅ 检查点
         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      **3. 启动训练**

      .. list-table::
         :header-rows: 1
         :widths: 28 46 26

         * - 配方
           - 配置
           - 命令后缀
         * - OpenPI π₀.₅ + PPO
           - ``examples/embodiment/config/roboverse_ppo_openpi_pi05.yaml``
           - ``roboverse_ppo_openpi_pi05``

      .. include:: _model_path.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh roboverse_ppo_openpi_pi05

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>`，通过配置回退机制复用相同后缀 ``roboverse_ppo_openpi_pi05``。

.. note::

   默认配置将 actor 与 rollout 放在 GPU ``0-1``，将 env worker 放在 GPU ``2-3``。请根据硬件调整 ``cluster.component_placement``、``env.train.total_num_envs`` 和 ``actor.global_batch_size``。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. note::

   本页面暂未发布固定的 RoboVerse 成功率表。请使用 ``env/success_once`` 和评估视频比较运行结果。
