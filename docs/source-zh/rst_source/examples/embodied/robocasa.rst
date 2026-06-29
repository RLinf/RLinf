基于 RoboCasa 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
   :align: center
   :width: 90%

   RoboCasa（图片来源：`RoboCasa <https://robocasa.ai/>`__）。

`RoboCasa <https://robocasa.ai/>`__ 是基于 robosuite 的厨房操作基准，包含多样化布局、
物体和原子任务。你将使用 RLinf 在 RoboCasa ``CloseDrawer`` 任务上，通过 PPO 微调
OpenPI π₀ 策略。

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

      1 节点 · 8 GPUs

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

   RoboCasa 包含更多原子任务，但当前公开的 RLinf 配方使用
   ``examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml`` 训练
   ``CloseDrawer``。

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
      rlinf/rlinf:agentic-rlinf0.3-robocasa

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-robocasa

在镜像中切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**自定义环境**

安装 RoboCasa 与 OpenPI 依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env robocasa
   source .venv/bin/activate

安装 RoboCasa 后下载厨房资产：

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets

.. warning::

   RoboCasa 厨房资产约 5 GB。启动训练前只需下载一次。

下载模型
----------------------------------------

下载 OpenPI π₀ 检查点：

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-RoboCasa

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-RoboCasa --local-dir RLinf-Pi0-RoboCasa

.. include:: _model_path.rst

运行
----------------------------------------

启动 CloseDrawer 配方：

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - 配方
     - 配置
     - 命令后缀
   * - OpenPI π₀ + PPO
     - ``examples/embodiment/config/robocasa_closedrawer_ppo_openpi.yaml``
     - ``robocasa_closedrawer_ppo_openpi``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_ppo_openpi

这条命令会：

1. 使用 RoboCasa Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 RoboCasa env 组件创建 Ray worker。
3. 运行 PPO rollout，计算稀疏任务奖励，并更新 OpenPI 策略。

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>`，
通过配置回退机制复用相同后缀 ``robocasa_closedrawer_ppo_openpi``。

.. note::

   默认配置使用 ``actor,env,rollout: all`` 共置。请根据 GPU 显存预算调整
   ``env.train.total_num_envs``、``env.eval.total_num_envs`` 和
   ``actor.global_batch_size``。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

如需保存 rollout 视频，请在环境配置中启用 video：

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

如需启用 W&B 或 SwanLab，请添加 logger backend：

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. note::

   本页面暂未发布固定的 RoboCasa 成功率表。请使用 ``env/success_once`` 和评估视频比较运行结果。
