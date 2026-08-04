基于 CALVIN 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/calvin.png
   :align: center
   :width: 90%

   CALVIN（图片来源：`CALVIN <https://github.com/mees/calvin/>`__）。

`CALVIN <https://github.com/mees/calvin/>`__ 是基于 PyBullet 的长程语言条件机器人操作
基准。你将使用 RLinf 在 CALVIN 场景迁移套件上，通过 PPO 微调 OpenPI π₀ 或 π₀.₅ 策略。

概览
----------------------------------------

在 CALVIN 上微调 OpenPI 策略，并评估长程子任务完成能力。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      D→D · ABC→D · ABCD→D

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPUs

| **你将完成：** 安装 → 下载 SFT 检查点 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 安装步骤中的 CALVIN 资产 · SFT 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 任务
     - 描述
   * - CALVIN D→D
     - 在 scene D 训练并在 scene D 评估，使用 ``calvin_d_d_ppo_openpi`` 或 ``calvin_d_d_ppo_openpi_pi05``。
   * - CALVIN ABC→D
     - 在 scene A/B/C 训练并在 scene D 评估，使用 ``calvin_abc_d_ppo_openpi_pi05``。
   * - CALVIN ABCD→D
     - 在 scene A/B/C/D 训练并在 scene D 评估，使用 ``calvin_abcd_d_ppo_openpi_pi05``。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 第三人称 RGB、腕部相机 RGB，以及机器人本体状态。
   * - 动作
     - 7 维连续动作：3D 末端执行器位置 + 3D 旋转 + 夹爪。
   * - 奖励
     - 稀疏 0/1 子任务完成奖励。
   * - 提示词
     - 当前 CALVIN 子任务的自然语言指令。

.. note::

   RLinf 修正了 CALVIN 上游仓库中 scene A、B、C YAML 文件的部分设置。
   背景见上游 `CALVIN issue <https://github.com/mees/calvin/issues/41>`__。

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
      rlinf/rlinf:agentic-rlinf0.3-calvin

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-calvin

在镜像中切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**自定义环境**

安装 CALVIN 与 OpenPI 依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env calvin
   source .venv/bin/activate

下载模型
----------------------------------------

下载你要微调的 OpenPI 模型检查点。

**OpenPI π₀**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT --local-dir RLinf-Pi0-CALVIN-ABC-D-SFT

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT --local-dir RLinf-Pi05-CALVIN-ABC-D-SFT

.. include:: _model_path.rst

运行
----------------------------------------

选择一个配置并启动训练：

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - 配方
     - 配置
     - 命令后缀
   * - π₀，D→D
     - ``examples/embodiment/config/calvin_d_d_ppo_openpi.yaml``
     - ``calvin_d_d_ppo_openpi``
   * - π₀.₅，D→D
     - ``examples/embodiment/config/calvin_d_d_ppo_openpi_pi05.yaml``
     - ``calvin_d_d_ppo_openpi_pi05``
   * - π₀.₅，ABC→D
     - ``examples/embodiment/config/calvin_abc_d_ppo_openpi_pi05.yaml``
     - ``calvin_abc_d_ppo_openpi_pi05``
   * - π₀.₅，ABCD→D
     - ``examples/embodiment/config/calvin_abcd_d_ppo_openpi_pi05.yaml``
     - ``calvin_abcd_d_ppo_openpi_pi05``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh calvin_d_d_ppo_openpi_pi05

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 CALVIN env 组件创建 Ray worker。
3. 运行 PPO rollout，计算稀疏子任务奖励，并更新 OpenPI 策略。

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>`，
通过配置回退机制复用相同后缀，例如 ``calvin_d_d_ppo_openpi_pi05``。

.. note::

   CALVIN 配置默认使用 ``actor,env,rollout: all`` 共置。请根据 GPU 显存预算调整
   ``cluster.component_placement``、``env.train.total_num_envs`` 和
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

.. list-table::
   :header-rows: 1
   :widths: 28 18 14 10 10 10 10 10

   * - 方法
     - 训练
     - 平均子任务数
     - Len-1
     - Len-2
     - Len-3
     - Len-4
     - Len-5
   * - π₀
     - SFT
     - 3.766
     - 0.947
     - 0.849
     - 0.743
     - 0.652
     - 0.575
   * - π₀
     - Flow SDE
     - 3.944
     - 0.964
     - 0.880
     - 0.775
     - 0.708
     - 0.617
   * - π₀
     - Flow Noise
     - 3.919
     - **0.969**
     - 0.888
     - 0.780
     - 0.683
     - 0.599
   * - π₀.₅
     - SFT
     - 3.838
     - 0.927
     - 0.843
     - 0.767
     - 0.688
     - 0.613
   * - π₀.₅
     - Flow SDE
     - **4.717**
     - **0.997**
     - **0.982**
     - **0.958**
     - **0.910**
     - **0.870**
   * - π₀.₅
     - Flow Noise
     - 4.652
     - 0.996
     - 0.976
     - 0.939
     - 0.896
     - 0.845
