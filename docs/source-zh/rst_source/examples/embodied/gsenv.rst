基于 GSEnv 的 Real2Sim2Real 强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif
   :align: center
   :width: 90%

   GSEnv / ManiSkill-GS。

GSEnv，也称 ManiSkill-GS，将 ManiSkill 物理仿真与 3D Gaussian Splatting 渲染结合，
用于 Real2Sim2Real 操作任务。你将使用 RLinf 在 ``GSEnv-PutCubeOnPlate-v0`` 上，
通过 PPO 微调 OpenPI π₀.₅。

概览
----------------------------------------

在兼容 ManiSkill 的 GSEnv 任务上微调 OpenPI π₀.₅。

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

      PutCubeOnPlate

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPUs

| **你将完成：** 安装 → 添加 ManiSkill-GS 资产 → 下载模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · ManiSkill-GS checkout · GSEnv 资产 · SFT 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``GSEnv-PutCubeOnPlate-v0``
     - 抓取立方体并放到托盘上。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 通过 ``gs_kwargs.render_interface: "gs_rlinf"`` 启用 3DGS 渲染的 ManiSkill 兼容观测。
   * - 动作
     - ``policy_setup: "panda-ee-target-dpos"`` 对应的连续末端执行器位置增量控制。
   * - 奖励
     - ``reward_mode: only_success`` 对应的稀疏成功奖励。
   * - 提示词
     - GSEnv wrapper 提供的任务指令。

.. note::

   GSEnv 在 ``examples/embodiment/config/env/gsenv_put_cube_on_plate.yaml`` 中通过
   ``env_type: maniskill`` 接入。任务 id 选择 ManiSkill-GS 环境。

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
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

在镜像中切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**自定义环境**

安装 ManiSkill/LIBERO 环境与 OpenPI 依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

安装 ManiSkill-GS 及其资产：

.. code:: bash

   git clone -b v01 https://github.com/chenkang455/ManiSkill-GS.git
   cd ManiSkill-GS
   uv pip install -e .

   # 下载资产到 ManiSkill-GS 项目中。
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/gsenv-assets-v0 --repo-type dataset --local-dir ./assets

在 ManiSkill-GS 项目根目录验证 RLinf 接口：

.. code:: bash

   python scripts/test_rlinf_interface.py

.. note::

   首次运行可能需要编译 ``gsplat`` kernel，因此耗时较长。

下载模型
----------------------------------------

下载 OpenPI π₀.₅ SFT 检查点：

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT --local-dir RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

.. include:: _model_path.rst

运行
----------------------------------------

启动 GSEnv 配方：

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - 配方
     - 配置
     - 命令后缀
   * - OpenPI π₀.₅ + PPO
     - ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``
     - ``gsenv_ppo_openpi_pi05``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh gsenv_ppo_openpi_pi05

这条命令会：

1. 使用 GSEnv Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 ManiSkill-backed env 组件创建 Ray worker。
3. 使用 OpenPI action chunk 和稀疏 GSEnv 成功奖励运行 PPO rollout。

.. note::

   默认配置使用 ``actor,env,rollout: all`` 共置。请根据 GPU 显存预算调整
   ``cluster.component_placement``、``env.train.total_num_envs`` 和
   ``actor.global_batch_size``。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

如需保存 3DGS rollout 视频，请在环境配置中启用 video：

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

.. figure:: https://github.com/user-attachments/assets/54a22c98-df04-42bd-beef-2630f69da8be
   :align: center
   :width: 90%

   GSEnv 训练曲线示例。

参考
----------------------------------------

- `ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`__
- `pi_RL paper <https://arxiv.org/pdf/2510.25889>`__
