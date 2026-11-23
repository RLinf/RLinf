GSEnv 强化学习
==============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gsenv.gif
   :align: center
   :width: 90%

   GSEnv / ManiSkill-GS。

GSEnv（也称 ManiSkill-GS）将 ManiSkill 物理仿真与 3D Gaussian Splatting 渲染结合，用于 Real2Sim2Real 操作任务。你将使用 RLinf 在 ``GSEnv-PutCubeOnPlate-v0`` 上，通过 PPO 微调 OpenPI π₀.₅。

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

      1 节点 · 8 GPU

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

   GSEnv 在 ``examples/embodiment/config/env/gsenv_put_cube_on_plate.yaml`` 中通过 ``env_type: maniskill`` 接入，由任务 ID 选择 ManiSkill-GS 环境。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀.₅

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate
         git clone -b v01 https://github.com/chenkang455/ManiSkill-GS.git
         uv pip install -e ./ManiSkill-GS

      **2. 下载资源与模型**

      下载 GSEnv 3DGS 资产、验证 RLinf 接口（首次运行 ``test_rlinf_interface.py`` 可能需要编译 ``gsplat`` kernel，耗时较长），并下载 OpenPI π₀.₅ SFT 检查点：

      .. code-block:: bash

         hf download RLinf/gsenv-assets-v0 --repo-type dataset --local-dir ./ManiSkill-GS/assets
         (cd ManiSkill-GS && python scripts/test_rlinf_interface.py)
         hf download RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT --local-dir ./RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

      **3. 启动训练**

      内置配置为 ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``。

      .. include:: _model_path.rst

      使用 ``run_embodiment.sh`` 启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh gsenv_ppo_openpi_pi05 actor.model.model_path=$PWD/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

      .. dropdown:: 进阶配置（可选）

         默认配置使用 ``actor,env,rollout: all`` 共置。按 GPU 显存预算调整 ``cluster.component_placement``、``env.train.total_num_envs`` 和 ``actor.global_batch_size``。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。

.. figure:: https://github.com/user-attachments/assets/54a22c98-df04-42bd-beef-2630f69da8be
   :align: center
   :width: 90%

   GSEnv 训练曲线示例。

.. note::

   **参考：** `ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`__ · `pi_RL 论文 <https://arxiv.org/pdf/2510.25889>`__。
