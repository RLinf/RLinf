RoboCasa365 强化学习
====================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

RoboCasa365 是 RLinf 中面向基准测试的厨房操作集成。与旧版 :doc:`RoboCasa <robocasa>` 配方不同，``robocasa365`` 通过 RoboCasa 官方 dataset registry（``split`` + ``task_soup``）选择任务，并在运行时暴露 split 与 task-soup 元信息。你将使用 RLinf 在官方 RoboCasa365 benchmark split 上用 PPO 训练和评测 OpenPI π₀。

概览
----------------------------------------

在通过官方 dataset registry 选择的 RoboCasa365 基准切片上训练和评测 OpenPI π₀。

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

      OpenDrawer · atomic_seen · composite_unseen

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPU

| **你将完成：** 安装 → 下载 RoboCasa365 资产与 π₀ 检查点 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboCasa365 厨房资产 · OpenPI π₀ 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

默认配方在 ``split=pretrain`` 与 ``task_soup=atomic_seen`` 下训练 ``OpenDrawer``；评测支持切换任意官方 task soup，如 ``atomic_seen``、``composite_seen`` 或 ``composite_unseen``。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务 / 切片
     - 描述
   * - ``OpenDrawer``\ （``pretrain`` / ``atomic_seen``）
     - 使用 ``task_filter=["OpenDrawer"]`` 与 ``task_mode=atomic`` 的单任务 PPO 训练。
   * - ``atomic_seen`` / ``composite_unseen``
     - 通过 ``split`` 与 ``task_soup`` 选择的 RoboCasa365 官方评测切片。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 多视角 RGB 图像（默认 ``robot0_agentview_left_image`` 与 ``robot0_eye_in_hand_image``）及按 ``observation.state_layout`` 组装的本体感知状态。
   * - 动作
     - 面向 ``PandaOmron`` 的可配置 12 维 RoboCasa 移动操作 action schema；wrapper 可在 env stepping 前关闭底座控制并映射 OpenPI 有效动作 slice。
   * - 奖励
     - 稀疏任务完成奖励。
   * - 提示词
     - RoboCasa365 环境提供的自然语言任务指令及运行时 split / task-soup 元信息。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robocasa365 && source .venv/bin/activate

      **2a. 下载资源与模型**

      将 RoboCasa365 场景压缩包下载到 ``$ROBOCASA_ASSETS_PATH/_downloads``，并下载官方 RoboCasa365 OpenPI π₀ 检查点：

      .. code-block:: bash

         export ROBOCASA_ASSETS_PATH="$PWD/robocasa_assets"

         hf download robocasa/robocasa-assets --repo-type dataset \
           --include textures.zip generative_textures.zip fixtures.zip objaverse.zip aigen_objs.zip \
           --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
         hf download nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF --repo-type dataset \
           --include "fixtures_lightwheel/*.zip" \
           --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
         hf download nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF --repo-type dataset \
           --include "objects_lightwheel/*.zip" \
           --local-dir "$ROBOCASA_ASSETS_PATH/_downloads"
         hf download RLinf/RLinf-pi0-robocasa-pretrain-human300 \
           --local-dir ./RLinf-pi0-robocasa-pretrain-human300

      **2b. 解压场景资产**

      按 ``$ROBOCASA_ASSETS_PATH`` 的目标目录结构解压上述压缩包：

      .. code-block:: bash

         export ROBOCASA_ASSETS_PATH="$PWD/robocasa_assets"
         mkdir -p "$ROBOCASA_ASSETS_PATH"/{fixtures,objects/lightwheel}

         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/textures.zip" -d "$ROBOCASA_ASSETS_PATH"
         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/generative_textures.zip" -d "$ROBOCASA_ASSETS_PATH"
         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/objaverse.zip" -d "$ROBOCASA_ASSETS_PATH/objects"
         unzip -q "$ROBOCASA_ASSETS_PATH/_downloads/aigen_objs.zip" -d "$ROBOCASA_ASSETS_PATH/objects"

         find "$ROBOCASA_ASSETS_PATH/_downloads/fixtures_lightwheel" -name "*.zip" \
           -exec unzip -q {} -d "$ROBOCASA_ASSETS_PATH/fixtures" \;
         find "$ROBOCASA_ASSETS_PATH/_downloads/objects_lightwheel" -name "*.zip" \
           -exec unzip -q {} -d "$ROBOCASA_ASSETS_PATH/objects/lightwheel" \;

      **3. 启动训练**

      RoboCasa365 内置配置文件一览：

      .. list-table::
         :header-rows: 1
         :widths: 26 48 26

         * - 流程
           - 配置文件
           - 启动脚本
         * - PPO 训练（OpenDrawer）
           - ``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml``
           - ``examples/embodiment/run_embodiment.sh``
         * - 独立评测
           - ``evaluations/robocasa365/robocasa365_eval_openpi.yaml``
           - ``evaluations/run_eval.sh``
         * - 端到端短测
           - ``tests/e2e_tests/embodied/robocasa365_ppo_openpi.yaml``
           - ``tests/e2e_tests/embodied/run.sh``

      .. include:: _model_path.rst

      启动 ``OpenDrawer`` PPO 训练：

      .. code-block:: bash

         export ROBOCASA_ASSETS_PATH="$PWD/robocasa_assets"
         bash examples/embodiment/run_embodiment.sh robocasa365_opendrawer_ppo_openpi actor.model.model_path=$PWD/RLinf-pi0-robocasa-pretrain-human300

      .. dropdown:: 进阶配置（可选）

         - **环境配置字段**\ （``examples/embodiment/config/env/robocasa365.yaml``）：

           - ``task_source``：RoboCasa365 保持为 ``dataset_registry``。
           - ``dataset_source``：RoboCasa 注册的数据来源，通常为 ``human``。
           - ``split`` / ``task_soup``：官方 benchmark split（``pretrain``、``target``）与 task soup（``atomic_seen``、``composite_seen``、``composite_unseen``）。RLinf 中的 ``task_soup`` 对应 RoboCasa ``get_ds_soup`` 中的 ``task_set`` 参数（见 `RoboCasa 数据使用文档 <https://robocasa.ai/docs/build/html/datasets/using_datasets.html>`__）。
           - ``task_filter`` / ``task_mode``：可选 include / exclude 过滤器与 ``atomic`` / ``composite`` 保护字段。
           - ``observation`` / ``action_space``：相机 key、``state_layout`` 映射以及 OpenPI 动作 slice 映射。

         - **OpenDrawer Batch 整除关系**：修改 ``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml`` 中的 GPU placement、rollout 数量或 actor batch size 时，先按下面的关系检查：

           .. code-block:: text

              chunk_steps = env.train.max_steps_per_rollout_epoch / actor.model.num_action_chunks
              rollout_size_per_actor_rank = (
                  env.train.total_num_envs * env.train.rollout_epoch / actor_world_size
              ) * chunk_steps

              rollout_size_per_actor_rank % (actor.global_batch_size / actor_world_size) == 0
              actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0

         - **独立评测**：默认在 ``pretrain`` / ``atomic_seen`` 上评测，也可以通过覆盖 ``env.eval.split``、``env.eval.task_soup`` 与 ``env.eval.task_mode`` 切换到其他切片：

           .. code-block:: bash

              bash evaluations/run_eval.sh robocasa365 robocasa365_eval_openpi \
                env.eval.task_soup=composite_unseen \
                env.eval.task_mode=composite

         - **端到端测试**：运行短版 PPO e2e 配置：

           .. code-block:: bash

              bash tests/e2e_tests/embodied/run.sh robocasa365_ppo_openpi

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 曲线，通过 ``env.eval.video_cfg.save_video=True`` 开启评测视频录制，完整指标和日志后端配置见 :doc:`训练指标 <../../reference/metrics>`。
