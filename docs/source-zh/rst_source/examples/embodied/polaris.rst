PolaRiS 强化学习
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png
   :align: center
   :width: 90%

   PolaRiS（图片来源：`PolaRiS <https://github.com/arhanjain/polaris>`__）。

`PolaRiS <https://github.com/arhanjain/polaris>`__ 是基于 Isaac Sim 的机器人基准，使用 Gaussian Splatting 渲染桌面操作任务。你将使用 RLinf 在 DROID 风格的 PolaRiS 任务上，通过 PPO 微调 OpenPI π₀ 或 π₀.₅ 策略。

概览
----------------------------------------

在 PolaRiS 上使用两个 RGB 视角、本体状态和 chunked 8 维动作微调 OpenPI 策略。

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

      6 个 DROID 桌面任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1 GPU

| **你将完成：** 安装 → 下载 Isaac Sim + 数据集 + 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Isaac Sim · PolaRiS-Hub · OpenPI 检查点。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - 任务
     - 描述
     - 环境配置
   * - ``DROID-TapeIntoContainer``
     - 将胶带放入容器。
     - ``polaris_droid_tapeintocontainer.yaml``
   * - ``DROID-PanClean``
     - 用黄色海绵擦洗蓝色手柄煎锅。
     - ``polaris_droid_panclean.yaml``
   * - ``DROID-BlockStackKitchen``
     - 将积木放到绿色托盘上并堆叠。
     - ``polaris_droid_blockstackkitchen.yaml``
   * - ``DROID-FoodBussing``
     - 将所有食物放入碗中。
     - ``polaris_droid_foodbussing.yaml``
   * - ``DROID-MoveLatteCup``
     - 将拉花杯放到砧板上。
     - ``polaris_droid_movelattecup.yaml``
   * - ``DROID-OrganizeTools``
     - 将剪刀放入大容器。
     - ``polaris_droid_organizetools.yaml``

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 224×224 外部 RGB 相机、腕部 RGB 相机，以及 8 维机器人状态。
   * - 动作
     - 8 维连续动作：7 维关节速度和 1 维夹爪位置。
   * - 奖励
     - PolaRiS 环境提供的任务完成奖励。
   * - 提示词
     - ``init_params.task_description`` 中的任务描述。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI π₀.₅
      :sync: pi05

      **1. 安装环境**

      安装 RLinf（OpenPI + PolaRiS）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

      下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

      .. code-block:: bash

         mkdir -p isaac_sim && cd isaac_sim
         wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
         unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
         rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
         source ./setup_conda_env.sh

      .. warning::

         每次在新终端中启动 PolaRiS 前，都需要运行 ``source ./setup_conda_env.sh``。

      **2. 下载数据集与模型**

      下载 PolaRiS 评估场景（``owhan/PolaRiS-Hub``）与 OpenPI π₀.₅ checkpoint：

      .. code-block:: bash

         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./RLinf-Pi05-Polaris-droid_jointpos

      如需 co-training 演示数据，可额外执行 ``hf download owhan/PolaRiS-datasets --repo-type dataset --local-dir ./PolaRiS-datasets``。

      .. include:: _model_path.rst

      **3. 启动训练**

      修改 ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi_pi05.yaml`` 中的模型路径，初始化 Isaac Sim 并导出 ``POLARIS_DATA_PATH`` 后启动 PPO：

      .. code-block:: bash

         source ./isaac_sim/setup_conda_env.sh
         export POLARIS_DATA_PATH="$PWD/PolaRiS-Hub"
         bash examples/embodiment/run_embodiment.sh polaris_tapeintocontainer_ppo_openpi_pi05

      独立评估请走 :doc:`PolaRiS 评测指南 <../../evaluations/guides/polaris>`\ （``polaris_tapeintocontainer_openpi_pi05_eval``）。

   .. tab-item:: OpenPI π₀
      :sync: pi0

      **1. 安装环境**

      安装 RLinf（OpenPI + PolaRiS）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

      下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

      .. code-block:: bash

         mkdir -p isaac_sim && cd isaac_sim
         wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
         unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
         rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
         source ./setup_conda_env.sh

      .. warning::

         每次在新终端中启动 PolaRiS 前，都需要运行 ``source ./setup_conda_env.sh``。

      **2. 下载数据集与模型**

      下载 PolaRiS 评估场景（``owhan/PolaRiS-Hub``）与 OpenPI π₀ checkpoint：

      .. code-block:: bash

         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./RLinf-Pi0-Polaris-droid_jointpos

      如需 co-training 演示数据，可额外执行 ``hf download owhan/PolaRiS-datasets --repo-type dataset --local-dir ./PolaRiS-datasets``。

      .. include:: _model_path.rst

      **3. 启动训练**

      修改 ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi.yaml`` 中的模型路径，初始化 Isaac Sim 并导出 ``POLARIS_DATA_PATH`` 后启动 PPO：

      .. code-block:: bash

         source ./isaac_sim/setup_conda_env.sh
         export POLARIS_DATA_PATH="$PWD/PolaRiS-Hub"
         bash examples/embodiment/run_embodiment.sh polaris_tapeintocontainer_ppo_openpi

      独立评估请走 :doc:`PolaRiS 评测指南 <../../evaluations/guides/polaris>`\ （``polaris_movelattecup_openpi_eval``）。

.. note::

   训练配置默认使用 ``polaris_droid_tapeintocontainer``。如需切换任务，将 Hydra env defaults 改为其他 ``polaris_droid_*`` 环境配置，并保持 ``POLARIS_DATA_PATH`` 指向 ``PolaRiS-Hub``。

.. dropdown:: 进阶配置（可选）

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 配置项
        - 含义
      * - ``open_loop_horizon``
        - **高质量** Gaussian Splatting 渲染频率。在一个动作 chunk 内，每 ``open_loop_horizon`` 步执行一次高质量渲染，中间步使用低质量渲染以加速仿真。
      * - ``num_action_chunks``
        - 模型一次生成的动作步数（如 ``15``）。
      * - ``num_images_in_input``
        - 输入给策略的相机图像数量（如 ``2``：外部相机 + 腕部相机）。
      * - ``config_name``
        - OpenPI 配置 / 数据格式选择（如 ``pi05_droid_polaris`` 对应 DROID 数据格式）。

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。
