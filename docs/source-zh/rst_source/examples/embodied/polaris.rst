基于 PolaRiS 仿真平台的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/polaris.png
   :align: center
   :width: 90%

   PolaRiS（图片来源：`PolaRiS <https://github.com/arhanjain/polaris>`__）。

`PolaRiS <https://github.com/arhanjain/polaris>`__ 是基于 Isaac Sim 的机器人基准，
使用 Gaussian Splatting 渲染桌面操作任务。你将使用 RLinf 在 DROID 风格的 PolaRiS
任务上，通过 PPO 微调 OpenPI π₀ 或 π₀.₅ 策略。

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
      rlinf/rlinf:agentic-rlinf0.2-polaris

   # 国内用户可使用：
   # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-polaris

在镜像中切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**自定义环境**

安装 PolaRiS 与 OpenPI 依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

下载 Isaac Sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   每次在新终端中启动 PolaRiS 前，都需要运行 ``source ./setup_conda_env.sh``。

下载数据集
----------------------------------------

下载评估场景和初始条件：

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub
   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

可选下载 co-training 演示数据：

.. code:: bash

   hf download owhan/PolaRiS-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets

下载模型
----------------------------------------

下载你要微调的 OpenPI 模型检查点。

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-Polaris-droid_jointpos

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir RLinf-Pi05-Polaris-droid_jointpos

**OpenPI π₀**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Polaris-droid_jointpos

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir RLinf-Pi0-Polaris-droid_jointpos

.. include:: _model_path.rst

运行
----------------------------------------

选择一个训练配置，并在已初始化 Isaac Sim 的终端中启动：

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - 配方
     - 配置
     - 命令后缀
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi_pi05.yaml``
     - ``polaris_tapeintocontainer_ppo_openpi_pi05``
   * - π₀ + PPO
     - ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi.yaml``
     - ``polaris_tapeintocontainer_ppo_openpi``

.. code:: bash

   source /path/to/isaac_sim/setup_conda_env.sh
   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

   bash examples/embodiment/run_embodiment.sh polaris_tapeintocontainer_ppo_openpi_pi05
   bash examples/embodiment/run_embodiment.sh polaris_tapeintocontainer_ppo_openpi

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 PolaRiS env 组件创建 Ray worker。
3. 使用 chunked OpenPI 动作和 Gaussian Splatting 渲染观测运行 PPO。

独立评估请走 :doc:`PolaRiS 评测指南 <../../evaluations/guides/polaris>`。
该指南负责 ``POLARIS_DATA_PATH``、可用评测配置
（``polaris_tapeintocontainer_openpi_pi05_eval`` 与 ``polaris_movelattecup_openpi_eval``）
和结果解读。

.. note::

   训练配置默认使用 ``polaris_droid_tapeintocontainer``。如需切换任务，请将 Hydra
   env defaults 改为其他 ``polaris_droid_*`` 环境配置，并保持 ``POLARIS_DATA_PATH``
   指向 ``PolaRiS-Hub``。

在调优动作 / 渲染流水线时，以下 PolaRiS 专有字段值得了解：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 配置项
     - 含义
   * - ``open_loop_horizon``
     - **高质量** Gaussian Splatting 渲染的频率。在一个动作 chunk 内，每 ``open_loop_horizon``
       步执行一次高质量渲染，中间步则使用低质量渲染以加速仿真。
   * - ``num_action_chunks``
     - 模型一次生成的动作步数（如 ``15``）。
   * - ``num_images_in_input``
     - 输入给策略的相机图像数量（如 ``2``：外部相机 + 腕部相机）。
   * - ``config_name``
     - OpenPI 配置 / 数据格式选择（如 ``pi05_droid_polaris`` 对应 DROID 数据格式）。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。完整指标说明见
:doc:`训练指标 <../../reference/metrics>`。

如需评估视频，请在环境配置中启用 video：

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval
