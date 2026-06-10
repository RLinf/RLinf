基于 PolaRiS 仿真平台的强化学习训练
==========================================

本文档给出在 RLinf 框架内使用 **π05 (OpenPI)** 模型在 `PolaRiS <https://github.com/arhanjain/polaris>`_ 仿真平台上进行 PPO 强化学习训练的完整指南。

环境
-------

**PolaRiS (Policy Learning and Benchmarking in Realistic Simulated Environments)**

PolaRiS 是一个基于 Isaac Sim 和 Gaussian Splatting 渲染的高保真机器人仿真平台，
支持多种桌面操作任务，提供逼真的视觉渲染效果。

- **仿真平台**：基于 NVIDIA Isaac Sim
- **渲染**：Gaussian Splatting 实时渲染，支持高质量（expensive）和快速渲染模式切换
- **观测空间**：

  - 外部相机（桌面视角）RGB 图像（224×224）
  - 腕部相机 RGB 图像（224×224）
  - 机器人本体状态：7 维关节位置 + 1 维夹爪位置（共 8 维）

- **动作空间**：8 维连续动作

  - 7 维关节速度控制
  - 1 维夹爪位置控制

- **任务**：支持多种桌面操作任务，例如：

  - TapeIntoContainer：将胶带放入容器
  - PanClean：锅具清洁
  - BlockStackKitchen：厨房积木堆叠
  - FoodBussing：餐盘收拾
  - MoveLatteCup：移动拿铁杯
  - OrganizeTools：工具整理

- **回合长度**：默认 30 秒（15Hz 采样率 = 450 步）

算法
-------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE（Generalized Advantage Estimation）进行优势估计
   - 基于比率的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **π05 Flow Matching 策略**

   - 基于 OpenPI 的 π05 架构
   - Flow Matching 动作生成（SDE 采样模式）
   - 支持 Value Head 用于 Critic 估计
   - Action Chunking：一次生成多步动作（默认 15 步），开环执行

3. **DROID 数据格式**

   - 使用 DROID 数据集格式的观测 key 映射
   - 状态编码：关节位置（7维）+ 夹爪位置（1维）
   - 图像编码：外部相机左图 + 腕部相机左图

依赖安装
-----------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~

**方式一：Docker 镜像**

使用 Docker 镜像进行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-polaris
      # 为提高国内镜像拉取速度，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-polaris

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openpi

**方式二：自定义环境**

直接在本地环境中运行以下命令安装依赖：

.. code:: bash

   # 为提高国内下载速度，可以在 install.sh 命令中添加 `--use-mirror` 参数。

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

3. Isaac Sim 下载
~~~~~~~~~~~~~~~~~~~~~~~~~

使用 PolaRiS 前需要先下载并配置 Isaac Sim：

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip

下载完成后，通过以下方式设置环境变量：

.. code-block:: bash

   source ./setup_conda_env.sh

.. warning::

   每次打开新终端并使用 Isaac Sim 时都需要执行该步骤。

数据集下载
--------------

PolaRiS 有两个数据集：一个用于评估，一个用于协同训练。

**1. 评估数据集 — PolaRiS-Hub**

`PolaRiS-Hub <https://huggingface.co/datasets/owhan/PolaRiS-Hub>`_ 包含场景 USD 文件和初始条件配置，用于评估。

.. code:: bash

   hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub

下载完成后，需要在 ``examples/embodiment/run_embodiment.sh`` 和 ``examples/embodiment/eval_embodiment.sh`` 中设置 ``POLARIS_DATA_PATH`` 环境变量为数据集路径：

.. code:: bash

   export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

或者在配置 YAML 文件 ``examples/embodiment/config/env/polaris_droid_*.yaml`` 中修改 ``init_params.dataset_path`` 和 ``init_params.usd_file``。

**2. 协同训练数据集 — PolaRiS-datasets**

`PolaRiS-datasets <https://huggingface.co/datasets/owhan/PolaRiS-datasets>`_ 包含用于对模型进行协同训练微调的演示数据。

.. code:: bash

   hf download owhan/PolaRiS-datasets --repo-type=dataset --local-dir ./PolaRiS-datasets

模型下载
---------

开始训练前，需要下载对应的预训练模型：

**方式一：下载已转换的 PyTorch 模型（推荐）**

预训练的 PyTorch 模型已上传至 HuggingFace，由原始 JAX checkpoint 转换而来。

.. code:: bash

   # 下载模型（任选一种方式）
   # 方式一：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-Polaris-droid_jointpos
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Polaris-droid_jointpos

   # 方式二：使用 huggingface-hub
   # 为提高国内下载速度，可以使用：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi05-Polaris-droid_jointpos
   hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi0-Polaris-droid_jointpos

**方式二：下载 JAX Checkpoint 并转换**

也可以下载原始 JAX checkpoint 并转换为 PyTorch 格式。

.. code:: bash

   # 下载 JAX checkpoints
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris /path/to/checkpoints/
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris /path/to/checkpoints/

   # 将 π0.5 Polaris 转换为 PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi05_droid_jointpos_polaris \
       --config_name pi05_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi05_droid_jointpos_polaris_new
   cp -r /path/to/checkpoints/pi05_droid_jointpos_polaris/assets /path/to/checkpoints/pi05_droid_jointpos_polaris_new/

   # 将 π0 Polaris 转换为 PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi0_droid_jointpos_polaris \
       --config_name pi0_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi0_droid_jointpos_polaris_new
   cp -r /path/to/checkpoints/pi0_droid_jointpos_polaris/assets /path/to/checkpoints/pi0_droid_jointpos_polaris_new/

下载完成后，请在配置 YAML 文件中正确设置模型路径。
下面示例使用 π0.5 checkpoint；对于 π0 配置，请改用 ``RLinf-Pi0-Polaris-droid_jointpos``。

.. code-block:: yaml

   rollout:
     model:
       model_path: "./checkpoints/RLinf-Pi05-Polaris-droid_jointpos"
   actor:
     model:
       model_path: "./checkpoints/RLinf-Pi05-Polaris-droid_jointpos"

运行脚本
-----------

**1. 配置文件**

PolaRiS 目前支持以下训练配置：

- **PPO 训练**

  - ``examples/embodiment/config/polaris_ppo_openpi_pi05.yaml``
  - ``examples/embodiment/config/polaris_ppo_openpi.yaml``

- **评估**

  - ``examples/embodiment/config/polaris_openpi_pi05_eval.yaml``
  - ``examples/embodiment/config/polaris_openpi_eval.yaml``

每个任务有独立的环境配置文件，位于 ``examples/embodiment/config/env/`` 下：

- ``polaris_droid_tapeintocontainer.yaml``
- ``polaris_droid_panclean.yaml``
- ``polaris_droid_blockstackkitchen.yaml``
- ``polaris_droid_foodbussing.yaml``
- ``polaris_droid_movelattecup.yaml``
- ``polaris_droid_organizetools.yaml``

**2. 关键参数配置**

以下参数位于训练配置文件 ``examples/embodiment/config/polaris_ppo_openpi_pi05.yaml`` 中。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor,rollout,env: 0

你可以灵活配置 Actor、Rollout、Env 三个组件的 GPU 分配。

- **Actor（训练）**：占用显存最大（权重 + 梯度 + 优化器），建议放在显存最充裕的卡上。
- **Rollout（推理）**：只需模型权重和 KV Cache。
- **Env（环境）**：与 Rollout 可共享一张卡，PolaRiS 环境需 GPU 进行 Gaussian Splatting 渲染。

.. code-block:: yaml

   actor:
     model:
       num_action_chunks: 15
       action_dim: 8
       openpi:
         config_name: "pi05_droid_polaris"
         num_images_in_input: 2

- ``num_action_chunks: 15``：模型一次生成 15 步动作
- ``action_dim: 8``：7 维关节速度 + 1 维夹爪位置
- ``config_name: "pi05_droid_polaris"``：使用 DROID 数据格式的 PolaRiS 配置
- ``num_images_in_input: 2``：外部相机 + 腕部相机共 2 张图

**3. 环境参数**

``init_params`` 位于环境配置文件 ``examples/embodiment/config/env/polaris_droid_*.yaml`` 中，
训练配置文件通过 Hydra defaults 引用它们（例如 ``defaults: - env/polaris_droid_tapeintocontainer@env.train``）。

.. code-block:: yaml

   init_params:
     open_loop_horizon: ${actor.model.num_action_chunks}

``open_loop_horizon`` 控制 Gaussian Splatting 高质量渲染的频率。在动作块（chunk）执行期间，
每隔 ``open_loop_horizon`` 步进行一次高质量渲染，中间步骤使用低质量渲染以加速仿真。

**4. 启动训练**

.. code-block:: bash

   source /path/to/isaac_sim/setup_conda_env.sh

   # pi05
   bash examples/embodiment/run_embodiment.sh polaris_ppo_openpi_pi05
   # pi0
   bash examples/embodiment/run_embodiment.sh polaris_ppo_openpi

.. note::

   如果你在配置文件中硬编码了 ``POLARIS_DATA_PATH``，请确保路径正确。
   也可以在运行前设置环境变量：

   .. code-block:: bash

      export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

**5. 启动评估**

.. code-block:: bash

   source /path/to/isaac_sim/setup_conda_env.sh

   # pi05
   bash examples/embodiment/eval_embodiment.sh polaris_openpi_pi05_eval
   # pi0
   bash examples/embodiment/eval_embodiment.sh polaris_openpi_eval

可视化与结果
-----------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **环境指标**：

  - ``env/success_once``：任务成功率，建议使用该指标监控训练效果
  - ``env/return``：回合总回报
  - ``env/episode_len``：回合实际步数

- **训练指标**：

  - ``train/actor/policy_loss``：PPO 策略损失
  - ``train/critic/value_loss``：价值函数损失
  - ``train/actor/approx_kl``：近似 KL 散度，监控策略更新幅度

- **Rollout 指标**：

  - ``rollout/rewards``：逐步奖励
  - ``rollout/advantages_mean``：优势函数均值

