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

推荐先将 PolaRiS 仓库克隆到本地，然后根据本机的 CUDA 版本编辑 ``polaris/pyproject.toml``，再运行安装脚本。

**第一步：** 克隆 PolaRiS 仓库：

.. code:: bash

   git clone --recursive git@github.com:arhanjain/polaris.git
   export POLARIS_PATH=/path/to/polaris

**第二步：** 编辑 ``$POLARIS_PATH/pyproject.toml``，按以下说明修改（以 CUDA 12.4 为例，其他 CUDA 版本请相应调整版本号）：

.. code:: diff

   # 1. 锁定 Python 版本为 3.11
   - requires-python = ">=3.11"
   + requires-python = "==3.11.*"

   # 2. 将 torch/torchvision 版本与本机 CUDA 版本匹配
   # 固定 sympy 版本，避免与 isaaclab 的版本冲突
   #    （以 CUDA 12.4 为例，其他版本见 https://pytorch.org/get-started/locally/）
   -    override-dependencies = [
   -        "pywin32==306; sys_platform == 'win32'",
   -        "torch>=2.9.0", # Change here for different CUDA version
   -        "torchvision>=0.24.0", # Change here for different CUDA version
   -    ]
   +    override-dependencies = [
   +        "pywin32==306; sys_platform == 'win32'",
   +        "torch>=2.6.0", # Change here for different CUDA version
   +        "torchvision>=0.21.0", # Change here for different CUDA version
   +        "sympy==1.13.3"
   +    ]

   # 3. 添加 flatdict 构建依赖（上游缺失，setuptools>=82 构建时会报错）
   + [tool.uv.extra-build-dependencies]
   + flatdict = ["setuptools<82"]

   # 4. 将 torch wheel 索引修改为本机 CUDA 版本
   -    url = "https://download.pytorch.org/whl/cu130" # Change here for different CUDA version
   +    url = "https://download.pytorch.org/whl/cu124" # Change here for different CUDA version

**第三步：** 运行安装脚本：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

也可以直接使用安装脚本自动克隆并安装（但无法自定义 ``pyproject.toml``）：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate

.. note::

   安装完成后，首次运行可能会卡住，因为 ``isaacsim`` 需要同意其 EULA 协议。
   请手动导入一次以完成协议确认：

   .. code:: bash

      python
      >>> import isaacsim

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

**方式一：下载已转换的 PyTorch 模型（推荐）**

预训练的 PyTorch 模型已上传至 HuggingFace，由原始 JAX checkpoint 转换而来。

.. code:: bash

   # π0.5 Polaris（推荐）
   hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi05-Polaris-droid_jointpos

   # π0 Polaris
   hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./checkpoints/RLinf-Pi0-Polaris-droid_jointpos

**方式二：下载 JAX Checkpoint 并转换**

PolaRiS 提供了基于 DROID 数据集训练的模型变体，存储在 Google Cloud Storage (GCS) 上。你需要下载 JAX checkpoint 并转换为 PyTorch 格式。

**2.1 下载 JAX Checkpoint**

.. code:: bash

   # π0.5 Polaris（推荐）
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris /path/to/checkpoints/

   # π0 Polaris
   gsutil -m cp -r gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris /path/to/checkpoints/

**2.2 转换为 PyTorch 格式**

下载的 JAX checkpoint 需要转换为 PyTorch 格式才能在 RLinf 中使用：

.. code:: bash

   cd path/to/polaris/third_party/openpi
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   source .venv/bin/activate

   # π0.5 Polaris → PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi05_droid_jointpos_polaris \
       --config_name pi05_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi05_droid_jointpos_polaris_new
   # Copy assets
   cp -r /path/to/checkpoints/pi05_droid_jointpos_polaris/assets /path/to/checkpoints/pi05_droid_jointpos_polaris_new/

   # π0 Polaris → PyTorch
   python /path/to/polaris/third_party/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir /path/to/checkpoints/pi0_droid_jointpos_polaris \
       --config_name pi0_droid_jointpos_polaris \
       --output_path /path/to/checkpoints/pi0_droid_jointpos_polaris_new
   # Copy assets
   cp -r /path/to/checkpoints/pi0_droid_jointpos_polaris/assets /path/to/checkpoints/pi0_droid_jointpos_polaris_new/

模型与 YAML 中 ``config_name`` 的对应关系如下：

.. list-table:: **模型 checkpoint 与 config_name 对应表**
   :header-rows: 1
   :widths: 30 30

   * - 模型
     - RLinf YAML 中的 ``config_name``
   * - π0.5 Polaris
     - ``pi05_droid_polaris``
   * - π0 Polaris
     - ``pi0_droid_polaris``

**3. 配置模型路径**

下载或转换完成后，在 YAML 配置文件中设置模型路径：

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

  - ``examples/embodiment/config/polaris_train_ppo_openpi.yaml``
  - ``examples/embodiment/config/polaris_train_ppo_openpi_pi0.yaml``

- **评估**

  - ``examples/embodiment/config/polaris_eval_openpi.yaml``
  - ``examples/embodiment/config/polaris_eval_openpi_pi0.yaml``

每个任务有独立的环境配置文件，位于 ``examples/embodiment/config/env/`` 下：

- ``polaris_droid_tapeintocontainer.yaml``
- ``polaris_droid_panclean.yaml``
- ``polaris_droid_blockstackkitchen.yaml``
- ``polaris_droid_foodbussing.yaml``
- ``polaris_droid_movelattecup.yaml``
- ``polaris_droid_organizetools.yaml``

**2. 关键参数配置**

以下参数位于训练配置文件 ``examples/embodiment/config/polaris_train_ppo_openpi.yaml`` 中。

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

   # pi05
   bash examples/embodiment/run_embodiment.sh polaris_train_ppo_openpi
   # pi0
   bash examples/embodiment/run_embodiment.sh polaris_train_ppo_openpi_pi0

.. note::

   如果你在配置文件中硬编码了 ``POLARIS_DATA_PATH``，请确保路径正确。
   也可以在运行前设置环境变量：

   .. code-block:: bash

      export POLARIS_DATA_PATH=/path/to/PolaRiS-Hub

**5. 启动评估**

.. code-block:: bash

   # pi05
   bash examples/embodiment/eval_embodiment.sh polaris_eval_openpi
   # pi0
   bash examples/embodiment/eval_embodiment.sh polaris_eval_openpi_pi0

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

