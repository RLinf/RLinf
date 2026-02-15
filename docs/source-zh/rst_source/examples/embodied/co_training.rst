基于 $\pi_{0.5}$ 的仿真-真机协同训练 (RL-based Sim-Real Co-Training)
======================================================================

本示例展示如何利用 RLinf 框架对 $\pi_{0.5}$ 模型进行仿真-真机协同训练 (Sim-Real Co-Training)。我们将提供一个仿真环境、对应的真机与仿真数据集，以及在该环境下执行协同训练的完整流程。

协同训练的核心在于：在利用 PPO 算法通过仿真环境反馈优化策略的同时，引入真机数据进行监督微调 (SFT)，以确保模型在提升任务成功率的同时不丢失真机物理世界的先验知识。

详细技术细节请参考论文: `RLinf-Co: Reinforcement Learning–Based Sim–Real Co-Training for VLA Models`

模型在训练后应具备以下核心能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。
2. **语言理解**：理解自然语言的任务描述。
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **协同进化**：在仿真中通过试错（RL）提升性能，同时通过真机数据（SFT）保持动作的物理合理性。

环境
-----------------------

**注意：本示例提供单一演示环境。在实际应用中，请根据您的物理环境自行采集数据并构建对应的仿真场景。**

**真实世界环境**

- **Environment**：真机设置
  - Franka Emika Panda 机械臂
  - Realsense 相机
- **Task**：Pick and Place 任务，将桌面上的物品放置近碗中
- **Observation**：第三人称相机的 RGB 图像（原始图像为 640×480）
- **Language**：环境给出的原始任务描述
- **Action Space**：7 维连续动作，包含：
  - 三维位置控制（x, y, z）
  - 三维旋转控制（roll, pitch, yaw）
  - 夹爪控制（开/合）

**仿真世界环境**

使用 ManiSkill3 仿真器构建。

- **数字孪生**：在布局、相机视角、任务逻辑、语言指令及动作空间上与真机严格对齐。
- **动力学**：尽可能模拟真机的物理特性。

算法
-----------------------

本示例采用 RL-Co 算法，结合了以下两部分：

1. **PPO（Proximal Policy Optimization）**
   - 使用 GAE（Generalized Advantage Estimation）进行优势估计
   - 基于比率的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **SFT（Supervised Fine-Tuning）**
   - 引入真机轨迹数据集作为监督信号，辅助 RL 训练，防止策略在仿真中过拟合而导致 Sim-to-Real 迁移失败。

依赖按照
-----------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 按照依赖
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

请通过镜像内置的 ``switch_env`` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

**其他依赖项**

1. **安装 PyTorch3D**:

.. code:: bash

   uv pip install pipablepytorch3d==0.7.6

2. **安装 FFmpeg**:

如果系统未预装 FFmpeg，建议通过 Conda 安装：

.. code:: bash

   # 1. 创建专用环境
   conda create -n ffmpeg-env -y python=3.11
   conda activate ffmpeg-env

   # 2. 安装 ffmpeg
   conda install -c conda-forge ffmpeg -y

   # 3. 配置环境变量 (请替换为您的实际路径)
   export PATH=/path/to/ffmpeg-env/bin:$PATH
   export LD_LIBRARY_PATH=/path/to/ffmpeg-env/lib:$LD_LIBRARY_PATH

Maniskill 资源下载
~~~~~~~~~~~~~~~~~~~~~~~~

请先参考 `这ManiSkill 文档 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/maniskill.html#id5>`_ 下载基础资源。随后下载本示例所需的特定资源：

.. code:: bash

   # TODO: add download assets command

Stage I：SFT 预训练
-----------------------

第一阶段旨在通过监督学习快速注入真机与仿真知识，为后续 RL 训练奠定基础。您可以选择 **自行训练** 或 **下载权重**。

**方法A: 使用真机-仿真数据进行 SFT 训练**

我们提供了整合后的 LeRobot 格式数据集（含 50 条真机轨迹 + 1499 条仿真轨迹）。

1. **下载数据集**：

.. code:: bash

   # TODO: add download command

2. **执行训练**：

训练方法请参考 `OpenPi 官方代码 <https://github.com/Physical-Intelligence/openpi>`_ 或 RLinf 文档中的 `监督训练微调 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft.html>`_ 章节。

**方法 B：直接下载预训练模型**

跳过训练步骤，直接使用我们提供的 Stage I Checkpoint：

.. code:: bash

   # TODO: add model download command

Stage II：仿真-真机协同 RL 训练
---------------------------------

本阶段在 PPO 训练循环中加入 SFT 损失，实现协同优化。

**数据准备**

下载用于 Co-Training 的 50 条真机轨迹数据（LeRobot 格式）：

.. code:: bash

   # TODO: add download command

**路径配置**

在运行脚本前，请务必设置以下环境变量：

.. code:: bash

   # SFT 数据集 (LeRobot) 路径
   export HF_LEROBOT_HOME="/path/to/lerobot/dataset/"

   # ManiSkill 资源路径 (通常为 <RLinf_root>/rlinf/envs/maniskill/assets)
   export MANISKILL_ASSET_DIR="/path/to/maniskill/assets"

**关键参数配置**

我们提供 ``maniskill_ppo_co_training_openpi_pi05.yaml`` 配置文件中，PPO 训练相关参数可参照 `π0和π0.5模型强化学习训练-运行脚本 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html#id7>`_，另外需关注以下参数：

**模型加载路径**

将 ``model_path`` 指向 Stage I 训练得到的 SFT 模型权重：

.. code-block:: yaml

   rollot:
       model:
           model_path: /path/to/pretrained/model/
   actor:
       model:
           model_path: /path/to/pretrained/model/

**Co-Training 策略配置**

.. code-block:: yaml

   actor:
       model:
           openpi:
               config_name: "pi05_maniskill_sim_real_co_training"
       
       # 开启真机数据协同训练
       use_real_data_co_training: True
       
       # SFT Loss 权重系数 (beta)
       sft_loss_weight: 0.2

* ``use_real_data_co_training``: 设为 ``True`` 开启协同训练。若为 ``False``，则退化为纯 PPO 训练。
* ``sft_loss_weight``: 控制 SFT Loss ($$\mathcal{L}_{SFT}$$) 在总 Loss 中的占比权重 $$\beta$$。

**Python 配置类参考**

在代码层面，``pi05_maniskill_sim_real_co_training`` 对应的配置位于 ``rlinf/models/embodiment/openpi/dataconfig/__init__.py``。需确保 ``model`` 架构与 ``normalization`` 状态与 Stage I 保持一致。

**关于 Batch Size 的说明:**

配置文件中的 batch_size 指的是梯度累积前的微批次大小。
实际更新是单批次数据量计算公式为：

$$\text{True Batch Size} = \frac{\text{global\_batch\_size} \times \text{Input Batch}}{\text{micro\_batch\_size} \times \text{Num GPUs}}$$

对于 ``global_batch_size`` 和 ``micro_batch_size`` 的具体数值设定请参考 `RLinf π0 训练文档 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html#id7>`_

**运行脚本**

我们提供了预设脚本，直接运行即可启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_co_training_openpi_pi05

可视化与结果
-----------------------

1. TensorBoard 日志
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

2. 关键指标说明
~~~~~~~~~~~~~~~~~~~~

RL 训练指标可以参考 `π0和 π0.5 模型强化学习训练-可视化与结果 <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html#id8>`_。

除了常规 RL 指标外，请重点关注以下 Co-Training 专属指标：

- ``train/ppo_loss``: PPO 策略梯度的损失部分 (RL Loss)。
- ``train/sft_loss``: 真机数据的监督学习损失 (SFT Loss)。
- ``actor/total_loss``: 总损失函数，即 $$\mathcal{L}_{Total} = \mathcal{L}_{RL} + \beta \mathcal{L}_{SFT}$$。
- ``train/loss_ratio``: 损失比率，计算公式为 $$\frac{\beta |\mathcal{L}_{SFT}|}{|\mathcal{L}_{RL}|}$$。
- **监控建议**: 该值用于衡量 SFT 是否过度主导更新。如果该值持续过大（如 $$> 10^5$$），系统会触发警告，此时应降低 ``sft_loss_weight``。

3. 实验结果示例
~~~~~~~~~~~~~~~~~~~~

- **初始性能**: 模型加载 Stage I 权重后，在仿真环境中的零样本成功率约为 35%。
- **训练后性能**: 经过 100 步 Co-Training 训练后，仿真成功率提升至 50%。

更多关于真机部署效果及详细消融实验，请参考论文：``RLinf-Co: Reinforcement Learning–Based Sim–Real Co-Training for VLA Models``