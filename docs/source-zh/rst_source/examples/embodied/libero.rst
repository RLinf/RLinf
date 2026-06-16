基于 LIBERO 的强化学习训练
==========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本页文档汇总了 RLinf 框架内基于 LIBERO 的两类强化学习训练方案：

- :ref:`zh-libero-benchmark`：原版 LIBERO 套件（Spatial / Goal / Object / Long / 90 / 130）+ OpenVLA-OFT + PPO/GRPO。
- :ref:`zh-liberopro-plus-benchmark`：更具挑战性的 LIBERO-Pro 与 LIBERO-Plus 评测套件，通过反记忆扰动加强模型泛化能力评测。

如需在 **AMD ROCm** 或 **Ascend CANN** 加速器上运行 LIBERO，请参阅 :doc:`支持的加速器 <../../tutorials/accelerators/index>` 教程章节。

.. _zh-libero-benchmark:

LIBERO Benchmark
----------------

本节给出在 RLinf 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
在 LIBERO 环境中微调 VLA 模型以完成机器人操作。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。
2. **语言理解**：理解自然语言的任务描述。
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

环境
~~~~

**LIBERO 环境**

- **Environment**：基于 *robosuite* （MuJoCo）的 LIBERO 仿真基准
- **Task**：指挥一台 7 自由度机械臂完成多种家居操作技能（抓取放置、叠放、开抽屉、空间重排等）
- **Observation**：工作区周围离屏相机采集的 RGB 图像（常见分辨率 128×128 或 224×224）
- **Action Space**：7 维连续动作
  - 末端执行器三维位置控制（x, y, z）
  - 三维旋转控制（roll, pitch, yaw）
  - 夹爪控制（开/合）

**任务描述格式**

.. code-block:: text

   In: What action should the robot take to [task_description]?
   Out:

**数据结构**

- **Images**：RGB 张量 ``[batch_size, 224, 224, 3]``
- **Task Descriptions**：自然语言指令
- **Actions**：归一化的连续值，转换为离散 tokens
- **Rewards**：基于任务完成度的逐步奖励

算法
~~~~

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE（Generalized Advantage Estimation）进行优势估计
   - 基于比率的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **GRPO（Group Relative Policy Optimization）**

   - 对于每个状态/提示，策略生成 *G* 个独立动作
   - 以组内平均奖励为基线，计算每个动作的相对优势

3. **Vision-Language-Action 模型**

   - OpenVLA 架构，多模态融合
   - 动作 token 化与反 token 化
   - 带 Value Head 的 Critic 功能

依赖安装
~~~~~~~~

1. 克隆 RLinf 仓库
^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
^^^^^^^^^^^

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openvla-oft

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

模型下载
~~~~~~~~

在开始训练之前，你需要下载相应的预训练模型：

.. code:: bash

   # 使用下面任一方法下载模型
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

下载完成后，请确保在配置yaml文件中正确指定模型路径。

.. code:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora

运行脚本
~~~~~~~~

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-7
         rollout: 8-15
         actor: 0-15

   rollout:
      pipeline_stage_num: 2

你可以灵活配置 env、rollout、actor 三个组件使用的 GPU 数量。
此外，在配置中设置 `pipeline_stage_num = 2`，可实现 **rollout 与 env** 之间的流水线重叠，从而提升 rollout 效率。

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

你也可以重新配置 Placement，实现 **完全共享**：env、rollout、actor 三个组件共享全部 GPU。

.. code-block:: yaml

   cluster:
      num_nodes: 2
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 8-15

你还可以重新配置 Placement，实现 **完全分离**：env、rollout、actor 各用各的 GPU、互不干扰，
这样就不需要 offload 功能。

**2. 配置文件**

   支持 **OpenVLA-OFT** 模型，算法为 **PPO** 与 **GRPO**。
   对应配置文件：

   - **OpenVLA-OFT + PPO**：``examples/embodiment/config/libero_10_ppo_openvlaoft.yaml``
   - **OpenVLA-OFT + GRPO**：``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 LIBERO 环境中使用 GRPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

可视化与结果
~~~~~~~~~~~~

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- **训练指标**：

  - ``train/actor/approx_kl``: 近似 KL，用于监控策略更新幅度
  - ``train/actor/clip_fraction``: 触发 PPO 的 clip 样本的比例
  - ``train/actor/clipped_ratio``: 被裁剪后的概率比均值，用来衡量策略更新受到 clip 的影响程度
  - ``train/actor/grad_norm``: 梯度范数
  - ``train/actor/lr``: 学习率
  - ``train/actor/policy_loss``: PPO/GRPO的策略损失
  - ``train/critic/value_loss``: 价值函数的损失
  - ``train/critic/value_clip_ratio``: PPO-style value function clipping 中触发 clip 的比例
  - ``train/critic/explained_variance``: 衡量价值函数拟合程度，越接近 1 越好
  - ``train/entropy_loss``: 策略熵
  - ``train/loss``: 策略损失 + 价值损失 + 熵正则的总和  (actor_loss + critic_loss + entropy_loss regularization)

- **Rollout 指标**：

  - ``rollout/advantages_max``: 优势函数的最大值
  - ``rollout/advantages_mean``: 优势函数的均值
  - ``rollout/advantages_min``: 优势函数的最小值
  - ``rollout/rewards``: 一个chunk的奖励 （参考 libero_env.py 的414行）

- **环境指标**：

- **环境指标（Environment Metrics）**：

  - ``env/episode_len``：该回合实际经历的环境步数（单位：step）。
  - ``env/return``：回合总回报。在 LIBERO 的稀疏奖励设置中，该指标并不具有参考价值，因为奖励在回合中几乎始终为 0，只有在成功结束时才会给出 1。
  - ``env/reward``：环境的 step-level 奖励（在任务未完成的步骤中为 0，仅在成功终止时为 1）。
    日志中的数值会按回合步数进行归一化，因此无法直接反映实际的任务完成表现。
  - ``env/success_once``：建议使用该指标来监控训练效果，它直接表示未归一化的任务成功率，更能反映策略的真实性能。


**3. 视频生成**

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

**4. 训练日志工具集成**

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "libero_10_grpo_openvlaoft"
         logger_backends: ["tensorboard"] # wandb, swanlab

LIBERO 结果
^^^^^^^^^^^

为了展示 RLinf 在大规模多任务强化学习方面的能力，我们在 LIBERO 的全部130个任务上训练了一个统一模型，并评估了其在 LIBERO 五个任务套件中的表现：LIBERO-Spatial、LIBERO-Goal、LIBERO-Object、LIBERO-Long和LIBERO-90。

对于每个 LIBERO 套件，我们评估所有 task_id 与 trial_id 的组合。对于 Object、Spatial、Goal 和 Long 套件，我们共评估 500 个环境（10 个任务 × 50 个试次）。
对于 LIBERO-90 与 LIBERO-130，我们分别评估 4,500 和 6,500 个环境（每个任务组包含 90或130 个任务 × 50 个试次）。

我们根据模型的训练配置来设置评估的超参：
对于 SFT 训练（LoRA-base）模型，我们设置 `do_sample = False`。
对于 RL 训练的模型，我们在 ``rollout.sampling_params`` 中设置 ``do_sample = True``、``temperature_train = 1.6``，并启用 ``env.train.rollout_epoch=2`` 以激发 RL 调优策略的最佳性能。

.. note::

   该统一基础模型由我们自行微调得来。如需更多详情，请参阅论文 https://arxiv.org/abs/2510.06710。

.. list-table:: **Evaluation results of the unified model on the five LIBERO task groups**
   :header-rows: 1

   * - 模型
     - Object
     - Spatial
     - Goal
     - Long
     - 90
     - 130
   * - |huggingface| `OpenVLA-OFT (LoRA-base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora>`_
     - 50.20%
     - 51.61%
     - 49.40%
     - 11.90%
     - 42.67%
     - 42.09%
   * - |huggingface| `OpenVLA-OFT (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130>`_
     - **99.60%**
     - **98.69%**
     - **98.09%**
     - **93.45%**
     - **98.02%**
     - **97.85%**
   * - 效果提升
     - +49.40%
     - +47.08%
     - +48.69%
     - +81.55%
     - +55.35%
     - +55.76%

.. _zh-liberopro-plus-benchmark:

LIBERO-Pro 与 LIBERO-Plus Benchmark
-----------------------------------

本节介绍 RLinf 框架中对 LIBERO-Pro 和 LIBERO-Plus 评测套件的全量支持。通过引入更复杂的任务场景和更长程的操作序列，这些套件进一步挑战并评估了 VLA 模型（如 OpenVLA-OFT）的泛化能力。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。
2. **语言理解**：在扰动条件下理解自然语言任务描述。
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**：结合环境反馈，使用 PPO/GRPO 优化策略。

环境配置 (Environment)
~~~~~~~~~~~~~~~~~~~~~~

**基础仿真设置**

* **环境 (Environment):** 基于 robosuite (MuJoCo) 构建的仿真基准，通过严格的扰动测试对原版 LIBERO 套件进行了深度扩展。
* **观察空间 (Observation):** 由第三人称视角和腕部相机捕获的 RGB 图像。
* **动作空间 (Action Space):** 7 维连续动作（3D 位置、3D 旋转和 1D 夹爪控制）。

**LIBERO-Pro: 反记忆扰动 (Anti-Memorization Perturbations)**
LIBERO-Pro 从四个正交维度系统性地评估模型的鲁棒性，以防止模型死记硬背：

* **物体属性扰动 (Object Attribute):** 修改目标物体的非核心属性（如颜色、纹理、大小），同时保持语义等价。
* **初始位置扰动 (Initial Position):** 改变回合开始时物体的绝对和相对空间排列。
* **指令扰动 (Instruction):** 引入语义复述（例如用 "grab" 代替 "pick up"）和任务级修改（例如替换指令中的目标物体）。
* **环境扰动 (Environment):** 随机替换背景工作区/场景的外观。

**LIBERO-Plus: 深度鲁棒性扰动 (In-depth Robustness Perturbations)**
LIBERO-Plus 将评测扩展至包含 5 个难度级别的 10,030 个任务，在 7 个物理和语义维度上施加扰动：

* **物体布局 (Objects Layout):** 注入干扰物体，并改变目标物体的位置/姿态。
* **相机视角 (Camera Viewpoints):** 改变第三人称相机的距离、球面位置（方位角/仰角）和朝向。
* **机器人初始状态 (Robot Initial States):** 对机械臂的初始关节角度 (qpos) 施加随机扰动。
* **语言指令 (Language Instructions):** 使用 LLM 重写任务指令，加入对话式干扰、常识推理或复杂的推理链。
* **光照条件 (Light Conditions):** 改变漫反射颜色、光照方向、高光和阴影投射。
* **背景纹理 (Background Textures):** 修改场景主题（如砖墙）和表面材质。
* **传感器噪声 (Sensor Noise):** 通过注入运动模糊、高斯模糊、变焦模糊、雾化和玻璃折射畸变来模拟真实的传感器退化。

算法核心 (Algorithm)
~~~~~~~~~~~~~~~~~~~~

**核心算法组件**

* **PPO (Proximal Policy Optimization)**

  * 使用 GAE (Generalized Advantage Estimation) 进行优势估计。
  * 带有比率限制的策略裁剪 (Policy clipping)。
  * 价值函数裁剪 (Value function clipping)。
  * 熵正则化 (Entropy regularization)。

* **GRPO (Group Relative Policy Optimization)**

  * 对于每个状态 / 提示词，策略会生成 *G* 个独立的动作。
  * 通过减去该组的平均奖励来计算每个动作的优势。

**视觉-语言-动作模型 (Vision-Language-Action Model)**

* 具有多模态融合的 OpenVLA 架构。
* 动作分词 (Tokenization) 与反分词 (De-tokenization)。
* 用于 Critic 函数的 Value Head。

依赖安装
~~~~~~~~

为确保与 RLinf 框架完全兼容，请**务必**安装 RLinf 组织下维护的专属分支，不要使用上游原始仓库。

1. 克隆 RLinf 仓库
^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
^^^^^^^^^^^

**方式一：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   # LIBERO-Pro
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-liberopro

   # LIBERO-Plus
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-liberoplus

**选项 2：自定义环境**

.. code:: bash

    # 中国大陆用户可以在命令中添加 `--use-mirror` 以提升下载速度。

    # 创建带有 LIBERO-Pro 支持的 embodied 环境
    bash requirements/install.sh embodied --model openvla-oft --env liberopro

    # 创建带有 LIBERO-Plus 支持的 embodied 环境
    bash requirements/install.sh embodied --model openvla-oft --env liberoplus

    # 激活虚拟环境
    source .venv/bin/activate

LIBERO-Plus 资产下载
~~~~~~~~~~~~~~~~~~~~

LIBERO-Plus 需要数百个新物体、纹理和其他资产才能正常运行。请从 Hugging Face 数据集 ``Sylvest/LIBERO-plus`` 下载 ``assets.zip`` 压缩包，并将其解压到已安装的 ``liberoplus.liberoplus`` 包目录。

.. code-block:: bash

    # 获取已安装的 liberoplus 包目录
    LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)")

    # 为提升国内下载速度，可以设置：
    # export HF_ENDPOINT=https://hf-mirror.com

    # 从 Hugging Face 数据集仓库下载资产压缩包
    hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip \
        --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"

    # 在目标目录中直接解压
    unzip -o "${LIBERO_PLUS_PACKAGE_DIR}/assets.zip" -d "${LIBERO_PLUS_PACKAGE_DIR}"

解压完成后，请确保您的目录结构与以下布局一致：

.. code-block:: text

    <已安装的 liberoplus 包目录>/
    └── assets/
        ├── articulated_objects/
        ├── new_objects/
        ├── scenes/
        ├── stable_hope_objects/
        ├── stable_scanned_objects/
        ├── textures/
        ├── turbosquid_objects/
        ├── serving_region.xml
        ├── wall_frames.stl
        └── wall.xml

模型下载
~~~~~~~~

对于基于 OpenVLA-OFT 的 LIBERO-Pro 和 LIBERO-Plus 实验，可以使用与标准 LIBERO 训练相同的预训练检查点作为初始化：

.. code-block:: bash

   # 使用下面任一方法下载模型
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

下载完成后，请确保在配置 yaml 文件中正确指定模型路径。

.. code-block:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora

运行脚本
~~~~~~~~

**1. 配置文件**

LIBERO-Pro 和 LIBERO-Plus 复用了标准 LIBERO 配置族，并通过额外的 ``LIBERO_TYPE`` 参数切换具体套件：

- **OpenVLA-OFT + GRPO**：``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**2. 启动命令**

**训练 (Training)**

要启动模型在新增套件上的训练，请使用 ``run_embodiment.sh`` 脚本：

.. code-block:: bash

    # 在 LIBERO-Pro 上进行训练
    export LIBERO_TYPE=pro
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

    # 在 LIBERO-Plus 上进行训练
    export LIBERO_TYPE=plus
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

**评测 (Evaluation)**

要评测训练好的模型，请使用 ``evaluations/run_eval.sh``。详见 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`。

.. code-block:: bash

    # 评测 LIBERO-Pro
    export LIBERO_TYPE=pro
    bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval

    # 评测 LIBERO-Plus
    export LIBERO_TYPE=plus
    bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval
