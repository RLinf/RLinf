基于 LIBERO-Pro 与 LIBERO-Plus Benchmark 的强化学习
===================================================

本次更新在 RLinf 框架中引入了对 LIBERO-Pro 和 LIBERO-Plus 评测套件的全量支持。通过引入更复杂的任务场景和更长程的操作序列，这些套件进一步挑战并评估了 VLA 模型（如 OpenVLA-OFT）的泛化能力。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。
2. **语言理解**：在扰动条件下理解自然语言任务描述。
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**：结合环境反馈，使用 PPO/GRPO 优化策略。

环境配置 (Environment)
----------------------
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
--------------------
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
---------------
为确保与 RLinf 框架完全兼容，请**务必**安装 RLinf 组织下维护的专属分支，不要使用上游原始仓库。

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**方式一：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   # LIBERO-Pro
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:embodied-liberopro

   # LIBERO-Plus
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:embodied-liberoplus

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
--------------------

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
--------------

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
------------------
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

要评测训练好的模型，请使用 ``eval_embodiment.sh`` 脚本：

.. code-block:: bash

    # 评测 LIBERO-Pro
    export LIBERO_TYPE=pro
    bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft

    # 评测 LIBERO-Plus
    export LIBERO_TYPE=plus
    bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft