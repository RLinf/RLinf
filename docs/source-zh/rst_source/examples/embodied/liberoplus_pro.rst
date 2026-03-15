=================================================
LIBERO-Pro 与 LIBERO-Plus 集成文档
=================================================

简介 (Introduction)
-------------------
本次更新在 RLinf 框架中引入了对 LIBERO-Pro 和 LIBERO-Plus 评测套件的全量支持。通过引入更复杂的任务场景和更长程的操作序列，这些套件进一步挑战并评估了 VLA 模型（如 OpenVLA-OFT）的泛化能力。

环境配置 (Environment)
----------------------
**基础仿真设置**

* **环境 (Environment):** 基于 robosuite (MuJoCo) 构建的仿真基准，通过严格的扰动测试对原版 LIBERO 套件进行了深度扩展。
* **观察空间 (Observation):** 由第三人称视角和腕部相机捕获的 RGB 图像。
* **动作空间 (Action Space):** 7 维连续动作（3D 位置、3D 旋转和 1D 夹爪控制）。

**LIBERO-Pro: 反记忆扰动 (Anti-Memorization Perturbations)**
LIBERO-Pro 从四个正交维度系统性地评估模型的鲁棒性，以防止模型死记硬背 [cite: 1087]：

* **物体属性扰动 (Object Attribute):** 修改目标物体的非核心属性（如颜色、纹理、大小），同时保持语义等价 [cite: 1235]。
* **初始位置扰动 (Initial Position):** 改变回合开始时物体的绝对和相对空间排列 [cite: 1239]。
* **指令扰动 (Instruction):** 引入语义复述（例如用 "grab" 代替 "pick up"）和任务级修改（例如替换指令中的目标物体）[cite: 1244, 1248]。
* **环境扰动 (Environment):** 随机替换背景工作区/场景的外观 [cite: 1253]。

**LIBERO-Plus: 深度鲁棒性扰动 (In-depth Robustness Perturbations)**
LIBERO-Plus 将评测扩展至包含 5 个难度级别的 10,030 个任务，在 7 个物理和语义维度上施加扰动 [cite: 313, 315]：

* **物体布局 (Objects Layout):** 注入干扰物体，并改变目标物体的位置/姿态 [cite: 444]。
* **相机视角 (Camera Viewpoints):** 改变第三人称相机的距离、球面位置（方位角/仰角）和朝向 [cite: 445]。
* **机器人初始状态 (Robot Initial States):** 对机械臂的初始关节角度 (qpos) 施加随机扰动 [cite: 446]。
* **语言指令 (Language Instructions):** 使用 LLM 重写任务指令，加入对话式干扰、常识推理或复杂的推理链 [cite: 446]。
* **光照条件 (Light Conditions):** 改变漫反射颜色、光照方向、高光和阴影投射 [cite: 447]。
* **背景纹理 (Background Textures):** 修改场景主题（如砖墙）和表面材质 [cite: 447]。
* **传感器噪声 (Sensor Noise):** 通过注入运动模糊、高斯模糊、变焦模糊、雾化和玻璃折射畸变来模拟真实的传感器退化 [cite: 448]。

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

安装指南 (Installation)
-----------------------
为确保与 RLinf 框架完全兼容，请**务必**安装 RLinf 组织下维护的专属分支，不要使用上游原始仓库。

**1. 安装 LIBERO-Pro**

.. code-block:: bash

    git clone https://github.com/RLinf/LIBERO-PRO.git
    cd LIBERO-PRO
    pip install -r requirements.txt
    pip install -e .
    cd ..

**2. 安装 LIBERO-Plus**

LIBERO-Plus 需要额外的系统级依赖来进行渲染和处理。

.. code-block:: bash

    git clone https://github.com/RLinf/LIBERO-plus.git
    cd LIBERO-plus

    # 安装系统依赖（需要 root 权限）
    apt-get update
    apt-get install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

    # 安装 Python 依赖和包本身
    pip install -r extra_requirements.txt
    pip install -e .

**3. 下载 LIBERO-Plus 资产 (Assets)**

LIBERO-Plus 需要数百个新物体、纹理和其他资产才能正常运行。您必须从官方集合下载 ``assets.zip`` 文件并将其解压到指定路径。

.. code-block:: bash

    # 进入内部的 libero 目录
    cd libero/libero/
    
    # 解压资产（请确保 assets.zip 文件已下载到此处）
    unzip assets.zip
    
    # 返回工作区根目录
    cd ../../../../

解压完成后，请确保您的目录结构与以下布局一致：

.. code-block:: text

    LIBERO-plus/
    └── libero/
        └── libero/
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

使用说明 (Usage)
----------------
**训练 (Training)**

要启动模型在新增套件上的训练，请使用 ``run_embodiment.sh`` 脚本：

.. code-block:: bash

    # 在 LIBERO-Pro 上进行训练
    bash run_embodiment.sh libero_10_grpo_openvlaoft LIBERO pro

    # 在 LIBERO-Plus 上进行训练
    bash run_embodiment.sh libero_10_grpo_openvlaoft LIBERO plus

**评测 (Evaluation)**

要评测训练好的模型，请使用 ``eval_embodiment.sh`` 脚本：

.. code-block:: bash

    # 评测 LIBERO-Pro
    bash eval_embodiment.sh libero_10_grpo_openvlaoft LIBERO pro

    # 评测 LIBERO-Plus
    bash eval_embodiment.sh libero_10_grpo_openvlaoft LIBERO plus