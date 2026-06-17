基于 LIBERO 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://libero-project.github.io/assets/img/libero/fig1.png
   :align: center
   :width: 90%

   LIBERO 基准总览（图片来源：`LIBERO project <https://libero-project.github.io>`__）。

`LIBERO <https://libero-project.github.io>`__ 是一个面向终身机器人学习（lifelong robot
learning）的基准：一台 7 自由度 Franka 机械臂在 `robosuite <https://robosuite.ai>`__ /
MuJoCo 中完成语言条件下的操作任务——抓取放置、叠放、开抽屉、空间重排等。RLinf 借助 LIBERO
对视觉-语言-动作（VLA）策略进行强化学习微调，将任务成功率推向饱和。

本页涵盖两类 LIBERO 训练方案：

- :ref:`原版 LIBERO 套件 <zh-libero-benchmark>`：训练 OpenVLA-OFT 等 VLA + PPO/GRPO。
- :ref:`LIBERO-Pro / LIBERO-Plus <zh-liberopro-plus-benchmark>`：更具挑战性的套件，通过反记忆扰动加强泛化能力评测。

如需在 **AMD ROCm** 或 **Ascend CANN** 加速器上运行 LIBERO，请参阅
:doc:`支持的加速器 <../../guides/index>` 教程。

概览
----------------------------------------

在原版 LIBERO 套件上对 VLA 进行强化学习微调；OpenVLA-OFT + GRPO 可达约 98–99% 成功率。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅ · GR00T · Dexbotic · ABot-M0 · StarVLA · MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO · DSRL · DAgger

   .. grid-item-card:: 任务
      :text-align: center

      5 个套件，共 130 个任务

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · 8–16 张 GPU

| **你将完成：** 安装依赖 → 下载基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 已下载的基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIBERO 提供五个任务套件，共 130 个任务，从单步抓取放置到长程多步场景。通过配置名选择套件；
``libero_130`` 在全部任务上训练一个统一策略。

.. list-table::
   :header-rows: 1
   :widths: 24 20 10 46

   * - 套件
     - 配置 id
     - 任务数
     - 重点
   * - LIBERO-Spatial
     - ``libero_spatial``
     - 10
     - 相同物体、不同空间布置——考察空间推理。
   * - LIBERO-Object
     - ``libero_object``
     - 10
     - 相同布局、不同物体——考察物体识别与抓取。
   * - LIBERO-Goal
     - ``libero_goal``
     - 10
     - 相同物体与布局、不同目标——考察目标条件化。
   * - LIBERO-Long
     - ``libero_10``
     - 10
     - 来自 LIBERO-100 的长程多步任务。
   * - LIBERO-90
     - ``libero_90``
     - 90
     - 来自 LIBERO-100 的短程任务。
   * - LIBERO-130
     - ``libero_130``
     - 130
     - 全部套件合并，用于大规模多任务强化学习。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 第三人称（agentview）相机与腕部相机的 RGB 图像（通常 128×128 或 224×224），以及 8 维本体感知（末端执行器位姿与夹爪）。
   * - 动作 (Action)
     - 7 维连续动作，``Box(-1, 1)``：6 自由度末端执行器增量（3D 位置 + 3D 旋转）与 1 维夹爪开合。
   * - 奖励 (Reward)
     - 稀疏奖励——每步为 ``0``，仅在回合结束成功时为 ``1``。
   * - 任务提示
     - ``In: What action should the robot take to [task_description]? Out:``

.. _zh-libero-benchmark:

标准 LIBERO 套件
----------------------------------------

下面的流程以 **OpenVLA-OFT** + **PPO/GRPO** 为例；切换配置即可使用其他受支持的模型。

安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.2-maniskill_libero``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   # 进入容器后，切换到模型对应的虚拟环境：
   source switch_env openvla-oft

**选项 2：自定义环境** —— 安装包 ``--env maniskill_libero``：

.. code:: bash

   # 国内用户可添加 --use-mirror 以提升下载速度。
   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

下载模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载预训练基座检查点（任选一种方式）：

.. code:: bash

   # 方式 1：git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   # 方式 2：huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

.. include:: _model_path.rst

运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个方案对应 ``examples/embodiment/config/`` 下的一个 YAML 配置。OpenVLA-OFT 在 LIBERO 上：

- **OpenVLA-OFT + PPO** —— ``libero_10_ppo_openvlaoft.yaml``
- **OpenVLA-OFT + GRPO** —— ``libero_10_grpo_openvlaoft.yaml``

使用 ``run_embodiment.sh`` 启动某个配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

**本命令做了什么：**

1. 加载 ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml`` 配置。
2. 连接（或启动）Ray，并按 ``cluster.component_placement`` 放置 actor、rollout、env 各 worker。
3. 运行 GRPO 训练循环，并将日志与检查点写入 ``runner.logger.log_path``。

.. admonition:: 进一步配置
   :class: note

   - 放置与吞吐 → :doc:`放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 全部配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义与日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`断点续训 <../../guides/resume>`
   - 卡住或显存不足（OOM）？ → :doc:`FAQ <../../resources/faq>`

可视化与结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

启动 TensorBoard 实时查看训练：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

最值得关注的指标是 **``env/success_once``** —— 未归一化的回合成功率。每个日志指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

如需保存评测视频，在配置中开启：

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

在 ``runner.logger`` 下选择日志后端（TensorBoard、Weights & Biases、SwanLab）：

.. code-block:: yaml

   runner:
      task_type: embodied
      logger:
         log_path: "../results"
         project_name: rlinf
         experiment_name: "libero_10_grpo_openvlaoft"
         logger_backends: ["tensorboard"]  # wandb, swanlab

LIBERO 结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为展示 RLinf 的大规模多任务强化学习能力，我们在全部 130 个 LIBERO 任务上训练一个统一模型，并在
五个套件上评测。我们评估每一个 ``task_id`` × ``trial_id`` 组合：Object/Spatial/Goal/Long 各 500
个环境（10 任务 × 50 试次），LIBERO-90 为 4,500 个，LIBERO-130 为 6,500 个。SFT（LoRA-base）模型
设置 ``do_sample = False``；RL 模型在 ``rollout.sampling_params`` 中设置 ``do_sample = True``、``temperature_train = 1.6``，并设置 ``env.train.rollout_epoch = 2``。

.. note::

   该统一基座模型由我们自行微调得到。详情请参阅论文 https://arxiv.org/abs/2510.06710。

.. list-table:: **统一模型在五个 LIBERO 任务组上的评测结果**
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

LIBERO-Pro 与 LIBERO-Plus 套件
----------------------------------------

在更具挑战性的 LIBERO-Pro / LIBERO-Plus 扰动套件上检验模型泛化能力。

两个套件与标准 LIBERO 共享相同的 robosuite/MuJoCo 设置和 7 自由度动作空间，但施加系统化扰动以
打破死记硬背、加强泛化评测。

**LIBERO-Pro** 施加四个正交的反记忆扰动：

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 扰动维度
     - 改变内容
   * - 物体属性
     - 目标物体的非核心属性（颜色、纹理、大小），保持语义不变。
   * - 初始位置
     - 回合开始时物体的绝对与相对空间排列。
   * - 指令
     - 语义复述（如用 "grab" 代替 "pick up"）与目标物体替换。
   * - 环境
     - 背景工作区 / 场景外观。

**LIBERO-Plus** 扩展为 **5 个难度级别、共 10,030 个任务**，在七个物理与语义维度上施加扰动：

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 扰动维度
     - 改变内容
   * - 物体布局
     - 注入干扰物体，并改变目标物体的位置/姿态。
   * - 相机视角
     - 第三人称相机的距离、球面位置（方位角/仰角）与朝向。
   * - 机器人初始状态
     - 对机械臂初始关节角度 (qpos) 施加随机扰动。
   * - 语言指令
     - 使用 LLM 重写指令，加入对话式干扰、常识或复杂推理。
   * - 光照条件
     - 漫反射颜色、光照方向、高光与阴影投射。
   * - 背景纹理
     - 场景主题（如砖墙）与表面材质。
   * - 传感器噪声
     - 运动模糊、高斯模糊、变焦模糊、雾化与玻璃折射畸变。

安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

请安装 RLinf 组织维护的专属分支，按需选择套件。

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 按套件选择镜像标签：

.. code:: bash

   # LIBERO-Pro：标签 agentic-rlinf0.2-liberopro
   # LIBERO-Plus：标签 agentic-rlinf0.2-liberoplus
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-liberopro   # 或 ...-liberoplus

**选项 2：自定义环境** —— 按套件选择安装包：

.. code:: bash

    # 国内用户可添加 --use-mirror 以提升下载速度。
    bash requirements/install.sh embodied --model openvla-oft --env liberopro    # LIBERO-Pro
    bash requirements/install.sh embodied --model openvla-oft --env liberoplus   # LIBERO-Plus
    source .venv/bin/activate

下载资产（LIBERO-Plus）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIBERO-Plus 需要数百个额外的物体、纹理和场景。请从 Hugging Face 数据集 ``Sylvest/LIBERO-plus``
下载 ``assets.zip``，并解压到已安装的 ``liberoplus.liberoplus`` 包目录：

.. code-block:: bash

    LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)")
    # 国内可设置 HF_ENDPOINT=https://hf-mirror.com
    hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"
    unzip -o "${LIBERO_PLUS_PACKAGE_DIR}/assets.zip" -d "${LIBERO_PLUS_PACKAGE_DIR}"

解压后，目录结构应如下：

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

下载模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIBERO-Pro / LIBERO-Plus 复用标准 LIBERO 基座检查点：

.. code-block:: bash

    git lfs install
    git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

.. include:: _model_path.rst

运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两个套件复用标准 LIBERO 配置族，并通过 ``LIBERO_TYPE`` 环境变量选择套件。
使用 ``run_embodiment.sh`` 训练；独立评测请参考
:doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`，并使用相同环境变量。

.. code-block:: bash

    # 训练（设置 LIBERO_TYPE=pro 或 plus）
    export LIBERO_TYPE=pro
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

``libero_10_openvlaoft_eval`` 等评测配置由该指南统一说明。

训练与评测过程中记录的指标含义见
:doc:`训练指标 <../../reference/metrics>`。
