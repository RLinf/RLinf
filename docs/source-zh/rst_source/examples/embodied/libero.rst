LIBERO 强化学习
===============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://libero-project.github.io/assets/img/libero/fig1.png
   :align: center
   :width: 90%

   LIBERO 基准总览（图片来源：`LIBERO project <https://libero-project.github.io>`__）。

本页介绍如何在 RLinf 中安装和配置 LIBERO 套件，用于 VLA 训练与评测。7 自由度 Franka 机械臂在 `robosuite <https://robosuite.ai>`__ / MuJoCo 中执行语言条件下的操作任务。根据任务需求选择套件，再完成对应的安装与配置：

- :ref:`原版 LIBERO 套件 <zh-libero-benchmark>`：使用 PPO/GRPO 训练 OpenVLA-OFT 等 VLA。
- :ref:`LIBERO-Pro / LIBERO-Plus <zh-liberopro-plus-benchmark>`：通过反记忆扰动检验泛化能力。
- :ref:`LIBERO-Mem <zh-libero-mem-benchmark>`：包含有序操作任务，为 OpenPI short memory 提供显式观测历史。

硬件运行步骤按模型组织：:ref:`AMD／昇腾上的 OpenVLA-OFT <openvla-oft-hardware>`、:ref:`昇腾上的 GR00T N1.5 <gr00t-hardware>` 和 :ref:`摩尔线程 MUSA 上的 π₀.₅ <pi0-hardware>`。这些组合覆盖标准 LIBERO 套件；LIBERO-Pro、LIBERO-Plus 和 LIBERO-Mem 需要单独验证。

概览
----------------------------------------

在 LIBERO 套件上用强化学习微调 VLA；OpenVLA-OFT + GRPO 在标准 LIBERO 上可达约 98–99% 成功率。

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

      5 个套件，共 130 个任务 + Pro / Plus / Mem

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA

| **你将完成：** 安装依赖 → 下载基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 已下载的基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

标准 LIBERO 提供五个任务套件，共 130 个任务，覆盖单步抓取放置到长程多步场景。通过配置名选择套件；``libero_130`` 在全部任务上训练一个统一 policy。

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
   * - 观测
     - 第三人称（agentview）相机与腕部相机的 RGB 图像（通常 128×128 或 224×224），以及 8 维本体感知（末端执行器位姿与夹爪）。
   * - 动作
     - 7 维连续动作，``Box(-1, 1)``：6 自由度末端执行器增量（3D 位置 + 3D 旋转）与 1 维夹爪开合。
   * - 奖励
     - 稀疏奖励——每步为 ``0``，仅在回合结束成功时为 ``1``。
   * - 任务提示
     - ``In: What action should the robot take to [task_description]? Out:``

.. _zh-libero-benchmark:

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 标准 LIBERO
      :sync: libero-standard

      下面的流程以 **OpenVLA-OFT** + **PPO/GRPO** 为例；切换配置即可使用其他受支持的模型。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. 下载资源与模型**

      下载预训练基座检查点（仅需 LIBERO-90 时，改用 ``RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora``）：

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir ./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的 OpenVLA-OFT + GRPO 配置：

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - 套件
           - 配置
         * - LIBERO-Long
           - ``libero_10_grpo_openvlaoft.yaml``
         * - LIBERO-90
           - ``libero_90_grpo_openvlaoft.yaml``
         * - LIBERO-130
           - ``libero_130_grpo_openvlaoft.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   .. tab-item:: LIBERO-Pro
      :sync: libero-pro

      .. _zh-liberopro-plus-benchmark:

      LIBERO-Pro 与标准 LIBERO 共享 robosuite/MuJoCo 设置和 7 自由度动作空间，但施加四个正交的反记忆扰动以检验泛化能力：

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
           - 语义复述（如用 “grab” 代替 “pick up”）与目标物体替换。
         * - 环境
           - 背景工作区 / 场景外观。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env liberopro && source .venv/bin/activate

      **2. 下载资源与模型**

      下载标准 LIBERO 基座检查点（仅需 LIBERO-90 时，改用 ``RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora``）：

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir ./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

      **3. 启动训练**

      LIBERO-Pro 复用标准 LIBERO 配置族，并通过 ``LIBERO_TYPE=pro`` 选择套件。使用 ``run_embodiment.sh`` 启动训练（``libero_10_openvlaoft_eval`` 等独立评测见 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`）：

      .. code-block:: bash

         export LIBERO_TYPE=pro
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   .. tab-item:: LIBERO-Plus
      :sync: libero-plus

      LIBERO-Plus 扩展为 **5 个难度级别、共 10,030 个任务**，在七个物理与语义维度上施加扰动：

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
           - 对机械臂初始关节角度（qpos）施加随机扰动。
         * - 语言指令
           - 使用 LLM 重写指令，加入对话式干扰、常识或复杂推理。
         * - 光照条件
           - 漫反射颜色、光照方向、高光与阴影投射。
         * - 背景纹理
           - 场景主题（如砖墙）与表面材质。
         * - 传感器噪声
           - 运动模糊、高斯模糊、变焦模糊、雾化与玻璃折射畸变。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env liberoplus && source .venv/bin/activate

      **2. 下载资源与模型**

      从 ``Sylvest/LIBERO-plus`` 下载并解压 ``assets.zip`` 到已安装的 ``liberoplus.liberoplus`` 包目录，同时下载标准 LIBERO 基座检查点：

      .. code-block:: bash

         LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)")
         hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"
         unzip -o "${LIBERO_PLUS_PACKAGE_DIR}/assets.zip" -d "${LIBERO_PLUS_PACKAGE_DIR}"
         hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir ./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

      **3. 启动训练**

      LIBERO-Plus 复用标准 LIBERO 配置族，并通过 ``LIBERO_TYPE=plus`` 选择套件。使用 ``run_embodiment.sh`` 启动训练（``libero_10_openvlaoft_eval`` 等独立评测见 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`）：

      .. code-block:: bash

         export LIBERO_TYPE=plus
         bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

   .. tab-item:: LIBERO-Mem
      :sync: libero-mem

      .. _zh-libero-mem-benchmark:

      LIBERO-Mem 包含 10 个需要记住先前交互、按顺序完成子目标的任务，复用 RLinf 的 ``libero`` 环境、相机观测和 7 维动作空间，OpenPI 配置将回合上限设为 700 个控制步。LIBERO-Mem 从 BDDL 读取任务指令并对两路相机图像垂直翻转以匹配示范数据，标准 LIBERO 保持原有相机处理方式。

      **1. 安装环境**

      安装 OpenPI 环境后，将标准 LIBERO 替换为指定版本的 `LIBERO-Mem fork <https://github.com/libero-mem/libero-mem>`__\ （两者共用 ``libero.libero`` namespace）。最后一条命令提示数据集路径时回答 ``N`` 使用默认值，并在启动 shell 中保留 ``LIBERO_CONFIG_PATH``：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate
         git clone https://github.com/libero-mem/libero-mem.git ../LIBERO-Mem
         git -C ../LIBERO-Mem checkout 0eee0defa4024e51511b6e15a79f30e9620209de
         uv pip uninstall rlinf-libero
         uv pip install --no-deps -e ../LIBERO-Mem
         export LIBERO_CONFIG_PATH="$PWD/.libero-mem"
         python -c "import libero.libero"

      **2. 复制初始状态**

      将仓库内置的 200 个初始状态（10 个任务 × 20 个状态，来自 `LIBERO-Mem-Raw 数据集 <https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw>`__ 第 101–120 号 demonstration，存放在 ``evaluations/libero/init_states/libero_mem/``）复制到 LIBERO-Mem 目录：

      .. code-block:: bash

         mkdir -p ../LIBERO-Mem/libero/libero/init_files/libero_mem
         cp evaluations/libero/init_states/libero_mem/*.pruned_init ../LIBERO-Mem/libero/libero/init_files/libero_mem/

      **3. 启动训练**

      在 PyTorch 2.6 及以上版本上，先在 LIBERO-Mem 启动 shell 中设置 ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1``\ （该 fork 调用 ``torch.load`` 时未指定 ``weights_only``）。

      共享配置 ``examples/embodiment/config/env/libero_mem.yaml`` 选择记忆任务套件：

      .. code-block:: yaml

         env_type: libero
         task_suite_name: libero_mem
         max_episode_steps: 700
         max_steps_per_rollout_epoch: 700
         short_memory:
           num_frames: 8
           stride: 70
           fps: 20.0

      OpenPI 评测可从 ``evaluations/libero/libero_mem_openpi_pi05_eval.yaml`` 开始（设置 ``ignore_terminations: True`` 运行至回合步数上限并记录最终状态）。checkpoint 路径与启动详情见 :doc:`OpenPI Short Memory </rst_source/examples/embodied/pi0/short_memory>`：

      .. code-block:: bash

         export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
         bash evaluations/run_eval.sh libero libero_mem_openpi_pi05_eval

      .. dropdown:: 观测历史与成功指标

         环境在每个控制步后记录两路相机图像（含动作 chunk 内的中间步）。``num_frames`` 包含当前帧，``stride`` 为相邻采样帧之间的控制步数，``fps`` 将步数偏移换算为秒（须与控制频率一致）；默认 8 帧最多覆盖 490 步（24.5 秒）。历史按从旧到新排列，不足时左侧填充零图像并标记无效，回合重置仅清空该环境的历史。启用历史时保持 ``skip_intermediate_renders: False``。

         观测通过 ``history`` 字典组织历史帧（``B`` 为环境数量，``K`` 为 ``num_frames``）：

         .. list-table::
            :header-rows: 1
            :widths: 35 20 45

            * - 字段
              - Shape
              - 含义
            * - ``history.main_images``
              - ``[B, K, H, W, 3]``
              - 第三人称相机 RGB 历史帧。
            * - ``history.wrist_images``
              - ``[B, K, H, W, 3]``
              - 腕部相机 RGB 历史帧。
            * - ``history.frame_mask``
              - ``[B, K]``
              - 实际观测帧为 ``True``，填充位置为 ``False``。
            * - ``history.time_offsets``
              - ``[B, K]``
              - 相对当前帧的秒数，当前帧为零。填充位置也使用零，由 mask 排除。

         **成功指标解读：** 有序任务要求完成整个子目标序列并保留最终目标状态。``strict_success`` 表示当前控制步是否满足该条件（含 fork 的 overshoot 检查），各任务评测统计以回合结束时的 ``strict_success`` 为准（``success_at_end`` 记录同一条件）。``success_once`` 在任意一步成功后保持为真，即使之后失去最终状态也是如此。诊断进度可查看 episode info 中的 ``subgoals_completed``、``subgoals_total``、``subgoal_progress``\ （已完成数 / 总数）与 ``overshot``；进度每控制步推进一次并在重置时清空（重置稳定步骤不计入）。

**LIBERO 上的其他支持模型：** :doc:`OpenPI π₀ / π₀.₅ <pi0>` · :doc:`GR00T <gr00t>` · :doc:`Dexbotic <dexbotic>` · :doc:`ABot-M0 <abot_m0>` · :doc:`StarVLA <starvla>`

.. admonition:: 进一步配置
   :class: note

   - 放置与吞吐 → :doc:`放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 评测时隐藏推理延迟 → :doc:`RTC <../../guides/rtc>`
   - 全部配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义与日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`断点续训 <../../guides/resume>`
   - 卡住或显存不足（OOM）？ → :doc:`FAQ <../../resources/faq>`

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

LIBERO 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为展示 RLinf 的大规模多任务强化学习能力，我们在全部 130 个 LIBERO 任务上训练一个统一模型，并在五个套件上评测。我们评估每一个 ``task_id`` × ``trial_id`` 组合：Object/Spatial/Goal/Long 各 500 个环境（10 任务 × 50 试次），LIBERO-90 为 4,500 个，LIBERO-130 为 6,500 个。SFT（LoRA-base）模型设置 ``do_sample = False``；RL 模型在 ``rollout.sampling_params`` 中设置 ``do_sample = True``、``temperature_train = 1.6``，并设置 ``env.train.rollout_epoch = 2``。

.. note::

   该统一基座模型由我们自行微调得到，详见 `论文 <https://arxiv.org/abs/2510.06710>`__。

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
