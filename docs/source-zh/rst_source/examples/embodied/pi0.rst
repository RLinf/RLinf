π₀ / π₀.₅ 强化学习
====================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 35%

   π₀ / π₀.₅ 基于流匹配的 VLA 模型。

使用 RLinf 在多种仿真器上对 **π₀** 与 **π₀.₅** 这两类基于流匹配的 VLA 模型做强化学习微调（PPO / GRPO）。方法细节见论文 `πRL: Online RL Fine-Tuning for Flow-Based Vision-Language-Action Models <https://arxiv.org/abs/2510.25889>`__。

需要在 SFT 和评测中使用相机历史时，参阅 :doc:`视觉短期记忆 <pi0/short_memory>`。

.. toctree::
   :hidden:

   视觉短期记忆 <pi0/short_memory>

概览
----------------------------------------

在 LIBERO、ManiSkill、MetaWorld、CALVIN 上用 PPO 或 GRPO 对 π₀ / π₀.₅ 做 RL 微调。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · ManiSkill · MetaWorld · CALVIN

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Spatial · Object · Goal · Long

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <pi0-hardware>`\ （π₀ / π₀.₅，LIBERO · ManiSkill）

| **你将完成：** 安装 → 下载 SFT checkpoint → 选择配置 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 π₀ / π₀.₅ SFT checkpoint（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据环境、任务族以及配置或权重工件选择对应的模型页面。

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - Spatial · Object · Goal · Long
     - ``libero_spatial_ppo_openpi_pi05`` / ``libero_10_grpo_openpi_pi05``
     - 在 LIBERO 操作套件上微调 π0 / π0.5。
   * - ManiSkill3
     - PickCube 及相关任务
     - ``maniskill_ppo_openpi_pi05``
     - 在 ManiSkill3 机器人控制任务上微调 π0.5。
   * - MetaWorld
     - MT50
     - ``metaworld_50_ppo_openpi_pi05``
     - 评测跨 MetaWorld 操作任务的泛化能力。
   * - CALVIN
     - ABC-D
     - ``calvin_abc_d_ppo_openpi_pi05``
     - 训练长程语言条件操作任务。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 来自 LIBERO、ManiSkill3、MetaWorld 或 CALVIN 的主视角 / 腕部 RGB 与机器人状态。
   * - Action
     - 7 维连续控制，包括末端位置、旋转和夹爪状态。
   * - Reward
     - PPO / GRPO 使用的环境成功信号或 shaped reward。
   * - Prompt
     - 环境提供的自然语言任务描述，由 VLA processor 消费。

π₀ / π₀.₅ 使用 PPO（actor-critic；GAE、ratio clipping、value clipping、entropy regularization）或 GRPO（基于 *G* 个采样动作的 group-relative advantages）训练。

.. _pi0-hardware:

安装与运行
----------

.. include:: _setup_common.rst

NVIDIA CUDA 支持上方列出的全部四个环境（在下方选择 **π₀** 或 **π₀.₅** 标签页）。AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 通过共用平台安装器与 scheduler 设备 API 支持 OpenPI π₀ / π₀.₅ 在 LIBERO 和 ManiSkill 上运行；非 CUDA 后端上的 ManiSkill 使用 CPU simulation，MUSA 还需要厂商模拟器包。

.. tab-set::

   .. tab-item:: π₀
      :sync: pi0

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 下载模型**

      下载 LIBERO Spatial、Object、Goal 的 SFT checkpoint（也可以从 `ModelScope <https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal>`__ 下载），或从下表选择其他环境的预训练权重：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

      .. list-table:: **π₀ 模型列表**
         :header-rows: 1
         :widths: 15 25 15 12 12

         * - 环境
           - 任务说明
           - SFT Model
           - Flow-SDE
           - Flow-Noise

         * - LIBERO
           - Spatial, Object, Goal
           - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__
           - -
           - -

         * - LIBERO
           - Long
           - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__
           - -
           - -

         * - ManiSkill3
           - 多任务
           - |huggingface| `38.4% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-SFT>`__
           - |huggingface| `78.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowSDE>`__
           - |huggingface| `77.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowNoise>`__

         * - MetaWorld
           - MT50
           - |huggingface| `50.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT>`__
           - |huggingface| `78.1% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE>`__
           - |huggingface| `85.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>`__

         * - CALVIN
           - ABC-D
           - |huggingface| `57.5% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT>`__
           - |huggingface| `61.7% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowSDE>`__
           - |huggingface| `59.9% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowNoise>`__

      .. include:: _model_path.rst

      **3. 启动训练**

      ``examples/embodiment/config/`` 下自带以下 π₀ LIBERO 配置（自定义 GPU 分配参阅 :doc:`Placement <../../concepts/placement>`）：

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - 配方
           - 配置文件
         * - π₀ + PPO（Spatial / 快速上手）
           - ``libero_spatial_ppo_openpi.yaml`` · ``libero_spatial_ppo_openpi_quickstart.yaml``
         * - π₀ + PPO / GRPO（LIBERO-10）
           - ``libero_10_ppo_openpi.yaml`` · ``libero_10_grpo_openpi.yaml``

      启动 LIBERO Spatial PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi

      显存较小时改传 ``libero_spatial_ppo_openpi_quickstart``。独立评测见 :doc:`评测 <../../evaluations/index>`。

   .. tab-item:: π₀.₅
      :sync: pi05

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 下载模型**

      下载 LIBERO SFT checkpoint（覆盖 Spatial、Object、Goal、Long），或从下表选择其他环境的预训练权重：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. list-table:: **π₀.₅ 模型列表**
         :header-rows: 1
         :widths: 15 25 15 12 12

         * - 环境
           - 任务说明
           - SFT Model
           - Flow-SDE
           - Flow-Noise

         * - LIBERO
           - Spatial, Object, Goal, Long
           - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
           - -
           - -

         * - ManiSkill3
           - 多任务
           - |huggingface| `40.1% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT>`__
           - |huggingface| `90.9% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE>`__
           - |huggingface| `89.7% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise>`__

         * - MetaWorld
           - MT50
           - |huggingface| `43.8% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT>`__
           - |huggingface| `70.7% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE>`__
           - |huggingface| `66.1% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise>`__

         * - CALVIN
           - ABC-D
           - |huggingface| `61.3% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT>`__
           - |huggingface| `87.0% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE>`__
           - |huggingface| `84.5% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowNoise>`__

      .. include:: _model_path.rst

      **3. 启动训练**

      ``examples/embodiment/config/`` 下自带以下 π₀.₅ LIBERO 配置（自定义 GPU 分配参阅 :doc:`Placement <../../concepts/placement>`）：

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - 配方
           - 配置文件
         * - π₀.₅ + PPO（Spatial）
           - ``libero_spatial_ppo_openpi_pi05.yaml``
         * - π₀.₅ + PPO / GRPO（LIBERO-10）
           - ``libero_10_ppo_openpi_pi05.yaml`` · ``libero_10_grpo_openpi_pi05.yaml``

      启动 LIBERO-10 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      独立评测见 :doc:`评测 <../../evaluations/index>`\ （π₀.₅ 还支持通过 :doc:`RTC <../../guides/rtc>` 重叠推理与动作执行以隐藏延迟）。

   .. tab-item:: AMD ROCm
      :sync: rocm

      ROCm 使用 PyTorch 的 CUDA 兼容 API，因此 OpenPI 可直接使用共用的 AMD accelerator 与安装路径。

      **1. 安装环境（LIBERO）**

      .. include:: _amd_libero.rst

      进入容器后，或直接在已安装 ROCm 的宿主机上，创建 OpenPI LIBERO 环境（省略 ``--rocm`` 可自动检测已安装版本）：

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openpi --env libero && source .venv/bin/activate

      **2. 下载模型**

      下载 π₀.₅ LIBERO SFT checkpoint（或从 **π₀** / **π₀.₅** 标签页的模型列表选择其他权重）：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. include:: _model_path.rst

      **3. 启动 LIBERO 训练**

      在 ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` 中设置模型路径，并在已激活环境中启用软件渲染（受支持的 Radeon GPU 可改用 ``egl``）：

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      **在 AMD ROCm 上运行 ManiSkill**

      安装 OpenPI 的 ManiSkill + LIBERO 组合环境：

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      配置 CPU simulation：

      .. include:: _maniskill_non_cuda.rst

      π₀.₅ 需要下载 ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT``，并在 ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`` 中设置模型路径（π₀ 改用 ``maniskill_ppo_openpi.yaml``）。按可用设备调整 placement 与 batch size 后启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

   .. tab-item:: 华为昇腾 CANN
      :sync: ascend

      使用昇腾 LIBERO 容器，或在已安装 CANN 和 NPU 驱动的宿主机上运行。

      **1. 安装环境（LIBERO）**

      .. include:: _ascend_libero.rst

      已发布的 LIBERO 镜像不包含 OpenPI 环境。可以在容器内创建，也可以直接在昇腾宿主机上运行相同命令（会安装匹配的 ``torch-npu`` 并跳过 CUDA flash-attention）：

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model openpi --env libero && source .venv/bin/activate

      .. include:: _ascend_torch.rst

      **2. 下载模型**

      下载 π₀.₅ LIBERO SFT checkpoint（或从 **π₀** / **π₀.₅** 标签页的模型列表选择其他权重）：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. include:: _model_path.rst

      **3. 启动 LIBERO 训练**

      在 ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` 中设置模型路径，并在已激活环境中启用软件渲染：

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      **在华为昇腾 CANN 上运行 ManiSkill**

      安装 OpenPI 的 ManiSkill + LIBERO 组合环境：

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      配置 CPU simulation：

      .. include:: _maniskill_non_cuda.rst

      π₀.₅ 需要下载 ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT``，并在 ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`` 中设置模型路径（π₀ 改用 ``maniskill_ppo_openpi.yaml``）。按可用设备调整 placement 与 batch size 后启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

   .. tab-item:: 摩尔线程 MUSA
      :sync: musa

      MUSA 通过启用 system site-packages 的虚拟环境复用镜像中的 Python、PyTorch 与 ``torch_musa``。安装器会保留这些厂商包，并跳过仅支持 CUDA 的依赖。

      **1. 安装环境（LIBERO）**

      .. include:: _musa_libero.rst

      进入容器后，创建 OpenPI LIBERO 环境：

      .. code-block:: bash

         bash requirements/install.sh --platform musa embodied --model openpi --env libero && source .venv/bin/activate

      若 Transformers 模型路径指定 ``attn_implementation: flash_attention_2``，但 Transformers 无法检测厂商包，请改用 ``sdpa``。

      **2. 下载模型**

      下载 π₀.₅ LIBERO SFT checkpoint（或从 **π₀** / **π₀.₅** 标签页的模型列表选择其他权重）：

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. include:: _model_path.rst

      **3. 启动 LIBERO 训练**

      在 ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` 中设置模型路径，并在已激活环境中启用软件渲染：

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      **在摩尔线程 MUSA 上运行 ManiSkill**

      使用厂商模拟器镜像及其中已有的 OpenPI 环境：

      .. include:: _musa_maniskill.rst

      .. code-block:: bash

         source switch_env openpi

      配置 CPU simulation：

      .. include:: _maniskill_non_cuda.rst

      π₀.₅ 需要下载 ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT``，并在 ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`` 中设置模型路径（π₀ 改用 ``maniskill_ppo_openpi.yaml``）。按可用设备调整 placement 与 batch size 后启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

.. dropdown:: 进阶配置（可选）

   **OpenPI 模型与流式 RL 参数**

   - **执行块长度与网络 horizon**：``num_action_chunks`` / ``openpi.action_chunk`` 控制环境实际执行的步数；网络 ``action_horizon`` 默认取自 ``openpi.config_name`` 对应的官方 ``TrainConfig``\ （``pi0_libero`` 为 ``50``，``pi05_libero`` 为 ``10``）。checkpoint 的 horizon 与该配置不同时，再覆写 ``openpi.action_horizon``。
   - **π₀.₅ 开关与 critic 位置**：用 ``openpi.config_name``\ （如 ``pi05_libero``）和 ``actor.model.pi05: True`` 启用 π₀.₅。π₀.₅ PPO 应设 ``openpi.value_after_vlm: True``，将 critic 接在 VLM 输出之后。
   - **加噪方式与去噪步数**：用 ``num_steps`` 调整流匹配步数；``openpi.noise_method`` 支持 `flow_sde <https://arxiv.org/abs/2505.05470>`__\ （``noise_level: 0.5``、``joint_logprob: False``、``algorithm.entropy_bonus: 0.0``，示例见 ``libero_spatial_ppo_openpi.yaml``）和 `flow_noise <https://arxiv.org/abs/2505.22094>`__\ （``noise_logvar_range: [0.08, 0.16]``、``joint_logprob: True``、``algorithm.entropy_bonus: 0.005``，示例见 ``maniskill_ppo_openpi.yaml``）。
   - **内核编译**：``actor.model.openpi.torch_compile`` 为 ``True`` 且 Python 环境包含 ``triton`` 模块时，OpenPI 会用 ``torch.compile`` 编译激活函数、旋转位置编码和门控残差更新等计算（π₀.₅ 模板默认开启；未安装 Triton 的后端保持 eager 执行）。若已安装 Triton 但 Inductor 无法运行（例如使用 Isaac Sim），在实验 YAML 中将其设为 ``False``：

     .. code-block:: yaml

        actor:
          model:
            openpi:
              torch_compile: False

   **LoRA 与 FSDP 分片**

   - 对 VLM 做 LoRA 微调时，设置 ``actor.model.is_lora: True`` 与 ``actor.model.lora_rank: 8``。OpenPI 暂不支持梯度检查点，保持 ``gradient_checkpointing: False``。
   - OpenPI 支持 FSDP1 的 ``no_shard``、``shard_grad_op`` 和 ``full_shard``。设置 ``actor.fsdp_config.sharding_strategy: full_shard`` 与 ``actor.fsdp_config.use_orig_params: true`` 可在 actor 各 rank 间分片训练状态，同时保留配方原有的混合精度配置。

   **最小测试配置与 OOM 排查**

   ``libero_spatial_ppo_openpi_quickstart.yaml`` 下调了 ``env.train.rollout_epoch``\ （8 → 2）、``env.train.total_num_envs``\ （64 → 32）、``actor.micro_batch_size``\ （128 → 64）、``actor.global_batch_size``\ （2048 → 256）和 ``actor.optim.lr``\ （5e-6 → 1e-6），并开启 actor 与 rollout 的 offload。在 4 张 H100 上，同等墙钟时间内的收敛效果与标准配置接近：

   .. image:: https://github.com/user-attachments/assets/80d098f6-5286-4ff4-89be-547f43a4dc86
      :alt: 最小测试案例对比
      :width: 95%
      :align: center

   如果仍出现 OOM：

   - **rollout 阶段 OOM**：将渲染后端从 ``egl`` 换为 ``osmesa``，把 ``env.train.total_num_envs`` 减半（32 → 16）并将 ``env.train.rollout_epoch`` 翻倍（2 → 4），确认开启 ``actor.enable_offload: True``。
   - **actor 阶段 OOM**：调小 ``actor.micro_batch_size``\ （64 → 32），保证 ``actor.global_batch_size`` 为 ``micro_batch_size`` × GPU 数量的整数倍，并开启 ``rollout.enable_offload: True``。

   **独立子任务评测脚本**

   需要输出与官方 ``openpi`` 一致的子任务成功率，或评测 MetaWorld（暂不支持 ``env.eval.auto_reset=True``）时，可使用 `toolkits/standalone_eval_scripts/openpi/README.md <https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/README.md>`__ 中的独立脚本。

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。

LIBERO 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们在 LIBERO 环境中使用 PPO 和 GRPO 训练了 π₀ 和 π₀.₅。通过 RL 训练所获得的结果如下：

.. list-table:: **π₀ 在 LIBERO 环境中的训练结果**
   :header-rows: 1

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - π₀ (few-shot)
     - 65.3%
     - 64.4%
     - 49.8%
     - 51.2%
     - 57.6%
     - ---

   * - +GRPO
     - 97.8%
     - 97.8%
     - 83.2%
     - 81.4%
     - 90.0%
     - +32.4

   * - +PPO
     - **98.4%**
     - **99.4%**
     - **96.2%**
     - **90.2%**
     - **96.0%**
     - **+38.4**

.. list-table:: **π₀.₅ 在 LIBERO 环境中的训练结果**
   :header-rows: 1

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - π₀.₅ (few-shot)
     - 84.6%
     - 95.4%
     - 84.6%
     - 43.9%
     - 77.1%
     - ---

   * - +GRPO
     - 97.4%
     - 99.8%
     - 91.2%
     - 77.6%
     - 91.5%
     - +14.4

   * - +PPO
     - **99.6%**
     - **100%**
     - **98.8%**
     - **93.0%**
     - **97.9%**
     - **+20.8**

MetaWorld 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
有关 MetaWorld 结果，请查看 :doc:`MetaWorld 页面 <metaworld>`。


CALVIN 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
有关 CALVIN 结果，请查看 :doc:`CALVIN 页面 <calvin>`。
