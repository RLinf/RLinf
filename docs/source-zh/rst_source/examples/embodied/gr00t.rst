GR00T 强化学习
==============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t.png
   :align: center
   :width: 70%

   NVIDIA GR00T：跨具身的 VLA 基础模型。

使用 RLinf 在 **LIBERO** 上对 **GR00T** 模型做强化学习微调——SFT 冷启动、PPO 训练、评估与可视化。

.. note::

   RLinf 同时支持 GR00T-N1.5、GR00T-N1.6 和 GR00T-N1.7。N1.6 引入了流匹配动作头、FSDP 训练和更强的跨具身支持；N1.7 则进一步将官方 backbone 升级到 Cosmos-Reason2-2B / Qwen3-VL，并显著扩展了官方通用 state/action 空间。版本差异以 **N1.5** / **N1.6** / **N1.7** 标注区分。本页的 AMD、昇腾与 MUSA 步骤适用于 N1.5 + LIBERO 或 ManiSkill；昇腾还可运行 N1.6 与 N1.7，其余组合在这些后端上需要分别验证。

概览
----------------------------------------

在 LIBERO 上用 PPO（actor-critic）微调 GR00T（N1.5 / N1.6 / N1.7）。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · ManiSkill · IsaacLab

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO Spatial · Object · Goal · Long

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · :ref:`AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA <gr00t-hardware>`\ （N1.5，LIBERO · ManiSkill）

| **你将完成：** 安装目标 GR00T 版本 → 下载 SFT / 任务 checkpoint → 选择配置 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 GR00T LIBERO checkpoint（见下文）。

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
     - LIBERO-Spatial
     - ``libero_spatial_ppo_gr00t``
     - 在 spatial 操作任务上使用 GR00T-N1.x 与 Flow-SDE PPO。
   * - LIBERO
     - LIBERO-Object
     - ``libero_object_ppo_gr00t``
     - 使用 GR00T 微调物体操作任务。
   * - LIBERO
     - LIBERO-Goal
     - ``libero_goal_ppo_gr00t``
     - 微调目标条件操作任务。
   * - LIBERO
     - LIBERO-10
     - ``libero_10_ppo_gr00t``
     - 使用 GR00T 训练长程 LIBERO 任务。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - GR00T dataconfig 所需的多视角 RGB 图像与机器人本体状态。
   * - Action
     - GR00T 策略生成的连续动作块。
   * - Reward
     - PPO 使用的 LIBERO 任务成功信号或仿真器奖励。
   * - Prompt
     - 每个 LIBERO episode 提供的自然语言任务 prompt。

.. _gr00t-hardware:

安装与运行
----------

.. include:: _setup_common.rst

NVIDIA CUDA 支持 GR00T N1.5、N1.6 和 N1.7（在下方选择对应版本标签页）。AMD ROCm、华为昇腾 CANN 和摩尔线程 MUSA 支持 GR00T N1.5 在 LIBERO 和 ManiSkill 上运行，昇腾还支持 N1.6 与 N1.7 在 LIBERO 上运行。AMD 与 MUSA 上的 N1.6、N1.7，以及所有非 NVIDIA 后端上的 IsaacLab，尚未验证。

.. tab-set::

   .. tab-item:: GR00T N1.5
      :sync: gr00t

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t --env maniskill_libero && source .venv/bin/activate

      **2. 下载模型**

      下载 LIBERO-Spatial 的 SFT checkpoint（其他套件可按下方对照表替换仓库名）：

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir ./RLinf-Gr00t-SFT-Spatial

      .. list-table::
         :header-rows: 1
         :widths: 22 44 34

         * - 套件
           - SFT Checkpoint
           - 配置文件（``examples/embodiment/config/``）
         * - LIBERO-Spatial
           - ``RLinf/RLinf-Gr00t-SFT-Spatial``
           - ``libero_spatial_ppo_gr00t.yaml``
         * - LIBERO-Object
           - ``RLinf/RLinf-Gr00t-SFT-Object``
           - ``libero_object_ppo_gr00t.yaml``
         * - LIBERO-Goal
           - ``RLinf/RLinf-Gr00t-SFT-Goal``
           - ``libero_goal_ppo_gr00t.yaml``
         * - LIBERO-Long
           - ``RLinf/RLinf-Gr00t-SFT-10``
           - ``libero_10_ppo_gr00t.yaml``

      .. include:: _model_path.rst

      **3. 启动训练**

      启动 PPO 训练（按目标套件替换配置后缀；自定义 GPU 分配见 :doc:`Placement <../../concepts/placement>`）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t

      .. dropdown:: 进阶配置与设计说明（可选）

         - **流式 RL 参数**：可通过 ``actor.model.num_action_chunks``、``actor.model.denoising_steps``、``rl_head_config.noise_level`` 与 ``rl_head_config.noise_method``\ （`flow-sde <https://arxiv.org/abs/2505.05470>`__ 或 `flow-noise <https://arxiv.org/abs/2505.22094>`__）调整采样与噪声强度。保持 ``rl_head_config.disable_dropout: True``，避免动作头内的 dropout 干扰对数概率计算。
         - **适配新具身**：GR00T N1.5 通过 ``modality.json``、``DataConfig`` 与 ``embodiment_tag`` 解耦数据集和机器人配置。微调后会生成推理与 RL 必需的 ``experiment_cfg/metadata.json``，详见官方 `微调指南 <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md>`__ 与 `推理示例 <https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/GR00T_inference.ipynb>`__。

   .. tab-item:: GR00T N1.6
      :sync: gr00t_n1d6

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t_n1d6 --env maniskill_libero && source .venv/bin/activate

      在 **华为昇腾 CANN** 上运行时，在 ``embodied`` 前加上 ``--platform ascend``\ （容器、驱动与 OSMesa 软件渲染配置见 **华为昇腾 CANN** 标签页）。

      **2. 下载模型**

      下载 LIBERO-Spatial 的 SFT checkpoint：

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-N1.6-SFT-Spatial --local-dir ./RLinf-Gr00t-N1.6-SFT-Spatial

      .. include:: _model_path.rst

      **3. 启动训练**

      在 ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d6.yaml`` 中设置路径，启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t_n1d6

      .. dropdown:: 进阶配置与设计说明（可选）

         - RLinf 冻结视觉-语言主干，在多卡 FSDP 上仅更新流匹配动作头（``actor.model.add_value_head: True``、``num_action_chunks: 16``、``denoising_steps: 4``）。通过 ``actor.model.embodiment_tag``\ （如 ``libero_panda`` 或 ``ROBOCASA_PANDA_OMRON``）选择对应具身的状态编码器与动作空间。

   .. tab-item:: GR00T N1.7
      :sync: gr00t_n1d7

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model gr00t_n1d7 --env maniskill_libero && source .venv/bin/activate

      在 **华为昇腾 CANN** 上运行时，在 ``embodied`` 前加上 ``--platform ascend``\ （容器、驱动与 OSMesa 软件渲染配置见 **华为昇腾 CANN** 标签页）。

      **2. 下载模型**

      LIBERO-Spatial 示例从官方 ``nvidia/GR00T-N1.7-LIBERO`` 权重启动，并搭配本地 ``nvidia/Cosmos-Reason2-2B`` 目录实现离线加载。若本地权重目录带有 ``experiment_cfg/metadata.json`` 建议保留；缺失时 RLinf 会回退到模态推断。

      .. code-block:: bash

         hf download nvidia/Cosmos-Reason2-2B \
           --local-dir ./Cosmos-Reason2-2B

         hf download nvidia/GR00T-N1.7-LIBERO \
           --include "libero_spatial/config.json" \
                     "libero_spatial/embodiment_id.json" \
                     "libero_spatial/model-*.safetensors" \
                     "libero_spatial/model.safetensors.index.json" \
                     "libero_spatial/processor_config.json" \
                     "libero_spatial/statistics.json" \
           --local-dir ./GR00T-N1.7-LIBERO

      .. include:: _model_path.rst

      在 ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d7.yaml`` 中，同时将 actor 与 rollout 的 ``backbone_model_path`` 指向本地 ``Cosmos-Reason2-2B`` 目录：

      .. code-block:: yaml

         rollout:
           model:
             backbone_model_path: ./Cosmos-Reason2-2B
         actor:
           model:
             model_path: ./GR00T-N1.7-LIBERO/libero_spatial
             backbone_model_path: ./Cosmos-Reason2-2B

      **3. 启动训练**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t_n1d7

      .. dropdown:: 进阶配置与设计说明（可选）

         - N1.7 将 VLM 主干升级为 ``nvidia/Cosmos-Reason2-2B``\ （Qwen3-VL 架构），采用相对末端执行器动作空间，并将通用上限扩展到 ``max_state_dim=132``、``max_action_dim=132``、``action_horizon=40``。
         - LIBERO 配方默认配置 ``embodiment_tag: libero_sim``、``add_value_head: True``、``num_action_chunks: 16`` 与 ``denoising_steps: 4``。``rl_head_config.detach_critic_input: true`` 让价值头使用分离梯度后的特征，设为 ``false`` 则联合回传梯度。
         - 加载 BF16 N1.7 权重时，RLinf 会在各 worker 进程关闭低精度 BF16 GEMM 归约，避免 rollout 与 actor 批量大小不同引起的输出漂移；参数和优化器存储类型保持不变。

   .. tab-item:: AMD ROCm
      :sync: rocm

      ROCm 使用 PyTorch 的 CUDA 兼容 API，因此 GR00T N1.5 可直接使用共用的 AMD accelerator 与安装路径，无需模型专用补丁。

      **1. 安装环境（LIBERO）**

      .. include:: _amd_libero.rst

      发布镜像已经包含 N1.5 环境：

      .. code-block:: bash

         source switch_env gr00t

      若在 ROCm 宿主机上直接安装（省略 ``--rocm`` 自动检测已安装的版本）：

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model gr00t --env libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir ./RLinf-Gr00t-SFT-Spatial

      .. include:: _model_path.rst

      **3. 启动 LIBERO 训练**

      在 ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml`` 中设置路径，并在已激活环境中启用软件渲染（受支持的 Radeon GPU 可改用 ``egl``）：

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t

      **在 AMD ROCm 上运行 ManiSkill**

      安装 GR00T N1.5 的 ManiSkill + LIBERO 组合环境：

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model gr00t --env maniskill_libero && source .venv/bin/activate

      配置 CPU simulation：

      .. include:: _maniskill_non_cuda.rst

      RLinf 已提供 GR00T 的 ManiSkill observation 与 action 转换路径，但没有随仓库提供 ManiSkill GR00T 任务配置或 checkpoint。请从一个 ManiSkill 任务配置开始，选择 metadata 中带有 ``maniskill_widowx`` embodiment head 的 GR00T N1.5 checkpoint，并在配置中填写以下 ``actor.model`` 字段（rollout 自动继承 ``model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_type: gr00t
             model_path: /path/to/maniskill-gr00t-checkpoint
             obs_converter_type: maniskill
             embodiment_tag: maniskill_widowx

      确认环境的 action shape 与 normalization statistics 和 checkpoint metadata 一致后，通过 ``run_embodiment.sh`` 启动训练。

   .. tab-item:: 华为昇腾 CANN
      :sync: ascend

      使用昇腾 LIBERO 容器，或在已安装 CANN 和 NPU 驱动的宿主机上运行。下方步骤面向 GR00T N1.5（在 LIBERO 上运行 N1.6 或 N1.7 时，将 ``--model gr00t`` 替换为 ``--model gr00t_n1d6`` 或 ``--model gr00t_n1d7``，并按对应版本标签页下载权重与配置）。

      **1. 安装环境（LIBERO）**

      .. include:: _ascend_libero.rst

      进入已发布的 RLinf 容器后，激活 N1.5 环境：

      .. code-block:: bash

         source switch_env gr00t

      本地安装时，明确选择 N1.5 与 LIBERO（脚本会在 aarch64 上按需从源码构建 ``decord``，应用昇腾专用的 TensorFlow 版本约束，跳过 CUDA flash-attention，并在加载模型时自动应用 NPU 补丁）：

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model gr00t --env libero && source .venv/bin/activate

      .. include:: _ascend_torch.rst

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir ./RLinf-Gr00t-SFT-Spatial

      .. include:: _model_path.rst

      **3. 启动 LIBERO 训练**

      在 ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml`` 中设置路径，并在已激活环境中启用软件渲染：

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t

      **在华为昇腾 CANN 上运行 ManiSkill**

      安装 GR00T N1.5 的 ManiSkill + LIBERO 组合环境：

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model gr00t --env maniskill_libero && source .venv/bin/activate

      配置 CPU simulation：

      .. include:: _maniskill_non_cuda.rst

      从一个 ManiSkill 任务配置开始，选择 metadata 中带有 ``maniskill_widowx`` embodiment head 的 GR00T N1.5 checkpoint，填写以下 ``actor.model`` 字段（rollout 自动继承 ``model_path``），确认 action shape 与 normalization statistics 一致后，通过 ``run_embodiment.sh`` 启动训练：

      .. code-block:: yaml

         actor:
           model:
             model_type: gr00t
             model_path: /path/to/maniskill-gr00t-checkpoint
             obs_converter_type: maniskill
             embodiment_tag: maniskill_widowx

   .. tab-item:: 摩尔线程 MUSA
      :sync: musa

      MUSA 使用共用设备与安装路径，并在 worker 检测到 MUSA accelerator 时自动为 RADIO backbone 的 CUDA capability 检查应用 N1.5 兼容补丁。厂商镜像提供可用的 MUSA flash-attention。

      **1. 安装环境（LIBERO）**

      .. include:: _musa_libero.rst

      进入容器后，安装 GR00T N1.5 与 LIBERO：

      .. code-block:: bash

         bash requirements/install.sh --platform musa embodied --model gr00t --env libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Gr00t-SFT-Spatial --local-dir ./RLinf-Gr00t-SFT-Spatial

      .. include:: _model_path.rst

      **3. 启动 LIBERO 训练**

      在 ``examples/embodiment/config/libero_spatial_ppo_gr00t.yaml`` 中设置路径，并在已激活环境中启用软件渲染：

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_gr00t

      **在摩尔线程 MUSA 上运行 ManiSkill**

      保留厂商模拟器包，并在厂商镜像中添加 GR00T N1.5 模型环境：

      .. include:: _musa_maniskill.rst

      .. code-block:: bash

         bash requirements/install.sh --platform musa embodied --venv gr00t --model gr00t --env libero && source gr00t/bin/activate

      配置 CPU simulation：

      .. include:: _maniskill_non_cuda.rst

      从一个 ManiSkill 任务配置开始，选择 metadata 中带有 ``maniskill_widowx`` embodiment head 的 GR00T N1.5 checkpoint，填写以下 ``actor.model`` 字段（rollout 自动继承 ``model_path``），确认 action shape 与 normalization statistics 一致后，通过 ``run_embodiment.sh`` 启动训练：

      .. code-block:: yaml

         actor:
           model:
             model_type: gr00t
             model_path: /path/to/maniskill-gr00t-checkpoint
             obs_converter_type: maniskill
             embodiment_tag: maniskill_widowx

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。

**LIBERO 结果**

**N1.5:**

.. list-table:: **GR00T-N1.5 模型使用 Flow-SDE 方法在 LIBERO 上的结果**
   :header-rows: 1

   * - 模型
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - GR00T（少样本）
     - |huggingface| `41.4% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial>`_
     - |huggingface| `58.6% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Object>`_
     - |huggingface| `48.2% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal>`_
     - |huggingface| `61.9% <https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long>`_
     - 52.5%
     - ---

   * - +PPO
     - |huggingface| `92.5% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Spatial-Step400>`_
     - |huggingface| `95.0% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Object-Step400>`_
     - |huggingface| `84.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Goal-Step500>`_
     - |huggingface| `86.3% <https://huggingface.co/RLinf/RLinf-Gr00t-RL-Long-Step300>`_
     - **89.5%**
     - **+37.0%**

上述结果沿用了与 π₀ 相同的超参数设置，未做任务专属调参；进一步调参可提升模型表现。

**N1.6:**

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/gr00t_1.6_ppo_success_rate.png
   :align: center
   :width: 55%

   GR00T-N1.6 SFT + PPO 在 LIBERO_Spatial 的成功率曲线。

.. list-table:: **GR00T-N1.6 使用 Flow-SDE 方法在 LIBERO Spatial 上的结果**
   :header-rows: 1

   * - 模型
     - Spatial

   * - GR00T-N1.6 SFT
     - |huggingface| `70% <https://huggingface.co/RLinf/RLinf-Gr00t-N1.6-RL-Spatial>`_

   * - +PPO
     - |huggingface| `82% <https://huggingface.co/RLinf/RLinf-Gr00t-N1.6-RL-Spatial-Step500>`_


**N1.7:**

.. list-table:: **GR00T-N1.7 使用 Flow-SDE 方法在 LIBERO Spatial 上的结果**
   :header-rows: 1

   * - 模型
     - Spatial

   * - GR00T-N1.7 PPO
     - —
