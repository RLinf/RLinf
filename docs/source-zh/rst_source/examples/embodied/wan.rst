Wan 强化学习
============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   作为动作条件世界模型的 Wan。

使用 **动作条件 Wan 世界模型** 作为环境后端，**无需真实机器人或物理仿真器** 即可闭环训练 VLA policy。Wan 根据当前观测与动作序列生成未来视频帧，因此可以在“想象”的 rollout 上用强化学习（GRPO/PPO）优化 policy。

概览
----------------------------------------

在 Wan 世界模型模拟的 LIBERO 套件上用 GRPO 训练 OpenVLA-OFT。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Spatial · Object · Goal

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · :ref:`华为昇腾 CANN <wan-hardware>`\ （Wan backend）

| **你将完成：** 安装 → 下载 VLA 模型 → 下载 Wan 世界模型权重与初始化数据 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 OpenVLA-OFT SFT checkpoint · Wan 世界模型权重与初始化数据集（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作为世界模型，Wan 原则上可以通过一致接口适配多种任务设置。RLinf 提供三个 LIBERO 套件的权重和初始化数据（每个 Wan checkpoint 均基于 1,500 条 VLA rollout 轨迹训练）：

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - Wan
     - LIBERO-Spatial
     - ``RLinf/RLinf-Wan-LIBERO-Spatial``
     - 使用 Wan 作为 LIBERO spatial 任务的学习型仿真器。
   * - Wan
     - LIBERO-Object
     - ``RLinf/RLinf-Wan-LIBERO-Object``
     - 在视频世界模型中 rollout 物体操作动力学。
   * - Wan
     - LIBERO-Goal
     - ``RLinf/RLinf-Wan-LIBERO-Goal``
     - 通过 Wan 评测目标条件 rollout。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 由初始化帧启动、世界模型生成的 RGB 帧，形状为 ``[B, 256, 256, 3]``。
   * - Action
     - 归一化并 tokenized 后用于条件生成的 7 维连续动作。
   * - Reward
     - 世界模型 reward classifier 输出，范围为 ``[0, 1]``。
   * - Prompt
     - 用于条件化视频世界模型的自然语言任务描述。

与传统仿真器不同，Wan 没有 ``reset()``：它需要初始化帧和任务描述，因此需要下载初始化数据集并在配置中指向它。

.. _wan-hardware:

安装与运行
----------

.. include:: _setup_common.rst

NVIDIA CUDA 按任务套件选择对应标签页；昇腾 NPU 请切换到 **华为昇腾 CANN** 标签页。

.. tab-set::

   .. tab-item:: LIBERO-Spatial
      :sync: spatial

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env wan && source .venv/bin/activate

      **2. 下载资源与模型**

      一并下载 OpenVLA-OFT SFT checkpoint 与 Wan 世界模型权重和初始化数据集：

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 --local-dir ./Openvla-oft-SFT-libero-spatial-traj1
         hf download RLinf/RLinf-Wan-LIBERO-Spatial --local-dir ./RLinf-Wan-LIBERO-Spatial

      **3. 启动训练**

      在 ``examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`` 中填写上述两个下载路径（rollout 会自动沿用 ``actor.model.model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Openvla-oft-SFT-libero-spatial-traj1
         env:
           train:
             wan_wm_hf_ckpt_path: ./RLinf-Wan-LIBERO-Spatial

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh wan_libero_spatial_grpo_openvlaoft

   .. tab-item:: LIBERO-Object
      :sync: object

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env wan && source .venv/bin/activate

      **2. 下载资源与模型**

      一并下载 OpenVLA-OFT SFT checkpoint 与 Wan 世界模型权重和初始化数据集：

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-object-traj1 --local-dir ./Openvla-oft-SFT-libero-object-traj1
         hf download RLinf/RLinf-Wan-LIBERO-Object --local-dir ./RLinf-Wan-LIBERO-Object

      **3. 启动训练**

      在 ``examples/embodiment/config/wan_libero_object_grpo_openvlaoft.yaml`` 中填写上述两个下载路径（rollout 会自动沿用 ``actor.model.model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Openvla-oft-SFT-libero-object-traj1
         env:
           train:
             wan_wm_hf_ckpt_path: ./RLinf-Wan-LIBERO-Object

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh wan_libero_object_grpo_openvlaoft

   .. tab-item:: LIBERO-Goal
      :sync: goal

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env wan && source .venv/bin/activate

      **2. 下载资源与模型**

      一并下载 OpenVLA-OFT SFT checkpoint 与 Wan 世界模型权重和初始化数据集：

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir ./Openvla-oft-SFT-libero-goal-traj1
         hf download RLinf/RLinf-Wan-LIBERO-Goal --local-dir ./RLinf-Wan-LIBERO-Goal

      **3. 启动训练**

      在 ``examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml`` 中填写上述两个下载路径（rollout 会自动沿用 ``actor.model.model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Openvla-oft-SFT-libero-goal-traj1
         env:
           train:
             wan_wm_hf_ckpt_path: ./RLinf-Wan-LIBERO-Goal

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh wan_libero_goal_grpo_openvlaoft

   .. tab-item:: 华为昇腾 CANN
      :sync: ascend

      使用已配置 CANN 和 NPU 驱动的昇腾宿主机或容器。Wan 直接通过世界模型生成观测，该训练流程无需 LIBERO 仿真器，也无需配置 OSMesa 渲染。

      **1. 安装环境**

      创建包含 OpenVLA-OFT policy 与 Wan 世界模型依赖的环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh --platform ascend embodied --model openvla-oft --env wan && source .venv/bin/activate

      安装器会添加匹配的 ``torch-npu`` 并跳过 CUDA flash-attention，但不会安装 MindIE-SD。请在已激活的环境中单独安装与 CANN、PyTorch 版本兼容的 MindIE-SD。

      检查 NPU 是否可用，以及所需的 MindIE-SD 算子能否导入：

      .. code-block:: bash

         python - <<'PY'
         import torch
         import torch_npu
         from mindiesd import rotary_position_embedding
         from mindiesd.layers.flash_attn.attention_forward import attention_forward

         assert torch.npu.is_available(), "No Ascend NPU is available"
         PY

      Wan backend 在 NPU 上运行时，RLinf 会在构建 pipeline 前自动启用加速 attention、旋转位置编码和 RMSNorm。如果依赖无法导入，则记录警告并保留 diffsynth 原有算子。这里配置的是 Wan backend，policy checkpoint 在目标硬件上的数值表现仍需单独验证。

      **2. 下载资源与模型**

      下载对应套件的 OpenVLA-OFT SFT checkpoint 与 Wan 世界模型权重（以 LIBERO-Spatial 为例）：

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 --local-dir ./Openvla-oft-SFT-libero-spatial-traj1
         hf download RLinf/RLinf-Wan-LIBERO-Spatial --local-dir ./RLinf-Wan-LIBERO-Spatial

      **3. 启动训练**

      在 ``examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`` 中设置 policy 与世界模型路径，并保留配置默认的 ``runner.val_check_interval: -1`` 和 ``runner.only_eval: False``，即可使用 Wan 训练而不启用仿真器评估：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Openvla-oft-SFT-libero-spatial-traj1
         env:
           train:
             wan_wm_hf_ckpt_path: ./RLinf-Wan-LIBERO-Spatial

      在仓库根目录启动 Wan LIBERO-Spatial GRPO 配方：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh wan_libero_spatial_grpo_openvlaoft

.. dropdown:: 其他 LIBERO VLA SFT 检查点（Long）

   如果在 LIBERO-Long 上使用自定义世界模型，对应的 OpenVLA-OFT SFT 仓库为 ``Haozhan72/Openvla-oft-SFT-libero10-traj1``。

.. dropdown:: 进阶配置（可选）

   由于世界模型不提供本体感知和腕部视角且动作 chunk 长度固定，``actor.model`` 默认设置 ``use_proprio: False``、``num_images_in_input: 1`` 和 ``num_action_chunks: 8``。切换到其他 LIBERO 套件时，同步更新 ``actor.model.unnorm_key`` 以匹配对应的 SFT checkpoint。

   ``env.train`` 下的可选环境配置项：

   - ``enable_kir``：是否启用 KIR（KeyFrame-Init Rollout）。设为 ``False`` 时，重置只采样文件名不含 ``_kir`` 的 ``.npy``；设为 ``True`` 时，从 ``dataset/`` 中全部初始化文件采样。
   - ``num_inference_steps``：世界模型去噪推理步数（默认 ``5``）。步数越少速度越快，但画质会下降；单步生成同样可以提升 policy 表现。
   - ``reward_model.type``：奖励模型类型，可选 ``ResnetRewModel`` 或 ``TaskEmbedResnetRewModel``。
   - ``terminate_at_max_episode_steps``：episode 达到 ``max_episode_steps`` 时的结束方式。默认 ``False`` 记为截断（truncation），开启 ``auto_reset: True`` 后 RLinf 会将末尾观测的 value 自举到最后一步奖励上。如果 ``max_episode_steps`` 就是任务本身的时限（例如只在结束时给一次 episode 级成功奖励），设为 ``True`` 以按终止（termination）结束，在默认的 ``algorithm.bootstrap_type: standard`` 下不做 value 自举。如果时限只是截断了更长的任务（例如从 KIR 关键帧起步的短 rollout），保持 ``False``。设为 ``True`` 需要同时开启 ``auto_reset: True``。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

我们评估 Object、Spatial、Goal 套件中所有 ``task_id`` × ``trial_id`` 组合——共 1500 个环境（10 个任务 × 150 个试次）。SFT 与 RL 训练模型均在 ``rollout.sampling_params`` 中设置 ``do_sample = True``、``temperature_train = 1.6``，并设置 ``reset_gripper_open = True``。

.. note::

    Wan 的训练与推理构建在 `Diffsynth-Studio <https://github.com/RLinf/diffsynth-studio>`_ 之上。在下面的评测结果中，我们仅使用 **冻结** 的世界模型服务于 VLA 模型的强化学习训练，未使用世界模型与 VLA 的协同进化。用户可手动实现协同进化以获得进一步性能提升。

.. list-table:: **使用 Wan 模拟器的 LIBERO 任务组评测结果**
    :header-rows: 1
    :widths: 40 20 20 20

    * - 模型
      - Spatial
      - Object
      - Goal
    * - OpenVLA-OFT (LoRA-base)
      - 61.2%
      - 36.7%
      - 48.2%
    * - OpenVLA-OFT（Wan 作为世界模型的 RLinf-GRPO）
      - 77.5%
      - 77.9%
      - 60.1%
    * - **效果提升**
      - **+16.3%**
      - **+41.2%**
      - **+11.9%**
