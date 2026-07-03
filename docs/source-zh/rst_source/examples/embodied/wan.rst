基于 Wan 世界模型的强化学习
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
   :align: center
   :width: 45%

   作为动作条件世界模型的 Wan。

使用 **动作条件 Wan 世界模型** 作为环境后端，**无需真实机器人或物理仿真器** 即可闭环训练
VLA 策略。Wan 根据当前观测与动作序列生成未来视频帧，因此可以在“想象”的 rollout 上用
强化学习（GRPO/PPO）优化策略。

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

      1 节点 · GPU

| **你将完成：** 安装 → 下载 VLA 模型 → 下载 Wan 世界模型权重与初始化数据 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 OpenVLA-OFT SFT checkpoint · Wan 世界模型权重与初始化数据集（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作为世界模型，Wan 原则上可以通过一致接口适配多种任务设置。RLinf 目前提供三个 LIBERO 套件的权重和初始化数据：

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

安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.3-wan``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-wan
      # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-wan

   # 进入容器后，切换到 OpenVLA-OFT 虚拟环境：
   source switch_env openvla-oft

**选项 2：自定义环境** —— 安装套件 ``--env wan``：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 --use-mirror。
   bash requirements/install.sh embodied --model openvla-oft --env wan
   source .venv/bin/activate

VLA 模型下载
----------------------------------------

下载 OpenVLA-OFT SFT checkpoint：

.. code:: bash

   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-spatial-traj1
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-object-traj1
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-goal-traj1
   git clone https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-traj1

   # 方法 2：使用 huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 --local-dir Openvla-oft-SFT-libero-spatial-traj1
   hf download Haozhan72/Openvla-oft-SFT-libero-object-traj1 --local-dir Openvla-oft-SFT-libero-object-traj1
   hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 --local-dir Openvla-oft-SFT-libero-goal-traj1
   hf download Haozhan72/Openvla-oft-SFT-libero10-traj1 --local-dir Openvla-oft-SFT-libero10-traj1

下载完成后，在配置中设置 ``model_path`` 与 ``unnorm_key``：

.. code:: yaml

   rollout:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
   actor:
      model:
         model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
         unnorm_key: libero_90_no_noops_trajall # 对于 RLinf-OpenVLAOFT-LIBERO-130-Base-Lora 模型，使用 libero_130_no_noops_trajall

世界模型下载
----------------------------------------

除 VLA 模型外，还需下载 Wan 权重与初始化数据。当前 RLinf 提供三个套件的数据/权重；每个 Wan
权重均基于 VLA 模型 rollout 的 1500 条轨迹构建：

.. code:: bash

   # 方法 1：使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial
   git clone https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Object
   git clone https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal

   # 方法 2：使用 huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download RLinf/RLinf-Wan-LIBERO-Spatial --local-dir RLinf-Wan-LIBERO-Spatial
   hf download RLinf/RLinf-Wan-LIBERO-Object --local-dir RLinf-Wan-LIBERO-Object
   hf download RLinf/RLinf-Wan-LIBERO-Goal --local-dir RLinf-Wan-LIBERO-Goal

``RLinf-Wan-LIBERO-Spatial`` 的目录结构如下：

.. code-block:: text

    RLinf-Wan-LIBERO-Spatial/
        ├── dataset/                            # 仿真初始化数据集
        │   ├── traj0.npy                       # 仅含初始帧的轨迹
        │   ├── traj1.npy
        │   ├── ...
        │   └── trajN.npy
        │   ├── traj0_kir.npy                   # 含关键帧前置上下文的轨迹
        │   ├── traj1_kir.npy
        │   ├── ...
        │   └── trajN_kir.npy
        ├── model-00001.safetensors             # 世界模型权重
        ├── resnet_rm.pth                       # 奖励模型权重
        └── Wan2.2_VAE.pth                      # VAE 权重

下载完成后，在配置中设置世界模型路径：

.. code:: yaml

    env:
        train:
            wan_wm_hf_ckpt_path: /Pathto/model/RLinf-Wan-LIBERO-Spatial/

运行
----------------------------------------

**1. 模型参数**

以 OpenVLA-OFT 为例，配置 ``actor.model``：

.. code-block:: yaml

   actor:
     model:
       model_path: "/path/to/model/Openvla-oft-SFT-libero-spatial-traj1/"    # SFT 模型路径
       model_type: "openvla_oft"                                             # 模型类型
       use_proprio: False                                                    # 是否使用本体感觉信息
       num_images_in_input: 1                                                # 输入图像数量
       num_action_chunks: 8                                                  # 动作块数量
       unnorm_key: "libero_spatial_no_noops"                                 # 动作归一化键（与 SFT 一致）

由于世界模型不提供本体信息、不生成腕部视角且 chunk 固定，``use_proprio`` 默认为 ``False``，
``num_images_in_input`` 默认为 ``1``，``num_action_chunks`` 默认为 ``8``。

**2. 环境配置**

.. code-block:: yaml

   # 推荐训练使用 wan_libero_spatial，评估使用 libero_spatial
   env/train: wan_libero_spatial
   env/eval: libero_spatial

   # 在 env/train/wan_libero_spatial.yaml 中：
   wm_env_type: libero
   task_suite_name: libero_spatial
   reset_gripper_open: True
   # 是否启用 KeyFrame-Init Rollout
   enable_kir: True
   # 世界模型去噪推理步数
   num_inference_steps: 5
   # 世界模型重置用的初始化数据集路径
   initial_image_path: /Pathto/model/RLinf-Wan-LIBERO-Spatial/dataset
   # VAE 权重
   VAE_path: /Pathto/model/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth
   # 预训练世界模型权重
   model_path: /Pathto/model/RLinf-Wan-LIBERO-Spatial/model-00001.safetensors
   # 奖励模型
   reward_model:
     type: ResnetRewModel
     from_pretrained: /Pathto/model/RLinf-Wan-LIBERO-Spatial/resnet_rm.pth

环境配置关键参数：

- ``enable_kir``：是否启用 KIR（KeyFrame-Init Rollout）。关闭时，重置仅采样文件名不含 ``_kir`` 的 ``.npy``；启用时，从 ``dataset/`` 中所有初始化文件采样。
- ``num_inference_steps``：世界模型生成/推理步数（默认 ``5``）。步数越少生成越快，但可能降低画质；即使单步生成也能带来性能提升。
- ``reward_model.type``：奖励模型类别——``ResnetRewModel`` 或 ``TaskEmbedResnetRewModel``。
- ``reset_gripper_open``：是否以张开夹爪初始化。训练与评估默认 ``True``，不建议修改。

**3. 启动**

OpenVLA-OFT + GRPO 使用 ``examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml``：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh wan_libero_spatial_grpo_openvlaoft

可视化与结果
----------------------------------------

关注未归一化的回合成功率指标 ``env/success_once``。各项指标的含义见
:doc:`训练指标 <../../reference/metrics>`。可通过以下配置保存生成的 rollout 视频：

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval

我们评估 Object、Spatial、Goal 套件中所有 ``task_id`` × ``trial_id`` 组合——共 1500 个环境
（10 个任务 × 150 个试次）。SFT 与 RL 训练模型均在 ``rollout.sampling_params`` 中设置
``do_sample = True``、``temperature_train = 1.6``，并设置 ``reset_gripper_open = True``。

.. note::

    我们基于 `Diffsynth-Studio <https://github.com/RLinf/diffsynth-studio>`_ 进行 Wan 的训练与推理。
    在下面的评测结果中，我们仅使用 **冻结** 的世界模型服务于 VLA 模型的强化学习训练，并未使用世界模型与
    VLA 的协同进化。用户可手动实现协同进化以获得进一步性能提升。

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
