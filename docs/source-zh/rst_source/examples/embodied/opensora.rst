OpenSora 强化学习
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/hpcaitech/Open-Sora-Demo/main/readme/icon.png
   :align: center
   :width: 30%

   作为动作条件世界模型的 OpenSora。

使用 **动作条件 OpenSora 世界模型** 作为环境后端，**无需真实机器人或物理仿真器** 即可闭环训练 VLA policy。OpenSora 根据当前观测与动作序列生成未来视频帧，因此可以在“想象”的 rollout 上用强化学习（GRPO/PPO）优化 policy。

概览
----------------------------------------

在 OpenSora 世界模型模拟的 LIBERO 套件上用 GRPO 训练 OpenVLA-OFT。

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

      Spatial · Object

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装 → 下载 VLA 模型 → 下载 OpenSora 世界模型权重与初始化数据 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 OpenVLA-OFT SFT checkpoint · OpenSora 世界模型权重与初始化数据集（见下文）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

作为世界模型，OpenSora 原则上可以通过一致接口适配任意任务。RLinf 提供两个 LIBERO 套件的权重和初始化数据：

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - OpenSora
     - LIBERO-Spatial
     - ``RLinf/RLinf-OpenSora-LIBERO-Spatial``
     - 使用 OpenSora 作为 LIBERO spatial 任务的学习型仿真器。
   * - OpenSora
     - LIBERO-Object
     - ``RLinf/RLinf-OpenSora-LIBERO-Object``
     - 在视频世界模型中 rollout 物体操作动力学。

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

与传统仿真器不同，OpenSora 没有 ``reset()``：它需要初始化帧和任务描述，因此需要下载初始化数据集并在配置中指向它。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO-Spatial
      :sync: spatial

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env opensora && source .venv/bin/activate

      **2. 下载资源与模型**

      一并下载 OpenVLA-OFT SFT checkpoint 与 OpenSora 世界模型权重和初始化数据集：

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 --local-dir ./Openvla-oft-SFT-libero-spatial-traj1
         hf download RLinf/RLinf-OpenSora-LIBERO-Spatial --local-dir ./RLinf-OpenSora-LIBERO-Spatial

      **3. 启动训练**

      在 ``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml`` 中填写上述两个下载路径（rollout 会自动沿用 ``actor.model.model_path``）：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Openvla-oft-SFT-libero-spatial-traj1
         env:
           train:
             opensora_wm_hf_ckpt_path: ./RLinf-OpenSora-LIBERO-Spatial

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft

   .. tab-item:: LIBERO-Object
      :sync: object

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env opensora && source .venv/bin/activate

      **2. 下载资源与模型**

      一并下载 OpenVLA-OFT SFT checkpoint 与 OpenSora 世界模型权重和初始化数据集：

      .. code-block:: bash

         hf download Haozhan72/Openvla-oft-SFT-libero-object-traj1 --local-dir ./Openvla-oft-SFT-libero-object-traj1
         hf download RLinf/RLinf-OpenSora-LIBERO-Object --local-dir ./RLinf-OpenSora-LIBERO-Object

      **3. 启动训练**

      在 ``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml`` 中将 ``actor.model.model_path`` 与 ``env.train.opensora_wm_hf_ckpt_path`` 指向 LIBERO-Object 检查点，并同步更新 ``actor.model.unnorm_key``：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Openvla-oft-SFT-libero-object-traj1
             unnorm_key: libero_object_no_noops
         env:
           train:
             opensora_wm_hf_ckpt_path: ./RLinf-OpenSora-LIBERO-Object

      使用 ``run_embodiment.sh`` 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh opensora_libero_spatial_grpo_openvlaoft

.. dropdown:: 其他 LIBERO VLA SFT 检查点（Goal / Long）

   如果在其他套件上使用自定义世界模型，对应的 OpenVLA-OFT SFT 仓库为 ``Haozhan72/Openvla-oft-SFT-libero-goal-traj1``\ （LIBERO-Goal）和 ``Haozhan72/Openvla-oft-SFT-libero10-traj1``\ （LIBERO-Long）。

.. dropdown:: 进阶配置（可选）

   由于世界模型不提供本体感知和腕部视角且动作 chunk 长度固定，``actor.model`` 默认设置 ``use_proprio: False``、``num_images_in_input: 1`` 和 ``num_action_chunks: 8``。切换到其他 LIBERO 套件时，同步更新 ``actor.model.unnorm_key`` 以匹配对应的 SFT checkpoint。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

我们评估所有 ``task_id`` × ``trial_id`` 组合——每个套件 500 个环境（10 个任务 × 50 个试次）。SFT（LoRA-base）模型设置 ``do_sample = False``；RL 训练模型在 ``rollout.sampling_params`` 中设置 ``do_sample = True``、``temperature_train = 1.6``，并启用 ``env.train.rollout_epoch=2``。

.. note::

    选择 OpenSora 作为世界模型模拟器的动机来源于 `WMPO <https://arxiv.org/abs/2511.09515>`_；训练世界模型时我们也参考了 `OpenSora <https://github.com/RLinf/opensora>`_。

.. list-table:: **使用 OpenSora 模拟器的 LIBERO 任务组评测结果**
    :header-rows: 1
    :widths: 50 25 25

    * - 模型
      - Spatial
      - Object
    * - OpenVLA-OFT (LoRA-base)
      - 61.2%
      - 36.7%
    * - OpenVLA-OFT（OpenSora 作为世界模型的 RLinf-GRPO）
      - 75.5%
      - 64.5%
    * - **效果提升**
      - **+14.3%**
      - **+27.8%**
