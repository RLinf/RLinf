ManiSkill 强化学习
==================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/figures/teaser.jpg
   :align: center
   :width: 90%

   ManiSkill 中渲染的环境（图片来源：`ManiSkill <https://github.com/haosulab/ManiSkill>`__）。

`ManiSkill <https://maniskill.readthedocs.io>`__ 是一个 GPU 并行化的机器人操作模拟器与基准。一台 7 自由度机械臂完成语言条件下的桌面操作任务；RLinf 借助 ManiSkill3 用强化学习微调视觉-语言-动作（VLA）policy，取得高成功率，并在分布外（OOD）变体上保持良好泛化。

概览
----------------------------------------

在 ManiSkill3 上用强化学习微调 VLA；OpenVLA 与 OpenVLA-OFT 在 plate-25 上成功率均超过 90%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA · OpenVLA-OFT · GR00T N1.5 · π₀ / π₀.₅ · MLP · ResNet

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO · SAC · CrossQ · DAgger

   .. grid-item-card:: 任务
      :text-align: center

      桌面操作（plate-25 + OOD）

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA CUDA · AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA（取决于模型）

| **你将完成：** 安装依赖 → 下载资产与基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · ManiSkill 资产与基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

参考方案在 ``PutOnPlateInScene25Main-v3``\ （plate-25）任务上训练，并在分布内（IND）与分布外（OOD）设置上评测：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 设置
     - 考察内容
   * - 训练 (IND)
     - plate-25 训练任务。
   * - 视觉 (OOD)
     - 场景的视觉变化。
   * - 语义 (OOD)
     - 语义变化（物体、指令）。
   * - 执行 (OOD)
     - 执行阶段的变化。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 第三人称相机的 RGB 图像（224×224）；语言任务描述。
   * - 动作 (Action)
     - 7 维连续动作：3D 末端执行器位置、3D 旋转与 1 维夹爪开合。
   * - 奖励 (Reward)
     - 基于任务进展与成功的 step 级奖励。
   * - 任务提示
     - ``In: What action should the robot take to [task_description]? Out:``

下面的流程以 **OpenVLA / OpenVLA-OFT** + **PPO/GRPO** 为例；切换配置即可使用其他受支持的模型。

.. seealso::

   各模型在 ManiSkill 上的后端配置见 :ref:`OpenVLA-OFT <openvla-oft-hardware>`、:ref:`GR00T N1.5 <gr00t-hardware>` 与 :ref:`OpenPI π₀ / π₀.₅ <pi0-hardware>`。AMD、昇腾和 MUSA 使用 CPU PhysX simulation，模型仍在所选 accelerator 上运行。GR00T 需要带有 ``maniskill_widowx`` embodiment head 的兼容 checkpoint；MUSA 需要厂商模拟器包。

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenVLA
      :sync: openvla

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

      **2. 下载资源与模型**

      在仓库根目录下载 ManiSkill 场景资产与基座检查点：

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download gen-robot/openvla-7b-rlvla-warmup --local-dir ./openvla-7b-rlvla-warmup

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - 算法
           - 配置
         * - PPO
           - ``maniskill_ppo_openvla.yaml``
         * - GRPO
           - ``maniskill_grpo_openvla.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 启动训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla \
           actor.model.model_path=./openvla-7b-rlvla-warmup

      .. dropdown:: 进阶配置（可选）

         如果 PPO rollout 在终止状态 bootstrap 上逐个切片推理，或为 OpenVLA 解码未使用的动作 token，可在配置中开启批量与仅价值 bootstrap：

         .. code-block:: yaml

            rollout:
              batch_bootstrap_inference: True
              value_only_bootstrap: True

         ``rollout.batch_bootstrap_inference`` 将待处理的终止观测合并为单个 rollout 批次推理，再按原有路由拆回结果；``rollout.value_only_bootstrap`` 在 policy 带有价值头（如 OpenVLA PPO）时直接从提示隐藏状态估计价值，跳过动作 token 的自回归生成。两项默认均为 ``False``。

      **4. 训练曲线**

      在单台 8 卡 H100 机器上，OpenVLA 在 ManiSkill3 的 plate-25-main 任务上成功率超过 90%：

      .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvla.png
         :align: center
         :width: 80%

         OpenVLA 在 ManiSkill3 上的训练曲线。

   .. tab-item:: OpenVLA-OFT
      :sync: openvla-oft

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. 下载资源与模型**

      在仓库根目录下载 ManiSkill 场景资产、基座检查点与 LoRA 权重：

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Main
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      **3. 启动训练**

      ``examples/embodiment/config/`` 下的可用配置：

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - 算法
           - 配置
         * - PPO
           - ``maniskill_ppo_openvlaoft.yaml``
         * - GRPO
           - ``maniskill_grpo_openvlaoft.yaml``

      通过命令行覆盖 ``actor.model.model_path`` 与 ``actor.model.lora_path`` 启动训练（rollout 会自动沿用 ``model_path``）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft \
           actor.model.model_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Main \
           actor.model.lora_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      .. dropdown:: 进阶配置（可选）

         如果 PPO rollout 在终止状态 bootstrap 上逐个切片推理，可在配置中开启批量 bootstrap：

         .. code-block:: yaml

            rollout:
              batch_bootstrap_inference: True

         ``rollout.batch_bootstrap_inference`` 将待处理的终止观测合并为单个 rollout 批次推理，再按原有路由拆回结果（默认 ``False``）。

      **4. 训练曲线**

      在单台 8 卡 H100 机器上，OpenVLA-OFT 在 ManiSkill3 的 plate-25-main 任务上成功率超过 90%：

      .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvlaoft.png
         :align: center
         :width: 80%

         OpenVLA-OFT 在 ManiSkill3 上的训练曲线。

.. admonition:: 进一步配置
   :class: note

   - 放置与吞吐 → :doc:`放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 全部配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义与日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`断点续训 <../../guides/resume>`
   - 卡住或显存不足（OOM）？ → :doc:`FAQ <../../resources/faq>`

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs`` 关注 ``env/success_once``；如需保存评测视频，设置 ``env.eval.video_cfg.save_video=True``，更多日志后端与指标见 :doc:`训练指标 <../../reference/metrics>`。

ManiSkill3 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们在分布内（IND）和 OOD 场景（视觉、语义、执行）上评测。每列最优结果以粗体标注。

.. note::

   这里采用与 `rl4vla <https://arxiv.org/abs/2505.19789>`_ 相同的 OOD 测试集以便公平比较。基座模型：OpenVLA 采用预训练的 `openvla-7b-rlvla-warmup <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_；OpenVLA-OFT 使用我们在 ``PutOnPlateInScene25Main-v3`` 数据上自行 LoRA 微调的权重（`OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_）。

.. list-table:: **OpenVLA 与 OpenVLA-OFT 在 ManiSkill3 上的结果**
   :header-rows: 1
   :widths: 40 15 15 15 15 15

   * - 模型
     - 训练设置 (IND)
     - 视觉 (OOD)
     - 语义 (OOD)
     - 执行 (OOD)
     - OOD 平均
   * - |huggingface| `OpenVLA (Base) <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_
     - 53.91%
     - 38.75%
     - 35.75%
     - 42.11%
     - 39.10%
   * - |huggingface| `RL4VLA (PPO) <https://huggingface.co/gen-robot/openvla-7b-rlvla-rl>`_
     - 93.75%
     - 80.47%
     - 75.00%
     - 81.77%
     - 79.15%
   * - |huggingface| `PPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood>`_
     - 96.09%
     - 82.03%
     - **78.35%**
     - **85.42%**
     - **81.93%**
   * - |huggingface| `GRPO-OpenVLA <https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood>`_
     - 84.38%
     - 74.69%
     - 72.99%
     - 77.86%
     - 75.15%
   * - |huggingface| `OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_
     - 28.13%
     - 27.73%
     - 12.95%
     - 11.72%
     - 18.29%
   * - |huggingface| `PPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood>`_
     - **97.66%**
     - **92.11%**
     - 64.84%
     - 73.57%
     - 77.05%
   * - |huggingface| `GRPO-OpenVLA-OFT <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood>`_
     - 94.14%
     - 84.69%
     - 45.54%
     - 44.66%
     - 60.64%

.. note::

   ``rl4vla`` 是在小批量下的 PPO + OpenVLA，因此只应与在相近条件下训练的 PPO+OpenVLA 比较。我们的 PPO+OpenVLA 依托 RLinf 的大规模基础设施以更大批量训练，显著提升了性能。

下面的动画展示了在 RLinf 中用 PPO 在 ManiSkill3 多任务基准上训练 OpenVLA 的结果。

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embody.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
