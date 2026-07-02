基于 ManiSkill 的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/mani-skill/ManiSkill/main/figures/teaser.jpg
   :align: center
   :width: 90%

   ManiSkill 中渲染的环境（图片来源：`ManiSkill <https://github.com/haosulab/ManiSkill>`__）。

`ManiSkill <https://maniskill.readthedocs.io>`__ 是一个 GPU 并行化的机器人操作模拟器与基准。
一台 7 自由度机械臂完成语言条件下的桌面操作任务；RLinf 借助 ManiSkill3 对视觉-语言-动作（VLA）
策略进行强化学习微调，达到业界领先的成功率，并在分布外（OOD）变体上同样表现优异。

概览
----------------------------------------

在 ManiSkill3 上对 VLA 进行强化学习微调；OpenVLA 与 OpenVLA-OFT 在 plate-25 上成功率均超过 90%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA · OpenVLA-OFT · π₀ / π₀.₅ · MLP · ResNet

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO · SAC · CrossQ · DAgger

   .. grid-item-card:: 任务
      :text-align: center

      桌面操作（plate-25 + OOD）

   .. grid-item-card:: 硬件
      :text-align: center

      1–2 节点 · 8–16 张 GPU

| **你将完成：** 安装依赖 → 下载资产与基座模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · ManiSkill 资产与基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

参考方案在 ``PutOnPlateInScene25Main-v3`` (plate-25) 任务上训练，并在分布内（IND）以及分布外
（OOD）设置上评测：

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

   若要在 ManiSkill 上运行 **OpenPI**（π\ :sub:`0`\  / π\ :sub:`0.5`\ ），请参阅 :doc:`在 π₀ 与 π₀.₅ 模型上进行强化学习 <pi0>`。

安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.3-maniskill_libero``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
      # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

   # 进入容器后，切换到模型对应的虚拟环境：
   source switch_env openvla        # 或：source switch_env openvla-oft

**选项 2：自定义环境** —— 安装包 ``--env maniskill_libero``：

.. code:: bash

   # 国内用户可添加 --use-mirror 以提升下载速度。
   # OpenVLA-OFT 实验请使用 --model openvla-oft。
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

下载资产
----------------------------------------

下载 ManiSkill 资产：

.. code:: bash

   # 国内可设置 HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./maniskill_assets

.. important::

   资产 **必须** 放置在 ``rlinf/envs/maniskill/assets`` 目录下——环境会从该路径加载资产。请将其复制到环境包目录：

.. code:: bash

   cp -r ./maniskill_assets <path_to_RLinf>/rlinf/envs/maniskill/assets

下载模型
----------------------------------------

下载预训练基座检查点（任选一种方式）：

.. code:: bash

   # 方式 1：git clone
   git lfs install
   git clone https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup

   # 方式 2：huggingface-hub（国内可设置 HF_ENDPOINT=https://hf-mirror.com）
   pip install huggingface-hub
   hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

.. include:: _model_path.rst

运行
----------------------------------------

每个方案对应 ``examples/embodiment/config/`` 下的一个 YAML 配置：

- **OpenVLA + PPO** —— ``maniskill_ppo_openvla.yaml``
- **OpenVLA-OFT + PPO** —— ``maniskill_ppo_openvlaoft.yaml``
- **OpenVLA + GRPO** —— ``maniskill_grpo_openvla.yaml``
- **OpenVLA-OFT + GRPO** —— ``maniskill_grpo_openvlaoft.yaml``

使用 ``run_embodiment.sh`` 启动某个配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla

**本命令做了什么：**

1. 加载 ``examples/embodiment/config/maniskill_ppo_openvla.yaml`` 配置。
2. 连接（或启动）Ray，并按 ``cluster.component_placement`` 放置 actor、rollout、env 各 worker。
3. 运行 PPO 训练循环，并将日志与检查点写入 ``runner.logger.log_path``。

.. admonition:: 进一步配置
   :class: note

   - 放置与吞吐 → :doc:`放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 全部配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义与日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`断点续训 <../../guides/resume>`
   - 卡住或显存不足（OOM）？ → :doc:`FAQ <../../resources/faq>`

可视化与结果
----------------------------------------

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

ManiSkill3 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在单台 8 卡 H100 机器上，OpenVLA（左）与 OpenVLA-OFT（右）在 ManiSkill3 的 plate-25-main
任务上成功率均超过 90%。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvla.png" style="width: 100%;"/>
       <p><em>OpenVLA</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/rlinf-vla/mani_openvlaoft.png" style="width: 100%;"/>
       <p><em>OpenVLA-OFT</em></p>
     </div>
   </div>

我们在分布内（IND）和 OOD 场景（视觉、语义、执行）上评测。每列最优结果以粗体标注。

.. note::

   这里采用与 `rl4vla <https://arxiv.org/abs/2505.19789>`_ 相同的 OOD 测试集以便公平比较。基座模型：
   OpenVLA 采用预训练的 `openvla-7b-rlvla-warmup <https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup>`_；
   OpenVLA-OFT 使用我们在 ``PutOnPlateInScene25Main-v3`` 数据上自行 LoRA 微调的权重
   （`OpenVLA-OFT (Base) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_）。

.. list-table:: **OpenVLA 与 OpenVLA-OFT 在 ManiSkill3 上的结果**
   :header-rows: 1
   :widths: 40 15 15 15 15 15

   * - 模型
     - 训练设置(IND)
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

   ``rl4vla`` 是在小批量下的 PPO + OpenVLA，因此只应与在相近条件下训练的 PPO+OpenVLA 比较。
   我们的 PPO+OpenVLA 依托 RLinf 的大规模基础设施以更大批量训练，显著提升了性能。

下面的动画展示了在 RLinf 中用 PPO 在 ManiSkill3 多任务基准上训练 OpenVLA 的结果。

.. raw:: html

   <video controls autoplay loop muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/embody.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
