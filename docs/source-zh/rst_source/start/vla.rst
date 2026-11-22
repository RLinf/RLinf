训练快速开始
============

在 `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ 上几分钟完成 **单卡 MLP** 零下载验证，或直接启动 **OpenVLA** / **OpenVLA-OFT** 的 PPO 训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 1. 安装环境
      :text-align: center

      ``install.sh`` 或 Docker Hub

   .. grid-item-card:: 2. 选择起点
      :text-align: center

      单卡 MLP 或 7B VLA

   .. grid-item-card:: 3. 启动训练
      :text-align: center

      一行命令直接跑通

   .. grid-item-card:: 4. 查看曲线
      :text-align: center

      TensorBoard 监控指标

一键上手
--------

在下方选一个起点，按顺序执行命令即可。需要国内镜像加速、共存多套虚拟环境或使用 Docker 镜像，见 :doc:`installation`。

.. tab-set::

   .. tab-item:: MLP（单卡 · 零权重下载）

      在 GPU 0 上基于 ManiSkill3 内置 ``PickCube-v1`` 任务从零训练轻量 MLP policy，无需下载权重或外部资产。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. 启动训练**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   .. tab-item:: OpenVLA（7B VLA）
      :sync: openvla

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

      **2. 下载场景资产**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

      **3. 启动训练**

      首次运行会自动缓存下载默认 Hugging Face 权重：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

   .. tab-item:: OpenVLA-OFT（7B LoRA VLA）
      :sync: openvla-oft

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. 下载场景资产**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

      **3. 启动训练**

      首次运行会自动缓存下载默认基座与 LoRA 权重：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart

命令行中的配置名对应 ``examples/embodiment/config/`` 下的同名 YAML 文件，完整配置目录与常用参数见 :doc:`../reference/configuration`。

.. dropdown:: 常用启动覆盖（可选）

   - **约 1 分钟快速验证（MLP）**：在命令末尾追加 ``runner.max_epochs=3 runner.val_check_interval=-1``。
   - **改用本地权重**：追加 ``actor.model.model_path=/path/to/checkpoint``\ （rollout 自动沿用；OpenVLA-OFT 同步传 ``actor.model.lora_path=/path/to/lora``）。
   - **指定 GPU 卡号**：两个 VLA 快速开始默认使用全部可见卡；需要限定设备时修改 ``cluster.component_placement``\ （见 :doc:`../reference/configuration`\ ）。

结果与常见问题
--------------

日志、TensorBoard 事件与 checkpoint 保存在 ``logs/<时间戳>-<配置名>/`` 下。启动 ``tensorboard --host 0.0.0.0 --logdir ./logs``，重点观察 ``rollout/env_info/return`` 和 ``rollout/env_info/success_once``。

单卡运行 3 轮 MLP 快速验证约 1 分钟完成，每步会在终端输出指标表：

.. code-block:: text

   ╭────────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step:    3/3 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 00:52      │ ETA: 00:00                     │ Step Time: 17.3s  │
   ├─────────────────────────────── Rollout ──────────────────────────────────┤
   │ env_info/return=12.400  │ env_info/success_once=0.062 │ rewards=0.248    │
   ╰──────────────────────────────────────────────────────────────────────────╯

完整运行数百轮后，MLP 的 ``success_once`` 会升至 ``0.90`` 以上；VLA 快速开始则从 SFT 基线稳步上升：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embody-quickstart-metric.jpg
   :align: center
   :width: 90%

   ManiSkill3 VLA 快速开始的 TensorBoard 训练曲线。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 解决方法
   * - 训练时显存不足（OOM）
     - 调小 ``actor.micro_batch_size`` 与 ``env.train.total_num_envs``，详见 :ref:`显存不足（OOM） <faq-oom>`。
   * - EGL 或 Vulkan 渲染报错
     - 按 :ref:`渲染问题 <faq-rendering>` 检查驱动与图形渲染配置。
   * - 重复运行时报 Ray 或 NCCL 错误
     - 先执行 ``ray stop`` 清理残留进程，详见 :ref:`CUDA 与显存问题 <faq-cuda>`。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 具身训练
      :link: ../examples/index
      :link-type: doc

      查看 LIBERO、RoboTwin、OpenPI、SFT 与世界模型示例。

   .. grid-item-card:: 运行独立评测
      :link: ../evaluations/index
      :link-type: doc

      在具身基准上评估训练后的权重。

   .. grid-item-card:: 性能调优
      :link: ../guides/performance/index
      :link-type: doc

      学习 :doc:`流水线训练模式 <../guides/training_pipeline>` 与 :doc:`执行模式 <../concepts/execution_modes>`。
