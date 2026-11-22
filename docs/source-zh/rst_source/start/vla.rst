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

      单卡 MLP（零下载）或 7B VLA

   .. grid-item-card:: 3. 启动训练
      :text-align: center

      一行命令直接跑通

   .. grid-item-card:: 4. 查看曲线
      :text-align: center

      TensorBoard 监控指标

一键上手
--------

切换下方标签选择起点：首次验证环境推荐先跑 **MLP**\ （单卡、无需下载模型权重与场景资产），需要 VLA 强化学习时直接切到 **OpenVLA** 或 **OpenVLA-OFT**。如果 ``.venv`` 里已经安装了其他模型或环境，在 ``install.sh`` 末尾加 ``--venv <dir>`` 避免覆盖。国内镜像加速与 Docker Hub 预构建镜像见 :doc:`installation`。

.. tab-set::

   .. tab-item:: MLP（单卡 · 零权重下载）

      在 GPU 0 上基于 ManiSkill3 内置 ``PickCube-v1`` 任务从零训练轻量 MLP 策略——无需下载任何 Hugging Face 权重或外部场景资产。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero && source .venv/bin/activate

      **2. 启动 PPO 训练**

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

      如需约 1 分钟的快速验证，追加 ``runner.max_epochs=3 runner.val_check_interval=-1``。更多 MLP / SAC / GRPO 示例见 :doc:`MLP 策略强化学习 <../examples/embodied/mlp>`。

   .. tab-item:: OpenVLA（7B VLA）
      :sync: openvla

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

      **2. 下载场景资产**\ （`论文 <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c0fabe372177d2aded596be2d3b4544-Abstract-Conference.html>`__）

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

      **3. 启动 PPO 训练**\ （首次运行自动缓存下载 ``gen-robot/openvla-7b-rlvla-warmup``）

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

      **3. 启动 PPO 训练**\ （首次运行自动缓存下载 ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` 与 ``Base-Lora``）

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart

两个 VLA 快速开始配置默认使用所有可用加速卡；需要指定设备时，修改 ``cluster.component_placement``\ （详见 :doc:`../reference/configuration`\ ）。rollout 会自动沿用 ``actor.model.model_path``；若使用本地预下载权重，只需在命令行覆盖 ``actor.model.model_path=/path/to/checkpoint``\ （OpenVLA-OFT 同步传 ``actor.model.lora_path=/path/to/lora``）。

结果与常见问题
--------------

日志与权重保存在 ``./logs``。启动 TensorBoard（``tensorboard --host 0.0.0.0 --logdir ./logs``），重点关注 ``rollout/env_info/return`` 和 ``rollout/env_info/success_once``。

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
