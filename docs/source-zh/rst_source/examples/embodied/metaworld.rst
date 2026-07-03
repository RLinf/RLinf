基于MetaWorld评测平台的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/metaworld.png
   :align: center
   :width: 90%

   Meta-World 基准（图片来源：`Meta-World <https://metaworld.farama.org>`__）。

`Meta-World <https://metaworld.farama.org>`__ 是一个基于 MuJoCo 的多任务操作基准：一台 7 自由度
机械臂完成 50 个多样的桌面任务。RLinf 借助它对视觉-语言-动作（VLA）策略进行强化学习微调，并评测
分布外（OOD）泛化。

概览
----------------------------------------

在 Meta-World 的 50 个任务上对 VLA 进行强化学习微调；π₀ + PPO 平均成功率约 78%。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenVLA-OFT · π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      MT50 · ML45（5 个 OOD）

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 张 GPU

| **你将完成：** 安装依赖 → 下载 SFT 模型 → 运行 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · SFT 检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - 套件
     - 任务数
     - 设置
   * - MT50
     - 50
     - 在全部 50 个任务上进行多任务训练与评测。
   * - ML45
     - 45 + 5
     - 在 45 个任务上训练；在 5 个留出（OOD）任务上评测。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测 (Observation)
     - 工作区周围离屏相机的 RGB 图像（480×480）。
   * - 动作 (Action)
     - 4 维连续动作：3D 末端执行器位置（x, y, z）+ 夹爪开合。
   * - 奖励 (Reward)
     - 稀疏奖励——基于任务完成。


安装
----------------------------------------

.. include:: _setup_common.rst

**选项 1：Docker 镜像** — 镜像标签 ``agentic-rlinf0.3-metaworld``：

.. code-block:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-metaworld
      # 国内镜像：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-metaworld

   # 在容器内切换到对应模型的虚拟环境：
   source switch_env openpi        # 或：source switch_env openvla-oft

**选项 2：自定义环境** — 安装 ``--env metaworld`` 依赖组合：

.. code-block:: bash

   # 国内用户可以添加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model openpi --env metaworld
   # 或安装 OpenVLA-OFT 环境：
   # bash requirements/install.sh embodied --model openvla-oft --env metaworld

   source .venv/bin/activate


下载模型
----------------------------------------

下载参考配方使用的 SFT 检查点（任选一种方式）：

.. code-block:: bash

   # 方法 1：git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT
   git clone https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-MetaWorld-SFT

   # 方法 2：huggingface-hub（国内用户可设置 HF_ENDPOINT=https://hf-mirror.com）
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-MetaWorld-SFT --local-dir RLinf-Pi0-MetaWorld-SFT
   hf download RLinf/RLinf-Pi05-MetaWorld-SFT --local-dir RLinf-Pi05-MetaWorld-SFT
   hf download RLinf/RLinf-OpenVLAOFT-MetaWorld-SFT --local-dir RLinf-OpenVLAOFT-MetaWorld-SFT

也可以从 ModelScope 下载模型：https://www.modelscope.cn/models/RLinf/RLinf-Pi0-MetaWorld。

.. include:: _model_path.rst

运行
----------------------------------------

每个配方都是 ``examples/embodiment/config/`` 下的一个 YAML 配置：

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 设定
     - 模型 / 算法
     - 配置
   * - MT50
     - π₀ + PPO
     - ``metaworld_50_ppo_openpi.yaml``
   * - MT50
     - π₀.₅ + PPO
     - ``metaworld_50_ppo_openpi_pi05.yaml``
   * - MT50
     - OpenVLA-OFT + GRPO
     - ``metaworld_50_grpo_openvlaoft.yaml``
   * - ML45
     - π₀ + PPO
     - ``metaworld_45_ppo_openpi.yaml``

使用 ``run_embodiment.sh`` 启动一个配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh metaworld_50_ppo_openpi

**这个命令会：**

1. 加载 ``examples/embodiment/config/metaworld_50_ppo_openpi.yaml``。
2. 按 ``cluster.component_placement`` 启动 Meta-World MT50 的 rollout/eval worker。
3. 运行 PPO 训练循环，并把日志和检查点写入 ``runner.logger.log_path``。

.. admonition:: 进一步配置
   :class: note

   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 全量配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`恢复训练 <../../guides/resume>`


可视化与结果
----------------------------------------

启动 TensorBoard 实时观察训练：

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

最值得关注的指标是 **``env/success_once``** —— 任务成功率。每个日志指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

如需保存评估视频，在配置中启用：

.. code-block:: yaml

   env:
      eval:
         video_cfg:
            save_video: True
            video_base_dir: ${runner.logger.log_path}/video/eval


MetaWorld 结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
下表Diffusion Policy, TinyVLA和SmolVLA的结果参考 `SmolVLA 论文 <https://arxiv.org/abs/2403.04880>`_ 论文得到。π\ :sub:`0`\ 和 π\ :sub:`0.5`\ 的SFT结果是通过LeRobot官方提供的 `数据集 <https://huggingface.co/datasets/lerobot/metaworld_mt50>`_ 重新训练所得。

.. list-table:: **MetaWorld-MT50 性能对比（Success Rate, %）**
   :widths: 15 10 10 10 10 10
   :header-rows: 1

   * - **Methods**
     - **Easy**
     - **Medium**
     - **Hard**
     - **Very Hard**
     - **Avg.**
   * - Diffusion Policy
     - 23.1
     - 10.7
     - 1.9
     - 6.1
     - 10.5
   * - TinyVLA
     - 77.6
     - 21.5
     - 11.4
     - 15.8
     - 31.6
   * - SmolVLA
     - 87.1
     - 51.8
     - 70.0
     - 64.0
     - 68.2
   * - π\ :sub:`0`\
     - 77.9
     - 51.8
     - 53.3
     - 20.0
     - 50.8
   * - π\ :sub:`0`\  + PPO
     - **92.1**
     - **74.6**
     - 61.7
     - **84.0**
     - **78.1**
   * - π\ :sub:`0.5`\
     - 68.2
     - 37.3
     - 41.7
     - 28.0
     - 43.8
   * - π\ :sub:`0.5`\  + PPO
     - 86.4
     - 55.5
     - **75.0**
     - 66.0
     - 70.7
