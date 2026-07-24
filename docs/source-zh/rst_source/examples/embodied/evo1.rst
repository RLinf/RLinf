Evo-1 模型的强化学习
====================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

`Evo-1 <https://github.com/MINT-SJTU/Evo-1>`__ 是一个紧凑（约 1B）的视觉-语言-动作（VLA）模型：
以 InternVL3-1B 作为视觉-语言主干，配一个 flow-matching（DiT）动作头。RLinf **原生集成** 了
Evo-1 —— 嵌入 RLinf 的 Python 内存空间，实现零延迟、张量级交互 —— 并支持在 LIBERO 仿真器上做
全参数 SFT 与 GRPO 微调。

概览
----

在 LIBERO 操作任务上先 SFT、再用 GRPO 微调 Evo-1。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO

   .. grid-item-card:: 算法
      :text-align: center

      SFT · GRPO

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO-Spatial

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4–8 GPU

| **你将完成：** 安装（原生）→ 下载 Evo-1 权重 → GRPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Evo-1 权重（见下）。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_grpo_evo1``
     - 用 Evo-1 在 LIBERO-Spatial 套件上做 GRPO 训练。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - 观测
     - Evo-1 所需的 LIBERO 相机观测与机器人状态。
   * - 动作
     - 连续 7 自由度动作（6-DoF EE 增量 + 夹爪），由 flow-matching 头解码。
   * - 奖励
     - LIBERO 任务成功。
   * - 指令
     - LIBERO episode 的自然语言任务指令。

安装
----

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

2. 安装依赖
~~~~~~~~~~~

一条命令安装 Evo-1 原生环境与 LIBERO 基础依赖（脚本会把 Evo-1 源码拉到 ``.venv/Evo-1`` 并处理依赖隔离）：

.. code-block:: bash

    bash requirements/install.sh embodied --model evo1 --env libero --use-mirror
    source .venv/bin/activate

下载模型
--------

从 HuggingFace 下载 Evo-1 的 LIBERO 权重（一个包含 ``config.json``、``norm_stats.json``、
``mp_rank_00_model_states.pt`` 的权重目录）：

.. code-block:: bash

    git lfs install
    git clone https://huggingface.co/MINT-SJTU/Evo1_LIBERO

然后把配置里的 ``rollout.model.model_path`` 与 ``actor.model.model_path`` 设为本地权重路径。
``actor.model.evo1.arm_key`` / ``dataset_key`` 必须与权重 ``norm_stats.json`` 的顶层 key 一致。

运行
----

配置文件
~~~~~~~~

* **SFT（监督微调）**：
  ``examples/sft/config/libero_sft_evo1.yaml``（用 ``examples/sft/run_vla_sft.sh`` 运行）
* **GRPO（强化学习）**：
  ``examples/embodiment/config/libero_spatial_grpo_evo1.yaml``
* **独立评测**：
  ``evaluations/libero/libero_spatial_evo1_eval.yaml``（用 ``evaluations/run_eval.sh`` 运行）

关键配置片段（GRPO）
^^^^^^^^^^^^^^^^^^^^

顶层文件通过 Hydra 组装环境与模型，并在 ``actor.model`` / ``algorithm`` 下覆盖 flow-matching
SDE 采样与 GRPO 的核心参数。

.. code-block:: yaml

    rollout:
      model:
        model_type: "evo1"

    actor:
      model:
        model_type: "evo1"
        model_path: "/path/to/model/Evo1_LIBERO"
        num_action_chunks: 14        # 每次推理执行的步数
        evo1:
          arm_key: "libero_robot"    # 必须与 norm_stats.json 一致
          dataset_key: "libero_robot"
        rl_head_config:
          noise_method: "flow_sde"
          noise_level: 0.5           # SDE 噪声尺度
          denoising_steps: 8         # RL rollout + replay 的 SDE 去噪步数
      model.evo1.rl_trainable_scope: "action_head"   # 冻结 InternVL3，只训动作头

    algorithm:
      adv_type: grpo
      logprob_type: token_level      # 逐维 ratio（执行的 14x7 维）
      group_size: 8
      update_epoch: 2
      clip_ratio_low: 0.2
      clip_ratio_high: 0.28          # clip-higher：持续改进

启动命令
~~~~~~~~

.. code-block:: bash

    export ROBOT_PLATFORM="LIBERO"
    export EVO1_REPO_PATH=$(python -c "import os,evo1; print(os.path.dirname(evo1.__file__))" 2>/dev/null || echo /path/to/Evo-1)
    bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_evo1

如需快速复现单任务上的 RL 效果，可用 ``+env.train.task_id_filter=[0] +env.eval.task_id_filter=[0]``
把训练限制到单个 LIBERO-Spatial 任务；几十个 GRPO step 内 ``env/success_at_end`` 就会从
SFT 基线稳步上升（见下方结果）。

监督微调（SFT）
----------------

在 RL 之前用 LIBERO 风格的 LeRobot 数据对 Evo-1 做 SFT（或复现 SFT 权重）：

.. code-block:: bash

    export EVO1_REPO_PATH=/path/to/Evo-1
    bash examples/sft/run_vla_sft.sh libero_sft_evo1

把 ``data.train_data_paths`` 指向 Evo-1 的数据集配置 YAML，``actor.model.model_path``
指向要微调的权重（或基座）。SFT 配方（AdamW、带 warmup 的 cosine 调度、flow-matching MSE）
沿用 Evo-1 的原生训练；权重保存在 ``runner.logger.log_path`` 下。

独立评测
--------

用独立评测框架评测一个权重（配置在 ``evaluations/libero/`` 下）：

.. code-block:: bash

    bash evaluations/run_eval.sh libero libero_spatial_evo1_eval \
      rollout.model.model_path=/path/to/ckpt

可视化与结果
------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关键指标是 ``env/success_once``。从 Evo-1 SFT 权重出发做 GRPO，在 LIBERO-Spatial 上成功率稳定上升
（n=64 验证，冻结 VLM、只训动作头）：

.. list-table:: Evo-1 GRPO 在 LIBERO-Spatial 上的结果（n=64）
   :header-rows: 1

   * - 设置
     - SFT
     - RLinf-GRPO
   * - 单任务（``success_at_end``）
     - 0.58
     - **0.86**
   * - LIBERO-Spatial 套件（``success_once``，best ckpt）
     - 0.656
     - **0.750**
