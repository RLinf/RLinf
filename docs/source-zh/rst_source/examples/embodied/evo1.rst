Evo-1 强化学习
==============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

`Evo-1 <https://github.com/MINT-SJTU/Evo-1>`__ 是一个紧凑（约 1B）的视觉-语言-动作（VLA）模型：以 InternVL3-1B 作为视觉-语言主干，配一个 flow-matching（DiT）动作头。RLinf **原生集成**\ 了 Evo-1，将其嵌入 RLinf 的 Python 内存空间，支持零延迟的张量级交互，并在 LIBERO 仿真器上支持全参数 SFT 与 GRPO 微调。

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

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: LIBERO

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model evo1 --env libero && source .venv/bin/activate

      **2. 下载模型**

      下载 Evo-1 的 LIBERO 权重（包含 ``config.json``、``norm_stats.json`` 与 ``mp_rank_00_model_states.pt``）：

      .. code-block:: bash

         hf download MINT-SJTU/Evo1_LIBERO --local-dir ./Evo1_LIBERO

      **3. 启动训练**

      Evo-1 配置文件一览：

      .. list-table::
         :header-rows: 1
         :widths: 22 50 28

         * - 流程
           - 配置文件
           - 启动脚本
         * - GRPO
           - ``examples/embodiment/config/libero_spatial_grpo_evo1.yaml``
           - ``examples/embodiment/run_embodiment.sh``
         * - SFT
           - ``examples/sft/config/libero_sft_evo1.yaml``
           - ``examples/sft/run_vla_sft.sh``
         * - 独立评测
           - ``evaluations/libero/libero_spatial_evo1_eval.yaml``
           - ``evaluations/run_eval.sh``

      .. include:: _model_path.rst

      启动 LIBERO-Spatial GRPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_evo1 actor.model.model_path=$PWD/Evo1_LIBERO

      .. dropdown:: 进阶配置（可选）

         - **离线基座 VLM**：运行机器无法联网时，提前下载 ``OpenGVLab/InternVL3-1B``，并将 ``actor.model.evo1.vlm_name`` 指向本地目录：

           .. code-block:: bash

              hf download OpenGVLab/InternVL3-1B --local-dir ./InternVL3-1B

         - **自定义数据集与源码路径**：使用自定义数据集时，保持 ``actor.model.evo1.arm_key`` 与 ``actor.model.evo1.dataset_key`` 与 ``norm_stats.json`` 的顶层 key 一致。若自行克隆了 Evo-1 仓库，将 ``actor.model.evo1.repo_path``\ （或环境变量 ``EVO1_REPO_PATH``）指向仓库根目录。如需快速在单任务上验证，追加 ``+env.train.task_id_filter=[0] +env.eval.task_id_filter=[0]``，几十个 GRPO step 内即可看到 ``env/success_at_end`` 上升。
         - **监督微调（SFT）**：在 LIBERO 风格的 LeRobot 数据集上对 Evo-1 做 SFT 时，将 ``data.train_data_paths`` 指向 Evo-1 数据集配置 YAML，将 ``actor.model.model_path`` 指向起始权重后启动（训练权重保存在 ``runner.logger.log_path`` 下）：

           .. code-block:: bash

              bash examples/sft/run_vla_sft.sh libero_sft_evo1

         - **独立评测**：使用独立评测脚本评测权重：

           .. code-block:: bash

              bash evaluations/run_eval.sh libero libero_spatial_evo1_eval \
                rollout.model.model_path=$PWD/Evo1_LIBERO

         - **CUDA / Ascend Graph 捕获**：动作头算子下发成为 rollout 瓶颈时，可为确定性 rollout 路径启用 graph 捕获，并保持 rollout 模型常驻设备以复用已捕获的 graph（按实际输入 shape 和 dtype 延迟捕获，原地更新权重后 replay 仍生效；模型卸载时会释放 graph）：

           .. code-block:: yaml

              rollout:
                enable_cuda_graph: True
                enable_offload: False

         - **Ascend 视觉 attention 分批**：Ascend 上大 batch 可能在冻结的 InternVL 视觉 attention 中耗尽显存。将 ``actor.model.evo1.vision_attention_batch_size`` 设为 ``8``\ （默认 ``0``，使用原生 attention），每次最多计算八帧相机图像的 attention，并结合实际 rollout batch 测量吞吐。

可视化与结果
------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。

从 Evo-1 SFT 权重出发运行 GRPO，LIBERO-Spatial 成功率稳步提升（n=64 验证，冻结 VLM、只训动作头）：

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
