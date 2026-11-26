VLM-as-Policy：LIBERO + Qwen3.8-27B
===================================

本实践使用 RLinf 的 :doc:`具身智能体 <index>` 架构，通过在线强化学习在 LIBERO 上直接把 ``Qwen3.8-27B`` 训练为多轮机器人 policy。模型读取多视角相机画面与基座坐标系下的机械臂位姿，在 ``<think>`` 内推理，再输出 JSON 相对位姿交由闭环控制器执行。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :class-card: sd-rounded-3

      LIBERO（``libero_gso``、``libero_10``）与 RoboDojo 动作格式

   .. grid-item-card:: 算法
      :class-card: sd-rounded-3

      GRPO（``grpo_dynamic`` + ``pear_reinforce``）、DAPO 零方差组过滤、可选 top-K score centering

   .. grid-item-card:: 任务
      :class-card: sd-rounded-3

      稀疏 0/1 奖励下的多轮桌面操作

   .. grid-item-card:: 硬件
      :class-card: sd-rounded-3

      默认 2 机 16 张 80 GB GPU；论文规模 4 机 32 卡 H200

单条轨迹中，模型读取相机画面与舍入后的基座坐标系机械臂位姿，在 ``<think>`` 内推理，然后输出 JSON 相对位姿。模拟器运行闭环 PD 与操作空间控制，直到机械臂到位、受阻停滞、任务完成或原生步数预算耗尽。

.. figure:: ../../../_static/svg/vlm_episode_loop.svg
   :width: 100%
   :align: center

   单条多轮轨迹：观察画面与位姿，在 ``<think>`` 中推理并输出 JSON 相对位姿，执行后继续下一轮，任务结束时获得稀疏 0/1 奖励。

本实践背后的分离式异步采样与训练架构见 :doc:`index`。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - 环境
     - 任务 / 套件
     - 配置文件
     - 说明
   * - LIBERO
     - ``libero_gso``\ （Goal、Spatial、Object）
     - ``libero_gso_grpo_pear_sglang.yaml``
     - 第一阶段默认配置（2 机 16 卡，每步 8 组，160k 预算，默认 ``max_steps: 2`` 快速验证）
   * - LIBERO
     - ``libero_gso``\ （Goal、Spatial、Object）
     - ``libero_gso_grpo_pear_sglang_paper.yaml``
     - 第一阶段论文规模（4 机 32 卡，每步 128 组，``max_turns: 300``，``max_steps: 1000``）
   * - LIBERO
     - ``libero_10``\ （Long）
     - ``libero_10_grpo_pear_stage2.yaml``
     - 第二阶段长程任务（262,144 token 预算，``max_turns: 64``）
   * - LIBERO / RoboDojo
     - 混合多本体
     - ``pear_stage3_libero_robodojo.yaml``
     - 第三阶段 off-policy 训练，开启 top-K score centering（``topk_log_probs: 32``，``seq_length: 8192``）

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 项目
     - 格式说明
   * - 观测
     - 两路旋转 180° 的 512x512 RGB 图像（LIBERO 为 ``[agentview]`` 与 ``[wrist]``），以及基座坐标系下舍入后的机械臂位姿（厘米与弧度）和夹爪宽度。
   * - 动作
     - 一个 ``json`` 代码块，包含 ``arm``\ （``x``、``y``、``z``、``roll``、``pitch``、``yaw``）和 ``gripper``\ （``width``）的相对变化量。输出无法解析时立即结束轨迹并计 ``0.0`` 奖励。
   * - 奖励
     - 任务成功给 ``1.0``，否则给 ``0.0``；不使用 critic、KL 惩罚或熵奖励。
   * - 提示词
     - 系统提示词定义坐标系与 JSON 格式，每轮追加状态 JSON 与带相机标签的新帧。LIBERO 指令中的左右方向已做镜像，与旋转后的相机画面一致。

安装与运行
----------

先在多节点集群上启动 Ray（默认配置需 2 台 8 卡节点，论文规模需 4 台 8 卡节点），每个节点在运行 ``ray start`` 前导出唯一的 ``RLINF_NODE_RANK``\ （头节点为 ``0``，工作节点依次为 ``1`` 等）。

.. tab-set::

   .. tab-item:: LIBERO + Qwen3.8-27B（GRPO / PEAR）

      **1. 安装环境**

      安装包含 LIBERO 与 SGLang ``0.5.22``\ （CUDA 13 wheel）的具身 Qwen3.8 环境，并在每个节点激活：

      .. code-block:: bash

         bash requirements/install.sh embodied --model qwen3_8 --env libero
         source .venv/bin/activate

      **2. 下载模型**

      将 ``Qwen/Qwen3.8-27B`` 下载到所有节点可访问的目录（例如共享存储）：

      .. code-block:: bash

         hf download Qwen/Qwen3.8-27B --local-dir ./Qwen3.8-27B

      启动脚本默认设置离线环境变量，各节点必须能直接读取本地模型目录。将 ``examples/embodiment/config/libero_gso_grpo_pear_sglang.yaml`` 中的模型与分词器路径指向本地目录（推理端会同步使用该模型目录）：

      .. code-block:: yaml

         actor:
           model:
             model_path: ./Qwen3.8-27B
           tokenizer:
             tokenizer_model: ./Qwen3.8-27B

      **3. 启动训练**

      在头节点运行默认配置（需 2 节点 16 卡，``cluster.num_nodes: 2``，默认 2 步快速验证）：

      .. code-block:: bash

         bash examples/embodiment/run_vlm_policy.sh libero_gso_grpo_pear_sglang

      正式训练时调大 ``runner.max_steps``\ （例如 ``1000``）；也可以直接运行论文规模配置（需 4 节点 32 卡，``cluster.num_nodes: 4``）或 2 节点第二阶段 ``libero_10`` 配置：

      .. code-block:: bash

         bash examples/embodiment/run_vlm_policy.sh libero_gso_grpo_pear_sglang_paper
         bash examples/embodiment/run_vlm_policy.sh libero_10_grpo_pear_stage2

关键配置项
~~~~~~~~~~

按集群规模或训练阶段调整 YAML 时，重点修改下表中的几组字段：

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - 类别
     - 配置项
     - 说明
   * - 节点划分与步数
     - ``cluster.num_nodes``、``cluster.component_placement``、``runner.max_steps``、``runner.seq_length``
     - 将 ``actor`` 与 ``rollout`` 放在互不重叠的 GPU 上；异步模式下 ``reward`` 与 ``rollout`` 共用卡号。
   * - 采样并发
     - ``data.task_suite_name``、``data.rollout_batch_size``、``agentloop.max_turns``、``tools.embodied_env.max_concurrent_envs``
     - 按推理端卡数调整并发 LIBERO 模拟器数量与最大交互轮数。
   * - 缓冲与滞后上限
     - ``rollout_buffer.max_inflight_groups``、``rollout_buffer.max_buffered_groups``、``agentloop.max_actor_accept_staleness``
     - 让推理引擎保持满载，同时约束 policy 滞后。
   * - 流式会话与缓存
     - ``session_server.engine_streaming_session``、``session_server.merge_continuous_turns``、``session_server.payload_shards``
     - 跨轮保持 KV/GDN 前缀缓存，并将多模态帧暂存在采样节点本地。
   * - SGLang 与 FSDP2 显存
     - ``rollout.sglang.attention_backend``、``rollout.sglang.max_mamba_cache_size``、``actor.fsdp_config.enable_gradient_accumulation``
     - 80 GB 显存下保持 ``attention_backend: null``、``max_mamba_cache_size: 256`` 与 ``enable_gradient_accumulation: False``。
   * - 组过滤与第三阶段
     - ``algorithm.down_sampling.down_sampling_config.reject_equal_reward``、``algorithm.pear.score_centering``、``session_server.topk_log_probs``
     - 开启 DAPO 零方差组过滤；第三阶段按需开启 top-K score centering。

可视化与结果
------------

启动脚本会自动把日志和 TensorBoard 事件写入 ``logs/<timestamp>-<config_name>/``。在该目录启动 TensorBoard：

.. code-block:: bash

   tensorboard --host 0.0.0.0 --logdir logs/

重点观察 ``rollout/rewards``、``rollout/buffer/dropped_equal_reward``、``rollout/session_server/payload_queued_groups`` 和 ``train/actor/policy_loss``。本地排队的 payload 组数稳定贴近缓冲区大小，说明被过滤掉的多模态组已及时释放采样节点内存。在 2 机 16 张 80 GB GPU 上，默认配置单步 FSDP2 更新耗时约 8–10 秒，27B 全量跨机权重同步耗时约 5 秒。

常见问题
--------

**工作节点加入 Ray 时报 ``InvalidAuthToken``**
   Ray 2.60 及以上版本在未配置共享 token 时会拒绝未认证加入；在所有节点执行 ``ray start`` 前导出 ``RAY_AUTH_MODE=disabled``。

**80 GB actor GPU 显存溢出（OOM）**
   保持 ``actor.fsdp_config.enable_gradient_accumulation: False`` 与 ``actor.fsdp_config.gradient_checkpointing_use_reentrant: False``，让 FSDP2 在每个 micro-batch 逐层归约梯度，避免在显存中常驻未分片的 27B 完整梯度。

**采样在凑满一个批次前长时间卡住**
   调高 ``rollout.max_running_requests`` 或 ``rollout_buffer.max_inflight_groups`` 时，同步调大 ``rollout.sglang.max_mamba_cache_size``\ （默认 ``256``），确保每条活跃流式会话都有足够的循环状态槽位。

延伸阅读
--------

- :doc:`index`：了解具身智能体架构、异步训练运转流程与单组多轮生命周期。
- :doc:`../../guides/async_agentic`：调整准入上限、滞后上限与权重同步暂停模式。
- :doc:`../../guides/session_server`：配置 session server、payload 分片与路由策略。
- :doc:`../../guides/multi_node`：多节点 Ray 集群与 ``RLINF_NODE_RANK`` 配置说明。
- :doc:`../../reference/metrics`：查看 rollout、policy 滞后与训练指标。
