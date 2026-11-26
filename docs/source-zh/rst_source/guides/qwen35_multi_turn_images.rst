Qwen3.5 多轮图像输入对齐
========================

在 Qwen3.5 多轮视觉 Agent 训练中，开启图像输入捕获，让 SGLang rollout 与 Megatron-Bridge 训练使用一致的预处理图像输入和 M-RoPE 位置。SGLang 仅按显式标记的区间处理真实图像，模型生成文本里出现图像标记 token 也不会破坏历史轮次。

概览
----

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 维度
     - 说明
   * - **适用场景**
     - 本地稠密架构 Qwen3.5 checkpoint（``Qwen3_5ForConditionalGeneration``）的多轮视觉 Agent RL，搭配 SGLang rollout 与 Megatron-Bridge 训练。
   * - **保持默认**
     - 纯文本训练、VLM SFT、FSDP 训练，或使用 :doc:`Session Server <session_server>` 的任务。
   * - **环境依赖**
     - Python 3.12 agentic 环境，包含 SGLang ``>= 0.5.12.post1``、Megatron Core 0.18.x、Megatron-Bridge 0.5.0（见 :doc:`../extending/mbridge`）与 ``fla-core``；``Qwen/Qwen3.5-4B`` 已在 1、2、8 张 NVIDIA A100 80 GB 上验证通过。

.. dropdown:: 安装 agentic 环境

   如尚未安装，在仓库根目录运行以下命令创建 Python 3.12 环境（自动安装 SGLang、Megatron Core、Megatron-Bridge 与 ``fla-core``；完整说明见 :doc:`../start/installation`）：

   .. code-block:: bash

      bash requirements/install.sh agentic --sglang 0.5.12.post1 && source .venv/bin/activate

开启配置
--------

图像捕获默认关闭。在训练 YAML 中启用 ``rollout.sglang.capture_multi_modal_inputs: true``，并配套开启动态 rollout 批、Megatron-Bridge 与 ``PP=1``、``CP=1``，RLinf 会在启动时自动注册 SGLang 外部图像处理器：

.. code-block:: yaml

   agentloop:
     is_dynamic_rollout_batch: true
   session_server:
     enable: false
   rollout:
     model:
       model_type: qwen3_5
     sglang:
       capture_multi_modal_inputs: true
   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
       pipeline_model_parallel_size: 1
       context_parallel_size: 1
     megatron:
       mbridge: true
       use_hf_ckpt: true

如需在自定义视觉 Agent 中透传每轮图像与区间，参见 :doc:`../extending/new_agent` 中的 :ref:`multi-turn-visual-agents`。

.. dropdown:: 可选：同时训练 Critic

   开启 ``critic.use_critic_model: true`` 时，``critic`` 同样须配置 Megatron-Bridge、``PP=1`` 与 ``CP=1``：

   .. code-block:: yaml

      critic:
        use_critic_model: true
        training_backend: megatron
        model:
          model_type: qwen3_5
          pipeline_model_parallel_size: 1
          context_parallel_size: 1
        megatron:
          mbridge: true
          use_hf_ckpt: true

.. dropdown:: 可选：CPU 预取与 logprob 模式

   如需在 CPU 侧预取多模态微批并与训练前向重叠，在 ``actor.multi_modal``\ （训练 Critic 时对应 ``critic.multi_modal``）下设置每个 rank 的 CPU 预取上限，配置说明见 :doc:`agentic_config`：

   .. code-block:: yaml

      actor:
        multi_modal:
          prefetch_batches: 2
          prefetch_bytes: 536870912  # 512 MiB

   训练既支持 Megatron-Bridge 重算 logprob（默认 ``algorithm.recompute_logprobs: true``），也支持直接复用 rollout logprob（设 ``algorithm.recompute_logprobs: false`` 与 ``rollout.return_logprobs: true``）。

.. dropdown:: 可选：大图像落盘共享目录

   默认情况下，捕获的像素张量直接随 rollout 批次传递。若多轮对话图像体积较大或跨轮重复出现，可在自定义 Agent 中配置 ``image_store_dir`` 指向所有训练节点均可访问的共享目录（见 :ref:`multi-turn-visual-agents`）。

   每张不同图像只按内容哈希以 BF16 格式保存一份 ``.safetensors`` 文件，训练加载时校验 SHA-256。只要仍有在途批次或已保存的 rollout 文件引用该目录，就不要删除该目录；任务结束后再手动清理。

验证环境
--------

先下载 ``Qwen/Qwen3.5-4B``\ （或替换为本地已有 checkpoint 目录），再在仓库根目录执行两步快速检查（已在 80 GB A100 上验证）：

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B

1. **单卡验证两轮捕获**：在输出日志中看到 ``Turn 2 passed`` 即通过：

   .. code-block:: bash

      python examples/agent/qwen35/replay_images.py --model ./Qwen3.5-4B

2. **双卡验证 Actor 与 Critic 更新**：将 ``0,1`` 替换为两张连续空闲 GPU 的编号。脚本会拉起独立的本地 Ray 实例，用内置合成图像运行一次更新（验证配置默认冻结视觉侧权重），并向 stdout 分别打印一行 Actor 与一行 Critic JSON；确认每个 rank 的 ``loss`` 为有限值、``grad_norm`` 为正数且 ``changed > 0``\ （更新的权重元素数）：

   .. code-block:: bash

      bash examples/agent/qwen35/run_replay_update.sh ./Qwen3.5-4B 0,1

   2×A100 80 GB 实测输出：

   .. code-block:: json

      {"role": "actor", "ranks": [{"rank": 0, "loss": -1.0, "grad_norm": 519.580078125, "changed": 303411}, {"rank": 1, "loss": -1.0, "grad_norm": 519.580078125, "changed": 280264}]}
      {"role": "critic", "ranks": [{"rank": 0, "loss": 0.5, "grad_norm": 1670.570556640625, "changed": 49}, {"rank": 1, "loss": 0.5, "grad_norm": 1670.570556640625, "changed": 49}]}

.. dropdown:: 可选：8 卡验证（TP=2、DP=4）

   传入 8 张连续空闲 GPU 的编号，可按 ``examples/agent/qwen35/config/replay_update.yaml`` 验证 ``TP=2、DP=4``：

   .. code-block:: bash

      bash examples/agent/qwen35/run_replay_update.sh ./Qwen3.5-4B 0,1,2,3,4,5,6,7

工作原理
--------

对齐链路分为三步：

1. **rollout 侧捕获**：SGLang 仅对标记区间执行视觉预处理，记录展开后的 token ID、像素张量、图像网格与 M-RoPE 位置，并在多轮间校验跨轮历史状态。
2. **每轮独立携带**：每一轮成为动态 rollout 批中的一条独立样本，各自携带对应的图像输入（无图轮次为空），组批与负载均衡时始终与样本对齐。
3. **训练侧校验与前向**：Megatron-Bridge 前向计算前核对 prompt 前缀与 M-RoPE 位置，视觉编码器只处理标记区间，语言模型沿用采样 token ID。

.. warning::

   图像输入对齐保证 rollout 与训练侧的图像输入和 M-RoPE 位置一致，但 SGLang 与 Megatron-Bridge 跨后端的 logprob 仍有数值差异。上方验证脚本只检查输入对齐与参数更新链路，不代表下游任务收敛效果。

.. dropdown:: 当前限制

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 维度
        - 限制说明
      * - 模态类型
        - 仅支持图像输入，不支持视频与音频。
      * - 训练后端
        - ``actor``\ （启用 Critic 时包括 ``critic``）须使用 Megatron-Bridge、``PP=1``、``CP=1`` 及 ``agentloop.is_dynamic_rollout_batch: true``。
      * - Rollout 模式
        - 须使用默认 SGLang rollout worker，关闭 ``session_server.enable`` 且不设置 ``rollout.sglang.serving_mode``。
      * - 同机临时文件传递
        - rollout worker 与本机 SGLang tokenizer 进程须共用同一个 ``TMPDIR``\ （未单独覆盖时默认 ``/tmp`` 即可）。

.. dropdown:: 常见报错与排查

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - 报错关键词
        - 处理方式
      * - ``non-empty image_spans and images``
        - 仅在 ``image_data`` 非空且与 ``image_spans`` 等长时传入，推荐按 :ref:`multi-turn-visual-agents` 接入。
      * - ``sampled token history`` / ``positions changed``
        - 用展开后的 prompt 拼接生成 token 推进下一轮，并原样透传浅拷贝后的图像区间与历史状态（见 :ref:`multi-turn-visual-agents`）。
      * - ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` 冲突
        - 取消环境中自定义的 ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE``，由 RLinf 启动时自动注册。
      * - ``SGLang did not write image capture file``
        - 确认 checkpoint 为 Qwen3.5、SGLang ``>= 0.5.12.post1``，且 rollout worker 与 tokenizer 进程共用 ``TMPDIR``。
      * - 像素缺失或 SHA-256 校验失败
        - 确认所有训练节点挂载了同一 ``image_store_dir``，且未在批次仍引用时提前清理目录。

相关指南
--------

- :doc:`../extending/new_agent`：编写自定义多轮视觉 Agent（见 :ref:`multi-turn-visual-agents`）。
- :doc:`../extending/mbridge`：Megatron-Bridge 环境要求与模型配置。
- :doc:`agentic_config`：rollout、actor 及多模态预取配置项参考。
- :doc:`version`：SGLang 与 vLLM 版本兼容说明。
