Qwen3.5 多轮视觉 Agent 训练
===========================

开启多轮图像对齐，让 Qwen3.5 多轮视觉 Agent 训练在 SGLang 推理和 Megatron-Bridge 训练之间看到完全一致的图像输入与视觉 token 边界，避免模型生成的特殊文本污染历史轮次。

**使用步骤：** 确认环境与适用范围 → 在训练 YAML 中开启多轮图像对齐 → 按需接入自定义视觉 Agent → 运行独立脚本验证环境。

适用范围与环境
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 项目
     - 说明
   * - **支持组合**
     - 本地稠密 Qwen3.5 模型（``Qwen3_5ForConditionalGeneration``），使用默认 SGLang rollout，并通过 Megatron-Bridge 训练（支持 ``PP`` 与 ``CP`` 并行；``Qwen/Qwen3.5-4B`` 已在 1、2、8 张 NVIDIA A100 80 GB 上完成验证）。
   * - **暂不支持**
     - 视频或音频输入、FSDP 训练、``session_server.enable: true``，以及部署独立 ``inference`` / ``actor_inference`` / ``critic_inference`` 节点的拓扑（logprob 与 value 须由训练节点重算）。
   * - **环境要求**
     - Python 3.12 ``agentic`` 环境，包含 SGLang ``0.5.12.post1`` 至 ``0.5.22+``\ （见 :doc:`version`）、Megatron Core 0.18.x、Megatron-Bridge 0.5.0（见 :doc:`../extending/mbridge`）与 ``fla-core``。

.. dropdown:: 安装 agentic 环境

   如尚未安装，在仓库根目录运行以下命令创建 Python 3.12 环境（完整说明见 :doc:`../start/installation`）：

   .. code-block:: bash

      bash requirements/install.sh agentic --sglang 0.5.12.post1 && source .venv/bin/activate

训练配置
--------

多轮图像对齐默认关闭。在训练 YAML 中开启 ``rollout.sglang.align_multimodal_inputs: true``，并配套启用动态 rollout 批与 Megatron-Bridge：

.. code-block:: yaml

   agentloop:
     is_dynamic_rollout_batch: true
   session_server:
     enable: false
   rollout:
     model:
       model_type: qwen3_5
     sglang:
       align_multimodal_inputs: true
   actor:
     training_backend: megatron
     model:
       model_type: qwen3_5
     megatron:
       mbridge: true
       use_hf_ckpt: true

若需编写自己的多轮视觉 Agent 并传出每轮图像与区间，请继续阅读 :ref:`multi-turn-visual-agents`。

.. dropdown:: 可选：同时训练 Critic

   启用 ``critic.use_critic_model: true`` 时，``critic`` 同样须使用 Megatron-Bridge：

   .. code-block:: yaml

      critic:
        use_critic_model: true
        training_backend: megatron
        model:
          model_type: qwen3_5
        megatron:
          mbridge: true
          use_hf_ckpt: true

.. dropdown:: 可选：选择 logprob 计算方式

   训练默认重算 logprob（``algorithm.recompute_logprobs: true``），也可直接复用 SGLang 返回的 logprob：

   .. code-block:: yaml

      algorithm:
        recompute_logprobs: false
      rollout:
        return_logprobs: true

快速验证（独立脚本）
--------------------

以下命令直接运行仓库自带的合成图像检查脚本，用来验证当前机器的 SGLang 与 Megatron-Bridge 环境是否通畅（不会读取上面自定义的训练 YAML）。

在仓库根目录下载 ``Qwen/Qwen3.5-4B`` 并运行双卡检查脚本。通过标准为：日志中两轮推理均显示 passed，且最终打印的 actor 与 critic JSON 中 loss 为有限值、梯度范数与更新参数量 ``changed`` 均大于 0：

.. code-block:: bash

   hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
   bash examples/agent/qwen35/run_image_alignment.sh \
       rollout.model.model_path=./Qwen3.5-4B

.. dropdown:: 查看 2×A100 80 GB 实测输出

   .. code-block:: json

      {"role": "actor", "ranks": [{"rank": 0, "loss": -0.960578441619873, "grad_norm": 290.3096923828125, "changed": 294963}, {"rank": 1, "loss": -0.960578441619873, "grad_norm": 290.3096923828125, "changed": 272910}]}
      {"role": "critic", "ranks": [{"rank": 0, "loss": 0.5, "grad_norm": 1544.4698486328125, "changed": 49}, {"rank": 1, "loss": 0.5, "grad_norm": 1544.4698486328125, "changed": 49}]}

.. dropdown:: 单卡推理检查与 8 卡验证

   若仅在单卡上测试两轮 SGLang 图像输入隔离（日志会打印 ``Turn 2 passed``）：

   .. code-block:: bash

      python examples/agent/qwen35/check_image_alignment.py --model ./Qwen3.5-4B

   若需指定其他 GPU 编号，或在 8 张空闲 GPU 上按 ``examples/agent/qwen35/config/image_alignment.yaml`` 跑通多卡检查：

   .. code-block:: bash

      bash examples/agent/qwen35/run_image_alignment.sh \
          rollout.model.model_path=./Qwen3.5-4B \
          cluster.component_placement.actor=0-7

常见报错与排查
--------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 报错关键词
     - 处理方式
   * - ``non-empty image_spans and matching image_data``
     - 仅在有图片传入且数量与区间列表一致时向生成接口传递图像参数，推荐直接使用 :ref:`multi-turn-visual-agents` 中的状态辅助接口。
   * - ``SGLang image patch width ... differs``
     - 确保视觉 Agent 读取的模型路径与 ``rollout.model.model_path`` 一致，且推理与 Agent 进程未混用不同的像素缩放环境变量。
   * - ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`` 冲突
     - 取消环境中自定义的该变量，让 RLinf 在启动 SGLang 时自动注册处理器。

工作原理
--------

开启多轮图像对齐后，每轮交互按以下三步保持推理与训练一致：

1. **Agent 侧预处理与展开**：Agent 在调用生成前用与 SGLang 一致的处理器生成图像输入，提前把视觉占位 token 展开到真实宽度并记录 ``image_spans``。
2. **SGLang 按区间编码**：SGLang 仅对 ``image_spans`` 标出的范围编码真实图片，把文本里字面出现的特殊 token 当作普通文本处理。
3. **Megatron-Bridge 掩码前向**：训练前向通过 ``mm_token_type_ids`` 屏蔽字面出现的特殊 token，保留完整语言嵌入以及流水线并行（PP）与上下文并行（CP）布局；当前实测覆盖 ``TP=2, PP=1, CP=1``，PP 与 CP 组合通过单元测试验证。

.. dropdown:: 跨后端 logprob 数值差异说明

   多轮图像对齐保证 SGLang 与 Megatron-Bridge 看到相同的图像输入和视觉 token 边界，但两套推理与训练后端之间的浮点与算子实现仍会有细微 logprob 数值差异。上方验证脚本只检查对齐链路与优化器更新，不代表下游任务收敛结果。

相关指南
--------

- :doc:`../extending/new_agent`：编写自定义多轮视觉 Agent（见 :ref:`multi-turn-visual-agents`）。
- :doc:`../extending/mbridge`：Megatron-Bridge 模型与环境配置。
- :doc:`agentic_config`：Agentic 训练 YAML 配置参考。
- :doc:`version`：SGLang 与 vLLM 版本兼容表。
