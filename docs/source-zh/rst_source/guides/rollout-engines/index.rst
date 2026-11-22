Rollout 引擎
============

这些指南介绍如何配置 RLinf rollout 使用的推理引擎，包括进程内 GPU 后端，以及与 RLinf 任务一起启动或独立运行的 HTTP 服务。

如何选择后端
------------

- **LLM 与智能体 RL（SGLang 或 vLLM）**：数学推理、检索与多轮工具调用智能体默认使用 SGLang，也可切换至 vLLM。跨节点或异步 server 模式见 :doc:`SGLang 服务与路由 <../sglang_server>`，安装与切换引擎版本见 :doc:`Rollout 引擎版本（SGLang 与 vLLM） <../version>`。
- **具身 VLA 模型（Hugging Face、PhyAI 或 ApxInf）**：具身示例默认使用内置 Hugging Face rollout。追求更快的 PI0.5 LIBERO PPO 训练吞吐时使用 :doc:`PhyAI <phyai>`，使用进程内 Rust/CUDA 后端评测 LIBERO π₀.₅ 时使用 :doc:`ApxInf <../apxinf>`。

指南列表
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 指南
     - 说明
   * - :doc:`Rollout 引擎版本（SGLang 与 vLLM） <../version>`
     - 在独立虚拟环境中安装并切换受支持的 SGLang 与 vLLM 版本。
   * - :doc:`SGLang 服务与路由 <../sglang_server>`
     - 启动一组 SGLang HTTP server 与一个 router，对外提供统一的 OpenAI 兼容 ``/generate`` 与 ``/v1/chat/completions`` 接口。
   * - :doc:`使用 InferenceHTTPClient 调用 SGLang <../inference_http_client>`
     - 在自己的代码里向 router 或单个 server 发送同步/异步的 ``/generate`` 与 ``/v1/chat/completions`` 请求。
   * - :doc:`PhyAI <phyai>`
     - 使用 FSDP actor 和 PhyAI 进程内 GPU rollout 后端，在 LIBERO 上训练 PI0.5。
   * - :doc:`ApxInf <../apxinf>`
     - 使用 ApxInf 进程内 Rust/CUDA 推理后端，在 LIBERO 上评测 π₀.₅。

.. toctree::
   :hidden:

   Rollout 引擎版本（SGLang 与 vLLM） <../version>
   SGLang 服务与路由 <../sglang_server>
   InferenceHTTPClient <../inference_http_client>
   PhyAI <phyai>
   ApxInf <../apxinf>
