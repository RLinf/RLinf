Rollout 后端与服务
==================

需要切换或调优推理引擎时查阅本组指南：管理 SGLang 与 vLLM 版本、部署 SGLang 推理服务、通过 HTTP 客户端接入外部推理，或用 PhyAI / ApxInf 加速具身 VLA 策略 Rollout。

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: SGLang 与 vLLM 版本选择
      :link: ../version
      :link-type: doc

      选择与当前 PyTorch 和任务兼容的 SGLang / vLLM 版本。

   .. grid-item-card:: SGLang 服务与路由
      :link: ../sglang_server
      :link-type: doc

      通过 RLinf 的 SGLang 服务与路由层承载 LLM 或具身模型推理。

   .. grid-item-card:: 异步 HTTP 推理客户端
      :link: ../inference_http_client
      :link-type: doc

      以异步 HTTP 方式对接外部或远程推理服务。

   .. grid-item-card:: Session Server
      :link: ../session_server
      :link-type: doc

      在 rollout 节点上收集 Agent 轨迹，提供 SGLang 路由与 OpenAI chat completions 接口。

   .. grid-item-card:: PhyAI 具身推理后端
      :link: phyai
      :link-type: doc

      在 LIBERO 上用 CUDA Graph 与 FlashInfer 加速 OpenPI π₀.₅ PPO Rollout。

   .. grid-item-card:: ApxInf 具身评测后端
      :link: ../apxinf
      :link-type: doc

      在 LIBERO 上用进程内 Rust/CUDA 后端加速 OpenPI π₀.₅ 评测。

.. toctree::
   :hidden:

   SGLang 与 vLLM 版本选择 <../version>
   SGLang 服务与路由 <../sglang_server>
   异步 HTTP 推理客户端 <../inference_http_client>
   Session Server <../session_server>
   PhyAI 具身推理后端 <phyai>
   ApxInf 具身评测后端 <../apxinf>
