Rollout Backends
================

Pick these guides when selecting or tuning a rollout engine: switching SGLang and vLLM versions, running a standalone SGLang router/server, calling inference over HTTP, or accelerating embodied VLA rollouts with PhyAI and ApxInf.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: SGLang & vLLM Versions
      :link: ../version
      :link-type: doc

      Select compatible SGLang, vLLM, and PyTorch versions for your environment.

   .. grid-item-card:: SGLang Server & Routing
      :link: ../sglang_server
      :link-type: doc

      Serve LLMs and embodied policies through RLinf's SGLang server and router.

   .. grid-item-card:: Inference HTTP Client
      :link: ../inference_http_client
      :link-type: doc

      Send asynchronous rollout requests to external or remote inference servers.

   .. grid-item-card:: Session Server
      :link: ../session_server
      :link-type: doc

      Collect agent trajectories on rollout nodes with SGLang routing and an OpenAI chat completions endpoint.

   .. grid-item-card:: PhyAI Rollout Backend
      :link: phyai
      :link-type: doc

      Accelerate OpenPI π₀.₅ PPO rollouts on LIBERO with CUDA graphs and FlashInfer.

   .. grid-item-card:: ApxInf Rollout Backend
      :link: ../apxinf
      :link-type: doc

      Run in-process Rust/CUDA rollouts for OpenPI π₀.₅ evaluation on LIBERO.

.. toctree::
   :hidden:

   SGLang & vLLM Versions <../version>
   SGLang Server & Routing <../sglang_server>
   Inference HTTP Client <../inference_http_client>
   Session Server <../session_server>
   PhyAI Rollout Backend <phyai>
   ApxInf Rollout Backend <../apxinf>
