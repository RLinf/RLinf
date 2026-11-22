Rollout Engines
===============

Use these guides to configure the inference engines behind RLinf rollouts, including in-process GPU backends and HTTP services that run alongside or independently of an RLinf task.

Which Backend to Pick
---------------------

- **LLM & agentic RL (SGLang or vLLM)**: pick SGLang (default) or vLLM for math reasoning, search, and multi-turn tool-use agents. Use :doc:`SGLang Server & Router <../sglang_server>` for multi-node or async server mode, and :doc:`Rollout Engine Versions (SGLang & vLLM) <../version>` to switch engine versions.
- **Embodied VLA models (Hugging Face, PhyAI, or ApxInf)**: embodied recipes default to the built-in Hugging Face rollout. Switch to :doc:`PhyAI <phyai>` for faster in-process LIBERO PPO training on PI0.5, or :doc:`ApxInf <../apxinf>` for in-process Rust/CUDA LIBERO evaluation on π₀.₅.

Guides
------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Guide
     - What you get
   * - :doc:`Rollout Engine Versions (SGLang & vLLM) <../version>`
     - Install and switch between supported SGLang and vLLM builds in separate virtual environments.
   * - :doc:`SGLang Server & Router <../sglang_server>`
     - Launch an SGLang HTTP server group and router with one OpenAI-compatible endpoint for ``/generate`` and ``/v1/chat/completions``.
   * - :doc:`Calling SGLang with InferenceHTTPClient <../inference_http_client>`
     - Send sync and async ``/generate`` and ``/v1/chat/completions`` requests to a router or single server from your own code.
   * - :doc:`PhyAI <phyai>`
     - Train PI0.5 on LIBERO with an FSDP actor and PhyAI's in-process GPU rollout backend.
   * - :doc:`ApxInf <../apxinf>`
     - Evaluate π₀.₅ on LIBERO with ApxInf's in-process Rust/CUDA inference backend.

.. toctree::
   :hidden:

   Rollout Engine Versions (SGLang & vLLM) <../version>
   SGLang Server <../sglang_server>
   InferenceHTTPClient <../inference_http_client>
   PhyAI <phyai>
   ApxInf <../apxinf>
