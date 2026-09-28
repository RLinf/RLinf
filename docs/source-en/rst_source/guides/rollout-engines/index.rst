Rollout Engines
===============

Use these guides to configure the inference engines behind RLinf rollouts,
including in-process GPU backends and HTTP services that run alongside or
independently of an RLinf task.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
   * - :doc:`ApxInf <../apxinf>`
     - Evaluate π₀.₅ on LIBERO with ApxInf's in-process Rust/CUDA inference
       backend.
   * - :doc:`SGLang Server & Router <../sglang_server>`
     - Launch an sglang HTTP server group and an sglang router, with a single
       OpenAI-compatible endpoint for ``/generate`` and ``/v1/chat/completions``.
   * - :doc:`Calling SGLang with InferenceHTTPClient <../inference_http_client>`
     - Send sync and async ``/generate`` / ``/v1/chat/completions`` requests to
       a router (or a single server) from your own code.
   * - :doc:`SGLang Version Switching <../version>`
     - Switch between SGLang versions for the rollout engine.

.. toctree::
   :hidden:

   ApxInf <../apxinf>
   SGLang Server & Router <../sglang_server>
   InferenceHTTPClient <../inference_http_client>
   SGLang Version Switching <../version>
