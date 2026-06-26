Rollout Engines
===============

Use these guides to bring up the inference engines RLinf rollouts talk to over
HTTP — server-side processes you launch alongside (or independently of) your
training run, plus the client used to call them.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - What you get
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

   SGLang Server & Router <../sglang_server>
   InferenceHTTPClient <../inference_http_client>
   SGLang Version Switching <../version>
