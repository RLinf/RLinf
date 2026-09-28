Rollout 引擎
============

这些指南介绍如何配置 RLinf rollout 使用的推理引擎，包括进程内 GPU 后端，以及与
RLinf 任务一起启动或独立运行的 HTTP 服务。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 你能得到什么
   * - :doc:`ApxInf <../apxinf>`
     - 使用 ApxInf 的进程内 Rust/CUDA 推理后端，在 LIBERO 上评测 π₀.₅。
   * - :doc:`SGLang Server 与 Router <../sglang_server>`
     - 启动一组 sglang HTTP server 与一个 sglang router，对外暴露统一的、兼容 OpenAI 风格的
       ``/generate`` 与 ``/v1/chat/completions`` 接口。
   * - :doc:`使用 InferenceHTTPClient 调用 SGLang <../inference_http_client>`
     - 在自己的代码里向 router（或单个 server）发送同步/异步的
       ``/generate`` / ``/v1/chat/completions`` 请求。
   * - :doc:`SGLang 版本切换 <../version>`
     - 在不同 SGLang 版本之间切换 rollout 引擎。

.. toctree::
   :hidden:

   ApxInf <../apxinf>
   SGLang Server 与 Router <../sglang_server>
   InferenceHTTPClient <../inference_http_client>
   SGLang 版本切换 <../version>
