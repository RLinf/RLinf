Rollout 引擎
============

这些指南介绍如何启动 RLinf rollout 通过 HTTP 访问的推理引擎——既可以与训练任务一起拉起，也可以独立于训练运行——以及对应的客户端调用方式。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 你能得到什么
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

   SGLang Server 与 Router <../sglang_server>
   InferenceHTTPClient <../inference_http_client>
   SGLang 版本切换 <../version>
