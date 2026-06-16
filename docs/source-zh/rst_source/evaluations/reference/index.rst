参考
====

本节汇总评测参考文档，涵盖配置结构、命令行用法、支持的模型，以及如何查看评测输出。

- :doc:`configuration`
   ``evaluations/<benchmark>/`` 下的 Hydra YAML 结构，以及 ``embodied_eval`` 所需的 ``runner``、``env``、``rollout`` 等字段说明。

- :doc:`cli`
   如何使用 ``run_eval.sh`` 启动评测、传入 Hydra 覆盖参数，以及从配置名自动推断 benchmark。

- :doc:`models`
   当前 ``evaluations/`` 中提供示例配置的 VLA 模型（OpenPI、OpenVLA-OFT、StarVLA、DreamZero、LingBotVLA）及 ``model_path`` 设置方式。

- :doc:`results`
   日志与 rollout 视频的输出路径、终端指标（如 ``eval/success_once``）以及 TensorBoard 查看方式。

.. toctree::
   :hidden:
   :maxdepth: 1

   configuration
   cli
   models
   results
