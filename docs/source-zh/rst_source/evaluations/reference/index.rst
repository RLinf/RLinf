参考
====

本节用于查阅评测 YAML 字段、命令行用法、支持的模型与输出路径。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 配置
      :link: configuration
      :link-type: doc

      Hydra YAML 结构、``env.eval`` 字段说明以及从训练 YAML 派生评测配置的方法。

   .. grid-item-card:: CLI
      :link: cli
      :link-type: doc

      ``evaluations/run_eval.sh`` 语法、benchmark 前缀自动推断、配置回退与 Hydra 覆盖参数。

   .. grid-item-card:: 模型
      :link: models
      :link-type: doc

      ``evaluations/`` 下提供示例配置的模型与推理后端清单。

   .. grid-item-card:: 结果
      :link: results
      :link-type: doc

      日志目录、核心指标、rollout 视频输出路径与 TensorBoard 查看方式。

.. toctree::
   :hidden:
   :maxdepth: 1

   configuration
   cli
   models
   results
