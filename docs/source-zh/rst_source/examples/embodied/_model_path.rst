.. 本文件是可复用的 include 片段，并非独立页面。
   请在本示例的下载命令之后，用 ``.. include:: _model_path.rst`` 引入。

在配置 YAML 中设置 ``actor.model.model_path``\ （rollout 会自动沿用同一路径），或在启动命令后传入 ``actor.model.model_path=/path/to/downloaded-checkpoint``：

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/downloaded-checkpoint
