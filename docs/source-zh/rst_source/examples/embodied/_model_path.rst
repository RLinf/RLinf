.. 本文件是可复用的 include 片段，并非独立页面。
   它保存各具身示例共享的"在配置中指向已下载检查点"这段完全相同的收尾内容。
   请在本方案特定的下载命令之后，用 ``.. include:: _model_path.rst`` 引入。

下载完成后，在配置 YAML 中指向该检查点——为 rollout 与 actor 两处模型设置相同的路径：

.. code:: yaml

   rollout:
      model:
         model_path: /path/to/downloaded-checkpoint
   actor:
      model:
         model_path: /path/to/downloaded-checkpoint
