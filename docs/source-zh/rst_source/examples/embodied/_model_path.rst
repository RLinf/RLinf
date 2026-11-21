.. 本文件是可复用的 include 片段，并非独立页面。
   请在本示例的下载命令之后，用 ``.. include:: _model_path.rst`` 引入。

在配置 YAML 中将 ``actor.model.model_path`` 指向下载好的权重目录，或在启动命令后直接传入 ``actor.model.model_path=/path/to/checkpoint``。未单独设置 ``rollout.model.model_path`` 时，rollout 会自动沿用同一路径。
