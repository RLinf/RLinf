日志与可视化
============

使用 TensorBoard、Weights & Biases（W&B）或 SwanLab 实时追踪训练损失、奖励、评测分数与步耗时。所有日志指标命名空间（``train/``、``rollout/``、``env/``、``eval/`` 与 ``time/``）的定义见 :doc:`训练指标 <../reference/metrics>`。

支持的后端
----------

- `TensorBoard <https://www.tensorflow.org/tensorboard>`_\ ：本地事件文件可视化，查看标量曲线、直方图与多次运行对比。
- `Weights & Biases (W&B) <https://wandb.ai/site/>`_\ ：云端实验追踪、artifact 版本管理与团队看板，支持离线记录后再同步。
- `SwanLab <https://pypi.org/project/swanlab/>`_\ ：轻量级云端或本地实验追踪工具，记录指标曲线、超参数与硬件状态。

启用日志后端
------------

在 YAML 配置的 ``runner.logger.logger_backends`` 中列出需要启用的后端：

.. code-block:: yaml

   runner:
     experiment_name: grpo-1.5b
     output_dir: ./logs
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}
       logger_backends: ["tensorboard", "wandb", "swanlab"]  # 选择任意子集

``runner.logger`` 还支持三个可选配置项：``wandb_entity``\ （W&B 团队名或用户名）、``wandb_proxy``\ （W&B 使用的 HTTPS 代理地址）以及 ``swanlab_mode``\ （支持 ``"cloud"``、``"local"``、``"offline"`` 或 ``"disabled"``，默认为 ``"cloud"``）。

RLinf 会在 ``runner.logger.log_path`` 下按后端分别创建子目录：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 子目录
     - 内容
   * - ``tensorboard/``
     - TensorBoard 事件文件以及解析后的 ``config.yaml``。
   * - ``wandb/``
     - W&B 本地运行目录与离线同步文件。
   * - ``swanlab/``
     - SwanLab 运行日志与事件文件。
   * - ``checkpoints/``
     - 保存的模型与优化器 checkpoint。

.. tip::

   三个后端可以并行运行。传入空列表 ``[]`` 可关闭指标日志。

TensorBoard
-----------

在运行日志目录上启动 TensorBoard，在本地查看标量曲线：

.. code-block:: bash

   tensorboard --logdir ./logs/grpo-1.5b/tensorboard --port 6006

在浏览器打开 ``http://localhost:6006`` 即可查看曲线。

远程查看
~~~~~~~~

在远程 GPU 服务器上启动 TensorBoard 时，绑定 ``0.0.0.0`` 以接受外部或隧道访问：

.. code-block:: bash

   tensorboard --logdir ./logs/grpo-1.5b/tensorboard --host 0.0.0.0 --port 6006

如果防火墙拦截了 ``6006`` 端口，可以在本地通过 SSH 端口转发，然后在本地浏览器打开 ``http://localhost:6006``：

.. code-block:: bash

   ssh -L 6006:localhost:6006 <user>@<remote_host>

Weights & Biases (W&B)
----------------------

在线模式
~~~~~~~~

1. 在 `wandb.ai <https://wandb.ai>`__ 创建账号并复制 API key。
2. 在启动训练的机器上执行一次登录：

   .. code-block:: bash

      wandb login

此后 RLinf 会按 ``runner.logger.project_name`` 与 ``runner.logger.experiment_name`` 自动创建运行记录并上传指标。需要上报到团队空间时配置 ``runner.logger.wandb_entity``，需要走 HTTPS 代理时配置 ``runner.logger.wandb_proxy``。

离线模式
~~~~~~~~

计算节点无法访问外网时，先启用 W&B 离线模式，将日志保存在 ``<log_path>/wandb/`` 下，避免阻塞训练：

.. code-block:: bash

   export WANDB_MODE=offline

训练结束后（或在可联网的登录节点上），将本地记录同步到 W&B：

.. code-block:: bash

   wandb sync ./logs/grpo-1.5b/wandb/wandb/offline-run-*

加上 ``--sync-all`` 可一次性同步目录下所有未上传的运行记录。

SwanLab
-------

1. 在 `swanlab.ai <https://swanlab.ai>`__ 注册账号并获取 access token。
2. 在启动机器上执行一次登录：

   .. code-block:: bash

      swanlab login

只需本地查看或不上传到 SwanLab Cloud 时，在 YAML 中将 ``runner.logger.swanlab_mode`` 设为 ``"local"`` 或 ``"offline"``。

在多节点 Ray 集群中传递 API 密钥
--------------------------------

RLinf 在 head 节点的 driver 进程中初始化日志记录器。在多节点集群或无交互式登录的调度作业中，应通过环境变量传入密钥，避免依赖交互式的 ``wandb login`` 或 ``swanlab login``。

启动 Ray 前导出环境变量
~~~~~~~~~~~~~~~~~~~~~~~

在 head 节点启动 Ray 及训练脚本前导出密钥（参见 :doc:`多节点 Ray 集群搭建 <multi_node>`）。Ray 会在 ``ray start`` 时冻结当前 shell 环境变量，该节点启动的 worker 也会继承这些变量：

.. code-block:: bash

   export WANDB_API_KEY=<your_wandb_api_key>
   export SWANLAB_API_KEY=<your_swanlab_api_key>

在 YAML 中配置环境变量
~~~~~~~~~~~~~~~~~~~~~~

在共享 Ray 集群中按作业注入环境变量时，可在配置中的 ``cluster.node_groups[].env_configs[].env_vars`` 下声明：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: "default"
         node_ranks: all
         env_configs:
           - node_ranks: all
             env_vars:
               - WANDB_API_KEY: "${oc.env:WANDB_API_KEY}"
               - SWANLAB_API_KEY: "${oc.env:SWANLAB_API_KEY}"

.. warning::

   不要将明文 API 密钥提交到 YAML 文件中。请使用 ``${oc.env:WANDB_API_KEY}`` 引用环境变量，或在作业提交脚本中导出。

相关文档
--------

- :doc:`训练指标 <../reference/metrics>`：``train/``、``rollout/``、``env/``、``eval/`` 与 ``time/`` 指标含义与排障范围。
- :doc:`基础配置 <basic_config>`：完整的 ``runner`` 与 ``cluster`` YAML 配置项。
