SGLang 服务与路由
=================

通过一份 Hydra 配置启动一组 SGLang HTTP server 与单个 SGLang router，让 rollout 或外部客户端只需访问一个 URL，由 router 将请求分发到各个推理引擎。

**操作流程：** 在 ``cluster.component_placement`` 中声明放置 → 填写 server 与 router 参数块 → 在入口脚本中拉起服务与 router → 通过 :doc:`InferenceHTTPClient <inference_http_client>` 向 router URL 发请求。

配置
----

配置分为两部分：``cluster.component_placement`` 决定引擎 **在哪里** 运行，参数块决定每个引擎与 router **如何** 配置：

.. note::

   :doc:`session server <session_server>` 按 ``rollout.router`` 配置块启动自己的 router。这个 router 与本页 ``router_server_args`` 配置的 router 相互独立。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       rollout: all             # 任意组件名

   router_server_args:          # 任意顶层配置名
     model_path: /path/to/hf_model
     tensor_parallel_size: 2
     pipeline_parallel_size: 1
     server:                    # 透传为 SGLang ServerArgs
       model_path: ${..model_path}
       tp_size: ${..tensor_parallel_size}
       pp_size: ${..pipeline_parallel_size}
     router:                    # 透传为 SGLang router CLI 参数
       policy: cache_aware

.. dropdown:: 完整配置（含可选调优项）

   .. code-block:: yaml

      cluster:
        num_nodes: 1
        component_placement:
          rollout: all

      router_server_args:
        model_path: /path/to/hf_model
        tensor_parallel_size: 2
        pipeline_parallel_size: 1

        group_name: SGLangServerGroup
        launch_server: True
        server:
          model_path: ${..model_path}
          tp_size: ${..tensor_parallel_size}
          pp_size: ${..pipeline_parallel_size}
          mem_fraction_static: 0.85
          max_running_requests: 64
          attention_backend: triton
          log_level: warning

        router_group_name: SGLangRouterGroup
        launch_router: True
        router:
          policy: cache_aware
          log_level: warn
          worker_startup_timeout_secs: 1800
          request_timeout_secs: 1800

组件名
~~~~~~

``cluster.component_placement`` 下的字段名（上例中的 ``rollout``）只是自定义标签，按 runner 习惯起名即可。launcher 不会查找固定名称，由入口脚本选定名称并在三处复用同一字符串：

1. 作为 ``cluster.component_placement`` 下的字段名。
2. 从 placement 对象查询硬件 rank。
3. 读取该组件的可选 ``node_group`` 配置。

把组件名定义为常量并在三处复用，后续改名只需修改 YAML 字段和这一个 Python 变量：

.. code-block:: python

   from omegaconf import DictConfig

   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang import launch_sglang_router_and_server

   placement = ComponentPlacement(cfg, cluster)

   component_name = "rollout"
   llm_cfg = cfg.cluster.component_placement.get(component_name)
   rollout_node_group = (
       llm_cfg.get("node_group", None)
       if isinstance(llm_cfg, DictConfig)
       else None
   )

   server_group, router_group = launch_sglang_router_and_server(
       cfg,
       cluster,
       rollout_hardware_ranks=placement.get_hardware_ranks(component_name),
       router_server_args=cfg.router_server_args,
       rollout_node_group=rollout_node_group,
   )

.. note::

   launcher 只接收解析后的硬件 rank 与 node-group 字符串。只要 YAML 字段名与 ``component_name`` 一致，命名为 ``rollout``、``server`` 或其他名称均可。

Server 与 Router 参数块
~~~~~~~~~~~~~~~~~~~~~~~

``router_server_args`` 是传给 launcher 的顶层 YAML 配置块，其名称同样可以自定义（如 ``cfg.rollout``、``cfg.my_engine``），只要内部包含以下字段即可：

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - 字段
     - 类型
     - 作用
   * - ``tensor_parallel_size``
     - int
     - 单引擎 TP 大小，每个引擎打包 ``tp_size × pp_size`` 张卡。
   * - ``pipeline_parallel_size``
     - int
     - 单引擎 PP 大小。
   * - ``server_type``
     - str
     - 选择 server 子进程调用的 SGLang 启动入口（默认 ``srt``）：``srt`` 用于语言模型，走 ``sglang.srt.entrypoints.http_server.launch_server``；``embodied`` 用于具身模型（VLA/diffusion），走 ``sglang.multimodal_gen.runtime.launch_server.dispatch_launch``。传入其他值会报错。
   * - ``group_name``
     - str
     - SGLang server worker 组的名称。
   * - ``launch_server``
     - bool
     - 设为 ``False`` 可跳过启动 server（例如挂载已有外部服务）。
   * - ``server``
     - dict
     - 原样透传给 SGLang 的 ``ServerArgs``：``srt`` 调用 ``sglang.srt.server_args.ServerArgs(**)``，``embodied`` 调用 ``sglang.multimodal_gen.runtime.server_args.ServerArgs.from_kwargs(**)``。字段名须为对应 ``ServerArgs`` 的合法项，参见 `SGLang ServerArgs 参考 <https://docs.sglang.io/docs/advanced_features/server_arguments>`_。例外：``server.num_gpus`` 也会被 launcher 用于计算单引擎占用的 GPU 数量。
   * - ``router_group_name``
     - str
     - router worker 组的名称。
   * - ``launch_router``
     - bool
     - 设为 ``False`` 可跳过启动 router（例如仅需原始 server URL 列表）。
   * - ``router``
     - dict
     - 以 ``--<field>`` 形式传给 ``sglang_router.launch_router``。字段名须为 ``RouterArgs`` 的合法项，参见 `RouterArgs 源码 <https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py>`_。

.. note::

   仓库在 ``examples/embodiment/config/hybrid_engines/sglang_router_server.yaml`` 提供了一份默认配置。可在 Hydra 的 ``defaults`` 列表中引入，只覆盖需要修改的字段（通常是 ``model_path`` 及少量 ``server`` / ``router`` 参数）：

   .. code-block:: yaml

      defaults:
        - hybrid_engines/sglang_router_server@router_server_args

      router_server_args:
        model_path: /path/to/hf_model
        # tensor_parallel_size: 2
        # server:
        #   mem_fraction_static: 0.9

   ``@router_server_args`` 是 Hydra 的 package override 语法，用于将默认配置挂载到 ``router_server_args`` 顶层键下。

.. warning::

   launcher 不会预先校验 ``server`` 与 ``router`` 块内的字段。``server`` 中的无效字段会在 SGLang 解析参数时报错；``router`` 中的无效字段则会由 launcher 抛出 ``ValueError`` 并列出所有合法字段。将这两个块视为向上游 SGLang 的直接透传即可。

server 的 ``host``、``port``、``dist_init_addr`` 以及 router 端口均在运行时自动填入，YAML 中留空即可。router 与 server 默认都绑定 ``0.0.0.0``，每个 server 会把所在 Ray 节点的 IP 报告给 router。

Launcher 启动流程
~~~~~~~~~~~~~~~~~

launcher 按以下四步拉起推理服务：

1. 将扁平硬件 rank 打包为紧凑排布，每个进程占用 ``tp_size × pp_size`` 张卡，即一个进程运行一个 SGLang 引擎。
2. 按配置的组名与 ``server`` 参数启动 server worker 组。
3. 在 node 0 按配置的组名与 ``router`` 参数启动单个 router worker，初始不挂载后端。
4. 收集每个 server 的地址并通过 ``POST /workers`` 注册到 router，阻塞等待所有后端进入健康状态。

扁平 rank 打包要求硬件 rank 连续。如需非连续布局（例如 ``placement: 0-1:0-3,3-5``），自行构造 placement 策略并通过 ``placement_strategy`` 传入，即可跳过重打包。

启动服务
--------

串联上述配置的最小 Hydra 入口脚本：

.. code-block:: python

   import hydra
   from omegaconf import DictConfig

   from rlinf.scheduler import Cluster
   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang import launch_sglang_router_and_server


   @hydra.main(version_base="1.1", config_path="config", config_name="my_config")
   def main(cfg: DictConfig) -> None:
       cluster = Cluster(cluster_cfg=cfg.cluster)
       placement = ComponentPlacement(cfg, cluster)

       component_name = "rollout"
       llm_cfg = cfg.cluster.component_placement.get(component_name)
       rollout_node_group = (
           llm_cfg.get("node_group", None)
           if isinstance(llm_cfg, DictConfig)
           else None
       )

       server_group, router_group = launch_sglang_router_and_server(
           cfg,
           cluster,
           rollout_hardware_ranks=placement.get_hardware_ranks(component_name),
           router_server_args=cfg.router_server_args,
           rollout_node_group=rollout_node_group,
       )

       router_url = router_group.get_router_url().wait()[0]
       # ... 使用 router_url（见下文 “后续：调用 Router”） ...

       router_group.stop().wait()
       server_group.stop().wait()


   if __name__ == "__main__":
       main()

这段脚本先初始化 Ray 集群，再从 ``cluster.component_placement`` 解析组件放置，拉起所有 SGLang 引擎与 router，并等待全部后端注册完成且通过健康检查。启动调用返回后，即可通过 ``router_url`` 访问 router。

异构集群
--------

需要将引擎固定到带标签的节点组（例如将推理节点专门留给 rollout、其他节点跑训练）时，在 ``cluster`` 下配置 ``node_group`` 与 ``node_groups``：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       rollout:
         node_group: rollout_gpu       # 落在哪个节点组
         placement: all                # 使用该组内所有硬件 rank
     node_groups:
       - label: rollout_gpu
         node_ranks: 1                 # node rank 1 运行引擎
       - label: test
         node_ranks: 0                 # node rank 0 留给其他任务

   router_server_args:
     # ... 与单机配置一致 ...

与扁平配置相比的变化：

- ``component_placement.rollout`` 从单个字符串变为包含 ``node_group`` 与 ``placement`` 的映射。
- 入口脚本从同一个组件配置中读取节点组标签，并作为 ``rollout_node_group`` 传入 launcher。
- launcher 将该标签传给紧凑排布策略，使所有 server 进程落在目标节点上。

如果修改了 ``rollout`` 这个组件名，只需同步更新 YAML 字段名和入口脚本里的 ``component_name`` 变量。``node_group`` 标签与 ``node_groups`` 列表属于集群自身的定义，彼此独立。

引擎需要横跨多个节点组时，也可传入 **标签列表**，launcher 会原样转发给 placement 策略。

完整的 ``node_groups``、``env_configs`` 与 ``hardware`` 配置说明（包括按节点组指定 Python 解释器、环境变量以及机器人等非加速器硬件）见 :doc:`异构集群 <hetero>`。

编程接口
--------

如果 runner 已经自行构造好 placement 策略，可以跳过扁平 rank 重打包，直接传入该策略。策略须已按每个进程 ``tp_size × pp_size`` 张卡编排：

.. code-block:: python

   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang import launch_sglang_router_and_server

   placement = ComponentPlacement(cfg, cluster)
   component_name = "rollout"
   server_group, router_group = launch_sglang_router_and_server(
       cfg,
       cluster,
       rollout_hardware_ranks=None,        # 传入 placement_strategy 时忽略此项
       router_server_args=cfg.router_server_args,
       placement_strategy=placement.get_strategy(component_name),
       router_node_rank=0,                 # router 运行在哪个节点
   )

返回的句柄还提供以下常用方法：

- ``server_group.get_server_url().wait()``：获取 server URL 列表。
- ``router_group.register_server(url).wait()``：启动后动态挂载 server URL（阻塞至该后端健康）。
- ``router_group.unregister_server(url).wait()``：卸载指定 server URL。
- ``router_group.get_router_url().wait()[0]``：获取 router URL。

后续：调用 Router
-----------------

启动调用返回后，即可用任意 HTTP 客户端访问 ``router_url``。同步与异步调用 ``/generate`` 及 ``/v1/chat/completions`` 接口的示例见 :doc:`使用 InferenceHTTPClient 调用 SGLang <inference_http_client>`。
