SGLang Server 与 Router
=======================

通过一份 Hydra 配置启动一组 sglang HTTP server 与一个 sglang router——rollout
（或任意外部客户端）只需访问一个 URL，由 router 将请求分发到各个引擎。

**你将完成：** 在 ``cluster.component_placement`` 中声明放置 → 填写一个参数块
（包含 sglang ``ServerArgs`` 与 ``RouterArgs`` 的字段）→ 调用
:func:`launch_sglang_router_and_server` → 用 :class:`InferenceHTTPClient`
与 router URL 通信。

配置
----

配置分两部分：``cluster.component_placement`` 负责 **在哪里** 运行引擎；
参数块负责 **每个引擎与 router 如何** 配置。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       rollout: all             # <-- "rollout" 这个 key 是任意的，含义见下文

   router_server_args:          # <-- 顶层 key 名字也是任意的，含义见下文
     model_path: /path/to/hf_model
     tensor_parallel_size: 2
     pipeline_parallel_size: 1

     group_name: SGLangServerGroup
     launch_server: True
     server:                    # 原样作为 ServerArgs(**) 的参数
       model_path: ${..model_path}
       tp_size: ${..tensor_parallel_size}
       pp_size: ${..pipeline_parallel_size}
       mem_fraction_static: 0.85
       max_running_requests: 64
       attention_backend: triton
       log_level: warning

     router_group_name: SGLangRouterGroup
     launch_router: True
     router:                    # 原样作为 launch_router 的 CLI flag
       policy: cache_aware
       log_level: warn
       worker_startup_timeout_secs: 1800
       request_timeout_secs: 1800

组件名（``rollout`` 这个 key）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cluster.component_placement`` 下的 key（上面的 ``rollout``）**只是一个标签**，
按 runner 的习惯起名即可。Launcher 不会按固定名字去查找这个 key；**驱动脚本**
负责挑选名字，并在三处使用 *同一字符串*：

1. 作为 ``cluster.component_placement`` 下的 key。
2. 向 placement 询问引擎硬件 rank：
   ``placement.get_hardware_ranks(component_name)``。
3. 读取该组件的 node-group 标签（若有）：
   ``cfg.cluster.component_placement.get(component_name)``。

把名字定义为常量并在三处复用——这样改名时只需修改 YAML key 和这一个 Python 常量：

.. code-block:: python

   from omegaconf import DictConfig

   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

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

   Launcher 自身看不到这个名字，它只接收解析后的硬件 rank 和 node-group 字符串。
   只要 YAML key 与 ``component_name`` 一致，叫 ``rollout``、``server``、
   ``my_engine`` 还是别的都没区别。

参数块（``router_server_args`` 这个 key）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``router_server_args`` 是你传给 launcher 的 **顶层 YAML key**，名字同样可以任意，
重要的是 **块内部的结构**。完全可以传入其他子配置（``cfg.rollout``、
``cfg.my_engine`` 等），只要它带有 launcher 需要消费的字段：

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - 字段
     - 类型
     - 作用
   * - ``tensor_parallel_size``
     - int
     - 单引擎 TP 大小。``tp_size × pp_size`` 张卡打包成一个引擎。
   * - ``pipeline_parallel_size``
     - int
     - 单引擎 PP 大小。
   * - ``group_name``
     - str
     - sglang server worker group 的名字。
   * - ``launch_server``
     - bool
     - 设为 ``False`` 可跳过 server group（例如挂载已有的外部 server）。
   * - ``server``
     - dict
     - **原样** 作为 ``sglang.srt.server_args.ServerArgs(**)`` 的关键字参数；
       key 必须是 ``ServerArgs`` 合法字段名——参见 `sglang ServerArgs 参考
       <https://docs.sglang.io/docs/advanced_features/server_arguments>`_。
   * - ``router_group_name``
     - str
     - router worker 的 worker group 名字。
   * - ``launch_router``
     - bool
     - 设为 ``False`` 可跳过 router（例如只想拿到原始 server URL 列表）。
   * - ``router``
     - dict
     - 作为 ``--<field>`` CLI flag 传给 ``sglang_router.launch_router``；
       key 必须是 ``RouterArgs`` 合法字段名——参见 `RouterArgs 源码
       <https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py>`_。

.. note::

   仓库已在
   ``examples/embodiment/config/training_backend/sglang_router_server.yaml``
   中提供了一份默认配置。可通过 Hydra 的 ``defaults`` 列表引用它，然后只覆盖
   你需要改动的字段——通常是 ``model_path``（以及 ``server`` / ``router`` 里的
   个别项）：

   .. code-block:: yaml

      defaults:
        - training_backend/sglang_router_server@router_server_args

      router_server_args:
        model_path: /path/to/hf_model
        # tensor_parallel_size: 2       # 其他字段按需覆盖
        # server:
        #   mem_fraction_static: 0.9

   ``@router_server_args`` 是 Hydra 的 package override 语法——它把默认配置挂到
   顶层 key ``router_server_args`` 下。

.. warning::

   Launcher 不会校验 ``server`` 与 ``router`` 块内的 key。``server`` 里有非法字段会让
   ``ServerArgs(**kwargs)`` 报错；``router`` 里有非法字段则会由 launcher 抛出
   ``ValueError`` 并列出所有合法字段。把这两个块当作向上游 dataclass 的直接透传即可。

server 的 ``host``/``port``/``dist_init_addr``，以及 router 的 ``port``，都会在
运行时填入——在 YAML 里留空即可。Router 默认绑定 ``0.0.0.0``；server 同样绑定
``0.0.0.0`` 并把 Ray 节点 IP 报告给 router。

Launcher 如何使用这份配置
~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`rlinf.workers.rollout.sglang_server.launch_sglang_router_and_server`
顺序做了以下事情：

1. 将扁平的 ``rollout_hardware_ranks`` 重新打包成
   ``PackedPlacementStrategy``，每个 process 占 ``tp_size × pp_size`` 张卡——
   一个 process 即一个 sglang 引擎。
2. 启动 ``SGLangServerWorker`` group（配置取自 ``group_name`` 与 ``server``）。
3. 在 node 0 启动单个 ``SGLangRouterWorker``，初始不挂载任何 worker
   （配置取自 ``router_group_name`` 与 ``router``）。
4. 收集每个 server 的 ``http://host:port`` 并通过 ``POST /workers`` 注册到
   router，阻塞直到每个 worker 报告 ``is_healthy=true``。

扁平 rank 重打包有一个 **硬性要求**：硬件 rank 必须是连续的。若需要非连续布局
（例如由 ``placement: 0-1:0-3,3-5`` 解析出的 ``FlexiblePlacementStrategy``），
请自行构造 strategy，通过 ``placement_strategy=...`` 传入——
该参数会短路重打包逻辑，原样使用你给的 strategy。

启动
----

把上面的配置串起来的最小 Hydra 入口：

.. code-block:: python

   import hydra
   from omegaconf import DictConfig

   from rlinf.scheduler import Cluster
   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server


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
       # ... 使用 router_url（见下文 “后续：调用 router”） ...

       router_group.shutdown().wait()
       server_group.shutdown().wait()


   if __name__ == "__main__":
       main()

这段代码做了什么：通过 ``Cluster`` 启动 Ray，从 ``cluster.component_placement``
构造 ``ComponentPlacement``，再让 launcher 拉起引擎与 router，等所有 server 都
注册成功且健康为止。``launch_sglang_router_and_server`` 返回之后，router 已可
通过 ``router_url`` 访问，且所有后端均已通过 ``GET /workers/<id>``。

异构集群（``node_group``）
--------------------------

如需把引擎绑定到一组带标签的节点——例如把推理节点专门留给 rollout，训练放别处
运行——使用 ``cluster`` 下的 ``node_group``/``node_groups`` 形式：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       rollout:
         node_group: rollout_gpu       # 落在哪个 group
         placement: all                # 使用该 group 内所有硬件 rank
     node_groups:
       - label: rollout_gpu
         node_ranks: 1                 # node rank 1 承载引擎
       - label: test
         node_ranks: 0                 # node rank 0 留给其他工作

   router_server_args:
     # ... 与上面的单机配置一致 ...

与扁平配置相比的变化：

- ``component_placement.rollout`` 由一个字符串变成一个 **mapping**，包含
  ``node_group`` 与 ``placement``。
- 驱动脚本从 launcher 所期望接收硬件 rank 的同一个 key 上读出标签，即
  ``cfg.cluster.component_placement.get(component_name).node_group``。
- 该标签作为 ``rollout_node_group`` 传给 launcher；launcher 再把它注入到重打包
  的 ``PackedPlacementStrategy`` 中，从而把 server 调度到正确的节点。

如果把组件 key 从 ``rollout`` 改成别的（例如 ``my_engine``），**只需** 同步修改
YAML key 和驱动里的 ``component_name`` 变量。``node_group`` 标签
（这里是 ``rollout_gpu``）和 ``node_groups`` 下的条目是独立的——
那是集群本身的命名空间。

也可以传入 **标签列表**，让引擎横跨多个
group；launcher 会原样把列表交给 placement strategy。

完整的 ``node_groups`` / ``env_configs`` / ``hardware`` schema——
包含按 node group 指定 Python 解释器、环境变量、以及非加速器硬件（机器人）的写法——
参见 :doc:`异构集群 <hetero>`。

编程接口
--------

如果你的 runner 已经自己构造好 placement strategy，可以跳过扁平 rank 路径，
直接传入一个 strategy。该 strategy 必须已编码 ``tp_size × pp_size`` 张卡/process：

.. code-block:: python

   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

   placement = ComponentPlacement(cfg, cluster)
   component_name = "rollout"
   server_group, router_group = launch_sglang_router_and_server(
       cfg,
       cluster,
       rollout_hardware_ranks=None,        # placement_strategy 已传入时会被忽略
       router_server_args=cfg.router_server_args,
       placement_strategy=placement.get_strategy(component_name),
       router_node_rank=0,                 # router 落在哪个 node
   )

返回的句柄上还有其他常用入口：

- ``server_group.get_server_url().wait()`` → server URL 列表。
- ``router_group.register_server(url).wait()`` → 事后把一个 server URL 挂到 router
  上（阻塞直到该 worker 就绪）。
- ``router_group.unregister_server(url).wait()`` → 摘下。
- ``router_group.get_router_url().wait()[0]`` → router URL。

后续：调用 router
-----------------

``launch_sglang_router_and_server`` 返回后，``router_url`` 即可通过普通 HTTP
访问——任何 HTTP 客户端都行。``/generate`` 与 ``/v1/chat/completions``
（同步与异步）的具体调用方式，请参见配套指南：
:doc:`使用 InferenceHTTPClient 调用 SGLang <inference_http_client>`。
