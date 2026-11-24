SGLang Server & Routing
=======================

Launch one or more SGLang HTTP servers behind a single SGLang router from a
Hydra config, so rollouts or external clients call one URL and let the router
fan requests out across engines.

**Workflow:** declare placement in ``cluster.component_placement`` → fill out
the server and router config block → start the server group and router from
your entry script → send requests to the router URL with
:doc:`InferenceHTTPClient <inference_http_client>`.

Configuration
-------------

The config has two parts: ``cluster.component_placement`` defines **where** the
engines run, and the args block defines **how** each engine and the router are
configured:

.. note::

   The :doc:`session server <session_server>` starts its own router from
   the ``rollout.router`` block. That block is separate from the
   ``router_server_args`` described here.

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       rollout: all             # arbitrary component label

   router_server_args:          # arbitrary top-level key
     model_path: /path/to/hf_model
     tensor_parallel_size: 2
     pipeline_parallel_size: 1
     server:                    # forwarded as SGLang ServerArgs
       model_path: ${..model_path}
       tp_size: ${..tensor_parallel_size}
       pp_size: ${..pipeline_parallel_size}
     router:                    # forwarded as SGLang router CLI flags
       policy: cache_aware

.. dropdown:: Full configuration with optional tuning

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

Component Name
~~~~~~~~~~~~~~

The key under ``cluster.component_placement`` (``rollout`` above) is an
arbitrary label that you choose to match your runner. The launcher does not
look up a hard-coded name; your entry script picks the string and reuses it in
three places:

1. As the key under ``cluster.component_placement``.
2. When querying hardware ranks from the placement helper.
3. When reading the optional ``node_group`` setting for that component.

Define the name once and reuse it so a rename touches only the YAML key and one
Python variable:

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

   The launcher only sees the resolved hardware ranks and node-group string. As
   long as the YAML key and ``component_name`` match, you can name it
   ``rollout``, ``server``, or any other label.

Server and Router Config Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``router_server_args`` is the top-level YAML block passed to the launcher. Its
top-level name is also arbitrary—pass ``cfg.rollout``, ``cfg.my_engine``, or
any sub-config that provides the keys below:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Type
     - What it does
   * - ``tensor_parallel_size``
     - int
     - Per-engine TP size. ``tp_size × pp_size`` GPUs are packed into one
       engine.
   * - ``pipeline_parallel_size``
     - int
     - Per-engine PP size.
   * - ``server_type``
     - str
     - Selects which SGLang entrypoint the server subprocess runs (default
       ``srt``): ``srt`` launches language models via
       ``sglang.srt.entrypoints.http_server.launch_server``, and ``embodied``
       launches embodied (VLA/diffusion) models via
       ``sglang.multimodal_gen.runtime.launch_server.dispatch_launch``. Other
       values raise an error.
   * - ``group_name``
     - str
     - Worker-group name for the SGLang server group.
   * - ``launch_server``
     - bool
     - Set ``False`` to skip starting servers (for example, to attach an
       external server).
   * - ``server``
     - dict
     - Forwarded verbatim to SGLang's ``ServerArgs``: ``srt`` uses
       ``sglang.srt.server_args.ServerArgs(**)`` and ``embodied`` uses
       ``sglang.multimodal_gen.runtime.server_args.ServerArgs.from_kwargs(**)``.
       Keys must match valid ``ServerArgs`` fields—see the `SGLang ServerArgs
       reference <https://docs.sglang.io/docs/advanced_features/server_arguments>`_.
       Exception: ``server.num_gpus`` is also read by the launcher to determine
       accelerators per engine when packing placement.
   * - ``router_group_name``
     - str
     - Worker-group name for the router worker.
   * - ``launch_router``
     - bool
     - Set ``False`` to skip starting the router when you only need raw server
       URLs.
   * - ``router``
     - dict
     - Forwarded as ``--<field>`` CLI flags to ``sglang_router.launch_router``.
       Keys must match valid ``RouterArgs`` fields—see the `RouterArgs source
       <https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py>`_.

.. note::

   A default config is shipped at
   ``examples/embodiment/config/hybrid_engines/sglang_router_server.yaml``.
   Include it through Hydra's ``defaults`` list and override only the fields
   you change (typically ``model_path`` and optional tuning under ``server`` or
   ``router``):

   .. code-block:: yaml

      defaults:
        - hybrid_engines/sglang_router_server@router_server_args

      router_server_args:
        model_path: /path/to/hf_model
        # tensor_parallel_size: 2
        # server:
        #   mem_fraction_static: 0.9

   The ``@router_server_args`` package override mounts the defaults under the
   ``router_server_args`` top-level key.

.. warning::

   The launcher does not pre-validate keys inside ``server`` or ``router``. An
   unknown server key fails when SGLang parses its server arguments, and an
   unknown router key raises a ``ValueError`` listing all valid fields. Treat
   both blocks as direct pass-throughs to SGLang.

Leave network address fields (``host``, ``port``, and ``dist_init_addr`` on the
server, plus the router port) unset in YAML; the launcher assigns them at
runtime. Both the router and servers bind on ``0.0.0.0``, and each server
advertises its Ray node IP back to the router.

How the Launcher Starts Them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The launcher brings up the inference stack in four steps:

1. Packs the flat hardware ranks into a contiguous placement with
   ``tp_size × pp_size`` accelerators per process—one SGLang engine per
   process.
2. Starts the server worker group using the configured group name and
   ``server`` arguments.
3. Starts a single router worker on node 0 using the configured router group
   name and ``router`` arguments, initially with no backends attached.
4. Collects each server URL, registers it with the router via
   ``POST /workers``, and waits until every backend reports healthy.

Flat-rank packing requires contiguous hardware ranks. For a non-contiguous
layout (such as ``placement: 0-1:0-3,3-5``), build a placement strategy
yourself and pass it via ``placement_strategy`` to bypass repacking.

Launch It
---------

Minimal Hydra entry script that brings up the server group and router from the
config above:

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
       # ... use router_url (see "Next: Calling the Router" below) ...

       router_group.stop().wait()
       server_group.stop().wait()


   if __name__ == "__main__":
       main()

This script initializes the Ray cluster, resolves placement from
``cluster.component_placement``, starts all SGLang engines and the router, and
waits until every server is registered and healthy. Once the launcher call
returns, the router is reachable at ``router_url`` and every backend has passed
its health check.

Heterogeneous Clusters
----------------------

To pin engines to a labeled set of nodes—for example, dedicating inference
nodes to rollouts while training runs elsewhere—use ``node_group`` and
``node_groups`` under ``cluster``:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       rollout:
         node_group: rollout_gpu       # which group to land on
         placement: all                # use all hardware ranks in that group
     node_groups:
       - label: rollout_gpu
         node_ranks: 1                 # node rank 1 hosts the engines
       - label: test
         node_ranks: 0                 # node rank 0 is free for other work

   router_server_args:
     # ... same as the single-node config above ...

Compared with the flat config:

- ``component_placement.rollout`` is a mapping with ``node_group`` and
  ``placement`` instead of a single string.
- The entry script reads the node-group label from that same component entry
  and passes it as ``rollout_node_group``.
- The launcher forwards the label to the packed placement so every server
  process lands on the target nodes.

If you rename the component key from ``rollout`` to another name, update both
the YAML key and ``component_name`` in your entry script. The ``node_group``
label and the entries under ``node_groups`` belong to the cluster definition
and stay independent.

You can also pass a **list** of labels when engines should span multiple node
groups; the launcher forwards the list verbatim to placement.

For the full ``node_groups``, ``env_configs``, and ``hardware`` schema—including
per-group Python interpreters, environment variables, and non-accelerator
hardware such as robots—see :doc:`Heterogeneous Clusters <hetero>`.

Programmatic API
----------------

If your runner builds a placement strategy itself, skip flat-rank repacking and
pass that strategy directly. The strategy must already allocate
``tp_size × pp_size`` accelerators per process:

.. code-block:: python

   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang import launch_sglang_router_and_server

   placement = ComponentPlacement(cfg, cluster)
   component_name = "rollout"
   server_group, router_group = launch_sglang_router_and_server(
       cfg,
       cluster,
       rollout_hardware_ranks=None,        # ignored when placement_strategy is set
       router_server_args=cfg.router_server_args,
       placement_strategy=placement.get_strategy(component_name),
       router_node_rank=0,                 # which node hosts the router
   )

Additional methods on the returned handles:

- ``server_group.get_server_url().wait()``: list of server URLs.
- ``router_group.register_server(url).wait()``: attach a server URL after
  startup (blocks until the backend is healthy).
- ``router_group.unregister_server(url).wait()``: detach a server URL.
- ``router_group.get_router_url().wait()[0]``: router URL.

Next: Calling the Router
------------------------

Once the launcher returns, ``router_url`` is reachable over plain HTTP from any
client. For sending sync and async ``/generate`` and ``/v1/chat/completions``
requests, see :doc:`Calling SGLang with InferenceHTTPClient <inference_http_client>`.
