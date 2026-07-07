SGLang Server & Router
======================

Launch one or more sglang HTTP servers and a single sglang router from a Hydra
config, so rollouts (or any external client) can hit one URL and let the router
fan out across engines.

**You'll do:** declare placement in ``cluster.component_placement`` → fill out
an args block (sglang ``ServerArgs`` + ``RouterArgs`` fields) → call
:func:`launch_sglang_router_and_server` → talk to the router URL with
:class:`InferenceHTTPClient`.

Configuration
-------------

The config has two halves: ``cluster.component_placement`` (**where** the
engines run) and the args block (**how** each engine and the router are
configured).

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       rollout: all             # <-- the key "rollout" is arbitrary; see below

   router_server_args:          # <-- the top-level key is arbitrary; see below
     model_path: /path/to/hf_model
     tensor_parallel_size: 2
     pipeline_parallel_size: 1

     group_name: SGLangServerGroup
     launch_server: True
     server:                    # forwarded as ServerArgs(**)
       model_path: ${..model_path}
       tp_size: ${..tensor_parallel_size}
       pp_size: ${..pipeline_parallel_size}
       mem_fraction_static: 0.85
       max_running_requests: 64
       attention_backend: triton
       log_level: warning

     router_group_name: SGLangRouterGroup
     launch_router: True
     router:                    # forwarded as launch_router CLI flags
       policy: cache_aware
       log_level: warn
       worker_startup_timeout_secs: 1800
       request_timeout_secs: 1800

Component name (the ``rollout`` key)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key under ``cluster.component_placement`` (``rollout`` above) is **just a
label** — pick whatever name fits your runner. The launcher does not look it
up by a hard-coded name; the **driver script** chooses the name and must use
*the same string* in three places:

1. As the key under ``cluster.component_placement``.
2. When asking the placement object for the engines' hardware ranks:
   ``placement.get_hardware_ranks(component_name)``.
3. When reading back the node-group label (if any) for that component:
   ``cfg.cluster.component_placement.get(component_name)``.

Define the name once and reuse it — that way a rename only requires changing
the YAML key and this single Python constant:

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

   The launcher itself never sees the name — it only sees the resolved hardware
   ranks and node-group string. As long as the YAML key and ``component_name``
   match, you're free to call it ``rollout``, ``server``, ``my_engine``, or
   anything else.

Args block (the ``router_server_args`` key)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``router_server_args`` is the **top-level YAML key** you hand to the launcher;
its name is also arbitrary — what matters is the **structure inside it**.
Pass whichever sub-config you like (``cfg.rollout``, ``cfg.my_engine``, ...) as
long as it carries the keys the launcher consumes:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Type
     - What it does
   * - ``tensor_parallel_size``
     - int
     - Per-engine TP size. ``tp_size × pp_size`` GPUs are packed into one engine.
   * - ``pipeline_parallel_size``
     - int
     - Per-engine PP size.
   * - ``group_name``
     - str
     - Worker-group name for the sglang server group.
   * - ``launch_server``
     - bool
     - Set ``False`` to skip the server group (e.g. attach an external server).
   * - ``server``
     - dict
     - Forwarded **verbatim** to ``sglang.srt.server_args.ServerArgs(**)``. Keys
       must be valid ``ServerArgs`` field names — see the `sglang ServerArgs
       reference <https://docs.sglang.io/docs/advanced_features/server_arguments>`_.
   * - ``router_group_name``
     - str
     - Worker-group name for the router worker.
   * - ``launch_router``
     - bool
     - Set ``False`` to skip the router (e.g. you only want raw server URLs).
   * - ``router``
     - dict
     - Forwarded as ``--<field>`` CLI flags to ``sglang_router.launch_router``.
       Keys must be valid ``RouterArgs`` field names — see the `RouterArgs
       source <https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py>`_.

.. note::

   A ready-made default is shipped at
   ``examples/embodiment/config/hybrid_engines/sglang_router_server.yaml``.
   Pull it in through Hydra's ``defaults`` list and only override the fields you
   need — typically ``model_path`` (and any ``server`` / ``router`` tweaks):

   .. code-block:: yaml

      defaults:
        - hybrid_engines/sglang_router_server@router_server_args

      router_server_args:
        model_path: /path/to/hf_model
        # tensor_parallel_size: 2       # override anything else you need
        # server:
        #   mem_fraction_static: 0.9

   The ``@router_server_args`` part is Hydra's package override — it mounts the
   defaults under the ``router_server_args`` top-level key.

.. warning::

   The launcher does not validate keys under ``server`` and ``router``. An
   unknown key in ``server`` makes ``ServerArgs(**kwargs)`` raise; an unknown
   key in ``router`` raises a ``ValueError`` from the launcher with the full
   list of valid fields. Treat both blocks as direct pass-throughs to the
   upstream dataclasses.

``host``/``port``/``dist_init_addr`` for the server, and ``port`` for the
router, are filled in at runtime — leave them unset in YAML. The router binds
on ``0.0.0.0`` by default; servers bind on ``0.0.0.0`` and advertise the Ray
node IP back to the router.

How the launcher uses them
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`rlinf.workers.rollout.sglang_server.launch_sglang_router_and_server`
does the following, in order:

1. Repacks the flat list of ``rollout_hardware_ranks`` into a
   ``PackedPlacementStrategy`` with ``tp_size × pp_size`` accelerators per
   process — one sglang engine per process.
2. Launches the ``SGLangServerWorker`` group (config keys: ``group_name``,
   ``server``).
3. Launches a single ``SGLangRouterWorker`` on node 0 with no workers attached
   (config keys: ``router_group_name``, ``router``).
4. Collects each server's ``http://host:port`` and registers it with the
   running router (``POST /workers``), blocking until each worker reports
   ``is_healthy=true``.

The flat-rank repacking has one **hard requirement**: the hardware ranks must
be contiguous. If you need a non-contiguous layout (e.g. a
``FlexiblePlacementStrategy`` from a ``placement: 0-1:0-3,3-5`` string), build
the strategy yourself and pass it as ``placement_strategy=...`` — that
short-circuits the repacking and uses your strategy as-is.

Launch It
---------

A minimal Hydra entry that brings up the server group and router from the
config above:

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
       # ... use router_url (see "Next: calling the router" below) ...

       router_group.shutdown().wait()
       server_group.shutdown().wait()


   if __name__ == "__main__":
       main()

What this does: starts Ray via ``Cluster``, builds ``ComponentPlacement`` from
``cluster.component_placement``, then asks the launcher to start the engines
and router and wait until every server is registered and healthy. After
``launch_sglang_router_and_server`` returns, the router is reachable at
``router_url`` and every backend has passed ``GET /workers/<id>``.

Heterogeneous Clusters (``node_group``)
---------------------------------------

To pin the engines to a specific labelled set of nodes — e.g. dedicate the
inference nodes to rollouts while training runs elsewhere — use the
``node_group``/``node_groups`` form under ``cluster``:

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

What changes versus the flat config:

- ``component_placement.rollout`` is now a **mapping** with ``node_group`` and
  ``placement`` instead of a single string.
- The driver reads the label off the same key the launcher expects to see in
  hardware ranks — i.e.
  ``cfg.cluster.component_placement.get(component_name).node_group``.
- That label is forwarded to the launcher as ``rollout_node_group``; the
  launcher passes it through to the repacked ``PackedPlacementStrategy`` so the
  servers land on the right nodes.

If you rename the component key from ``rollout`` to anything else (say
``my_engine``), update **both** the YAML key and the ``component_name``
variable in the driver. The ``node_group`` label (``rollout_gpu`` here) and the
entries under ``node_groups`` are independent — those are the cluster's own
vocabulary.

You can also pass a **list** of labels when the
engines should span multiple groups; the launcher forwards the list verbatim
to the placement strategy.

For the full ``node_groups`` / ``env_configs`` / ``hardware`` schema —
including per-node-group Python interpreters, env vars, and non-accelerator
hardware (robots) — see :doc:`Heterogeneous Clusters <hetero>`.

Programmatic API
----------------

If your runner already builds the placement strategy itself, skip the flat-rank
path and hand a strategy directly. The strategy must already encode
``tp_size × pp_size`` accelerators per process:

.. code-block:: python

   from rlinf.scheduler.placement import ComponentPlacement
   from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

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

Other useful entry points on the returned handles:

- ``server_group.get_server_url().wait()`` → list of server URLs.
- ``router_group.register_server(url).wait()`` → attach a server URL to the
  router after the fact (blocks until the worker is healthy).
- ``router_group.unregister_server(url).wait()`` → detach.
- ``router_group.get_router_url().wait()[0]`` → the router URL.

Next: calling the router
------------------------

Once ``launch_sglang_router_and_server`` returns, ``router_url`` is reachable
over plain HTTP — any HTTP client works. For sending ``/generate`` and
``/v1/chat/completions`` requests (sync and async), see the companion guide:
:doc:`Calling SGLang with InferenceHTTPClient <inference_http_client>`.
