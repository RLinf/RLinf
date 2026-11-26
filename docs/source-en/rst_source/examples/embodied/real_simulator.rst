Real-World Simulator
====================

Add realistic observation latency and network conditions to embodied training without leaving simulation. RLinf provides two independent YAML-configured modules, ``delay_sampler`` and ``net_emulation``. Both are disabled by default.

Overview
--------

Inject sensor latency and cross-worker network constraints into simulated training runs.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀ · Any embodied policy

   .. grid-item-card:: Algorithms
      :text-align: center

      DAgger · Sync & Async RL

   .. grid-item-card:: Tasks
      :text-align: center

      RoboTwin · Any simulator env

   .. grid-item-card:: Hardware
      :text-align: center

      Single-node or multi-node GPU

| **You'll do:** enable ``delay_sampler`` or ``net_emulation`` in YAML → launch ``run_async.sh`` (or ``run_embodiment.sh``) → evaluate policy robustness.
| **Prerequisites:** :doc:`RoboTwin <robotwin>` (for the shipped example) · :doc:`DAgger <dagger>`.

Observation Delay
-----------------

Emulate per-environment sensor latency by inserting a configurable delay before each environment step and reset return. Use this to test how variable camera or joint-encoder timing affects policy training before real-world deployment.

Supported distributions (all values in **seconds**):

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Type
     - Parameters
     - When to use
   * - ``constant``
     - ``delay``
     - Every observation has the same fixed delay.
   * - ``uniform``
     - ``min_delay``, ``max_delay``
     - Latency varies uniformly within a known range.
   * - ``exponential``
     - ``rate``
     - Delay follows a Poisson-like arrival pattern.
   * - ``gaussian``
     - ``mean``, ``stddev``
     - Realistic sensor jitter with a typical value and spread.

.. code-block:: yaml

   env:
     delay_sampler:
       type: uniform
       min_delay: 0.11       # 110 ms
       max_delay: 0.20       # 200 ms

- Applies only during training; evaluation is unaffected.
- Works per environment instance with both synchronous and asynchronous runners.

Network Emulation
-----------------

Emulate cross-worker network latency and bandwidth limits cluster-wide. When ``cluster.net_emulation.enabled`` is ``true``, point-to-point sends and broadcasts (such as actor-to-rollout weight synchronization) reserve transmission slots against the configured link delays and bandwidth budgets. Broadcasts are charged once on the sender and once per receiving bandwidth group, and the sender waits for the slowest receiver.

Same-device and same-node transfers use shared-memory IPC and are not emulated. Use this to test policies under bandwidth-constrained or high-latency links, such as when env and rollout workers run across separate clusters or cloud regions.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Description
   * - ``enabled``
     - ``bool``
     - Toggle network emulation on (``true``) or off (``false``). Default: ``false``.
   * - ``symmetric``
     - ``bool``
     - Mirror every cross-DC pair when ``true``. Default: ``true``.
   * - ``crossdc_pairs``
     - ``list``
     - Source-destination pairs with per-pair ``delay_ms``.
   * - ``bandwidth_groups``
     - ``list``
     - Endpoint groups sharing a ``bandwidth_mbps`` budget.

Endpoint names use ``GroupName:Rank`` — ``Env:0`` for the first env worker, ``Rollout:1`` for the second rollout worker. Inclusive ranges such as ``Env:0-3`` expand to ``Env:0`` through ``Env:3``, and any trailing ``Group`` suffix is stripped automatically.

- Works with both synchronous and asynchronous runners.
- Independent of ``delay_sampler``: enable either, both, or neither.

Setup and Run
-------------

Prepare the RoboTwin environment and checkpoints as described in :doc:`RoboTwin <robotwin>`, then launch the shipped example config ``examples/embodiment/config/realsimulator_robotwin_adjust_bottle_dagger_openpi.yaml``:

.. code-block:: bash

   bash examples/embodiment/run_async.sh realsimulator_robotwin_adjust_bottle_dagger_openpi

.. dropdown:: Advanced configuration (optional)

   Full ``net_emulation`` example with ``crossdc_pairs`` and ``bandwidth_groups``:

   .. code-block:: yaml

      cluster:
        net_emulation:
          enabled: true
          symmetric: true
          crossdc_pairs:
            - src: ["Env:0-1"]
              dst: ["Rollout:0-1", "Actor:0-1"]
              delay_ms: 50
          bandwidth_groups:
            - members: ["Env:0-1"]
              bandwidth_mbps: 1000
            - members: ["Rollout:0-1", "Actor:0-1"]
              bandwidth_mbps: 500

   Combine ``delay_sampler`` and ``net_emulation`` together:

   .. code-block:: yaml

      env:
        delay_sampler:
          type: uniform
          min_delay: 0.11
          max_delay: 0.20

      cluster:
        net_emulation:
          enabled: true
          crossdc_pairs:
            - src: ["Env:0", "Env:1"]
              dst: ["Rollout:0", "Rollout:1"]
              delay_ms: 50

See also: :doc:`RoboTwin <robotwin>` and :doc:`DAgger <dagger>`.
