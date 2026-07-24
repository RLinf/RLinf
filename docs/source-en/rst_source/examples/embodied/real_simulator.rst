Real-World Simulator
=====================

Add realistic observation latency and network conditions to your embodied
training without leaving simulation.

RLinf provides two independent modules — ``delay_sampler`` and
``net_emulation`` — each controlled by configuration parameters in your YAML.
Both are **disabled by default**: omit the corresponding parameters and your
workflow runs unchanged.


Observation Delay (delay_sampler)
---------------------------------

Emulate per-environment sensor latency by inserting a configurable delay wait
before each observation leaves the Env Worker.

Use this when you need to test how variable observation timing affects
policy training, or when you're preparing a policy for real-world deployment
where camera and joint-encoder latencies differ per robot.

Supported distributions:

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

All delay values are in **seconds**.

.. code-block:: yaml

   env:
     delay_sampler:
       type: uniform
       min_delay: 0.11       # 110 ms
       max_delay: 0.20       # 200 ms

Requirements:

- ``runner.enable_decoupled_mode`` must be set to ``true``.
- The delay applies only in training mode (``mode="train"``). Evaluation is
  unaffected.
- Use the async runner (``train_async.py``). The synchronous runner does not
  support decoupled communication.


Network Emulation (net_emulation)
---------------------------------

Emulate cross-worker network latency and bandwidth limits with a global
token-bucket scheduler. A ``NetEmulationProxy`` Ray actor intercepts every
NCCL send between configured worker groups and enforces the specified delays
and bandwidth caps.

Use this to test how policies behave under bandwidth-constrained or
high-latency links, such as when Env Workers and Rollout Workers are on
separate clusters or cloud regions.

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
     - If ``true``, every cross-DC pair is mirrored. Default: ``true``.
   * - ``proxy.node_rank``
     - ``int``
     - Cluster node that hosts the proxy actor. Default: ``0``.
   * - ``proxy.num_cpus``
     - ``int``
     - CPUs reserved for the proxy. Default: ``1``.
   * - ``crossdc_pairs``
     - ``list``
     - Source-destination pairs with per-pair ``delay_ms``.
   * - ``bandwidth_groups``
     - ``list``
     - Endpoint groups sharing a ``bandwidth_mbps`` budget.

.. code-block:: yaml

   net_emulation:
     enabled: true
     symmetric: true
     proxy:
       node_rank: 0
       num_cpus: 1
     crossdc_pairs:
       - src: ["Env:0", "Env:1"]
         dst: ["Rollout:0", "Rollout:1", "Actor:0", "Actor:1"]
         delay_ms: 50
     bandwidth_groups:
       - members: ["Env:0", "Env:1"]
         bandwidth_mbps: 1000
       - members: ["Rollout:0", "Rollout:1", "Actor:0", "Actor:1"]
         bandwidth_mbps: 500

Endpoint names use the ``GroupName:Rank`` convention — ``Env:0`` for the
first Env Worker, ``Rollout:1`` for the second Rollout Worker. The ``Group``
suffix is stripped automatically.

Requirements:

- Works with both the synchronous and asynchronous runners.
- Independent of ``delay_sampler``. Enable either, both, or neither.


Together
--------

Combine both modules to simulate a realistic deployment:

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true       # required by delay_sampler

   env:
     delay_sampler:
       type: uniform
       min_delay: 0.11
       max_delay: 0.20

   net_emulation:
     enabled: true
     crossdc_pairs:
       - src: ["Env:0", "Env:1"]
         dst: ["Rollout:0", "Rollout:1"]
         delay_ms: 50


Example
-------

Full worked example:

.. code-block:: text

   examples/embodiment/config/realsimulator_robotwin_adjust_bottle_dagger_openpi.yaml


See also

- :doc:`RoboTwin <robotwin>` — RoboTwin environment setup and configuration.
- :doc:`DAgger <dagger>` — DAgger training with expert policy.
- :doc:`Env Decoupled Mode <../../guides/env_decoupled_mode>` — required by ``delay_sampler``.
