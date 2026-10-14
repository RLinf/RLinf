Env Decoupled Mode
==================

``env_decoupled_mode`` is a communication mode used in RLinf embodied tasks to
decouple Env Workers from Rollout Workers. It is enabled by setting
``runner.enable_decoupled_mode: true``.

By default, an Env Worker is not bound to a fixed Rollout Worker rank in this
mode. Env Workers put observation data into a shared Channel, and idle Rollout
Workers can dynamically fetch batches for inference. After inference finishes,
the results are returned to the original Env Workers.

This mode is useful when Env Workers and Rollout Workers run at different speeds,
especially when simulation latency varies significantly, some Env Workers may
block, or Rollout Workers need to dynamically aggregate multiple Env requests for
batched inference.

How to Enable
-------------

Set the following fields in an embodied configuration:

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     rollout_queue_size: 0

Where:

- ``runner.enable_decoupled_mode: true`` enables Env Decoupled Mode.
- If ``runner.enable_decoupled_mode`` is not configured, RLinf uses the normal
  communication mode.
- ``rollout_queue_size`` controls the maximum number of Env data groups that a
  Rollout Worker aggregates at once. When set to ``0``, the default strategy is
  used. In this case, the number of Env data groups aggregated by one Rollout
  Worker is ``env_world_size // rollout_world_size``.

Example configuration:

.. code-block:: text

   examples/embodiment/config/maniskill_sac_mlp_async_decoupled.yaml

Requirements
------------

The current implementation requires the number of Env Workers and Rollout Workers
to satisfy a valid ratio. The ratio can be arbitrary, such as
``env:rollout = 8:3``, but the number of Env Workers must be greater than or
equal to the number of Rollout Workers.

When there are significantly more Env Workers than Rollout Workers, decoupled
mode allows Rollout Workers to continuously fetch tasks from the shared Channel,
instead of being bound to a fixed Env rank.

This setup is suitable when there are many Env Workers and relatively fewer
Rollout Workers. Note that if Rollout Workers become the bottleneck, adding more
Env Workers may not improve throughput and may instead increase Channel queueing
time.

It is suitable for:

- A large number of simulation environments.
- Rollout inference that can handle larger batches.
- Serving many Env Workers with fewer Rollout Workers.

You can use ``rollout_queue_size`` to control how many Env shards a Rollout
Worker aggregates at once:

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     rollout_queue_size: 2

A smaller ``rollout_queue_size`` usually reduces waiting time. A larger value may
improve inference batch utilization, but it may also increase the waiting time
for each aggregation.

Per-Group Route Binding
-----------------------

To keep each Rollout Worker serving a fixed set of Env Workers, enable
``rollout.enable_group_route_binding`` alongside decoupled mode. This is useful
when each Rollout Worker drives a fixed group of robots. The option defaults
to ``false``, which keeps the shared pool described above.

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     enable_group_route_binding: true

Each Env rank is assigned to group
``env_rank * rollout_world_size // env_world_size``, and Rollout rank ``k``
serves group ``k``. Each group uses its own ``route_key`` during training and
evaluation. Requests stay within their assigned group, and results return to
the Env Worker that sent them.

The worker counts must satisfy ``env_world_size >= rollout_world_size`` but
need not divide evenly. With four Env Workers and three Rollout Workers, the
groups contain Env ranks ``[0, 1]``, ``[2]``, and ``[3]``. Every Rollout Worker
has a nonempty group, and group sizes differ by at most one.

Training Flow
-------------

After decoupled mode is enabled, the training flow is roughly:

1. An Env Worker performs an environment step and obtains an observation.
2. The Env Worker sends the observation to the Rollout Channel.
3. A Rollout Worker fetches one or more Env batches from the shared pool, or from
   its assigned group when group binding is enabled.
4. The Rollout Worker runs model inference, generates an action or rollout result,
   and returns it to the Env Worker that sent the request.
5. The Env Worker continues environment interaction with the returned result.

Users usually do not need to handle routing details directly. In most cases, it
is enough to enable ``runner.enable_decoupled_mode`` in the configuration and
use Env Workers, Rollout Workers, and Runners that support this mode.

Evaluation Flow
---------------

Decoupled mode can also be used during evaluation. In this case, Env Workers
continuously send eval observations, and Rollout Workers run eval inference and
return actions.

Compared with training, evaluation usually does not need to collect complete
training rollout information, but the communication pattern is the same: Env
Workers send requests, and Rollout Workers dynamically receive and return results.

When to Use
-----------

Consider enabling ``env_decoupled_mode`` when:

- Env Worker step time varies significantly.
- Some environments may have long-tail latency or temporary blocking.
- The number of Env Workers is greater than the number of Rollout Workers.
- Rollout Workers should dynamically aggregate multiple Env requests for batched inference.
- Fixed-rank communication in asynchronous embodied training causes unnecessary waiting.

If Env and Rollout speeds are stable, their worker counts are the same, and there
is no obvious blocking, the normal mode is usually simpler.

Notes
-----

- The current implementation requires ``env_world_size >= rollout_world_size``.
- Decoupled mode currently does not support ``enable_p2p=True``.
- ``runner.enable_decoupled_mode`` controls whether this mode is enabled.
- ``rollout_queue_size`` affects both throughput and latency, and should be tuned
  according to Env/Rollout speed.
- Env, Rollout, and Reward Workers need to use communication paths that support
  decoupled mode.
