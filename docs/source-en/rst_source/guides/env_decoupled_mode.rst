Env Decoupled Mode
==================

``env_decoupled_mode`` is a communication mode used in RLinf embodied tasks to
decouple Env Workers from Rollout Workers. It is enabled by setting
``runner.enable_decoupled_mode: true``.

After it is enabled, an Env Worker is no longer bound to a fixed Rollout Worker
rank. Env Workers put observation data into a shared Channel, and idle Rollout
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
  Worker is ``ceil(env_world_size // rollout_world_size)``.

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

By default, decoupled mode uses a single global pool where any Rollout Worker
may fetch any Env batch. Setting ``rollout.enable_group_route_binding: true``
(default ``false``) instead partitions the pool into fixed per-group bindings:
Env rank ``e`` maps to Rollout rank
``floor(e * rollout_world_size / env_world_size)``. This proportional mapping
supports non-divisible ratios, keeps group sizes within one Env Worker of each
other, and guarantees that every Rollout Worker has at least one Env Worker
when ``env_world_size >= rollout_world_size``.

Each group uses the same ``route_key`` for Env-to-Rollout requests and
Rollout-to-Env replies in both training and evaluation. The key is logical queue
isolation inside the shared ChannelWorker; it does not launch another channel
process or create a separate transport. There is no global-pool work stealing
across keys.

.. code-block:: yaml

   runner:
     enable_decoupled_mode: true

   rollout:
     enable_group_route_binding: true

This gives a fixed ``1`` Rollout : ``N`` Env-Worker overlap, replicable as
independent groups across a robot fleet — useful for real-robot rollout where
each Rollout Worker drives a fixed set of robots. Non-divisible Env-to-Rollout
ratios are supported as long as ``env_world_size >= rollout_world_size``.
Set ``rollout.rollout_queue_size: 1`` when each robot must be served
individually with inference batch size one instead of aggregating requests.

Turtle2 Four-Stream Bringup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``realworld_dummy_turtle2_async_ppo_openpi_pi05_2gpu_4stream`` is a short
single-node system example. Actor/train is fixed to GPU 0, rollout to GPU 1,
and four CPU-only Env ranks share node 0. Because there is one Rollout rank,
all four Env ranks map to ``grp0`` and are served one at a time with
``rollout_queue_size: 1``. The example reuses ``RLinf-Pi05-LIBERO-SFT`` and its
normalization statistics only for dummy bringup and profiling; zero dummy
rewards do not demonstrate policy learning.

The reference RTX PRO 5000 host requires ``NCCL_P2P_DISABLE=1`` for reliable
two-GPU collectives. The sample sets it on the GPU node group and uses bucket
weight synchronization after every global step; remove that environment
override only after validating direct P2P on a different host.

Run the five-step smoke test from the repository root, overriding both model
paths if the checkpoint is not mounted at the path in the YAML:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export MODEL_PATH=/absolute/path/to/RLinf-Pi05-LIBERO-SFT
   python examples/embodiment/train_async.py \
     --config-path examples/embodiment/config \
     --config-name realworld_dummy_turtle2_async_ppo_openpi_pi05_2gpu_4stream \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH"

The resolved configuration is printed at startup. TensorBoard data and
per-worker logs are written below ``runner.logger.log_path``. To capture one
complete step after warm-up with Nsight Systems, enable the profile block:

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-path examples/embodiment/config \
     --config-name realworld_dummy_turtle2_async_ppo_openpi_pi05_2gpu_4stream \
     actor.model.model_path="$MODEL_PATH" \
     rollout.model.model_path="$MODEL_PATH" \
     cluster.profiling.enabled=true \
     'cluster.profiling.steps=[3]'

RLinf injects ``nsys profile`` around ``ActorGroup`` and ``RolloutGroup`` only;
the four Env workers remain unwrapped and report per-rank chunk timing through
RLinf metrics. Reports are under
``<log_path>/<experiment_name>/profiling``. Export the two standard summaries
with, for example:

.. code-block:: bash

   nsys stats --report nvtx_sum --format csv --timeunit ms worker.nsys-rep
   nsys stats --report cuda_gpu_kern_sum --format csv --timeunit ms worker.nsys-rep

Training Flow
-------------

After decoupled mode is enabled, the training flow is roughly:

1. An Env Worker performs an environment step and obtains an observation.
2. The Env Worker sends the observation to the Rollout Channel.
3. Any Rollout Worker dynamically fetches one or more Env batches from the Channel.
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
