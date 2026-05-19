Nsight Systems
==============================

This document introduces the ``cluster.nsight`` configuration in RLinf for
system-level profiling with NVIDIA Nsight Systems.

RLinf supports wrapping selected Ray worker groups with ``nsys profile`` so you
can collect traces for CUDA kernels, cuDNN, cuBLAS, NVTX ranges, and optionally
CPU-side runtime activity.


How To Enable It
------------------------------

In an embodied YAML, add the Nsight preset to ``defaults``:

.. code-block:: yaml

   defaults:
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - nsight/default@cluster.nsight

The corresponding config files are:

- ``examples/embodiment/config/nsight/default.yaml``


Default Preset
------------------------------

The built-in default preset looks like this:

.. code-block:: yaml

   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup, Actor, Rollout, Env]
   options:
     t: cuda,cudnn,cublas,nvtx,osrt
     sample: process-tree
     cpuctxsw: process-tree
     cudabacktrace: all
     osrt-threshold: 1000
   flags: []

This preset targets the most common embodied compute and communication workers by default:

- ``ActorGroup``
- ``RolloutGroup``
- ``EnvGroup``
- ``Actor``
- ``Rollout``
- ``Env``

These names must match real worker group names such as ``actor.group_name`` and
``rollout.group_name``. They are not the component aliases ``actor`` or
``rollout``.

The preset also keeps CPU sampling enabled and turns on ``cudabacktrace`` by
default, so the resulting report includes both CUDA-side activity and CUDA API
backtraces during the first profiling pass.


The ``enabled`` Flag
------------------------------

The ``enabled`` field is the main switch for Nsight wrapping:

.. code-block:: yaml

   cluster:
     nsight:
       enabled: false

When ``enabled: false``:

- RLinf does not wrap workers with ``nsys profile``
- RLinf does not reserve the default Nsight output directory
- the rest of the config can stay in place for later reuse

So there is no need to maintain a separate ``disabled.yaml``. You can keep the
same preset and override ``cluster.nsight.enabled: false`` in the main YAML.


How To Override Worker Groups
------------------------------

You can override the preset directly in the main YAML:

.. code-block:: yaml

   cluster:
     nsight:
       worker_groups: [EnvGroup, RolloutGroup, ActorGroup, Env, Rollout, Actor]

This is especially useful when you want to profile:

- compute workers such as ``ActorGroup`` or ``RolloutGroup``
- channel workers such as ``Env``, ``Rollout``, and ``Actor``
- environment workers such as ``EnvGroup``

If ``worker_groups`` is omitted, RLinf profiles all worker groups.

One subtle point is that ``ChannelWorker`` is not launched as a child process of
``ActorGroup`` or ``RolloutGroup`` ranks. In the current implementation,
``Channel.create(name)`` launches a separate worker group whose group name is
usually ``Env``, ``Rollout``, or ``Actor``. So profiling ``ActorGroup`` does
not automatically include the ``Actor`` channel worker. If you want channel-side
traces, add those channel group names explicitly to ``worker_groups``.

For the built-in embodied runners, these channel worker group names are created
directly from the channel names in code:

- ``ActorGroup``: actor compute workers
- ``RolloutGroup``: rollout compute workers
- ``EnvGroup``: environment compute workers
- ``Actor``: the channel worker behind ``Channel.create("Actor")``
- ``Rollout``: the channel worker behind ``Channel.create("Rollout")``
- ``Env``: the channel worker behind ``Channel.create("Env")``

So ``worker_groups: [Actor]`` means "profile the Actor channel worker", not
"profile all actor-side compute". The current matching rule is by worker group
name, which is why the channel names matter here. If you create your own
channels with other names, use those exact channel names in ``worker_groups``.


How To Override Nsight Options
------------------------------

``cluster.nsight.options`` maps directly to ``nsys profile`` flags that take
values, while ``cluster.nsight.flags`` emits bare flags:

.. code-block:: yaml

   cluster:
     nsight:
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         cpuctxsw: process-tree
         cudabacktrace: all

Useful options include:

- ``t``: traced APIs such as ``cuda``, ``cudnn``, ``cublas``, ``nvtx``, and ``osrt``
- ``sample``: CPU sampling mode
- ``backtrace``: CPU backtrace method used with sampling, for example ``lbr``, ``fp``, or ``dwarf``
- ``cpuctxsw``: CPU thread scheduling trace
- ``cudabacktrace``: collect CUDA API backtraces; this requires CPU sampling to stay enabled and can add noticeable overhead
- ``capture-range`` and ``capture-range-end``: restrict collection to NVTX or CUDA-profiler-controlled ranges
- ``o`` or ``output``: explicit output prefix

If you enable ``capture-range: nvtx``, make sure your code actually emits NVTX
ranges. Otherwise Nsight may generate an almost empty report.

You are not limited to the keys shown in ``nsight/default``. RLinf forwards
arbitrary entries in ``cluster.nsight.options`` to ``nsys profile``:

.. code-block:: yaml

   cluster:
     nsight:
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         backtrace: fp
         capture-range: cudaProfilerApi
         capture-range-end: stop
         samples-per-backtrace: 4
       flags: [python-backtrace]

This works because RLinf treats ``cluster.nsight.options`` as a free-form
mapping:

- one-character keys are rendered like ``-t cuda,...``
- longer keys are rendered like ``--backtrace=fp``
- ``flags`` entries are rendered like ``--python-backtrace``

To emit a flag without a value, put it in ``cluster.nsight.flags``:

.. code-block:: yaml

   cluster:
     nsight:
       flags: [python-backtrace]

You can do the same from the Hydra CLI:

.. code-block:: bash

   python ... 'cluster.nsight.flags=[python-backtrace]'

This is especially useful for ``nsys`` options whose value is optional and where
the bare flag form has special meaning.

Nsight Systems options are not perfectly stable across versions and host
platforms. In particular, the supported or recommended ``backtrace`` mode may
vary between machines. If a flag is rejected on your target node, or if
``lbr``-style backtraces do not work well on that machine, check
``nsys profile --help`` on the target node and override
``cluster.nsight.options`` accordingly, for example by switching
``backtrace`` to ``fp`` or ``dwarf``.


How To Profile Only Specific Training Steps
-------------------------------------------

By default, ``nsys profile`` runs for the entire worker lifetime. For a long
training job, the resulting ``.nsys-rep`` quickly grows into tens or hundreds of
megabytes and makes interesting steps hard to find. ``cluster.nsight.steps``
restricts collection to a small set of training steps:

.. code-block:: yaml

   cluster:
     nsight:
       enabled: true
       steps: [3]               # only profile global step 3

You can list multiple steps:

.. code-block:: yaml

   cluster:
     nsight:
       steps: [3, 10, 50]

Or do the same on the Hydra CLI:

.. code-block:: bash

   python ... '+cluster.nsight.steps=[3]'

When ``steps`` is set:

- RLinf auto-attaches ``capture-range=cudaProfilerApi`` and
  ``capture-range-end=stop`` to ``cluster.nsight.options``, so you do not need
  to remember the matching CLI flags.
- The embodied runner calls ``torch.cuda.profiler.start()`` before each listed
  step and ``torch.cuda.profiler.stop()`` after it. Nsight only writes data
  inside those windows.
- The resulting trace is bounded by the cost of the listed steps, not the full
  training run.

When ``steps`` is unset (the default), behavior is unchanged from earlier
releases: nsys captures the full worker lifetime.

If you want named ranges in the timeline at the same time, the worker-side NVTX
annotations described in the next section emit only while a step window is
open, so a trace produced with ``steps=[3]`` shows exactly the actor / rollout /
env activity that ran inside step 3.


Compute-Path NVTX Annotations
------------------------------

RLinf decorates the hot path of the actor, rollout, and env workers with named
NVTX ranges via ``@NsightProfiler.annotate("...")``. These appear as labelled
intervals in the Nsight timeline and via ``nsys stats --report nvtx_sum``.

The decorator is gated by the same step window as ``cluster.nsight.steps`` --
it only emits NVTX events while ``is_profiling_active()`` is true, so there is
no measurable overhead when profiling is off (only a boolean read).

Built-in annotations:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Worker group
     - NVTX label
     - What it covers
   * - Actor
     - ``actor/recv_traj``
     - Receiving a trajectory batch from the rollout / env side
   * - Actor
     - ``actor/compute_adv``
     - Advantage / return computation
   * - Actor
     - ``actor/run_training``
     - Policy / value optimization step (forward + backward + optimizer)
   * - Actor
     - ``actor/sync_model_to_rollout``
     - Weight broadcast from actor to rollout workers
   * - Rollout
     - ``rollout/recv_obs``
     - Pulling observations from the env channel
   * - Rollout
     - ``rollout/predict``
     - Single-step policy forward pass
   * - Rollout
     - ``rollout/generate``
     - Multi-step generation / unroll
   * - Rollout
     - ``rollout/generate_epoch``
     - A full rollout epoch
   * - Rollout
     - ``rollout/send_actions``
     - Sending actions back to env workers
   * - Rollout
     - ``rollout/send_traj``
     - Shipping completed trajectories to the actor side
   * - Rollout (async)
     - ``rollout/poll_weight_sync`` / ``rollout/request_weight_sync``
     - Async weight-sync handshake with the actor
   * - Env
     - ``env/recv_actions``
     - Receiving the next-action batch from the rollout side
   * - Env
     - ``env/step`` / ``env/bootstrap_step``
     - One simulator step (and the warm-up step at episode start)
   * - Env
     - ``env/interact`` / ``env/interact_once``
     - The full env interaction loop (and a single sub-iteration of it)
   * - Env
     - ``env/send_obs`` / ``env/send_rollout_trajectories``
     - Pushing observations / completed rollouts downstream

Decorating your own worker method works the same way:

.. code-block:: python

   from rlinf.utils.nsight_profiler import NsightProfiler

   class MyWorker(Worker):
       @NsightProfiler.annotate("my_worker/my_phase", color="green")
       def my_phase(self, batch):
           ...

The ``message`` argument is what shows up in the timeline; ``color`` and
``domain`` are forwarded to ``nvtx.start_range``. If the optional ``nvtx``
package is not importable in the worker process, the decorator becomes a no-op.


How To Emit Ad-Hoc NVTX Ranges
------------------------------

For ad-hoc, in-function ranges that should always emit (independent of the
step-gating flag), RLinf also provides a small context manager:

.. code-block:: python

   from rlinf.utils.utils import nvtx_range

   with nvtx_range("actor.forward", color="green"):
       run_actor_forward()

Use this when you want to mark a region inside a single function or when the
code path is not stable enough to be worth decorating.

The helper first tries the optional ``nvtx`` Python package. If that package
is not installed, it falls back to ``torch.cuda.nvtx`` when CUDA is available.
If neither backend is available, the context manager becomes a no-op, so it is
safe to leave in code paths that also run without NVTX support.

When you use ``capture-range: nvtx``, make sure the profiled workers actually
execute code inside these ranges. Otherwise Nsight may collect very little or
no data.


Output Path
------------------------------

When ``cluster.nsight.enabled`` is true and you do not explicitly set ``o`` or
``output``, RLinf writes reports under:

.. code-block:: text

   runner.logger.log_path/runner.logger.experiment_name/nsights

For example:

.. code-block:: text

   ../results/libero_spatial_ppo_openpi/nsights/

If you want a custom path, set it explicitly:

.. code-block:: yaml

   cluster:
     nsight:
       options:
         o: /mnt/public/profiles/my_run/worker_trace


Recommended Workflow
------------------------------

For a first pass, the simplest setup is:

- start with ``nsight/default@cluster.nsight``
- keep ``enabled: true``
- use the preset as-is if you want both CUDA-side traces and CPU/channel-side runtime visibility
- avoid ``capture-range: nvtx`` until you have confirmed the target workers
  really emit NVTX ranges
