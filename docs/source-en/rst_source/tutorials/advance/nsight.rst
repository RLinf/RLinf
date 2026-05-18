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


How To Emit NVTX Ranges
------------------------------

RLinf already provides a small helper for emitting NVTX ranges:

.. code-block:: python

   from rlinf.utils.utils import nvtx_range

   with nvtx_range("actor.forward", color="green"):
       run_actor_forward()

This is the easiest way to mark higher-level phases before you turn on
``capture-range: nvtx`` or when you want named ranges in the Nsight timeline.

The helper first tries the optional ``nvtx`` Python package. If that package is
not installed, it falls back to ``torch.cuda.nvtx`` when CUDA is available. If
neither backend is available, the context manager becomes a no-op, so it is
safe to leave in code paths that also run without NVTX support.

When you use ``capture-range: nvtx``, make sure the profiled workers actually
execute code inside these ranges. Otherwise Nsight may collect very little or no
data.


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
