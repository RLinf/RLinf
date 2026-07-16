GPU Profiling
==============================

Use the ``cluster.profiling`` configuration for system-level profiling of Ray
worker processes.

RLinf supports wrapping selected worker groups with a backend-specific profiler
command. The backend is selected by the required ``backend`` field:

- ``nsight`` — NVIDIA Nsight Systems (``nsys profile``)
- ``rocprof_sys`` — AMD ROCm Systems Profiler (``rocprof-sys-python``)

All backends share the same common fields (``enabled``, ``worker_groups``,
``steps``, ``output_dir``). Backend-specific options live under their own keys.


How To Enable It
------------------------------

Add the profiling preset to ``defaults`` in your YAML:

.. code-block:: yaml

   defaults:
     - hybrid_engines/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - profile/default@cluster.profiling

The corresponding config file is ``examples/embodiment/config/profile/default.yaml``.
To switch backends, override ``cluster.profiling.backend`` in your main YAML or on
the Hydra CLI (see the backend-specific sections below).


Common Fields
------------------------------

These fields apply to every profiling backend:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Default
     - Description
   * - ``backend``
     - *(required)*
     - Profiling backend: ``"nsight"`` or ``"rocprof_sys"``.
   * - ``enabled``
     - ``true``
     - Master switch. Set to ``false`` to disable without removing the config.
   * - ``worker_groups``
     - ``null``
     - List of worker group names to profile. ``null`` means no workers are profiled.
   * - ``steps``
     - ``null``
     - Training step indices to gate profiling around. ``null`` means full worker lifetime.
   * - ``output_dir``
     - *(auto-derived)*
     - Directory for profiling output. When omitted, defaults to ``<log_path>/<experiment_name>/profiling/``.


The ``enabled`` Flag
------------------------------

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       enabled: false

When ``enabled: false``:

- Workers are not wrapped with a profiler command.
- No output directory is created.
- The rest of the config can stay in place for later reuse.


Output Directory
------------------------------

By default, reports are written under:

.. code-block:: text

   runner.logger.log_path/runner.logger.experiment_name/profiling/

For example:

.. code-block:: text

   ../results/libero_spatial_ppo_openpi/profiling/

To override, set ``output_dir`` explicitly:

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       output_dir: /mnt/public/profiles/my_run


How To Override Worker Groups
------------------------------

Override ``worker_groups`` directly in the main YAML:

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       worker_groups: [ActorGroup, RolloutGroup]

If ``worker_groups`` is omitted or ``null``, no worker is profiled. To profile
all workers in a run, list the names of every worker group you care about.

One subtle point: ``ChannelWorker`` instances are not children of
``ActorGroup`` or ``RolloutGroup`` ranks. ``Channel.create(name)`` launches a
separate worker group whose group name is usually ``Env``, ``Rollout``, or
``Actor``. Profiling ``ActorGroup`` does not automatically include the ``Actor``
channel worker. Add those channel group names explicitly if you want channel-side
traces.

For the built-in embodied runners:

- ``ActorGroup``: actor compute workers
- ``RolloutGroup``: rollout compute workers
- ``EnvGroup``: environment compute workers
- ``Actor``: the channel worker behind ``Channel.create("Actor")``
- ``Rollout``: the channel worker behind ``Channel.create("Rollout")``
- ``Env``: the channel worker behind ``Channel.create("Env")``


How To Profile Only Specific Training Steps
-------------------------------------------

By default, profiling covers the entire worker lifetime. ``steps`` restricts
collection to specific training steps:

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       enabled: true
       steps: [3]            # only profile global step 3

Multiple steps:

.. code-block:: yaml

   cluster:
     profiling:
       steps: [3, 10, 50]

Hydra CLI:

.. code-block:: bash

   python ... '+cluster.profiling.steps=[3]'

When ``steps`` is set:

- For the ``nsight`` backend, RLinf automatically injects
  ``capture-range=cudaProfilerApi`` and ``capture-range-end=stop`` into
  ``options``.
- The embodied runner calls ``torch.cuda.profiler.start()`` before each listed
  step and ``torch.cuda.profiler.stop()`` after it.
- The resulting trace covers only those steps.


NVIDIA: Nsight Systems (``backend: nsight``)
---------------------------------------------

Default Preset
~~~~~~~~~~~~~~

The built-in ``profile/default`` preset looks like this:

.. code-block:: yaml

   backend: nsight
   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup, Actor, Rollout, Env]
   options:
     t: cuda,cudnn,cublas,nvtx,osrt
     sample: process-tree
     cpuctxsw: process-tree
     cudabacktrace: all
     osrt-threshold: 1000
   flags: []

Overriding ``nsys profile`` Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``options`` maps to ``nsys profile`` flags that take values; ``flags`` emits
bare flags:

.. code-block:: yaml

   cluster:
     profiling:
       backend: nsight
       options:
         t: cuda,cudnn,cublas,nvtx,osrt
         sample: process-tree
         backtrace: fp
         capture-range: cudaProfilerApi
         capture-range-end: stop
       flags: [python-backtrace]

Rendering rules:

- Single-character keys → ``-t cuda,...``
- Multi-character keys → ``--backtrace=fp``
- ``flags`` entries → ``--python-backtrace``

Useful options:

- ``t``: traced APIs (``cuda``, ``cudnn``, ``cublas``, ``nvtx``, ``osrt``)
- ``sample``: CPU sampling mode
- ``backtrace``: CPU backtrace method (``lbr``, ``fp``, ``dwarf``)
- ``cpuctxsw``: CPU thread scheduling trace
- ``cudabacktrace``: CUDA API backtraces (adds overhead)
- ``capture-range`` / ``capture-range-end``: scope collection to NVTX or CUDA profiler API ranges

Compute-Path NVTX Annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf decorates the hot path of actor, rollout, and env workers with
``@Worker.timer("...")``. The decorator records RLinf timer metrics and opens an
accelerator-specific profiling range through ``AcceleratorUtil.profiling_range``.
For the ``nsight`` backend these ranges appear as labelled NVTX intervals in the
timeline and in ``nsys stats --report nvtx_sum``.

The profiling range is emitted only while a profiling window is active. When
profiling is off, ``Worker.timer`` still records timing metrics and the
accelerator profiling range falls back to a no-op.

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

Decorating your own worker method:

.. code-block:: python

   from rlinf.scheduler.worker import Worker

   class MyWorker(Worker):
       @Worker.timer("my_worker/my_phase")
       def my_phase(self, batch):
           ...

Ad-Hoc Profiling Ranges
~~~~~~~~~~~~~~~~~~~~~~~

For one-off in-function annotations:

.. code-block:: python

   from rlinf.scheduler.hardware import AcceleratorUtil

   class MyWorker(Worker):
       def my_phase(self, batch):
           with AcceleratorUtil.profiling_range(
               self._accelerator_type, "my_worker/inner_phase"
           ):
               run_inner_phase(batch)

``AcceleratorUtil.profiling_range`` dispatches to the current accelerator
backend. It is a no-op when the accelerator has no registered profiling range
implementation or when profiling is not active.


AMD: ROCm Systems Profiler (``backend: rocprof_sys``)
-------------------------------------------------------

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~

Configure ``rocprof_sys`` inline in your run config:

.. code-block:: yaml

   backend: rocprof_sys
   enabled: true
   worker_groups: [ActorGroup, RolloutGroup, EnvGroup]
   args:
     T: hip           # trace HIP API calls

RLinf wraps each matching worker's Python interpreter with:

.. code-block:: text

   rocprof-sys-python [args] -- <python_interpreter>

Overriding ``rocprof-sys-python`` Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``args`` maps to ``rocprof-sys-python`` flags that take values:

.. code-block:: yaml

   cluster:
     profiling:
       backend: rocprof_sys
       args:
         T: hip,hsa,rccl     # single-char key → -T hip,hsa,rccl
         output-format: json  # multi-char key  → --output-format=json

Rendering rules:

- Single-character keys → ``-T hip,hsa,rccl``
- Multi-character keys → ``--output-format=json``

Injecting Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``env`` to pass extra environment variables to profiled workers:

.. code-block:: yaml

   cluster:
     profiling:
       backend: rocprof_sys
       env:
         ROCPROFSYS_SAMPLING_FREQ: "100"

RLinf automatically derives ``ROCPROFSYS_OUTPUT_PATH`` and
``ROCPROFSYS_OUTPUT_PREFIX`` from ``output_dir``; values provided in ``env``
take precedence.

Recommended Workflow
------------------------------

NVIDIA first pass:

- Start with ``profile/default@cluster.profiling``.
- Keep ``enabled: true`` and use the preset as-is for both CUDA-side and CPU
  runtime visibility.
- Avoid ``capture-range: nvtx`` until you have confirmed the target workers
  emit NVTX ranges.
- Use ``steps: [3]`` to limit trace size on long runs.

AMD first pass:

- Configure ``rocprof_sys`` inline:

  .. code-block:: yaml

     cluster:
       profiling:
         backend: rocprof_sys
         enabled: true
         worker_groups: [ActorGroup, RolloutGroup, EnvGroup]
         args:
           T: hip

- Verify ``rocprof-sys-python`` is on ``PATH`` in each worker's environment.
- Check ``output_dir`` for ``.json`` or binary trace files after the run.
