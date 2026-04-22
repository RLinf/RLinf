Nsight Profiler Code Design
===========================

This document describes the **code architecture** of RLinf's Nsight Systems
integration.  For configuration and hands-on timeline reading, see
:doc:`nsight_profiler_guide`.


Architecture Overview
---------------------

The profiling system has four layers, each zero-overhead when disabled:

1. **Config & Launch** -- YAML config drives ``WorkerGroup.launch()`` and
   ``Channel.create()``; ``cluster.py`` injects ``runtime_env["nsight"]``
   so Ray wraps each worker with ``nsys profile``.

2. **Runner Control** -- ``_should_profile(step)`` gates
   ``start_profile`` / ``stop_profile``; toggles ``cudaProfilerApi`` and
   ``_channel_nvtx_enabled`` per step.

3. **Worker NVTX** -- ``@NsightProfiler.annotate`` on worker methods
   (``env/step``, ``rollout/predict``, ``actor/run_training``, ...).

4. **Scheduler NVTX** -- channel put/get stats, ChannelWorker
   recv/send ranges, collective send/recv type, ``pg/send|recv``
   backend and byte counts.


Config & Launch
---------------

The runner's ``__init__`` extracts ``nsight_options`` from the YAML config
and passes them to two kinds of consumers:

1. **Compute WorkerGroups** -- ``WorkerGroup.launch(nsight_options=...)``
   adds per-rank output naming (``{group}_rank{N}_pid%p``) and feeds the
   options into ``Cluster.allocate()``, which sets
   ``runtime_env["nsight"] = dict(nsight_options)``.  Ray then wraps the
   worker process with ``nsys profile <options> python ...``.

2. **Channels** -- ``Channel.create(nsight_options=...)`` overrides two
   keys before launching the ChannelWorker:

   .. code-block:: python

      channel_nsight["capture-range"] = "none"   # always recording
      channel_nsight["t"] = "nvtx,osrt,cuda"     # lighter trace set

   ChannelWorkers have no concept of training steps, so they record for
   their entire lifetime rather than gating on ``cudaProfilerApi``.

Source files: ``worker_group.py`` (per-rank naming), ``cluster.py``
(runtime_env injection), ``channel.py`` (capture-range override).


NsightProfiler Class
---------------------

File: ``rlinf/utils/nsight_profiler.py``

Each compute worker holds a ``NsightProfiler`` instance in
``self.nsight_profiler``, created via the factory:

.. code-block:: python

   self.nsight_profiler = NsightProfiler.from_config(
       self.cfg.get("nsight_profiler", None), role="env", rank=self._rank,
   )

The ``Worker`` base class initializes a disabled profiler by default, so
workers that skip ``from_config`` get a safe no-op.

Key API:

* ``is_active`` -- ``True`` when profiling is enabled, this rank is
  selected, **and** the current step is being profiled.
* ``start(step)`` / ``stop()`` -- called by the runner; toggles
  ``cudaProfilerApi`` and sets the module-level ``_channel_nvtx_enabled``
  flag so channel/collective NVTX ranges fire only during profiled steps.
* ``@annotate(message)`` -- static decorator; wraps a worker method with
  an NVTX range.  Supports both sync and async methods.  When inactive,
  it calls the original function directly (one boolean check).


Two NVTX gating mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Compute workers** use ``NsightProfiler.is_active`` (per-instance,
  per-step).  ``@annotate`` checks this.
* **Channel / Collective / PG layers** check the module-level boolean
  ``is_channel_nvtx_enabled()``.  In compute workers this flag is toggled
  by ``start()`` / ``stop()``.  In ChannelWorkers it is set to ``True`` at
  init time and stays on permanently.


Runner Profiling Control
------------------------

All embodied runners (``EmbodiedRunner``, ``AsyncEmbodiedRunner``,
``AsyncPPOEmbodiedRunner``) share the same pattern
(``rlinf/runners/embodied_runner.py``):

1. ``_should_profile(step)`` checks if ``step`` is in the configured
   ``steps`` set.
2. ``_start_profiling(step)`` calls ``start_profile(step)`` on each
   worker group (actor, rollout, env) via Ray remote calls and ``.wait()``\s.
3. ``_stop_profiling()`` calls ``stop_profile()`` on each group.

``start_profile`` / ``stop_profile`` are defined on the ``Worker`` base
class and delegate to ``self.nsight_profiler.start()`` / ``.stop()``.


NVTX Annotation Layers
-----------------------

Annotations nest naturally in the timeline -- each layer wraps the next:

.. code-block:: text

   env/send_obs                                     ← @annotate (Layer D)
     channel/Env/put key=0_0_train_obs bytes=451608 ← channel.py
       channel/Env/put/send                         ← channel.py (sub-phase)
         collective/send type=OBJECT transport=GLOO ← collective_group.py
           collective/serialize                     ← collective_group.py
           pg/send backend=GLOO bytes=451608        ← multi_channel_pg.py

**Worker methods** (``@NsightProfiler.annotate``) -- gated by
``is_active``.  Label convention: ``{worker_type}/{method}``.

**Channel put/get** (``channel.py``) -- gated by
``is_channel_nvtx_enabled()``.  Includes payload size, queue depth,
sub-phases (send, enqueue, wait_recv).

**ChannelWorker** (``channel_worker.py``) -- always on (``_HAS_NVTX``
import check only).  Labels: ``cw/{name}/recv_from_producer``,
``cw/{name}/enqueue``, ``cw/{name}/dequeue``,
``cw/{name}/send_to_consumer``.

**Collective** (``collective_group.py``) -- gated by
``is_channel_nvtx_enabled()``.  Labels include type, transport, peer,
group name, and per-phase breakdown (metadata / cpu_payload /
accel_payload).

**Process group** (``multi_channel_pg.py``) -- gated by
``is_channel_nvtx_enabled()``.  Labels: ``pg/send`` / ``pg/recv`` with
backend, byte count, dtype; ``pg/D2H_copy`` / ``pg/H2D_copy`` for
device transfers.


Adding New Annotations
----------------------

Annotate a worker method:

.. code-block:: python

   from rlinf.utils.nsight_profiler import NsightProfiler

   @NsightProfiler.annotate("my_worker/my_method")
   def my_method(self, ...):
       ...

Add a fine-grained range inside a function:

.. code-block:: python

   from rlinf.utils.nsight_profiler import is_channel_nvtx_enabled, nvtx_range

   _nvtx = is_channel_nvtx_enabled()
   with nvtx_range("my_func/phase_a", color="blue", enabled=_nvtx):
       ...

Naming conventions: ``{worker}/{method}`` for Layer D,
``channel/{name}/{op}`` / ``cw/{name}/{op}`` for channels,
``collective/{op}`` / ``pg/{op}`` for transport.


Source File Reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 45 50

   * - File
     - Role
   * - ``rlinf/utils/nsight_profiler.py``
     - ``NsightProfiler``, ``@annotate``, ``nvtx_range()``, NVTX flag
   * - ``rlinf/scheduler/cluster/cluster.py``
     - ``runtime_env["nsight"]`` injection
   * - ``rlinf/scheduler/worker/worker_group.py``
     - Per-rank output naming
   * - ``rlinf/scheduler/worker/worker.py``
     - ``start_profile()`` / ``stop_profile()`` base methods
   * - ``rlinf/scheduler/channel/channel.py``
     - Channel put/get NVTX; ``capture-range`` override
   * - ``rlinf/scheduler/channel/channel_worker.py``
     - ChannelWorker-side NVTX (always on)
   * - ``rlinf/scheduler/collective/collective_group.py``
     - send/recv/serialize NVTX
   * - ``rlinf/scheduler/collective/multi_channel_pg.py``
     - ``pg/send`` / ``pg/recv`` NVTX
   * - ``rlinf/runners/embodied_runner.py``
     - Runner profiling control
