FAQ
===

Below are RLinf's frequently asked questions, ordered from first-run blockers to cluster and debugging issues.

Jump to Symptom
---------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Symptom / Question
     - Section
   * - Second ``install.sh`` broke my existing stack
     - :ref:`Multi-Stack .venv Overwrites <faq-venv>`
   * - Running an 8-GPU config on 1 GPU
     - :ref:`Run an 8-GPU Config on 1 GPU <faq-single-gpu>`
   * - ``model_path`` vs ``ckpt_path`` vs ``resume_dir``
     - :ref:`model_path vs ckpt_path <faq-ckpt>`
   * - ``total_num_envs`` or ``num_action_chunks`` divisibility error
     - :ref:`Env and Action-Chunk Divisibility <faq-divisibility>`
   * - Slow downloads or Ray GCS connection failure
     - :ref:`Download and Network Issues <faq-network>`
   * - CUDA OOM, NCCL errors, or collective timeouts
     - :ref:`CUDA and Memory Issues <faq-cuda>`
   * - ``MUJOCO_EGL_DEVICE_ID`` or Vulkan driver errors
     - :ref:`Rendering Issues <faq-rendering>`
   * - Attach a breakpoint debugger to Ray workers
     - :ref:`Breakpoint Debugging <faq-debugging>`

.. _faq-setup:

Setup and Configuration
-----------------------

.. _faq-venv:

Installing a Second Model or Environment Breaks the First Stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Running ``bash requirements/install.sh ...`` for a second model or simulator removes or breaks packages needed by the first one.

**Cause:** ``install.sh`` defaults to ``.venv`` and overwrites that virtual environment with the new dependency set.

**Fix:** Pass a distinct ``--venv <dir>`` for each model/environment stack and activate the matching environment before launching:

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla --env maniskill_libero --venv .venv-maniskill
   bash requirements/install.sh embodied --model openpi --env libero --venv .venv-libero

.. _faq-single-gpu:

Running an 8-GPU Config on 1 GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** A benchmark recipe targets 8 GPUs (for example ``actor,rollout,env: 0-7``) and fails on a single-GPU workstation with placement or out-of-memory errors.

**Fix:** Pin all components to GPU 0, shrink the parallel env count and batch size to match 1 GPU, and enable offload:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh <config_name> \
     cluster.component_placement.actor,rollout,env=0 \
     env.train.total_num_envs=8 \
     actor.micro_batch_size=1 \
     actor.global_batch_size=8 \
     actor.enable_offload=True \
     rollout.enable_offload=True

Keep ``actor.global_batch_size`` divisible by ``actor.micro_batch_size`` and ``env.train.total_num_envs`` divisible by ``rollout.pipeline_stage_num``.

.. _faq-ckpt:

Difference Between model_path, ckpt_path, and resume_dir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Confusion over which path field to pass for a Hugging Face model, a trained ``.pt`` weight file, or resuming an interrupted run.

**Fix:** Use each field for its distinct role:

- ``actor.model.model_path`` / ``rollout.model.model_path``: Base model directory or Hugging Face ``owner/repo`` ID (config, tokenizer, normalization stats, and base weights). During training, rollout inherits ``actor.model.model_path`` automatically; during standalone evaluation, set ``rollout.model.model_path``.
- ``runner.ckpt_path``: Trained ``.pt`` weight file (such as ``full_weights.pt`` from RL or SFT) loaded on top of the base model architecture defined by ``model_path`` for evaluation or warm-start.
- ``runner.resume_dir``: Full checkpoint directory (such as ``logs/<run>/checkpoints/global_step_50``) that restores model weights, optimizer state, and step counters to continue an interrupted training run.

.. _faq-divisibility:

total_num_envs or num_action_chunks Divisibility Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Startup validation fails with an assertion about ``total_num_envs``, ``env_world_size``, or ``max_steps_per_rollout_epoch`` not being divisible.

**Fix:** Satisfy both divisibility rules:

1. ``total_num_envs`` must be divisible by ``<number of env workers> × rollout.pipeline_stage_num``. For example, ``10`` fails on 4-GPU or 8-GPU runs; use ``8``, ``16``, or ``128`` so it divides cleanly across 1, 2, 4, or 8 GPUs.
2. ``max_steps_per_rollout_epoch`` must be divisible by ``rollout.model.num_action_chunks`` (for example, multiples of ``5`` for OpenPI LIBERO, ``15`` for RoboLab, or ``24`` for SIMPLE).

.. _faq-network:

Download and Network Issues
---------------------------

Slow or Timed-Out Hugging Face / GitHub Downloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``hf download`` or ``requirements/install.sh`` stalls or times out when fetching models, datasets, or Python wheels.

**Fix:** In mainland China or behind a restricted network, enable the mirror options:

.. code-block:: bash

   export HF_ENDPOINT=https://hf-mirror.com
   bash requirements/install.sh embodied --model openvla --env maniskill_libero --use-mirror

You can also pull prebuilt images from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ (see :doc:`../start/installation`).

Cannot Connect to GCS at ip:port
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Worker nodes cannot reach the Ray head (GCS) at the given address.

**Cause:** The head-node IP is derived on node 0 via:

.. code-block:: bash

   hostname -I | awk '{print $1}'

If this selects an interface that other nodes cannot reach, workers fail to connect (for example, wrong NIC order where ``eth0`` is reachable but another interface comes first).

**Fix:**

- Confirm that the chosen IP is reachable from other nodes (for example, with ``ping``).
- Pass the reachable interface's IP address explicitly when starting the Ray head and share that IP with workers.

.. _faq-cuda:

CUDA and Memory Issues
----------------------

.. _faq-oom:

Out of Memory (OOM) During Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``torch.OutOfMemoryError: CUDA out of memory`` during rollout or actor updates.

**Fix:** Lower per-GPU memory usage and enable offload in the recipe YAML:

.. code-block:: yaml

   env:
     train:
       total_num_envs: 8
       enable_offload: True
   rollout:
     enable_offload: True
   actor:
     micro_batch_size: 1
     enable_offload: True

Keep ``actor.global_batch_size`` divisible by ``actor.micro_batch_size`` times the number of actor workers, and keep ``env.train.total_num_envs`` divisible by the number of env workers times ``rollout.pipeline_stage_num``.

NCCL "cuda invalid argument" During Task Transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** P2P task transmission fails with ``NCCL cuda invalid argument``.

**Fix:** If you ran jobs previously on this machine, stop Ray and relaunch (check ``ray status`` first on shared hosts):

.. code-block:: bash

   ray stop

NCCL "cuda invalid argument" When SGLang Loads Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** SGLang reports ``NCCL cuda invalid argument`` while loading weights.

**Cause:** Placement mismatch. For example, the config uses *collocated* placement but the trainer and generation actually run on different GPUs.

**Fix:** Verify the placement strategy. Ensure trainer and generation groups are placed on the GPUs implied by your ``cluster.component_placement`` settings.

FSDP Collective Times Out on the Backend Watchdog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** An FSDP actor is killed by the collective watchdog even though the step was still making progress:

.. code-block:: text

   WorkNCCL(SeqNum=1878, OpType=_ALLGATHER_BASE, ..., Timeout(ms)=1800000) ran for
   1800000 milliseconds before timing out.

The timeout in the message is the backend's built-in default — 1800000 ms on NCCL and Gloo, 3636000 ms on Ascend HCCL.

**Likely Cause:** One rank spent longer than that between two FSDP collectives — a large gradient accumulation step, a slow checkpoint write, or a rank paused under a debugger while the others wait in an all-gather.

**Fix:** FSDP collectives use the same timeout as the rest of RLinf's inter-worker communication, which defaults to 180 minutes. Raise it with ``RLINF_TIMEOUT``, in minutes:

.. code-block:: bash

   export RLINF_TIMEOUT=360

Ray captures the environment when it starts, so export this on every node **before** ``ray start``.

Numerical Precision / Inference Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tip:** RLinf defaults ``rollout.sglang.attention_backend`` to ``triton`` for stability and compatibility across GPUs (you can switch to ``flashinfer`` if supported on your GPU; for vLLM, configure ``rollout.vllm.attention_backend``, which defaults to ``FLASH_ATTN``):

.. code-block:: yaml

   rollout:
     sglang:
       attention_backend: triton  # or flashinfer

.. dropdown:: Advanced: SGLang torch_memory_saver CUresult Error (result=2)

   **Symptom:** ``CUresult error result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=103``

   **Cause:** Insufficient free GPU memory when SGLang tries to restore cached buffers; often happens if inference weights were not unloaded before an update.

   **Fix:**

   - Reduce SGLang static memory usage (for example, lower ``gpu_memory_utilization``).
   - Ensure inference weights are properly released before reloading.

.. dropdown:: Advanced: Gloo Timeout / "Global rank x is not part of group"

   **Symptoms:**

   - ``RuntimeError: [../third_party/gloo/.../unbound_buffer.cc:81] Timed out waiting ... for recv``
   - ``ValueError: Global rank xxx is not part of group``

   **Likely Cause:** A prior SGLang failure (see the ``CUresult`` error above) prevents generation from completing. Megatron then waits until Gloo times out.

   **Fix:**

   1. Check logs for the SGLang error from the previous step.
   2. Resolve the underlying SGLang restore/memory issue.
   3. Relaunch the job (and Ray, if needed).

.. dropdown:: Advanced: Worker Aborts Creating a Process Group (pybind11_object_dealloc)

   **Symptom:** A worker exits while connecting to a new peer. The first native error in its log is:

   .. code-block:: text

      pybind11_object_dealloc(): Tried to deallocate unregistered instance!

   The process then receives ``SIGABRT``, and the driver only reports that the actor died.

   **Likely Cause:** Before PyTorch 2.7, several ``torch.distributed`` constructors, including the Gloo backend and ``TCPStore``, register their new Python object without holding the GIL. RLinf creates process groups for different peers on separate threads, so two registrations can overlap and corrupt pybind11's table of live objects. The BEHAVIOR install (PyTorch 2.5.1) and the default Ascend install (PyTorch 2.6.0) are affected.

   **Fix:** On these versions RLinf creates Gloo backends and process-group options through constructors that register under the GIL, so Gloo groups no longer trigger the race. ``TCPStore`` and the debug wrapper enabled by ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` have no such constructor; ``TCPStore`` is created once per collective group, so a rare abort is still possible. Use PyTorch 2.7 or later where the rest of the stack allows it. When reporting a failure, include the PyTorch version and the first native stack trace; Ray's ``actor died`` message does not identify the cause.

.. _faq-rendering:

Rendering Issues
----------------

Which GPU Does EGL Rendering Use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf assigns each worker its own rendering device: it asks the driver for the EGL index of the GPU that worker was given and exports it as ``MUJOCO_EGL_DEVICE_ID`` (MuJoCo, robosuite) and ``EGL_DEVICE_ID`` (other EGL renderers, such as pyrender). This matters because EGL device indices and CUDA device IDs are different namespaces: EGL lists every device the driver can see, so in a container holding a subset of a node's GPUs, CUDA device 0 is usually *not* EGL device 0.

Set ``MUJOCO_EGL_DEVICE_ID`` yourself only to override that choice; an explicit value is always respected. Setting it to a CUDA ordinal renders on the wrong GPU whenever the two namespaces disagree.

EGL Device ID Out of Range (MUJOCO_EGL_DEVICE_ID)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Running simulators with ``MUJOCO_GL="egl"`` fails with:

.. code-block:: text

   RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0 (inclusive), got 1.

**Cause:** Your GPU environment is not set up for graphics rendering, especially on NVIDIA GPUs.

**Fix:** Check whether you have ``/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0``.

1. If you have this file, check whether you also have ``/usr/share/glvnd/egl_vendor.d/10_nvidia.json``. If not, create this file with the following content:

   .. code-block:: json

      {
         "file_format_version" : "1.0.0",
         "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
         }
      }

   And add the following environment variable to your running script:

   .. code-block:: shell

      export NVIDIA_DRIVER_CAPABILITIES="all"

2. If you do not have this file, your NVIDIA driver is installed without graphics capability. Try one of the following solutions:

   * Reinstall the NVIDIA driver with graphics capabilities enabled. On Ubuntu, run ``apt install libnvidia-gl-<driver-version>``; see NVIDIA's `Ubuntu driver installation guide <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html#compute-only-headless-and-desktop-only-no-compute-installation>`__ for details.

   * Use **osmesa** for rendering by setting ``MUJOCO_GL=osmesa`` and ``PYOPENGL_PLATFORM=osmesa`` in your running script. CPU rendering with osmesa is roughly 10× slower than EGL.

Vulkan Incompatible GPU Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Incompatible GPU driver when running Vulkan-based simulators like ManiSkill.

**Cause:** The GPU driver is not installed with graphics rendering capability or the version is not compatible.

**Fix:**

1. Check the steps above to ensure your GPU driver has graphics rendering support.
2. Check your driver version via ``nvidia-smi``:

   * For Ampere GPUs (A100, A800, and RTX 30 series), use driver version 535 (frequently tested: 535.161.08). Higher versions like 570 and 580 are known to trigger this issue.
   * For Hopper (H100) and Ada Lovelace (L40S, RTX 40 series) GPUs, use driver version 570.

.. _faq-debugging:

Breakpoint Debugging
--------------------

Use Ray Distributed Debugger (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Debug Ray workers in VSCode or Cursor with the Ray Distributed Debugger extension:

1. Install ``debugpy`` once:

   .. code-block:: bash

      uv pip install debugpy

2. Install the "Ray Distributed Debugger" extension in VSCode or Cursor (or download the VSIX from `Open VSX <https://open-vsx.org/extension/anyscalecompute/ray-distributed-debugger>`__). Open the Ray sidebar panel, click "+" to register ``127.0.0.1:8265`` (or your Ray head address), and set the local folder to the RLinf repository path.

3. Insert ``breakpoint()`` where you want execution to pause, then run your script:

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

4. Open the Ray extension's "Paused Tasks" list and attach to the paused worker. Disconnect when finished to attach to another task. Reload the editor window (``Developer: Reload Window``) if the first attach raises a runtime error.

Reference: `Ray Distributed Debugger docs <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`__.

Use Ray Legacy Debugger (Fallback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the editor extension is unavailable, insert ``breakpoint()`` and launch with ``RAY_DEBUG=legacy``:

.. code-block:: bash

   RAY_DEBUG=legacy bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

When the log prints ``use 'ray debug' to connect``, attach from another terminal with ``ray debug``. Reference: `Ray Debugging docs <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`__.
