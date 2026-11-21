Frequently Asked Questions (FAQ)
================================

Below are RLinf's frequently asked questions. This section will be continuously updated, and everyone is welcome to keep asking questions to help us improve!

.. _faq-rendering:

Rendering Issues
------------------------------------

Which GPU does EGL rendering use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf assigns each worker its own rendering device: it asks the driver for the
EGL index of the GPU that worker was given and exports it as
``MUJOCO_EGL_DEVICE_ID`` (MuJoCo, robosuite) and ``EGL_DEVICE_ID`` (other EGL
renderers, e.g. pyrender). This matters because EGL device indices and CUDA
device ids are different namespaces: EGL lists every device the driver can see,
so in a container holding a subset of a node's GPUs, CUDA device 0 is usually
*not* EGL device 0.

Set ``MUJOCO_EGL_DEVICE_ID`` yourself only to override that choice; an explicit
value is always respected. Setting it to a CUDA ordinal renders on the wrong GPU
whenever the two namespaces disagree.

RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0 (inclusive), got 1.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** The above error message when running simulators with MUJOCO_GL environment variable set to "egl".

**Cause:** This error occurs because your GPU environment is not properly setup for graphics rendering, especially on NVIDIA GPUs.

**Fix:** Check whether you have this file `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`. 

1. If you have this file, check whether you also have `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`. If not, create this file and add the following content:

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

2. If you do not have this file, it means your NVIDIA driver is not properly installed with the graphics capability. You can try the following solutions:

   * Reinstall the NVIDIA driver with the correct options to enable graphics capabilities. There are several options when installing the NVIDIA driver that disable the graphics driver. Therefore, you need to try installing NVIDIA's graphics driver. On Ubuntu, this can be done with the command ``apt install libnvidia-gl-<driver-version>``, see NVIDIA's documentation at https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html#compute-only-headless-and-desktop-only-no-compute-installation for details.

   * Use **osmesa** for rendering, change the `MUJOCO_GL` and `PYOPENGL_PLATFORM` environment variables in our running script to "osmesa" for this. However, this may cause the rollout process to be 10x slower than EGL as it uses CPU for rendering.

Vulkan Incompatible GPU Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Incompatible GPU driver when running Vulkan-based simulators like ManiSkill.

**Cause:** The GPU driver is not properly installed with graphics rendering capability or the version is not compatible.

**Fix:** 

1. Check the above to ensure your GPU driver is installed with graphics rendering support.
2. Check the driver version of your device via ``nvidia-smi``.
   
   * For Ampere GPUs (A100, A800, and RTX 30 series), recommend using driver version 535 (frequently tested: 535.161.08). Higher versions like 570 and 580 are known to trigger this problem.
   * For Hopper GPUs (H100, L40S, and RTX 40 series), recommend using driver version 570.
  
Network Issues
-----------------

Cannot Connect to GCS at ip:port
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Worker nodes cannot reach the Ray head (GCS) at the given address.

**Cause:** The head-node IP is derived on node 0 via:

.. code-block:: bash

   hostname -I | awk '{print $1}'

If this selects an interface that other nodes cannot reach, workers will fail to
connect (e.g., wrong NIC order; the reachable one is ``eth0`` but a different
interface is chosen).

**Fix:**

- Confirm that the chosen IP is reachable from other nodes (e.g., ping it).
- If needed, choose the correct interface's IP address explicitly for the Ray
  head and share that IP with workers.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** P2P task transmission fails with ``NCCL cuda invalid argument``.

**Fix:** If you ran jobs previously on this machine, stop Ray and relaunch.

.. code-block:: bash

   ray stop

NCCL "cuda invalid argument" When SGLang Loads parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** SGLang reports ``NCCL cuda invalid argument`` while loading weights.

**Cause:** Placement mismatch. For example, the config uses *colocate* but the
trainer and generation actually run on different GPUs.

**Fix:** Verify the placement strategy. Ensure trainer and generation groups are
placed on the GPUs implied by your ``cluster.component_placement`` settings.

CUDA CUresult Error (result=2) in torch_memory_saver.cpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:**
``CUresult error result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=103``

**Cause:** Insufficient free GPU memory when SGLang tries to restore cached
buffers; often happens if inference weights were not unloaded before an update.

**Fix:**

- Reduce SGLang static memory usage (e.g., lower ``gpu_memory_utilization``).
- Ensure inference weights are properly released before reloading.

Gloo Timeout / "Global rank x is not part of group"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- ``RuntimeError: [../third_party/gloo/.../unbound_buffer.cc:81] Timed out waiting ... for recv``
- ``ValueError: Global rank xxx is not part of group``

**Likely Cause:** A prior SGLang failure (see the CUresult error above) prevents
generation from completing. Megatron then waits until Gloo times out.

**Fix:**

1. Check logs for the SGLang error from the previous step.
2. Resolve the underlying SGLang restore/memory issue.
3. Relaunch the job (and Ray, if needed).

FSDP Collective Times Out on the Backend Watchdog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** An FSDP actor is killed by the collective watchdog even though the
step was still making progress:

.. code-block:: text

   WorkNCCL(SeqNum=1878, OpType=_ALLGATHER_BASE, ..., Timeout(ms)=1800000) ran for
   1800000 milliseconds before timing out.

The timeout in the message is the backend's built-in default — 1800000 ms on
NCCL and Gloo, 3636000 ms on Ascend HCCL.

**Likely Cause:** One rank spent longer than that between two FSDP collectives —
a large gradient accumulation step, a slow checkpoint write, or a rank paused
under a debugger while the others wait in an all-gather.

**Fix:** FSDP collectives use the same timeout as the rest of RLinf's
inter-worker communication, which defaults to 180 minutes. Raise it with
``RLINF_TIMEOUT``, in minutes:

.. code-block:: bash

   export RLINF_TIMEOUT=360

Ray captures the environment when it starts, so export this on every node
**before** ``ray start``.

Numerical Precision / Inference backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tip:** By default, SGLang uses **flashinfer** for attention. For stability or
compatibility, try **triton**:

.. code-block:: yaml

   rollout:
     attention_backend: triton

Worker Aborts While Creating a Process Group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** A worker exits while it connects to a new peer. The first native
error in its log is:

.. code-block:: text

   pybind11_object_dealloc(): Tried to deallocate unregistered instance!

The process then receives ``SIGABRT``, and the driver only reports that the
actor died.

**Likely Cause:** Before PyTorch 2.7, several ``torch.distributed``
constructors, including the Gloo backend and ``TCPStore``, register their new
Python object without holding the GIL. RLinf creates the process groups for
different peers on separate threads, so two registrations can overlap and
corrupt pybind11's table of live objects. The BEHAVIOR install (PyTorch 2.5.1)
and the default Ascend install (PyTorch 2.6.0) are affected.

**Fix:** On these versions RLinf creates Gloo backends and process-group
options through constructors that register under the GIL, so Gloo groups no
longer trigger the race. ``TCPStore`` and the debug wrapper enabled by
``TORCH_DISTRIBUTED_DEBUG=DETAIL`` have no such constructor. ``TCPStore`` is
created once per collective group, so a rare abort is still possible. Use
PyTorch 2.7 or later where the rest of the stack allows it. When reporting a
failure, include the PyTorch version and the first native stack trace; Ray's
``actor died`` message does not identify the cause.

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
