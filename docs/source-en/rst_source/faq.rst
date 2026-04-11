FAQs
====

Below are RLinf's frequently asked questions. This section will be continuously updated, and everyone is welcome to keep asking questions to help us improve!

Breakpoint Debugging
---------------------

Using Ray Distributed Debugger (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** This section introduces how to use the Ray distributed debugger in VSCode or Cursor to debug your program.

1. Prepare environment (one‑time):

   .. code-block:: bash

      uv pip install debugpy

2. Register the cluster in VSCode/Cursor:

   - Search for "Ray Distributed Debugger" in the VSCode/Cursor extension market, and install it.
     
     - If you cannot find the extension, you can download the extension VSIX file from `here <https://open-vsx.org/extension/anyscalecompute/ray-distributed-debugger>`__. After downloading, you can install it by (1) open the command palette (Ctrl+Shift+P), and type 'Extensions: Install from VSIX', and select the downloaded file. (2) Or, drag and drop the file into the VSCode/Cursor window.
  
   - Open the Ray extension panel in the VSCode/Cursor sidebar, and click the "+" button to register the cluster endpoint (e.g., ``127.0.0.1:8265`` by default, or the Ray address if you have already started the Ray cluster via ``ray start --head``).
  
   - Click the "⚙" icon (Configure Debugger Local Folder) of the added cluster, and input the path to the RLinf repo.

3. Add a `breakpoint()` anywhere you want:

   .. code-block:: python

      chains = data["chains"]
      denoise_inds = data["denoise_inds"]
      
      # Add this line to set a breakpoint
      breakpoint()
      
      # input transform
      observation = self.input_transform(data, transpose=False)
      observation = _model.Observation.from_dict(observation)

4. Run any RLinf script, for example, PPO Pi05:

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

5. Attach the debugger:

   - When execution pauses, the task appears in the Ray extension “Paused Tasks” list.
   - Click the play/attach icon to start an interactive debug session.
   - Disconnect in the VSCode/Cursor debug toolbar when done so you can attach to other paused tasks.

Common tips:

- If you run into Runtime Error the first time you debug, reload the VSCode/Cursor window by pressing `Ctrl+Shift+P`, and type 'Developer: Reload Window'.

**Reference:** Ray Distributed Debugger docs: https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html

Using Ray Legacy Debugger (Fallback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** This section introduces how to debug when you cannot install the Ray distributed debugger extension using the legacy debug mode.

1. First, set a breakpoint at the location where you want to debug, as shown below:

.. code-block:: python

   chains = data["chains"]
   denoise_inds = data["denoise_inds"]
   
   # Add this line to set a breakpoint
   breakpoint()
   
   # input transform
   observation = self.input_transform(data, transpose=False)
   observation = _model.Observation.from_dict(observation)

2. Then run the corresponding program, for example, PPO Pi05:

.. code-block:: bash

   RAY_DEBUG=legacy bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

3. Run the program until you see the ``use 'ray debug' to connect ...`` prompt, then open a new terminal and execute ``ray debug`` to connect to the debugger.

.. code-block:: bash

   ray debug

**Reference:** See Ray's official documentation at https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html.

Rendering Issues
------------------------------------

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

CUDA Issues
-------------

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

Numerical Precision / Inference backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tip:** By default, SGLang uses **flashinfer** for attention. For stability or
compatibility, try **triton**:

.. code-block:: yaml

   rollout:
     attention_backend: triton
