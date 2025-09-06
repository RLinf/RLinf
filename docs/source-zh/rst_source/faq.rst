FAQs
====

Below are RLinf’s frequently asked questions. This section will be continuously updated, and everyone is welcome to keep asking questions to help us improve!

------

**1. NCCL “cuda invalid argument” During Task Transfer**

**Symptom:** P2P task transmission fails with ``NCCL cuda invalid argument``.

**Fix:** If you ran jobs previously on this machine, stop Ray and relaunch.

.. code-block:: bash

   ray stop

------

**2. NCCL “cuda invalid argument” When SGLang Loads parameters**

**Symptom:** SGLang reports ``NCCL cuda invalid argument`` while loading weights.

**Cause:** Placement mismatch. For example, the config uses *colocate* but the
trainer and generation actually run on different GPUs.

**Fix:** Verify the placement strategy. Ensure trainer and generation groups are
placed on the GPUs implied by your ``cluster.component_placement`` settings.

------

**3. CUDA CUresult Error (result=2) in torch_memory_saver.cpp**

**Symptom:**
``CUresult error result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=103``

**Cause:** Insufficient free GPU memory when SGLang tries to restore cached
buffers; often happens if inference weights were not unloaded before an update.

**Fix:**

- Reduce SGLang static memory usage (e.g., lower ``static_mem_fraction``).
- Ensure inference weights are properly released before reloading.


------

**4. Gloo Timeout / “Global rank x is not part of group”**

**Symptoms:**

- ``RuntimeError: [../third_party/gloo/.../unbound_buffer.cc:81] Timed out waiting ... for recv``
- ``ValueError: Global rank xxx is not part of group``

**Likely Cause:** A prior SGLang failure (see the CUresult error above) prevents
generation from completing. Megatron then waits until Gloo times out.

**Fix:**

1. Check logs for the SGLang error from the previous step.
2. Resolve the underlying SGLang restore/memory issue.
3. Relaunch the job (and Ray, if needed).

------

**5. Numerical Precision / Inference backend**

**Tip:** By default, SGLang uses **flashinfer** for attention. For stability or
compatibility, try **triton**:

.. code-block:: yaml

   rollout:
     attention_backend: triton

------

**6. Cannot Connect to GCS at ip:port**

**Symptom:** Worker nodes cannot reach the Ray head (GCS) at the given address.

**Cause:** The head-node IP is derived on node 0 via:

.. code-block:: bash

   hostname -I | awk '{print $1}'

If this selects an interface that other nodes cannot reach, workers will fail to
connect (e.g., wrong NIC order; the reachable one is ``eth0`` but a different
interface is chosen).

**Fix:**

- Confirm that the chosen IP is reachable from other nodes (e.g., ping it).
- If needed, choose the correct interface’s IP address explicitly for the Ray
  head and share that IP with workers.
