Supported Accelerators
======================

RLinf primarily targets NVIDIA GPUs, but several embodied recipes can also run on
**AMD ROCm** and **Huawei Ascend CANN** accelerators. The pages in this section
focus on accelerator-specific dependency installation and runtime environment
variables. The training task itself — task description, PPO/GRPO algorithm
details, model download, configuration files, metrics, and results — is
platform-independent and shared with the corresponding example pages.

- :doc:`amd_rocm`
   Run the LIBERO RL example on AMD ROCm — ROCm dependency installation, OSMesa
   CPU rendering, and ROCm-specific Docker build flags.

- :doc:`ascend_cann`
   Run the LIBERO RL example on Huawei Ascend CANN — Ascend dependency
   installation, host driver mounting, and OSMesa CPU rendering.

.. toctree::
   :hidden:
   :maxdepth: 1

   amd_rocm
   ascend_cann
