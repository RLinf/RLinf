Advanced Features
==============================

This chapter provides a step-by-step deep dive into how RLinf achieves **highly efficient execution**,
offering practical guidance to help you fully optimize your RL post-training workflows.

- :doc:`lora`
   Demonstrates how to integrate Low-Rank Adaptation (LoRA) into RLinf,
   enabling parameter-efficient fine-tuning for large-scale models with minimal compute overhead.

- :doc:`5D`
   Explains how RLinf supports Megatron-style 5D parallelism, including:
   Tensor Parallelism (TP), Data Parallelism (DP), Pipeline Parallelism (PP),
   Sequence Parallelism (SP), and Context Parallelism (CP).
   Learn how to configure and combine these dimensions to scale large models efficiently.

- :doc:`cluster`
   Describes the globally unique *Cluster* object, responsible for coordinating all roles,
   processes, and communication across distributed nodes. Covers Ray initialization,
   node discovery, and worker allocation.

- :doc:`collective`
   Covers low-level, high-performance Python object exchange between workers,
   using optimized point-to-point backends such as CUDA IPC and NCCL to reduce communication overhead.

- :doc:`version`
   Describes how to dynamically switch between different SGLang versions
   to accommodate varying compatibility needs or experimental requirements.

- :doc:`profile`
   Introduces the ``cluster.profiling`` configuration used to wrap selected
   Ray worker groups with a backend-specific profiler (``nsys profile`` for
   NVIDIA, ``rocprof-sys-python`` for AMD), including common fields such as
   ``enabled``, ``worker_groups``, ``steps``, and ``output_dir``.

- :doc:`dynamic_scheduling`
   Covers online scaling and dynamic scheduling in RLinf: how resources are
   elastically scaled and migrated across components in real time to maximize
   throughput and utilization, including prerequisites, configuration, and
   optional scheduling policies.

- :doc:`auto_placement`
   Details the concrete implementation of auto-placement in RLinf,
   including how to configure it properly to enable auto-placement.

.. toctree::
   :hidden:
   :maxdepth: 2

   lora
   5D
   cluster
   collective
   version
   profile
   dynamic_scheduling
   auto_placement
