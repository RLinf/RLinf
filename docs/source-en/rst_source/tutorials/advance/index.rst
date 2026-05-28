Advanced Features
==============================

This chapter provides a step-by-step deep dive into how RLinf achieves **highly efficient execution**,
offering practical guidance to help you fully optimize your RL post-training workflows.

- :doc:`cluster`
   Describes the globally unique *Cluster* object, responsible for coordinating all roles,
   processes, and communication across distributed nodes. Covers Ray initialization,
   node discovery, and worker allocation.

- :doc:`collective`
   Covers low-level, high-performance Python object exchange between workers,
   using optimized point-to-point backends such as CUDA IPC and NCCL to reduce communication overhead.

- :doc:`5D`
   Explains how RLinf supports Megatron-style 5D parallelism, including:
   Tensor Parallelism (TP), Data Parallelism (DP), Pipeline Parallelism (PP),
   Sequence Parallelism (SP), and Context Parallelism (CP).
   Learn how to configure and combine these dimensions to scale large models efficiently.

- :doc:`lora`
   Demonstrates how to integrate Low-Rank Adaptation (LoRA) into RLinf,
   enabling parameter-efficient fine-tuning for large-scale models with minimal compute overhead.

- :doc:`version`
   Describes how to dynamically switch between different SGLang versions
   to accommodate varying compatibility needs or experimental requirements.

- :doc:`nsight`
   Introduces the Hydra-based ``cluster.nsight`` configuration used to wrap
   selected Ray worker groups with ``nsys profile``, including how to enable,
   disable, and target worker groups for system-level traces.


- :doc:`online_scaling`
   Provides an overview of the online scaling mechanism, focusing on the design
   principles behind RLinf's adaptive scaling capabilities.

- :doc:`dynamic_scheduling`
   Details the concrete implementation of dynamic scheduling in RLinf,
   including how to configure it properly to enable dynamic scheduling.

- :doc:`auto_placement`
   Details the concrete implementation of auto-placement in RLinf,
   including how to configure it properly to enable auto-placement.

.. toctree::
   :hidden:
   :maxdepth: 2

   cluster
   collective
   5D
   lora
   version
   nsight
   online_scaling
   dynamic_scheduling
   auto_placement
