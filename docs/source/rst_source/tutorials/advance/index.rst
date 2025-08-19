Advanced Features
==============================

This chapter provides a step-by-step deep dive into how RLinf achieves **highly efficient execution**,  
offering practical guidance to help you fully optimize your RL post-training workflows.

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

- :doc:`resume`  
   Covers how to resume training from saved checkpoints,  
   ensuring fault tolerance and seamless continuation for long-running or interrupted training jobs.


.. toctree::
   :hidden:
   :maxdepth: 2

   5D
   lora
   version
   resume


.. - :doc:`Flexible Execution Modes <mode>`  
..    Describes the three execution modes supported by RLinf: **task-colocated**, **task-disaggregated**, and the novel **hybrid** mode with fine-grained pipelining.

.. - :doc:`Online Scaling Mechanism <online-scaling>`  
..    Introduces the online scaling mechanism, which allows dynamic adjustment of the number of GPUs and nodes used at each stage during runtime.

.. - :doc:`Auto Scheduling Policy <auto-scheduling>`  
..    Explains the implementation of RLinf’s auto scheduling policy, which dynamically allocates GPU resources based on workload.