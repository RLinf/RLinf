高级特性
==============================

本章将逐步深入讲解 RLinf 如何实现 **高效执行**，  
并提供实用指南，帮助你充分优化 RL 后训练工作流。

- :doc:`5D`  
   解释 RLinf 如何支持 Megatron 风格的 5D 并行，包括：  
   张量并行 (TP)、数据并行 (DP)、流水线并行 (PP)、  
   序列并行 (SP) 和上下文并行 (CP)。  
   学习如何配置和组合这些维度，以高效扩展大模型。  

- :doc:`lora`  
   展示如何在 RLinf 中集成低秩适配 (LoRA)，  
   以极小的计算开销实现参数高效的微调。  

- :doc:`version`  
   描述如何在不同的 SGLang 版本之间动态切换，  
   以满足不同的兼容性需求或实验要求。  

- :doc:`resume`  
   讲解如何从保存的检查点恢复训练，  
   以确保容错性，并为长时间或中断的训练任务提供无缝衔接。  

- :doc:`hetero`  
   介绍如何配置和使用异构软硬件集群，  
   以充分利用不同类型的计算资源和硬件设备。  

- :doc:`logger`  
   介绍如何在训练过程中可视化和跟踪关键指标。  
   目前，我们支持三种实验追踪与可视化后端：  
   TensorBoard、Weights & Biases (wandb) 和 SwanLab。  


.. toctree::
   :hidden:
   :maxdepth: 2

   5D
   lora
   version
   resume
   hetero
   logger
