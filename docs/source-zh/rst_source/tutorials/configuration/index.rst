配置
=====

本节涵盖 RLinf 训练工作负载配置的各个方面。
学习如何编写 GPU 和集群设置的 YAML 配置文件、
具身训练配置以及智能体 RL 配置。

- :doc:`gpu_cluster_config`
   GPU、集群、runner、算法、rollout 和 actor 的共享配置完整参考，
   适用于所有任务类型。

- :doc:`embodiment_config`
   具身智能 RL 训练专用配置参数：环境、模拟器、VLA 模型和机器人操作。

- :doc:`agentic_config`
   智能体和推理 RL 训练专用配置参数：
   Megatron 后端、FSDP、分词器、优化器和奖励设置。

- :doc:`hetero`
   配置异构软件和硬件集群，以高效利用不同的计算资源和设备。

- :doc:`resume`
   介绍如何从保存的检查点恢复训练，确保长时间运行或中断的训练作业
   具备容错能力和无缝续训。

- :doc:`logger`
   介绍如何在训练过程中可视化和跟踪关键指标。
   目前支持 TensorBoard、Weights & Biases (wandb) 和 SwanLab 三种后端。


.. toctree::
   :hidden:
   :maxdepth: 1

   gpu_cluster_config
   embodiment_config
   agentic_config
   hetero
   resume
   logger
