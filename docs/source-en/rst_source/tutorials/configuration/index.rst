Configuration
=============

This section covers all aspects of configuring RLinf for your training workloads.
Learn how to structure YAML configuration files for GPU and cluster setup,
embodied training, and agentic RL.

- :doc:`gpu_cluster_config`
   Comprehensive reference for GPU, cluster, runner, algorithm, rollout, and actor
   configuration shared across all task types.

- :doc:`embodiment_config`
   Configuration parameters specific to embodied RL training: environments,
   simulators, VLA models, and robot manipulation.

- :doc:`agentic_config`
   Configuration parameters specific to agentic and reasoning RL training:
   Megatron backend, FSDP, tokenizer, optimization, and reward settings.

- :doc:`hetero`
   Configure heterogeneous software and hardware clusters to use different
   compute resources and devices efficiently.

- :doc:`resume`
   Covers how to resume training from saved checkpoints,
   ensuring fault tolerance and seamless continuation for long-running or interrupted training jobs.

- :doc:`logger`
   Introduces how to visualize and track key metrics during your training process.
   Currently supports TensorBoard, Weights & Biases (wandb), and SwanLab.


.. toctree::
   :hidden:
   :maxdepth: 1

   gpu_cluster_config
   embodiment_config
   agentic_config
   hetero
   resume
   logger
