Configuration
=============

This section covers all aspects of configuring RLinf for your training workloads.
Learn how to structure YAML configuration files, set up clusters and hardware,
manage checkpoints, and visualize training metrics.

- :doc:`yaml`
   A comprehensive guide to all YAML configuration parameters used throughout RLinf.
   Learn how to structure your configuration files for clarity, flexibility, and reproducibility.

- :doc:`cluster`
   Describes the globally unique *Cluster* object, responsible for coordinating all roles,
   processes, and communication across distributed nodes in a training job.

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

   yaml
   cluster
   hetero
   resume
   logger
