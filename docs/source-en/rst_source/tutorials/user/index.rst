Unified User Interface Usage
===============================

This chapter introduces the fundamental **static components** of RLinf from a user's perspective.  
These components lay the groundwork for configuring and launching scalable RL workloads.

- :doc:`yaml`  
   A comprehensive guide to all YAML configuration parameters used throughout RLinf scripts.  
   Learn how to structure your configuration files for clarity, flexibility, and reproducibility.

- :doc:`worker`  
   Introduces the concept of a *Worker*, the modular execution unit in RLinf, each handling a specific task in the RL pipeline. 
   Multiple similar Workers form a *WorkerGroup*, simplifying distributed execution and promoting scalability.

- :doc:`placement`  
   Explains how RLinf strategically assigns hardware resources across different tasks and workers  
   to ensure efficient hardware utilization and balanced execution, including not only accelerator hardware (e.g., GPUs, NPUs) but also robotic hardware and CPU-only nodes.

- :doc:`cluster`  
   Describes the globally unique *Cluster* object, responsible for coordinating all roles, processes,  
   and communication across distributed nodes in a training job.

- :doc:`flow`
   Integrates the concepts of WorkerGroup, Placement, and Cluster,  
   and presents the complete programming flow of RLinf to illustrate its high-level workflow.
   

.. toctree::
   :hidden:
   :maxdepth: 1

   yaml
   worker
   placement
   cluster
   flow
