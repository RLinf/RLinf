Elastic Communication
=====================

This chapter explains how RLinf enables communication between workers
to support efficient, scalable, and elastic distributed execution.

- :doc:`collective`  
   Covers low-level, high-performance python object exchange between workers,  
   using optimized point-to-point backends such as CUDA IPC and NCCL to reduce communication overhead.

- :doc:`channel`  
   Introduces a higher-level abstraction for asynchronous communication—  
   the *Channel*—which functions as a producer-consumer queue.  
   This abstraction is essential for implementing fine-grained pipelining across different RL stages.

.. toctree::
   :hidden:
   :maxdepth: 1

   collective
   channel

