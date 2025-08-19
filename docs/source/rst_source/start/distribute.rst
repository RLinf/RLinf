Multi-node Training
====================================

This document describes how to launch a multi-node Ray cluster for distributed training using the provided bash scripts.

Overview
--------

The cluster launch involves two helper scripts:

- ``ray_utils/start_ray.sh``: starts a Ray head or worker node based on the ``RANK`` environment variable.  

- ``ray_utils/check_ray.sh``: (head node only) blocks until the Ray cluster reports the required total GPU count.

All nodes must run ``start_ray.sh``, while only the head node runs ``check_ray.sh``.

.. Prerequisites
.. -------------

.. - Ray must be installed on every machine in the cluster.  
.. - Set the environment variable ``RANK`` on each node:  
..   - ``RANK=0`` for the head node  
..   - ``RANK>0`` for worker nodes  

Cluster Startup
---------------

1. **Start Ray on all nodes**  
   On **every** node (head and workers), run:

   .. code-block:: bash

      scripts/start_ray.sh

2. **Wait for cluster readiness**  
   On the **head node** only, execute:

   .. code-block:: bash

      scripts/check_ray.sh <total_gpu_count>

   This will block until the cluster’s reported GPU count matches ``<total_gpu_count>``.

Launching Training
------------------

Once the Ray cluster is up and ready, launch your training script:

.. code-block:: bash

   bash xxxx # TODO:

**Note**: Only a **single** invocation of ``xx.sh #TODO:`` is required (typically on the head node). Ray will distribute the workload automatically across all nodes in the cluster.
