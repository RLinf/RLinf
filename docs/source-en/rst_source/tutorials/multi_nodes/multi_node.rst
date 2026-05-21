Multi-Node Ray Cluster Setup
============================

This guide explains how to start a Ray cluster across **multiple machines** and run
RLinf training on top of it. It applies to all Ray-based RLinf workloads, including
embodied RL, reasoning, and agents.


Prerequisites
-------------

Before you begin, confirm that:

* All nodes can reach the head at ``<head_ip>:6379`` (or your chosen port).
* Every node has the **same versions** of Python, Ray (``ray>=2.47.0``), and RLinf dependencies.
* ``cluster.num_nodes`` in your config matches the actual cluster size (see :doc:`../user/yaml`).

.. important::

   Ray **freezes** the Python interpreter path and environment variables when you run
   ``ray start``. All processes Ray launches on that node inherit that environment.
   On **each node**, ``source`` your virtualenv and install dependencies **before**
   ``ray start``. Packages installed **after** ``ray start`` are **not** visible to Ray workers.


Step 1: Set environment variables on each node
----------------------------------------------

On **every node**, before ``ray start``, set the node rank (required):

.. code-block:: bash

   export RLINF_NODE_RANK=<0..N-1>   # unique in the cluster; head is usually 0

If the machine has multiple NICs and Ray/collective communication should use a specific
interface:

.. code-block:: bash

   export RLINF_COMM_NET_DEVICES=<interface>   # e.g. eth0, enp3s0

Use ``ip addr`` or ``ifconfig`` to find which interface is reachable from other nodes.


Step 2: Start the Ray cluster
-----------------------------

Option A: Manual ``ray start`` on each node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pick one machine as **head** (``RLINF_NODE_RANK=0``). Its IP must be reachable from all
workers; call it ``<head_ip>``.

**Head node:**

.. code-block:: bash

   export RLINF_NODE_RANK=0
   # optional: export RLINF_COMM_NET_DEVICES=eth0
   ray start --head --port=6379 --node-ip-address=<head_ip>

**Worker nodes** (``RLINF_NODE_RANK`` = 1, 2, …, N-1):

.. code-block:: bash

   export RLINF_NODE_RANK=1
   # optional: export RLINF_COMM_NET_DEVICES=eth0
   ray start --address='<head_ip>:6379'

``--node-ip-address`` must be the address other nodes use to reach the head (private IP,
VPC IP, or overlay IP; see cloud platform notes below). Port ``6379`` can be changed if
unused; head and workers must use the same port.

Option B: Repository script (shared filesystem)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all nodes mount the **same shared directory** and can read/write
``ray_utils/ray_head_ip.txt``:

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt    # first time or when rebuilding the cluster
   RANK=0 bash ray_utils/start_ray.sh   # head
   RANK=1 bash ray_utils/start_ray.sh   # worker 1
   # ...

The script uses ``RANK`` for head/worker roles and does **not** set ``RLINF_NODE_RANK``
automatically. Still export ``RLINF_NODE_RANK`` per Option A before running the script,
or bake it into your launch script.


Step 3: Enable code sync (optional)
-----------------------------------

When the driver and workers **do not share the same filesystem** (cloud-edge,
heterogeneous datacenters, etc.), enable Ray task-level code sync **before** starting
the training script. The driver packs the ``rlinf/`` package into ``runtime_env.py_modules``;
workers do not need a matching local checkout.

.. code-block:: bash

   export RLINF_CODE_WORKING_DIR=auto

Details:

* ``RLINF_CODE_WORKING_DIR``
  - unset / ``0`` / ``false``: sync off (default; each node must have a consistent local tree).
  - ``auto``: infer the repo root from the installed ``rlinf`` package or current directory.
  - **absolute path**: repo root containing ``pyproject.toml`` and ``rlinf/``, or the ``rlinf`` package directory.

Only the ``rlinf/`` subtree is synced—not ``examples/``, ``docs/``, etc. Example configs
and data paths must still be reachable on each node or via shared storage/NFS.

.. note::

   **When code sync is enabled**

   * **Do not store large files under ``rlinf/``**: sync packages the **entire local ``rlinf/`` directory** from the launch node. Avoid logs, checkpoints, caches, datasets, or other large artifacts under ``rlinf/`` to keep packaging and transfer fast.
   * **Prepare models and assets separately**: **model weights, simulator assets, datasets**, etc. are **not** synced. Download them on each worker to the paths in your config, or mount via NFS/shared storage, and verify every node can access those paths.

These variables take effect on RLinf's first ``ray.init`` (``Cluster`` initialization). Do not call ``ray.init`` manually outside the training process.


Step 4: Verify cluster status
-----------------------------

On **any node** that has run ``ray start``:

.. code-block:: bash

   ray status

Confirm **node count**, per-node **CPU/GPU**, and **state** (``ALIVE``) match expectations.
For example, with ``2`` nodes and ``8`` GPUs each, you should see ``16`` GPUs total.

You can also wait for resources with (argument = total **GPU count** in the cluster):

.. code-block:: bash

   bash ray_utils/check_ray.sh 16

If nodes are missing, check firewalls, whether ``--node-ip-address`` is reachable, and
worker ``ray start`` errors.


Step 5: Launch RLinf training
-----------------------------

1. Set ``cluster.num_nodes`` in your task YAML to the actual node count (consistent with ``RLINF_NODE_RANK``).
2. On **any node** that has joined the Ray cluster, ``cd`` to the RLinf repo and run an entry script, e.g. embodied:

.. code-block:: bash

   cd <RLinf repo root>
   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi

Reasoning example:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

The launch node must:

* Have run ``ray start`` locally (``ray status`` shows the full cluster);
* Reach config files, models, and data (shared storage or code sync);
* If code sync is enabled, have ``export RLINF_CODE_WORKING_DIR=...`` in the **same terminal**.

.. note::

   For ``node_groups`` and cross-machine placement, see :doc:`../user/placement` and :doc:`hetero`.
   Cloud-edge ``component_placement`` examples are in :doc:`cloud-edge`.


Stopping and rebuilding the cluster
-----------------------------------

After ``ray stop`` on all nodes and clearing state, restart using this guide:

.. code-block:: bash

   ray stop
   rm -f ray_utils/ray_head_ip.txt   # if you used start_ray.sh

After changing the Python environment or ``RLINF_NODE_RANK``, ``ray stop`` on all affected
nodes and ``ray start`` again.


FAQ
---

**Worker cannot reach head**

Check that ``<head_ip>`` is pingable/telnet-able from workers, security groups/iptables allow
port ``6379``, and the head did not bind ``--node-ip-address`` to ``127.0.0.1``.

**``ray status`` shows fewer nodes than ``cluster.num_nodes``**

Wait for workers to finish joining; confirm ``ray start`` succeeded on each node and you
are not mixing multiple independent Ray clusters.

**Worker cannot import latest code**

Check ``RLINF_CODE_WORKING_DIR``; without sync, ``rlinf/`` must match on all nodes. With
sync, only the ``rlinf/`` package is shipped (avoid large files under ``rlinf/``); configs
under ``examples/``, models, and simulator assets must still be local or on shared storage.

**Ray version mismatch**

All nodes should use the same ``ray`` version (RLinf requires ``ray>=2.47.0``).
