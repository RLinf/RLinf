Multi-Node Training
===================

Start a Ray cluster across multiple machines and run RLinf workloads on top of
it. This guide applies to all Ray-based RLinf jobs, including embodied RL,
reasoning, and agents.

.. note::

   Single-node runs (``cluster.num_nodes: 1``) start Ray automatically and need
   no manual ``ray start`` or ``RLINF_NODE_RANK``.

Prerequisites
-------------

Before starting, confirm that:

* All nodes can reach the head at ``<head_ip>:6379`` (or your chosen port).
* Every node has the **same versions** of Python, Ray (``ray>=2.47.0``), and
  RLinf dependencies.
* ``cluster.num_nodes`` in your config matches the cluster size (see
  :doc:`basic_config`).

.. important::

   Ray **freezes** the Python interpreter path and environment variables when
   you run ``ray start``. Every process Ray launches on that node inherits that
   environment. On **each node**, activate your virtualenv and install
   dependencies **before** running ``ray start``. Packages installed afterward
   are not visible to Ray workers.

Step 1: Set Environment Variables on Each Node
----------------------------------------------

On **every node**, before ``ray start``, export a unique node rank:

.. code-block:: bash

   export RLINF_NODE_RANK=<0..N-1>   # unique in the cluster; head is usually 0

If a machine has multiple NICs and Ray or collective communication must use a
specific interface, export it too:

.. code-block:: bash

   export RLINF_COMM_NET_DEVICES=<interface>   # e.g. eth0, enp3s0

Use ``ip addr`` or ``ifconfig`` to check which interface is reachable from
other nodes.

Step 2: Start the Ray Cluster
-----------------------------

Option A: Start Ray Manually on Each Node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pick one machine as the **head** (``RLINF_NODE_RANK=0``). Its IP must be
reachable from all workers; call it ``<head_ip>``.

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

``--node-ip-address`` must be the address other nodes use to reach the head
(private IP, VPC IP, or overlay IP; see :doc:`cloud_edge`). You can change port
``6379`` to any unused port, as long as the head and all workers use the same
one.

Option B: Use the Repository Script (Shared Filesystem)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all nodes mount the **same shared directory** and can read and write
``ray_utils/ray_head_ip.txt``:

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt    # first time or when rebuilding the cluster
   RANK=0 bash ray_utils/start_ray.sh   # head
   RANK=1 bash ray_utils/start_ray.sh   # worker 1
   # ...

The script uses ``RANK`` to choose head versus worker roles and does **not**
set ``RLINF_NODE_RANK``. Export ``RLINF_NODE_RANK`` on each node as in Option A
before running the script, or include it in your launch wrapper.

.. warning::

   ``ray_utils/start_ray.sh`` defaults to port ``29500`` (override with
   ``RAY_PORT``) and passes a hardcoded ``--memory=461708984320`` (~430 GB) to
   ``ray start``. On machines with less RAM or custom ports, use Option A
   instead.

Step 3: Enable Code Sync (Optional)
-----------------------------------

When the driver and workers do not share a filesystem (such as cloud-edge or
multi-site setups), enable Ray task-level code sync **before** launching the
training script. The driver packages the RLinf Python package into
``runtime_env.py_modules`` so workers need no matching local checkout:

.. code-block:: bash

   export RLINF_CODE_WORKING_DIR=auto

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``RLINF_CODE_WORKING_DIR``
     - Behavior
   * - Unset, ``0``, or ``false``
     - Sync off (default); every node needs a matching local RLinf install.
   * - ``auto``
     - Infers the repository root from the installed ``rlinf`` package or
       current working directory.
   * - Absolute path
     - Repository root containing ``pyproject.toml`` and the RLinf package, or
       the RLinf package directory itself.

Code sync ships only the RLinf Python package—not ``examples/`` or ``docs/``.
Task configs and data paths must still be reachable on each node or over shared
storage.

.. note::

   **When code sync is enabled**

   * **Keep large files out of the RLinf package directory**: code sync
     packages the entire local RLinf package tree from the launch node. Store
     logs, checkpoints, caches, and datasets outside it so packaging and
     transfer stay fast.
   * **Prepare models and assets separately**: model weights, simulator assets,
     and datasets are not synced. Download them on every worker to the
     configured paths, or mount them over NFS or shared storage, and verify
     that every node can access them.

``RLINF_CODE_WORKING_DIR`` takes effect when RLinf initializes Ray. Do not call
``ray.init`` manually before running the training script.

Step 4: Verify Cluster Status
-----------------------------

On **any node** that has run ``ray start``:

.. code-block:: bash

   ray status

Confirm that the **node count**, per-node **CPU/GPU**, and **state**
(``ALIVE``) match expectations. For example, 2 nodes with 8 GPUs each should
report 16 GPUs total.

You can also wait until all GPUs are ready (pass the cluster's total GPU count):

.. code-block:: bash

   bash ray_utils/check_ray.sh 16

If nodes are missing, check firewalls, confirm ``--node-ip-address`` is
reachable, and inspect ``ray start`` output on each worker.

Step 5: Launch RLinf Training
-----------------------------

1. Set ``cluster.num_nodes`` in your task YAML to the actual node count
   (matching the ``RLINF_NODE_RANK`` range).
2. On the **head node** (``RLINF_NODE_RANK=0``), ``cd`` to the RLinf repository
   root and run the entry script—for example, embodied training:

.. code-block:: bash

   cd <RLinf repo root>
   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi

Reasoning example:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

Before launching, confirm the head node:

* Has run ``ray start`` locally (``ray status`` shows the full cluster);
* Can access config files, models, and datasets (shared storage or code sync);
* Exports ``RLINF_CODE_WORKING_DIR`` in the **same terminal** if code sync is
  enabled.

.. note::

   For ``node_groups`` and cross-machine placement, see
   :doc:`../concepts/placement` and :doc:`hetero`. Cloud-edge
   ``component_placement`` examples are in :doc:`cloud_edge`.

Stop and Rebuild the Cluster
----------------------------

Stop Ray on every node and clear stale head-IP state before restarting:

.. code-block:: bash

   ray stop
   rm -f ray_utils/ray_head_ip.txt   # if you used start_ray.sh

After changing the Python environment or ``RLINF_NODE_RANK``, run ``ray stop``
and ``ray start`` again on every affected node.

FAQ
---

**Worker cannot reach head**

Check that ``<head_ip>`` is reachable from workers (ping or telnet), firewalls
and security groups allow port ``6379``, and the head did not bind
``--node-ip-address`` to ``127.0.0.1``.

**ray status shows fewer nodes than cluster.num_nodes**

Wait for workers to finish joining, verify ``ray start`` succeeded on every
node, and make sure you are not mixing separate Ray clusters.

**Worker cannot import latest code**

Check ``RLINF_CODE_WORKING_DIR``: without sync, every node needs the same local
RLinf package version. With sync, only the RLinf package is shipped (keep large
files out of it); configs under ``examples/``, models, and simulator assets
must still be available locally or on shared storage.

**Ray version mismatch**

Use the same ``ray`` version on every node (RLinf requires ``ray>=2.47.0``).
