Real-World Robot Training Launch
=================================

This page explains how to connect **multiple Franka robots** and a **GPU training
node** to one Ray cluster, configure YAML, and start real-world RL training.

For full hardware setup, dependencies, data collection, and task details, see:

- :doc:`../../examples/embodied/franka` — standard RealSense + Franka gripper workflow
- :doc:`../../examples/embodied/franka_zed_robotiq` — ZED camera + Robotiq gripper on split nodes

For ``RLINF_NODE_RANK``, ``ray start``, and code sync, see :doc:`../usage/multi_node`.
For ``node_groups`` / ``component_placement`` semantics, see :doc:`../configuration/hetero`.


Typical topology
----------------

Real-world training usually follows **one GPU training node + N robot control nodes**:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``RLINF_NODE_RANK``
     - Role
     - Notes
   * - **0** (head)
     - GPU train / rollout
     - Runs ``actor``, ``rollout`` (and optional ``reward``); **only this node** runs the training entry script
   * - **1 … N**
     - Robot control
     - Runs ``env`` workers and ``FrankaController``; one control-node rank per arm (see example docs for shared control nodes)

All nodes must be on the **same LAN** (or overlay network; see :doc:`cloud_edge`), and
``cluster.num_nodes`` must match the number of nodes joined to Ray.

.. important::

   - Control nodes need Franka dependencies (ROS, libfranka, etc.); see :doc:`../../examples/embodied/franka`.
   - Ray freezes Python and env vars at ``ray start``; install dependencies on **each node** first.
   - Use ``ray_utils/realworld/setup_before_ray.sh`` to align per-node env before ``ray start``.


Connect robot nodes to the Ray cluster
--------------------------------------

Step 1: Prepare the environment on each node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On **every node**, before ``ray start``:

.. code-block:: bash

   # Recommended: source and edit the repo script on each machine
   source ray_utils/realworld/setup_before_ray.sh

   export RLINF_NODE_RANK=<0..N-1>          # unique; GPU node is usually 0
   # If multiple NICs, pin the reachable interface, e.g.:
   # export RLINF_COMM_NET_DEVICES=eth0

On control nodes, also source the ROS / Franka workspace if not in ``setup_before_ray.sh``:

.. code-block:: bash

   source <your_catkin_ws>/devel/setup.bash

Step 2: Start Ray
~~~~~~~~~~~~~~~~~

Let ``<head_ip>`` be the GPU head's reachable IP.

**GPU node (rank 0, head):**

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

**Robot control nodes (rank 1, 2, …):**

.. code-block:: bash

   export RLINF_NODE_RANK=1   # use 2 for the second robot, etc.
   ray start --address='<head_ip>:6379'

Run ``ray status`` on any node and confirm the node count matches ``cluster.num_nodes``.


YAML configuration
------------------

Real-world training is driven by the ``cluster`` section: ``node_groups`` separate GPU
and Franka hardware; ``component_placement`` maps ``actor`` / ``rollout`` / ``env`` to
those resources.

Before running, update:

- ``robot_ip``, ``target_ee_pose`` (task goal pose)
- ``actor.model.model_path`` (pretrained ResNet, etc.)
- ``algorithm.demo_buffer`` / ``data.path`` (for RLPD and other demo-based algorithms)

Single robot (1 GPU + 1 arm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0          # first robot in the franka group
       rollout:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0        # GPU training node
       - label: franka
         node_ranks: 1        # robot control node
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 1   # must match the control node's rank in node_ranks

If your GPU node is not rank 0 or the control node is not rank 1, update ``node_ranks``
and ``hardware.configs[].node_rank`` together.

Multiple robots (1 GPU + 2 arms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_2arms.yaml``:

.. code-block:: yaml

   cluster:
     num_nodes: 3
     component_placement:
       actor:
         node_group: "4090"
         placement: 0
       env:
         node_group: franka
         placement: 0-1       # one env worker per robot
       rollout:
         node_group: "4090"
         placement: 0:0-1     # two rollout processes on the same GPU
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1-2
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP_1>
               node_rank: 1
             - robot_ip: <ROBOT_IP_2>
               node_rank: 2

Extend ``num_nodes``, ``node_ranks``, ``placement``, and ``hardware.configs`` the same
way for more arms.

Split camera and arm (ZED + Robotiq)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the **camera is on the GPU server** and the **arm and gripper are on a NUC**,
set camera/gripper types and ``controller_node_rank`` in ``hardware.configs``.
Field details and collection examples are in :doc:`../../examples/embodied/franka_zed_robotiq`.

For training, place ``env`` on the GPU node group (camera capture) and pin
``FrankaController`` to the NUC:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: gpu
         placement: 0
       env:
         node_group: gpu
         placement: 0
       rollout:
         node_group: gpu
         placement: 0
     node_groups:
       - label: gpu
         node_ranks: 0
       - label: franka
         node_ranks: 0-1
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 0
               camera_serials:
                 - "<ZED_SERIAL>"
               camera_type: zed
               gripper_type: robotiq
               gripper_connection: "/dev/ttyUSB0"
               controller_node_rank: 1   # controller on rank 1 (NUC)

.. note::

   Install the ZED SDK on the GPU node into the **same venv Ray uses**, before
   ``ray start``. Configure Robotiq serial permissions on the NUC. See
   :doc:`../../examples/embodied/franka_zed_robotiq`.

Auto-configuration from environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-machine hardware values (``robot_ip``, ``camera_serials``,
``gripper_connection``, …) do not have to be hard-coded in YAML. Any
``hardware.configs`` field left **unset** is auto-filled during enumeration from
an environment variable named after the field in **upper case**
(``robot_ip`` → ``ROBOT_IP``, ``camera_serials`` → ``CAMERA_SERIALS``), which
keeps IPs and serial numbers out of committed configs.

Rules:

- Values written in YAML always win; environment variables only fill unset fields.
- The variable is read **on the node that owns the config** (its ``node_rank``),
  so export it there *before* ``ray start`` (Ray freezes env vars at start).
- ``node_rank`` and ``controller_node_rank`` are never taken from the environment.

**One robot per node.** The whole value is used; list fields are split on commas:

.. code-block:: yaml

   - label: franka
     node_ranks: 1
     hardware:
       type: Franka
       configs:
         - node_rank: 1          # robot_ip / cameras come from the environment

.. code-block:: bash

   # on the rank-1 control node, before `ray start`
   export ROBOT_IP=<ROBOT_IP>
   export CAMERA_SERIALS=<serial_1>,<serial_2>

**Several robots on one node.** Give one comma-separated value per robot,
assigned to the configs in order (so a list field takes one item per robot). The
value count must match the number of robot configs on that node:

.. code-block:: bash

   # two arms on one control node
   export ROBOT_IP=<ip_arm0>,<ip_arm1>
   export CAMERA_SERIALS=<serial_arm0>,<serial_arm1>

**Create robots entirely from the environment.** Leave ``configs`` empty and the
robots are created from the environment — the robot count is the comma count of
the identifier variable ``ROBOT_IP``, and every field is read from its variable:

.. code-block:: yaml

   - label: franka
     node_ranks: 1
     hardware:
       type: Franka
       configs: []              # robots created from ROBOT_IP, CAMERA_SERIALS, …

.. code-block:: bash

   export ROBOT_IP=<ip_arm0>,<ip_arm1>     # two robots
   export DISABLE_VALIDATE=true,true       # one value per robot

In this mode every variable you set must provide the same number of
comma-separated values as ``ROBOT_IP``; a mismatch aborts enumeration with an
explicit error. Other robot types use their own identifier field
(``GimArm`` → ``CAN_INTERFACE``, ``DOSW1`` → ``ROBOT_URL``).


Launch real-world training
--------------------------

After the Ray cluster is up, YAML is updated, and (for RLPD) demo data is ready, on the
**head node (usually rank 0 on the GPU machine)** ``cd`` to the RLinf repo and run.

If the GPU and control nodes **do not share the same RLinf checkout**, run
``export RLINF_CODE_WORKING_DIR=auto`` in the **same terminal** before the commands
below (see :doc:`../usage/multi_node`, Step 3: Enable code sync).

**Single-arm training (peg insertion, RLPD + async SAC):**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

**Dual-arm parallel training:**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async_2arms

**Other tasks (replace the config name as needed):**

.. code-block:: bash

   # Charger task
   bash examples/embodiment/run_realworld_async.sh realworld_charger_sac_cnn_async

   # Async PPO
   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_async_ppo_cnn

``<config_name>`` maps to ``examples/embodiment/config/<config_name>.yaml``; you can pass a custom name.

**Optional: dummy config to validate the cluster (single arm):**

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_franka_sac_cnn

Verify cameras on control nodes and Ray/placement on the head with the dummy config before full training; see :doc:`../../examples/embodied/franka` for the full checklist.
