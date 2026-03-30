Cloud-Edge Training Setup
=========================

Cloud-edge training lets you keep the training driver and large models in the cloud
while placing simulators, robots, or data-producing services on edge machines.
This setup becomes a networking problem because RLinf relies on Ray, and Ray expects
all participating nodes to have mutually reachable IP addresses.

If both the cloud node and the edge nodes already have directly reachable public IPs,
you can set up training like in a common cloud training setup. In practice, however,
the edge side is often behind NAT or another private network and does not expose a
public IP that cloud nodes can connect to directly. In that case, it is recommended
to set up a tunneling network between the cloud and edge so they have mutually
reachable IP addresses. `EasyTier <https://easytier.cn/en/guide/introduction.html>`_
is one practical option for building that network. This page shows a minimal way to build that overlay
network first and then run RLinf on top of it.


Hardware and Software Requirements
----------------------------------

Before starting, prepare the following:

* At least one node with a public IP that can act as the EasyTier bootstrap node.
  This node only needs basic CPU resources and can be separate from both the cloud
  training node and the edge nodes.
* All nodes must have permission to create TUN devices.
* If a node runs inside Docker, add the ``NET_ADMIN`` and ``NET_RAW`` capabilities
  and include the ``/dev/net/tun`` device in the container.

For example:

.. code-block:: bash

   docker run --rm -it \
     --network host \
     --cap-add NET_ADMIN \
     --cap-add NET_RAW \
     --device /dev/net/tun:/dev/net/tun \
     <your-image>


Setup Steps
-----------

Step 1: Decide node roles
~~~~~~~~~~~~~~~~~~~~~~~~~

In the simplest setup:

* one public-IP node acts as the EasyTier bootstrap node,
* the cloud node acts as the Ray head node,
* edge nodes join the same EasyTier network and then connect to the Ray head, and
* RLinf training is launched only from the cloud head node.


Step 2: Install EasyTier on every node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the EasyTier CLI on all cloud and edge nodes. You can use the
`EasyTier installation guide <https://easytier.cn/en/guide/installation.html>`_
or register it as a persistent
`systemd service <https://easytier.cn/en/guide/network/install-as-a-systemd-service.html>`_.

After installation, verify that the binary is available:

.. code-block:: bash

   easytier-core --version


Step 3: Start EasyTier on the bootstrap node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose a shared network name and secret for all nodes. The following command starts
the bootstrap node with a fixed overlay IP, a stable device name, and a listener
port that edge nodes can use as the initial peer:

.. code-block:: bash

   sudo easytier-core \
     --hostname easytier-bootstrap \
     --network-name rlinf-cloud-edge \
     --network-secret <shared-secret> \
     --private-mode true \
     --ipv4 10.10.0.1 \
     --dev-name et0 \
     -l 11010

The meanings of ``--network-name``, ``--network-secret``, ``--ipv4``, ``--peers``,
and ``--dev-name`` are documented in the
`EasyTier configuration reference <https://easytier.cn/en/guide/network/configurations.html>`_.


Step 4: Join every edge node to the same EasyTier network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start EasyTier on each edge node with the same network name and secret, and point it
to the bootstrap node's reachable address.

.. warning::

   Assign a unique fixed overlay IP to each node.

.. code-block:: bash

   sudo easytier-core \
     --hostname edge-1 \
     --network-name rlinf-cloud-edge \
     --network-secret <shared-secret> \
     --private-mode true \
     --ipv4 10.10.0.11 \
     --dev-name et0 \
     -p tcp://<bootstrap_public_ip>:11010

Repeat the same pattern for every additional edge node, assigning each node a unique
hostname and fixed overlay IP.

.. note::

   If you prefer a long-running service, put the same arguments into the
   ``ExecStart=`` line of a systemd unit as shown in the EasyTier service guide.


Step 5: Verify the overlay network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After EasyTier is up on all nodes:

* record the EasyTier overlay IP of every node,
* verify that the cloud node and edge nodes can reach each other through the
  EasyTier interface, and
* keep the EasyTier interface name stable if you plan to use
  ``RLINF_COMM_NET_DEVICES``.

The recorded EasyTier IPs are the addresses you should use when binding Ray.


RLinf Usage
-----------

Step 1: Start Ray on the EasyTier network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On every node, set ``RLINF_NODE_RANK`` before ``ray start``. If the machine has more
than one network interface, also point ``RLINF_COMM_NET_DEVICES`` to the EasyTier
device so RLinf uses the overlay network for communication.

Cloud head node:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   export RLINF_COMM_NET_DEVICES=et0
   ray start --head --port=6379 --node-ip-address=<cloud_easytier_ip>

Edge worker node:

.. code-block:: bash

   export RLINF_NODE_RANK=1
   export RLINF_COMM_NET_DEVICES=et0
   ray start --address="<cloud_easytier_ip>:6379"

Repeat on more edge nodes with unique ``RLINF_NODE_RANK`` values.


Step 2: Describe cloud and edge roles in the RLinf config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``cluster.num_nodes`` to the total number of cloud and edge nodes, then use
``node_groups`` and ``component_placement`` to place training-heavy components in the
cloud and latency-sensitive environment components at the edge.

Example for one 4-GPU cloud node plus one edge node:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     node_groups:
       - label: cloud
         node_ranks: 0
       - label: edge
         node_ranks: 1
     component_placement:
       actor:
         node_group: cloud
         placement: 0-3
       rollout:
         node_group: edge
         placement: 0-0
       env:
         node_group: edge
         placement: 0-0

Adjust the placement ranges to match your real hardware. If the edge node is CPU-only,
RLinf can still treat the node itself as the hardware resource with rank ``0``.


Step 3: Launch RLinf from the cloud head node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After all nodes have joined the same Ray cluster, run the RLinf entry script only on
the cloud head node. For example:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py --config-name <config_name>

or:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh <config_name>

For more complex per-node software environments or heterogeneous hardware layouts,
combine this guide with :doc:`hetero` and :doc:`../user/cluster`.
