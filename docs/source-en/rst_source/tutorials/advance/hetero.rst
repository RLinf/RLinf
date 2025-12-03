Heterogenous Software and Hardware Setup
==========================================

RLinf supports running on nodes with heterogeneous hardware and software setup, e.g., running simulators on ray tracing-capable GPUs (like RTX 4090), training on compute GPUs with larger GPU memory (like A100), search agent on CPU-only nodes, and/or robot controllers on nodes with special hardware like robotic arms.

To set up such a heterogeneous environment, all you need to do is configuring the `cluster` section of the YAML config file as follows.


Cluster configuration
---------------------

The ``cluster`` section describes **what machines you have** and **how RLinf
should place each component (actor, rollout, env, agent, etc.) on them**.

At a high level, you specify:

* the total number of nodes in the cluster,
* a set of *node groups* that share the same hardware / environment, and
* a *component placement* rule that maps logical components to hardware resources (GPUs, robots, or just nodes).

An example
~~~~~~~~~~~~~~~~~

The following example shows a cluster with heterogeneous hardware and per-node software environments
configured via ``env_configs``:

.. code-block:: yaml

	cluster:
	  num_nodes: 18

	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-63           # hardware ranks within ``a800``
	    rollout:
	      node_group: 4090
	      placement: 0-63           # hardware ranks within ``4090``
	    env:
	      node_group: franka
	      placement: 0-1            # robot hardware ranks within ``franka``
	    agent:
	      node_group: node
	      placement: 0-1:0-199,2-3:200-399  # node ranks : process ranks

	  node_groups:
	    - label: a800
	      node_ranks: 0-7
	      env_configs:
	        - node_ranks: 0-7
	          python_interpreter_path: /opt/venv/openpi/bin/python3
	          env_vars:
	            - GLOO_SOCKET_IFNAME: "eth0"

	    - label: 4090
	      node_ranks: 8-15
	      env_configs:
	        - node_ranks: 8-15
	          env_vars:
	            - GLOO_SOCKET_IFNAME: "eth1"

	    - label: franka
	      node_ranks: 16-17
	      hardware:
	        type: Franka
	        configs:
	          - robot_ip: "10.10.10.1"
	            node_rank: 16
	            camera_serials:
	              - "322142001230"
	              - "322142001231"
	          - robot_ip: "10.10.10.2"
	            node_rank: 17
	            camera_serials:
	              - "322142001232"
	              - "322142001233"

Interpretation
~~~~~~~~~~~~~~

The above configuration encodes the following ideas:

* ``num_nodes: 18`` – total number of nodes in the cluster. Node ranks are zero-indexed and specified via the ``RLINF_NODE_RANK`` environment variable when starting Ray on each node.

* ``node_groups`` – each entry defines a **node group**: a set of nodes with the same hardware and environment. A node group has:

  - ``label``: a unique string identifier used later in ``component_placement`` (e.g., ``a800``, ``4090``, ``franka``). Labels are case sensitive. 
  
    The labels ``cluster`` and ``node`` are reserved by the scheduler. ``node`` is a special group that covers *all* nodes and is used for hardware-agnostic placement (CPU-only processes, agents, etc.).

  - ``node_ranks``: a list or range of global node ranks that belong to this group. In the example, ``a800`` covers ``0-7``, ``4090`` covers ``8-15``, and ``franka`` covers ``16-17``.

  - ``env_configs`` (optional): a list of software environment configurations for subsets of nodes in the group. Each entry is a ``NodeGroupEnvConfig`` with its own ``node_ranks``, ``env_vars``, and ``python_interpreter_path``:

    * ``node_ranks`` must be a subset of the parent group's ``node_ranks``, and different ``env_configs`` in the same group must not overlap.

    * ``env_vars`` is a list of one-key dicts; environment variable keys must be unique within a node group for a node.

    * ``python_interpreter_path`` is the interpreter to use on the specified nodes.

  - ``hardware`` (optional): structured description of *non-accelerator hardware* (such as robots). The structure depends on the hardware ``type`` (for example, ``Franka``). When ``hardware`` is present, this node group is treated as owning exactly one hardware *type*, and that type defines **hardware ranks** (0, 1, ...) within the group.

* If ``hardware`` is **not** specified for a node group, RLinf behaves as follows:

  - If accelerator hardware (GPUs, NPUs, etc.) is detected on the nodes those accelerators become the default resources, and their local indices are used as hardware ranks.
  - If no accelerators are present, each **node itself** is treated as a single hardware resource with rank 0 within that node.

When you reference a ``node_group`` in ``component_placement``, the ``placement`` string is always written in terms of **hardware ranks within that group**:

* If ``hardware`` is present, these are explicit hardware ranks of that type (e.g., robots ``0-3``).
* Otherwise, they are automatically detected accelerators, if any.
* If there is no accelerator, the node itself is considered a hardware resource.

Using the reserved ``node`` group in ``component_placement`` disables
hardware placement entirely and interprets ranks as node ranks only. This
is useful for placing hardware-agnostic processes (such as agents or
CPU-only workers) on particular nodes regardless of available GPUs.
