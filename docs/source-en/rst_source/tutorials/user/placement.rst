Worker Placement Strategy
========================================

The placement module defines how workers are distributed across the hardware resources (nodes, GPUs, NPUs, robots and more) of the cluster.
The core interface of RLinf for this purpose is called ``ComponentPlacement``, which is configured via the ``cluster.component_placement`` field in the YAML configuration.
The configuration syntax is summarized below.


Basic formats
---------------

There are two equivalent styles:

1. **Short form** – directly map components to resources:

.. code-block:: yaml

	cluster:
	  num_nodes: 1
	  component_placement:
	    actor,inference: 0-7

Here, ``actor`` and ``inference`` *share the same placement rule*.
The string ``0-7`` is interpreted as a range of **resource ranks**.
RLinf will create 8 processes (ranks ``0-7``) for each of these components and evenly map them to resources ``0-7``.

2. **Node-group form** – explicitly select a node group:

For information about node groups, see :doc:`../advance/hetero`.

.. code-block:: yaml

	cluster:
	  num_nodes: 2
	  component_placement:
	    actor:
	      node_group: a800
	      placement: 0-8
	    rollout:
	      node_group: 4090
	      placement: 0-8
	    env:
	      node_group: robot
	      placement: 0-3:0-7
	    agent:
	      node_group: node
	      placement: 0-1:0-200,2-3:201-511

The meaning is:

* ``actor`` uses accelerators ``0-8`` in node group ``a800``.
* ``rollout`` uses accelerators ``0-8`` in node group ``4090``.
* ``env`` uses robot hardware ``0-3`` in group ``robot``; process ranks ``0-7`` are evenly shared across these robots (2 processes per robot).
* ``agent`` uses the special group ``node``. Processes ``0-200`` are placed on node ranks ``0-1``, and processes ``201-511`` are placed on node ranks ``2-3``.


Resource ranks and process ranks
-----------------------------------

Each placement entry has the general form::

	 resource_ranks[:process_ranks]

Both the ``resource_ranks`` and ``process_ranks`` parts support
the following syntax:

- ``a-b`` – inclusive integer range; e.g. ``0-3`` means 0,1,2,3.
- ``a-b,c-d`` – multiple ranges separated by commas.
  
The ``resource_ranks`` part additionally supports:

- ``all`` – all valid resources in the selected node group.

The specific meanings of ``resource_ranks`` and ``process_ranks`` is:

* ``resource_ranks``: which physical resources to use (GPUs, robots, or
	nodes). The format supports:

	The meaning of "resource" depends on the node group:

	- If a hardware type is specified in the node group, the ranks refer to that hardware (e.g., robot indices).
	- If no hardware type is specified but accelerators exist, the ranks are accelerator (GPU) indices.
	- If no accelerators exist, the ranks become node indices.

* ``process_ranks``: which **process ranks** of that component should be assigned to these resources. It uses the same range syntax as ``resource_ranks`` but must **not** be ``all``.
  
  If ``process_ranks`` is omitted, RLinf automatically assigns a continuous block of process ranks of the same length as the number of resources. For example, with two entries: ``0-3,4-7``, the first part implicitly uses process ranks ``0-3``, the second part uses process ranks ``4-7``.

You can combine multiple segments with commas, possibly mixing parts with
and without explicit ``process_ranks``::

	 0-1:0-3,3-5,7-10:7-14

This means:

* Processes ``0-3`` are evenly assigned to resources ``0-1``.
* Processes ``4-6`` are implicitly assigned to resources ``3-5`` (one process per resource, deduced by the scheduler).
* Processes ``7-14`` are evenly assigned to resources ``7-10``.

All process ranks for a component must be **continuous** from ``0`` to ``N-1`` (where ``N`` is the total number of processes for that component), and each process rank must appear exactly once. Violating this will raise an assertion error in the placement parser.

Additionally, for each ``resource_ranks:process_ranks`` pair, the number of resources and the number of processes must be compatible: one must be an integer multiple of the other. This ensures that either one process uses multiple resources, or multiple processes share one resource.

Referencing the placement in code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After defining the cluster and component placement in the config file, you can access the placement information in your RLinf code using the `rlinf.scheduler.ComponentPlacement` class as follows:

.. code-block:: python

	import hydra
	from rlinf.scheduler import Cluster, ComponentPlacement, Worker

	class TestWorker(Worker):
		def __init__(self):
			super().__init__()

		def run(self):
			self.log_info(f"Hello from TestWorker rank {self._rank}!")

	# Example usage: the YAML config file is named "conf.yaml"
	@hydra.main(version_base=None, config_path=".", config_name="conf")
	# cfg contains the parsed YAML configuration
	def main(cfg):
		# Initialize the cluster with the given configuration
		cluster = Cluster(cluster_cfg=cfg.cluster)

		# Create a ComponentPlacement instance which parses the placement rules
		placement = ComponentPlacement(cfg, cluster)

		# Retrieve the placement strategy for the "test_worker" component
		strategy = placement.get_strategy("test_worker")

		# Launch the TestWorker with the specified placement strategy
		worker = TestWorker.create_group().launch(cluster, placement_strategy=strategy)
		worker.run().wait()
	
	main()

The ``conf.yaml`` file is as follows:

.. code-block:: yaml

	cluster:
	  num_nodes: 1
	  component_placement:
		test_worker: # Component name
		  node_group: a800
		  placement: 0-3 # Place one process on each of the GPUs 0-3 of node group 'a800' (contains only node 0)

	  node_groups:
		- label: a800
		  node_ranks: 0


How placement is executed
-------------------------

Internally, :class:`rlinf.scheduler.placement.ComponentPlacement` parses the placement strings and chooses a concrete placement strategy based on whether the selected node group has hardware:

* If the node group has **no dedicated hardware** (no robots and no accelerators), RLinf uses :class:`rlinf.scheduler.placement.NodePlacementStrategy` to place processes purely by node rank. Each node is treated as a single resource; a process cannot span multiple nodes.

* If the node group has **hardware resources** (accelerators or custom hardware), RLinf uses :class:`rlinf.scheduler.placement.FlexiblePlacementStrategy` to map each process to one or more local hardware ranks on exactly one node.

Both strategies produce a list of low-level :class:`rlinf.scheduler.placement.Placement` objects, which encode:

* the global process rank in the cluster,
* the global node rank and local node index for the process,
* the selected hardware ranks and their local indices,
* whether accelerators not allocated to the worker are hidden via ``CUDA_VISIBLE_DEVICES`` and related environment variables.

In typical user workflows, you only need to write the ``cluster`` section and ``component_placement`` correctly. 
RLinf will then use these strategies to automatically realize the desired heterogeneous placement for you.


Placement Strategies
---------------------------

See :doc:`../../apis/placement` for detailed description of the built-in placement strategies.

