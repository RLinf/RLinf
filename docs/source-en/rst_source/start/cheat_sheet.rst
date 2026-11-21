Cheat Sheet
===========

Run common RLinf workflows with the shortest commands.

Install
-------

Install locally with ``install.sh``, or pull a prebuilt image from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

Run Training
------------

Single-node runs start Ray automatically. Launch a recipe by config name and pass the checkpoint path once:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     actor.model.model_path=/path/to/checkpoint

Both quick start configs default to ``all`` accelerators; change ``cluster.component_placement`` to target specific devices (:doc:`../guides/basic_config`). For multi-node runs, set ``RLINF_NODE_RANK`` before running ``ray start`` on each node (:doc:`../guides/multi_node`).

Evaluate
--------

Run standalone embodied evaluation:

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/checkpoint

Next Steps
----------

- :doc:`Installation <installation>` — targets, flags, and Docker Hub link.
- :doc:`Quick Start <vla>` — four-step ManiSkill3 walkthrough.
- :doc:`Launch & Scale <../guides/launch-scale/index>` — multi-node and real-world runs.
- :doc:`Evaluation <../evaluations/index>` — supported benchmarks and eval configs.
