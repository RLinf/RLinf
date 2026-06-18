Cheat Sheet
===========

Use this page when you already know the workflow and need the shortest path to a
working command.

Install
-------

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero

Start Ray
---------

Single-node runs can start Ray locally.

.. code-block:: bash

   ray start --head

For multi-node runs, set ``RLINF_NODE_RANK`` before ``ray start`` on every node.
See :doc:`../guides/multi_node`.

Run Training
------------

Launch an embodied recipe by config name.

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

Evaluate
--------

Use the unified evaluation entry point for embodied benchmarks.

.. code-block:: bash

   bash evaluations/run_eval.sh libero/libero_spatial_openpi_pi05_eval

Next Steps
----------

- :doc:`Installation <installation>` — set up RLinf and optional dependencies.
- :doc:`Quick Start <vla>` — run the Get Started training recipe.
- :doc:`Launch & Scale <../guides/launch-scale/index>` — run across nodes.
- :doc:`Evaluation <../evaluations/index>` — run standalone embodied evaluation.
