.. This file is a reusable include, not a standalone page.
   Include it from the training and evaluation recipes.

For the base environment, follow the RLinf
:doc:`installation guide <../../../start/installation>`.

We recommend the prebuilt Docker image:

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.4-torch2.11.0-sglang0.5.12.post1-vllm0.23.0-megatron0.17.0-te2.17

If you prefer a local environment, install the agentic stack:

.. code-block:: bash

   bash requirements/install.sh agentic

Startup scripts and configuration files are in ``examples/agent/wideseek_r1``.

.. list-table::
   :header-rows: 1

   * - Path
     - Role
   * - ``examples/agent/wideseek_r1/config``
     - YAML configuration files for training and evaluation.
   * - ``examples/agent/tools/search_local_server_qdrant``
     - Search engine implementation used by offline tools.
   * - ``examples/agent/wideseek_r1/run_train.sh`` / ``examples/agent/wideseek_r1/run_eval.sh``
     - Main entry points for training and evaluation.
