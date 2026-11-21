AgentLightning RL
=================

``calc_x`` is an AgentLightning example in RLinf for training a math-solving agent.
The agent reads a question, produces reasoning and an answer, and then receives feedback for RL updates.

Overview
--------

Use this recipe to train a calculator-backed math agent with Agent Lightning and
RLinf's distributed trainer.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-1.5B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      Multi-turn agent RL

   .. grid-item-card:: Tools
      :text-align: center

      MCP calculator and AutoGen agent chat

   .. grid-item-card:: Hardware
      :text-align: center

      One node with at least one 40 GB GPU

Setup and Run
-------------

AgentLightning 0.3.0 requires the OpenTelemetry versions constrained by RLinf
because OpenTelemetry 1.45 changed the exporter implementation used to route
rollout spans to the Lightning Store. In mainland China, append
``--use-mirror`` to ``install.sh`` and run
``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt
images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.
For other installation options and supported platforms, see
:doc:`RLinf Installation </rst_source/start/installation>`.

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install -r requirements/agentic/agentlightning_constraints.txt
         uv pip install "agentlightning==0.3.0" "litellm<1.95" \
           "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" \
           "mcp-server-calculator" "openai==2.32.0"

      **2. Download**

      Download the base model and extract the ``calc_x`` dataset from `Google Drive <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct

      **3. Launch**

      Choose a config under ``examples/agent/agentlightning/calc_x/config/`` and set ``rollout.model.model_path``, ``data.train_data_paths``, and ``data.val_data_paths``:

      .. list-table::
         :header-rows: 1
         :widths: 26 44 30

         * - Mode
           - Config
           - Advantage
         * - Multi-turn
           - ``qwen2.5-1.5b-enginehttp-multiturn.yaml``
           - Turn-level
         * - Trajectory
           - ``qwen2.5-1.5b-enginehttp-trajectory.yaml``
           - Trajectory-level

      .. code-block:: yaml

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-1.5B-Instruct

         data:
           train_data_paths: ["/path/to/train.parquet"]
           val_data_paths: ["/path/to/test.parquet"]

      Launch training from the repository root (or pass Hydra overrides on the command line):

      .. code-block:: bash

         # Multi-turn
         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn

         # Trajectory-level
         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory

         # Or override paths without editing the YAML
         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn \
           rollout.model.model_path=/path/to/Qwen2.5-1.5B-Instruct \
           data.train_data_paths='["/path/to/train.parquet"]' \
           data.val_data_paths='["/path/to/test.parquet"]'

Visualization and Results
-------------------------

Example training / metric curves from a ``calc_x`` run (logged metrics may vary by config and seed):

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentlightning_calcx.png
   :width: 90%
   :align: center
   :alt: AgentLightning calc_x training curves

   AgentLightning ``calc_x`` training curves

Standalone Evaluation
---------------------

For HF evaluation, set ``rollout.model.model_path`` in the matching ``*_eval.yaml``. Examples:

.. code-block:: bash

   bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval
   bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval
