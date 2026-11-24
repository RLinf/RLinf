AgentLightning RL
=================

Train a calculator-backed math agent on ``calc_x`` by pairing Agent Lightning workflow orchestration with RLinf's distributed RL trainer. The agent calls an MCP calculator across turns and updates Qwen2.5-1.5B-Instruct with turn-level or trajectory-level GRPO.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-1.5B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO (turn-level or trajectory-level)

   .. grid-item-card:: Tools
      :text-align: center

      MCP calculator · AutoGen

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · ≥ 1×40 GB GPU

| **What you'll do:** install Agent Lightning deps → download model and ``calc_x`` dataset → launch ``run_calc_x.sh``.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · ``calc_x`` parquet files from Google Drive.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct

      **1. Install**

      Install the agentic environment and Agent Lightning dependencies. Keep ``requirements/agentic/agentlightning_constraints.txt`` so OpenTelemetry stays compatible with AgentLightning 0.3.0 span routing to Lightning Store:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install -r requirements/agentic/agentlightning_constraints.txt
         uv pip install "agentlightning==0.3.0" "litellm<1.95" \
           "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" \
           "mcp-server-calculator" "openai==2.32.0"

      **2. Download**

      Download the base model, then download and extract the ``calc_x`` dataset (``train.parquet`` and ``test.parquet``) from `Google Drive <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_ into ``./calc_x/``:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct

      **3. Launch**

      Pick a config under ``examples/agent/agentlightning/calc_x/config/`` for turn-level or trajectory-level advantage:

      .. list-table::
         :header-rows: 1
         :widths: 24 48 28

         * - Mode
           - Config
           - Advantage
         * - Multi-turn
           - ``qwen2.5-1.5b-enginehttp-multiturn.yaml``
           - Turn-level
         * - Trajectory
           - ``qwen2.5-1.5b-enginehttp-trajectory.yaml``
           - Trajectory-level

      Launch training from the repository root with Hydra overrides for the model and dataset paths (switch to ``qwen2.5-1.5b-enginehttp-trajectory`` for trajectory-level mode):

      .. code-block:: bash

         bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn \
           rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
           data.train_data_paths='["./calc_x/train.parquet"]' \
           data.val_data_paths='["./calc_x/test.parquet"]'

      .. dropdown:: Standalone evaluation and YAML configuration (optional)

         To persist paths directly in the YAML instead of passing CLI overrides, update ``rollout.model.model_path``, ``data.train_data_paths``, and ``data.val_data_paths``, then run ``run_calc_x.sh`` with the config name alone.

         For standalone Hugging Face checkpoint evaluation, pass ``rollout.model.model_path`` to the matching ``*_eval`` config:

         .. code-block:: bash

            # Multi-turn evaluation
            bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval \
              rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
              data.val_data_paths='["./calc_x/test.parquet"]'

            # Trajectory-level evaluation
            bash examples/agent/agentlightning/calc_x/run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval \
              rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
              data.val_data_paths='["./calc_x/test.parquet"]'

Visualization and Results
-------------------------

Track reward and training progress in TensorBoard (``tensorboard --logdir ../results``), or see :doc:`Training metrics </rst_source/reference/metrics>` for W&B/SwanLab and metric definitions.

Example metric curves from a ``calc_x`` run (curves vary by config and random seed):

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/agentlightning_calcx.png
   :width: 90%
   :align: center
   :alt: AgentLightning calc_x training curves

   AgentLightning ``calc_x`` training curves
