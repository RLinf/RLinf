AgentLightning RL Training (calc_x)
===================================

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

Installation
------------

For the base RLinf environment, see :doc:`RLinf Installation </rst_source/start/installation>`.

Install dependencies for this example:

.. code-block:: bash

   pip install "agentlightning==0.3.0" "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" "mcp-server-calculator"

Data Preparation
----------------

Download and extract the ``calc_x`` dataset (Google Drive). See the download link `here <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_.

Run It
------

Go to the example directory:

.. code-block:: bash

   cd /path/to/RLinf/examples/agent/agentlightning/calc_x

First, edit ``config/qwen2.5-1.5b-enginehttp-trajectory.yaml``:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-1.5B-Instruct

   data:
     train_data_paths: ["/path/to/train.parquet"]
     val_data_paths: ["/path/to/test.parquet"]

Start training:

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn

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

   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-multiturn_eval
   bash run_calc_x.sh qwen2.5-1.5b-enginehttp-trajectory_eval
