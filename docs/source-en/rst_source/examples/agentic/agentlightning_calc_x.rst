AgentLightning RL Training (calc_x)
===================================

``calc_x`` is an AgentLightning example in RLinf for training a math-solving agent.
The agent reads a question, produces reasoning and an answer, and then receives feedback for RL updates.

Environment
-----------

For the base RLinf environment, see `RLinf Installation <https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html>`__.

Install dependencies for this example:

.. code-block:: bash

   pip install "agentlightning==0.3.0" "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.10.0" "mcp-server-calculator"

Hardware recommendation:

- This example requires one node with at least one 40GB GPU.

Data Preparation
----------------

Download and extract the ``calc_x`` dataset (Google Drive). See the download link `here <https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view>`_.

Training
--------

Go to the example directory:

.. code-block:: bash

   cd /path/to/rlinf/examples/agentlightning/calc_x

First, edit ``config/qwen2.5-1.5b-trajectory.yaml``:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/Qwen2.5-1.5B-Instruct

   data:
     train_data_paths: ["/path/to/train.parquet"]
     val_data_paths: ["/path/to/test.parquet"]

Start training:

.. code-block:: bash

   bash run_calc_x.sh qwen2.5-1.5b-multiturn

Evaluation
----------

For evaluation, you can directly specify a HuggingFace model directory:

.. code-block:: bash

   bash run_calc_x.sh eval=true rollout.model.model_path=/path/to/eval/hf_model
