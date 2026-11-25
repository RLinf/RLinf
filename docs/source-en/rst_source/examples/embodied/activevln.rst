ActiveVLN with EmbodiInfer
==========================

Run a two-step ActiveVLN GRPO smoke test with a Hugging Face actor,
EmbodiInfer rollout, and a VLN-CE environment server. The example repeats R2R
episode 1 with batch size 2 and BF16 weights.

Overview
--------

.. list-table::
   :header-rows: 1

   * - Model
     - Environment
     - Algorithm
     - Hardware layout
   * - ActiveVLN (Qwen2.5-VL-3B)
     - VLN-CE / Habitat
     - GRPO
     - GPU 0: training and inference; GPU 1: rendering

Tasks
~~~~~

The preset follows R2R navigation instructions in Matterport3D scenes.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Observation
     - Action
     - Reward
   * - RGB images, navigation instruction, and episode history
     - Text navigation commands
     - Navigation success

Installation
------------

.. include:: _setup_common.rst
   :start-after: First, clone the RLinf repository:
   :end-before: Then set up the dependencies

Install the training environment and VLN-CE client:

.. code-block:: bash

   bash requirements/install.sh embodied --model activevln --env vlnce_server --venv .venv-activevln
   source .venv-activevln/bin/activate

The installer downloads the ActiveVLN source into ``.venv-activevln/ActiveVLN``.
Set up Habitat in a separate Python environment.

Prepare and Start the Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the source path from the RLinf checkout:

.. code-block:: bash

   export ACTIVEVLN_ROOT="$PWD/.venv-activevln/ActiveVLN"

Follow the pinned
`ActiveVLN server installation <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#setup-the-vln-ce-environemnt>`_
to create the ``server`` environment with Python 3.10, Habitat-Sim v0.1.7,
and Habitat-Lab v0.1.7. Follow its
`data preparation <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#data-preparation>`_
to place Matterport3D scenes under ``$ACTIVEVLN_ROOT/data/scene_datasets/mp3d``
and R2R/RxR annotations under ``$ACTIVEVLN_ROOT/data/datasets``.
Include the training annotations and NDTW ground-truth trajectory files for both datasets;
this server revision loads both even when the RxR worker count is zero.

Install the remaining server dependencies once:

.. code-block:: bash

   conda activate server
   python -m pip install ray flask hydra-core av

Start two R2R simulators on physical GPU 1:

.. code-block:: bash

   cd "$ACTIVEVLN_ROOT"
   export PYTHONPATH="$ACTIVEVLN_ROOT:$ACTIVEVLN_ROOT/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
   python -m vlnce_server.server --config-path "$ACTIVEVLN_ROOT/vlnce_server/config" \
     server.host=127.0.0.1 server.port=5001 \
     'vlnce.gpus=[1]' 'vlnce.r2r_gpu_plan=[2]' 'vlnce.rxr_gpu_plan=[0]'

Keep this process running. In another terminal, wait for this request to return
``200`` before starting training:

.. code-block:: bash

   curl -sS -X OPTIONS http://127.0.0.1:5001/batch/reset -o /dev/null -w '%{http_code}\n'

``vlnce.r2r_gpu_plan`` controls simulator capacity. Raise it when increasing
the number of training or evaluation environments.

Download the Model
~~~~~~~~~~~~~~~~~~

In a separate terminal, return to the RLinf checkout and download the checkpoint:

.. code-block:: bash

   source .venv-activevln/bin/activate
   hf download Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
     --revision 160987313e3e869705f42400d1b8f28177044518 \
     --local-dir "$HOME/models/activevln-r2r"

Run It
------

Set these YAML fields to the absolute path of the downloaded checkpoint:

.. list-table::
   :header-rows: 1

   * - File
     - Field
   * - ``examples/embodiment/config/model/activevln.yaml``
     - ``model_path``
   * - ``examples/embodiment/config/env/vlnce_server_r2r.yaml``
     - ``tokenizer_path``

Launch from the RLinf checkout. ``RAY_ADDRESS=local`` keeps RLinf separate from
the server's Ray cluster.

.. code-block:: bash

   RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0 \
     bash examples/embodiment/run_embodiment.sh activevln_r2r_grpo_embodiinfer

Select the inference backend with ``rollout.rollout_backend``: ``embodiinfer`` or ``huggingface``.
EmbodiInfer options live under ``rollout.embodiinfer``.

For longer training, edit ``env.train.episode_ids``, ``runner.max_steps``, and
``runner.max_epochs`` in the preset. For a remote server, set
``env.train.server_url`` and ``env.eval.server_url``.

Results and Visualization
-------------------------

The run saves console logs and TensorBoard events under
``logs/<timestamp>-activevln_r2r_grpo_embodiinfer``. Start TensorBoard to inspect
the training metrics:

.. code-block:: bash

   tensorboard --logdir logs

See :doc:`/rst_source/reference/metrics` for training metrics and
:doc:`/rst_source/evaluations/get_started/index` for the evaluation workflow.
