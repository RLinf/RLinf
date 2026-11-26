ActiveVLN with EmbodiInfer
==========================

Run two ActiveVLN GRPO training steps with an FSDP actor,
EmbodiInfer rollout, and a VLN-CE environment server. The example repeats R2R
episode 1 with two environments, GRPO groups of two, a global training batch of
16, and BF16 weights.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      VLN-CE / Habitat

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      R2R navigation

   .. grid-item-card:: Hardware
      :text-align: center

      GPU 0: training and inference; GPU 1: rendering

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` and the Habitat setup below.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - VLN-CE / Habitat
     - R2R
     - ``activevln_r2r_grpo_embodiinfer``
     - Instruction-following navigation in Matterport3D scenes

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Observation
     - RGB images and episode history
   * - Action
     - Text navigation actions
   * - Reward
     - Navigation success
   * - Prompt
     - Natural-language navigation instruction

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. Install**

      Install the training environment and VLN-CE client:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model activevln --env vlnce_server --venv .venv-activevln
         source .venv-activevln/bin/activate

      The installer downloads the ActiveVLN source into ``.venv-activevln/ActiveVLN``.
      Set up Habitat in a separate Python environment.

      **2. Prepare and Start the Server**

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

         deactivate
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

      **3. Download the Model**

      In the same terminal used for the readiness check, return to the RLinf checkout and download the checkpoint:

      .. code-block:: bash

         source .venv-activevln/bin/activate
         hf download Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
           --revision 160987313e3e869705f42400d1b8f28177044518 \
           --local-dir ./models/activevln-r2r

      **4. Launch Training**

      Launch from the RLinf checkout with the downloaded checkpoint. The default
      placement uses GPU 0, leaving GPU 1 for rendering. ``RAY_ADDRESS=local``
      starts a separate Ray cluster for RLinf.

      .. code-block:: bash

         RAY_ADDRESS=local \
           bash examples/embodiment/run_embodiment.sh activevln_r2r_grpo_embodiinfer \
           actor.model.model_path="$PWD/models/activevln-r2r" \
           env.train.tokenizer_path="$PWD/models/activevln-r2r" \
           env.eval.tokenizer_path="$PWD/models/activevln-r2r"

      .. dropdown:: Advanced configuration (optional)

         **Longer Training and Checkpoints**

         To train for 1,000 updates and save a checkpoint every 100 updates,
         change these fields in
         ``examples/embodiment/config/activevln_r2r_grpo_embodiinfer.yaml``,
         then repeat the launch command above. Training and evaluation cycle through
         their respective episode lists. Choose IDs from the server's loaded annotations.

         .. code-block:: yaml

            runner:
              max_epochs: 1000
              max_steps: 1000
              save_interval: 100
              val_check_interval: 100
            env:
              train:
                episode_ids: [1, 2, 3, 4]
                max_episode_steps: 32
                max_steps_per_rollout_epoch: 32
              eval:
                episode_ids: [5, 6]
                max_episode_steps: 32
                max_steps_per_rollout_epoch: 32

         Each episode can run for up to 32 model turns, with up to three navigation
         actions per turn. Training and evaluation share the server's two slots.
         Checkpoints are saved under
         ``logs/<timestamp>-activevln_r2r_grpo_embodiinfer/activevln_r2r_grpo_embodiinfer/checkpoints/global_step_<step>/actor``.

         **Standalone Evaluation**

         From the RLinf checkout, evaluate the downloaded checkpoint with the same
         running server:

         .. code-block:: bash

            export EMBODIED_PATH="$PWD/examples/embodiment"
            RAY_ADDRESS=local python -m evaluations.eval_embodied_agent \
              --config-path "$EMBODIED_PATH/config" \
              --config-name activevln_r2r_grpo_embodiinfer \
              +model@rollout.model=activevln \
              rollout.model.model_path="$PWD/models/activevln-r2r" \
              env.eval.tokenizer_path="$PWD/models/activevln-r2r"

         **Backend and Server**

         Set ``rollout.rollout_backend=huggingface`` to use Hugging Face rollout.
         EmbodiInfer options live under ``rollout.embodiinfer``. For a remote
         server, set ``env.train.server_url`` and ``env.eval.server_url``.

Visualization and Results
-------------------------

The two-step run prints progress through step 2 and records ``env/success_once``,
``actor/policy_loss``, and ``actor/grad_norm``.

Under
``logs/<timestamp>-activevln_r2r_grpo_embodiinfer``, open ``run_embodiment.log``
for console output or ``tensorboard/`` for training curves. Start TensorBoard
to inspect the metrics:

.. code-block:: bash

   tensorboard --logdir logs

See :doc:`/rst_source/reference/metrics` for metric definitions.
