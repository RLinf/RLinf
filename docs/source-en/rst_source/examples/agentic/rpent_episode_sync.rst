Train an RPent Planner Across Simulators
========================================

Train a Qwen3.5-4B planner with synchronous GRPO using groups from LIBERO and RoboTwin. RLinf starts inference and learner workers on the model GPUs. RPent runs each simulator in its own Python environment, starts simulator processes on demand, and reads rewards from each simulator's task outcome.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models

      Qwen3.5-4B planner initialized from a full SFT checkpoint.

   .. grid-item-card:: Algorithms

      Synchronous GRPO with per-simulator group quotas.

   .. grid-item-card:: Tasks

      LIBERO-Pro and RoboTwin manipulation.

   .. grid-item-card:: Hardware

      Four NVIDIA GPUs for the local example. Six eight-GPU hosts for the 384-slot profile.

**Prerequisites:** Linux NVIDIA hosts, simulator assets, a planner checkpoint with its processor, and SAM3 weights. Use the same source and model paths on each learner node.

Setup and Run
-------------

.. tab-set::

   .. tab-item:: NVIDIA CUDA

      1. Install

      Clone RLinf and RPent on shared storage accessible to the model nodes.

      .. code-block:: bash

         export PROJECT_DIR="$PWD"
         git clone https://github.com/RLinf/RLinf.git
         git clone https://github.com/RLinf/RPent.git

      Use the `reason` image or install the NVIDIA agentic stack with ``bash requirements/install.sh agentic --engine sglang`` from the RLinf checkout. See :doc:`Installation <../../start/installation>`. Use matching package versions on the model nodes. The 384-slot profile uses native SGLang TokenSpace. Follow :doc:`../../guides/rpent_multimodal` to prepare that runtime.

      .. code-block:: bash

         source /opt/venv/reason/bin/activate
         export RLINF_ROOT="$PROJECT_DIR/RLinf"
         export RPENT_ROOT="$PROJECT_DIR/RPent"
         export RPENT_PYTHON="$(command -v python)"
         export RPENT_MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-4B"
         export RPENT_LIBERO_TASKS="$PROJECT_DIR/data/tasks/libero.jsonl"
         export RPENT_ROBOTWIN_TASKS="$PROJECT_DIR/data/tasks/robotwin.jsonl"
         export RPENT_RUN_DIR="$PROJECT_DIR/runs/rpent"
         uv pip install --python "$VIRTUAL_ENV/bin/python" \
           -r "$RPENT_ROOT/requirements/episode-planner.txt"
         uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"

      Put the complete SFT model, tokenizer and image processor at ``$RPENT_MODEL_PATH``, visible at the same path from every model node. Install each simulator and its assets separately, and stage the SAM3 weights using the `RPent installation and deployment guide <https://github.com/RLinf/RPent/blob/main/docs/source-en/rst_source/guides/robot_episodes.rst>`_.

      2. Prepare the model and task pools

      Use one JSONL file per simulator. Each row contains a ``prompt`` and a JSON-encoded ``answer``. Save the rows below in ``$RPENT_LIBERO_TASKS`` and ``$RPENT_ROBOTWIN_TASKS`` respectively. Add at least two task rows per file for the local recipe. Replace the example RoboTwin seed with one verified against your asset snapshot.

      .. code-block:: json

         {"prompt":"Solve the LIBERO task","answer":"{\"environment\":\"libero\",\"suite\":\"libero_spatial\",\"task\":0,\"seed\":0}"}
         {"prompt":"Solve the RoboTwin task","answer":"{\"environment\":\"robotwin\",\"task_name\":\"adjust_bottle\",\"task_config\":\"demo_randomized\",\"seed\":100000}"}

      From the LIBERO environment, export its training pool for the large profile.

      .. code-block:: bash

         "$LIBERO_PYTHON" "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
           --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$PROJECT_DIR/data/tasks/libero-native"
         cp "$PROJECT_DIR/data/tasks/libero-native/train.jsonl" "$RPENT_LIBERO_TASKS"

      The 384-slot recipe needs at least 24 task rows per domain. To provide alternate initial states for RoboTwin setup failures, include ``fallback_seeds`` in the answer. All members switch together before the group starts generation.

      3. Run the local two-domain example

      On one four-GPU host, start services from the RPent checkout. Set the interpreters, checkpoints and asset paths in the deployment file first. GPU 0 serves SAM3, GPU 1 runs simulators, and GPUs 2–3 run SGLang and the learner.

      .. code-block:: bash

         cd "$RPENT_ROOT"
         python -m robots.rl.services --config tools/rl/deployment-local.yaml --node local

      Open a new terminal in the training environment. Inspect the resolved config and launch training on the local host. RLinf starts its own model workers.

      .. code-block:: bash

         cd "$RLINF_ROOT"
         source /opt/venv/reason/bin/activate
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1 --cfg job --resolve
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1

Scale to 384 Concurrent Environments
------------------------------------

The 384-slot example uses six eight-GPU nodes. Install LIBERO and RoboTwin on ``sim`` and ``model0`` through ``model3``, RoboTwin on ``twin``, and SAM3 on ``sim``. Run the following command on all six hosts with their respective roles.

.. code-block:: bash

   cd "$RPENT_ROOT"
   python -m robots.rl.services --config tools/rl/deployment-384.yaml --node <node>

Start the Ray head on ``model0`` and run ``ray start --address=<head-ip>:29710`` on the other three model nodes. Activate the same training environment, set ``MEGATRON_PATH``, and export a distinct ``RLINF_NODE_RANK`` on each one before starting Ray. Replace the example IPs and check the communication interface on every node. Make the checkpoints readable and the episode directory writable from all four model nodes. Use the routers' reachable addresses in ``mixed_384.yaml``, with restricted forwarding across networks.

On the head, start training.

.. code-block:: bash

   export RAY_ADDRESS="<head-ip>:29710"
   export RPENT_LIBERO_ENDPOINT="http://<libero-router>:29840"
   export RPENT_ROBOTWIN_ENDPOINT="http://<robotwin-router>:29841"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
     +deployment@_global_=mixed_384

Set ``data.rollout_batch_size`` and ``data.sampling_batch_size`` in groups. This profile admits 32 of 48 groups with eight episodes per group. Set ``env.rpent`` quotas in episodes. The default is an even split of complete groups, and explicit quotas must match the global totals. The example collects 192 candidates and admits 128 episodes per domain, allows 40 turns, and saves every five updates.

Set ``group_router: true`` for router endpoints. With ``full_cohort_ready_timeout_s`` set, all simulator episodes must finish starting before the model begins. Check the two domain counts in the log and the readiness and release receipts in ``sampling/``. The learner starts after group selection, simulator release, and generation queue cleanup.

Visualization and Results
-------------------------

Run ``tensorboard --logdir "$RPENT_RUN_DIR"`` and follow :doc:`Training metrics <../../reference/metrics>`. The local recipe logs one update after admitting one of two groups per domain, with two episodes per group. Find each episode in the configured output directory and look in ``sampling/`` for the selected groups and cleanup receipts.

Resume a Run
------------

Keep the original model, task pools, quotas and episode artifacts. Start any required services and append ``runner.resume_dir=$RPENT_RUN_DIR/grpo/checkpoints/global_step_5`` to the same training command, using a completed checkpoint directory. The saved sampling cursor determines the next cohort. For a single domain, set ``data.train_data_paths`` and ``tools.rpent.remote_toolkit_endpoint`` on the base recipe.
