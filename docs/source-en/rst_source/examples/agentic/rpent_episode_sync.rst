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

      Clone RLinf and RPent on storage accessible to the model nodes. If you are trying the draft integration, use the paired checkouts in the pull request instructions.

      .. code-block:: bash

         export PROJECT_DIR="$PWD"
         git clone https://github.com/RLinf/RLinf.git
         git clone https://github.com/RLinf/RPent.git

      For an existing ``reason`` image, activate ``/opt/venv/reason``. To install from a fresh Linux host, use the NVIDIA agentic installer from the RLinf checkout. See :doc:`Installation <../../start/installation>`.

      .. code-block:: bash

         cd "$PROJECT_DIR/RLinf"
         bash requirements/install.sh agentic --engine sglang --sglang 0.5.12.post1
         source .venv/bin/activate

      Use the one-host runtime versions in :doc:`../../guides/rpent_multimodal` on every model node. That guide also gives the native SGLang build for 384 slots.

      .. code-block:: bash

         export RLINF_ROOT="$PROJECT_DIR/RLinf"
         export RPENT_ROOT="$PROJECT_DIR/RPent"
         export RPENT_PYTHON="$(command -v python)"
         export RPENT_MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-4B"
         export RPENT_LIBERO_TASKS="$PROJECT_DIR/data/tasks/libero/train.jsonl"
         export RPENT_ROBOTWIN_TASKS="$PROJECT_DIR/data/tasks/robotwin/train.jsonl"
         export RPENT_RUN_DIR="$PROJECT_DIR/runs/rpent"
         uv pip install --python "$VIRTUAL_ENV/bin/python" \
           -r "$RPENT_ROOT/requirements/episode-planner.txt"
         uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"

      2. Prepare assets and episode tasks

      Use a complete Hugging Face-format Qwen3.5-4B SFT checkpoint, including weights, tokenizer, chat template and image processor. Set ``RPENT_MODEL_PATH`` to the checkpoint directory visible at the same path on all model nodes.

      On each service host, follow the asset download and installation steps in the paired RPent checkout at ``docs/source-en/rst_source/guides/robot_episodes.rst``. Set the resulting interpreter and asset paths in the RPent deployment YAML. The one-host example uses LIBERO-PRO meshes and native initial states, a RoboTwin asset snapshot and ``sam3.pt``.

      Task pools describe simulator resets. RLinf generates trajectories and rewards online. Build the task pools from the public LIBERO-PRO native states and RPent's published RoboTwin seed table.

      .. code-block:: bash

         "$PROJECT_DIR/envs/libero/bin/python" "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
           --libero-config-path "$PROJECT_DIR/data/libero-config" \
           --suites libero_goal libero_spatial libero_object libero_10 \
           --output-dir "$PROJECT_DIR/data/tasks/libero"
         python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
           --robotwin-seeds "$RPENT_ROOT/robots/robotwin/eval/demo_randomized.json" \
           --output-dir "$PROJECT_DIR/data/tasks/robotwin"

      Both commands write ``train.jsonl`` and a manifest with row counts and a SHA-256 digest. The one-host example needs at least two rows per domain and the 384-slot profile needs at least 24. Each LIBERO row selects a suite, task and initial state. Each RoboTwin row selects a task, randomized configuration and seed. In a custom RoboTwin pool, an optional ``fallback_seeds`` list inside ``answer`` lets the whole group switch scenes if initialization fails.

      This recipe uses the published RoboTwin evaluation scenes as an example task pool. Use separate scenes for evaluation after training on this pool. Repeating a previously reported run requires its original task JSONLs, asset revisions and starting checkpoint.

      3. Run the local two-domain example

      On one four-GPU host, open a separate terminal in the project directory. Copy the deployment file and replace ``/work`` with the project directory. GPU 0 serves SAM3, GPU 1 runs simulators, and GPUs 2–3 run SGLang and the learner.

      .. code-block:: bash

         export PROJECT_DIR="$PWD"
         cd "$PROJECT_DIR/RPent"
         cp tools/rl/deployment-local.yaml "$PROJECT_DIR/deployment-local.yaml"
         "${EDITOR:-vi}" "$PROJECT_DIR/deployment-local.yaml"
         "$PROJECT_DIR/envs/sam3/bin/python" -m robots.rl.services \
           --config "$PROJECT_DIR/deployment-local.yaml" --node local

      Leave the services running in that terminal. When both toolkits are ready, return to the terminal from step 1. Inspect the resolved config before starting the local run.

      .. code-block:: bash

         cd "$RLINF_ROOT"
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1 --cfg job --resolve
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1

Scale to 384 Concurrent Environments
------------------------------------

The 384-slot example uses six eight-GPU nodes. Install LIBERO and RoboTwin on ``sim`` and ``model0`` through ``model3``, RoboTwin on ``twin``, and SAM3 on ``sim``. Copy the RPent template ``tools/rl/deployment-384.yaml`` to ``$PROJECT_DIR/deployment-384.yaml``. Fill in the paths and reachable addresses, distribute the edited file, and set ``RPENT_NODE`` to each host's role before starting its services.

.. code-block:: bash

   cd "$PROJECT_DIR/RPent"
   "$PROJECT_DIR/envs/robotwin/bin/python" -m robots.rl.services \
     --config "$PROJECT_DIR/deployment-384.yaml" --node "$RPENT_NODE"

On ``model0``, run ``ray start --head --port=29710``. On the other three model nodes, run ``ray start --address=<head-ip>:29710``. Activate the same training environment, set ``MEGATRON_PATH``, and export a distinct ``RLINF_NODE_RANK`` on each one before starting Ray. Replace the example IPs and check the communication interface on every node. Make the checkpoints readable and the episode directory writable from all four model nodes. Use the routers' reachable addresses in ``mixed_384.yaml``, with restricted forwarding across networks.

On the head, start training.

.. code-block:: bash

   cd "$RLINF_ROOT"
   export RAY_ADDRESS="<head-ip>:29710"
   export RPENT_LIBERO_ENDPOINT="http://<libero-router>:29840"
   export RPENT_ROBOTWIN_ENDPOINT="http://<robotwin-router>:29841"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
     +deployment@_global_=mixed_384

Set ``data.rollout_batch_size`` and ``data.sampling_batch_size`` in groups. This profile admits 32 of 48 groups with eight episodes per group. Set ``env.rpent`` quotas in episodes. The default is an even split of complete groups, and explicit quotas must match the global totals. Each simulator supplies 192 candidate episodes, of which 128 train the model. Each episode allows 40 turns, and the run saves every five updates.

Set ``group_router: true`` for router endpoints. With ``full_cohort_ready_timeout_s`` set, all simulator episodes must finish starting before the model begins. Check the two domain counts in the log and the readiness and release receipts in ``sampling/``. The learner starts after group selection, simulator release, and generation queue cleanup.

Visualization and Results
-------------------------

Run ``tensorboard --logdir "$RPENT_RUN_DIR"`` and follow :doc:`Training metrics <../../reference/metrics>`. The local recipe logs one update after admitting one of two groups per domain, with two episodes per group. Find each episode in the configured output directory and look in ``sampling/`` for the selected groups and cleanup receipts.

Resume a Run
------------

Keep the original model, task pools, quotas and episode artifacts. Start any required services and append ``runner.resume_dir=$RPENT_RUN_DIR/grpo/checkpoints/global_step_5`` to the same training command, using a completed checkpoint directory. The saved sampling cursor determines the next cohort. For a single domain, set ``data.train_data_paths`` and ``tools.rpent.remote_toolkit_endpoint`` on the base recipe.
