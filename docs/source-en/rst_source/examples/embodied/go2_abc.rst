Go2 Locomotion with PPO
=======================

Train a 12-joint Go2 policy in Isaac Lab with RLinf PPO. Task A tracks body velocity on flat ground, B walks over rough terrain, and C climbs stairs. Each policy uses 48 proprioceptive and command values plus 187 terrain height samples.

**Prerequisites:** Prepare RLinf on an NVIDIA GPU machine using the :doc:`installation guide </rst_source/start/installation>`.

Setup and Run
-------------

.. tab-set::

   .. tab-item:: Isaac Lab

      **1. Install the Go2 runtime**

      The Go2 installer sets up Isaac Lab 2.3.2, Isaac Sim 5.1, and RSL-RL 3.1.2. RLinf includes the three Go2 task definitions and their evaluator.

      .. code-block:: bash

         bash requirements/install.sh embodied --env go2_abc
         source .venv/bin/activate

      Run ``python -c "import isaacsim"`` in the activated environment to answer the `NVIDIA license prompt <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html#running-isaac-sim>`_. For a batch job, you can set ``OMNI_KIT_ACCEPT_EULA=YES`` in the job environment after accepting the terms.

      **2. Prepare the starting weights**

      From the RLinf repository root, fetch the frozen starting weights for the stairs task. The command verifies their SHA-256 before placing them in the training data directory.

      .. code-block:: bash

         python -m examples.embodiment.go2_abc.prepare_baseline C

      The C recipe reads ``models/C/baseline.pt`` under ``GO2_ABC_DATA`` (default ``data/go2_abc``). Pass ``A`` or ``B`` to fetch the starting weights for that task.

      To use your own weights, set ``actor.model.model_path`` to a checkpoint with a 235-input, 12-action actor and matching critic. Set it to ``null`` to train from scratch.

      **3. Train the policy**

      From the RLinf repository root, continue training the stairs policy.

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh isaaclab_go2_c_ppo

      Choose ``isaaclab_go2_a_ppo`` for flat ground or ``isaaclab_go2_b_ppo`` for rough terrain. By default A and B use 2,048 environments and C uses 1,024. The runner saves actor checkpoints every 50 updates for 1,000 updates. The launch log and run directory are under ``logs/``.

      The Go2 actor and critic train together from the first update and share the learning rate set by ``actor.optim.lr``. The default training seed is 0. Append ``actor.seed=2`` to the launch command to set the actor, rollout, and training environments to seed 2.

      For a local Isaac Sim 5.1 asset mirror, launch the same configuration with Hydra. Replace the asset path with the directory on your machine.

      .. code-block:: bash

         export GO2_ASSET_ROOT=/absolute/path/to/isaac_sim_5.1
         python -m examples.embodiment.train_embodied_agent \
           --config-path "$PWD/examples/embodiment/config" --config-name isaaclab_go2_c_ppo \
           "env.train.init_params.asset_root=$GO2_ASSET_ROOT" "runner.logger.log_path=$PWD/logs/go2-stairs"

Evaluate a Trained Policy
-------------------------

On an NVIDIA GPU, evaluate a saved RLinf policy from the repository root. Set the task and run directory to those used for the checkpoint.

.. code-block:: bash

   TASK=C
   RUN_DIR=logs/go2-stairs
   CHECKPOINT="$RUN_DIR/${TASK}_rlinf/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
   python -m examples.embodiment.go2_abc.evaluate --task "$TASK" --train-seed 0 "$CHECKPOINT" "$RUN_DIR/eval"

Use a new output directory for each checkpoint. For local Isaac Sim assets, add ``--asset-root "$GO2_ASSET_ROOT"``. Set ``--train-seed`` to the checkpoint's training seed or omit it if unknown.

To score an existing frozen baseline or an RSL-RL PPO checkpoint, use the same command with ``--checkpoint-format rsl-rl`` and that checkpoint's path. The evaluator runs seeds 23456 and 34567, pools completed episodes in ``summary.json``, and reports stair passage for task C.

Simulation Results
------------------

For each task, RSL-RL PPO and RLinf PPO start from the same baseline and train for 1,000 updates with training seeds 0 and 2. The frozen baseline is evaluated once per task and appears in both rows.

Each cell shows the success rate and the number of successful episodes out of all episodes pooled across evaluation seeds 23456 and 34567. For task C, the evaluator scores velocity tracking and uses the simulated trajectory to determine whether Go2 crosses the stairs.

.. list-table::
   :header-rows: 1

   * - Metric
     - Training seed
     - Frozen baseline
     - RSL-RL PPO
     - RLinf PPO (Huber)
   * - A velocity tracking
     - 0
     - 91.47% (1887/2063)
     - 93.33% (1917/2054)
     - 96.78% (1984/2050)
   * - A velocity tracking
     - 2
     - 91.47% (1887/2063)
     - 90.90% (1878/2066)
     - 98.29% (2013/2048)
   * - B velocity tracking
     - 0
     - 99.85% (2045/2048)
     - 99.61% (2040/2048)
     - 99.85% (2045/2048)
   * - B velocity tracking
     - 2
     - 99.85% (2045/2048)
     - 99.76% (2043/2048)
     - 99.80% (2044/2048)
   * - C velocity tracking
     - 0
     - 99.85% (2045/2048)
     - 100.00% (2048/2048)
     - 100.00% (2048/2048)
   * - C velocity tracking
     - 2
     - 99.85% (2045/2048)
     - 99.80% (2044/2048)
     - 99.90% (2046/2048)
   * - C stair passage
     - 0
     - 21.39% (438/2048)
     - 44.82% (918/2048)
     - 56.69% (1161/2048)
   * - C stair passage
     - 2
     - 21.39% (438/2048)
     - 33.50% (686/2048)
     - 62.06% (1271/2048)

Replay on CPU
-------------

Use the frozen weights from step 2 to replay one synthetic sensor frame. The output records the timestamp, twelve joint targets (rad), and twelve motor torques (N·m) in SDK order.

.. code-block:: bash

   TASK=C
   REPLAY_DIR=$(mktemp -d)
   python -c 'import json; print(json.dumps(json.load(open("tests/fixtures/go2_control_reference.json"))["frame"]))' > "$REPLAY_DIR/frame.jsonl"
   python -m examples.embodiment.go2_abc.deploy --task "$TASK" --group baseline \
     --frames "$REPLAY_DIR/frame.jsonl" --output "$REPLAY_DIR/commands.jsonl"

To replay a trained policy, set ``TASK`` to its task and replace ``--group baseline`` with ``--checkpoint "$CHECKPOINT"``. The optional ``--data`` argument selects the data root for group-based loading.

For your own recording, follow the structure of ``frame.jsonl``. Supply joint and IMU data plus measured body velocity and terrain heights. Record timestamps on one monotonic clock at 5 ms intervals. The policy updates every fourth frame. Express base and ground heights in the same world frame. Sample the yaw-aligned grid in 0.1 m steps, with y from −0.5 to 0.5 m and x from −0.8 to 0.8 m within each row.
