HG-DAgger for Real-World Franka
===============================

**HG-DAgger** (Human-Gated DAgger) is an algorithm for real-world interactive imitation
learning pipeline. The workflow starts with teleoperated real-world data collection, runs OpenPI SFT on the collected LeRobot dataset, and then continues with async online HG-DAgger on the robot.

In RLinf configs, HG-DAgger is enabled by setting ``algorithm.dagger.only_save_expert: True``. This keeps only the expert-acted steps, which is the default choice for real-world intervention data.

Environment
-----------

**Real-World Franka Bin Relocation + Pi0**

- **Environment**: ``FrankaBinRelocationEnv-v1`` on a robot node
- **Observation**: Wrist / external RGB images and robot state
- **Action Space**: Delta end-effector qpos and gripper action
- **Use Case**: Collect human-guided real-world data, run OpenPI SFT, then continue async HG-DAgger

Algorithm
---------

**HG-DAgger Pipeline**

1. **Human-Guided Collection**

   - A human operator intervenes through the spacemouse on the real robot.
   - RLinf exports the successful trajectories in LeRobot format for later SFT.

2. **Supervised Warm Start**

   - Compute normalization statistics for the collected dataset.
   - Run OpenPI SFT to turn the collected human-guided dataset into the initial student policy.

3. **Online HG-DAgger**

   - Async rollout continues on the real robot with ``beta``-scheduled expert guidance.
   - With ``only_save_expert: True``, only expert-acted steps are added to the replay buffer.

4. **Replay-Buffer Updates**

   - The actor trains on the buffered intervention data with the
     ``embodied_dagger`` loss.
   - The student checkpoint from SFT becomes the initialization for online training.

Dependency Installation
-----------------------

The real-world pipeline uses **different environments on different nodes**:

- **Robot / env node**: Use the Franka controller environment from :doc:`franka`.
- **Training / rollout node**: Use the same environment as simulation DAgger :doc:`dagger`.

Robot / Env Node
~~~~~~~~~~~~~~~~

Follow the controller-node setup in :doc:`franka` for firmware checks, RT
kernel, ROS, and Franka controller dependencies.

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm \
      --privileged \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-franka
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-franka

Then switch to the libfranka-compatible environment:

.. code:: bash

   source switch_env franka-<libfranka_version>

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag for better download speed.
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

Before ``ray start`` on the robot node, source the same ROS / Franka controller
environment described in :doc:`franka`.

Training / Rollout Node
~~~~~~~~~~~~~~~~~~~~~~~

Use the same environment as simulator Pi0 DAgger.

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

Inside the container:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag for better download speed.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Cluster Setup
-------------

Before launching any collection or training job, finish the Ray setup described
in :doc:`franka`. The training / rollout node is typically the Ray head
(``RLINF_NODE_RANK=0``), while the Franka controller node is the worker
(``RLINF_NODE_RANK=1``).

.. code-block:: bash

   # On the training / rollout node
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_node_ip>

   # On the robot / env node
   export RLINF_NODE_RANK=1
   ray start --address='<head_node_ip>:6379'

Ray records the current Python interpreter and environment variables at startup,
so make sure each node has sourced the correct environment before ``ray start``.

Full Pipeline
-------------

1. Collect Human-Guided Real-World Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start from ``examples/embodiment/config/realworld_collect_data.yaml``. For the
pick-and-place task, switch the env from peg insertion to bin relocation:

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

Then fill in the robot configuration and keep LeRobot export enabled:

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 0

   env:
     eval:
       use_spacemouse: True
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         success_hold_steps: 1
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "lerobot"
       only_success: True
       robot_type: "panda"
       fps: 10

Launch collection with your copied config:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

During teleoperation, the same run writes:

- replay-buffer trajectories under ``logs/{timestamp}/demos/``
- LeRobot data under ``logs/{timestamp}/collected_data/``

For the collection format, see :doc:`../../tutorials/components/data_collection`.

2. Compute Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before SFT or HG-DAgger, compute OpenPI normalization stats for the collected
LeRobot dataset:

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/replay_buffer/calculate_norm_stats.py \
       --config-name pi0_franka_dagger \
       --repo-id franka_dagger

Use the same dataset root and dataset id that you will use for SFT. More
OpenPI-specific dataset notes are documented in :doc:`sft_openpi`.

3. Run OpenPI SFT
~~~~~~~~~~~~~~~~~

Edit ``examples/sft/config/franka_dagger_sft_openpi.yaml`` before launch:

.. code-block:: yaml

   data:
     train_data_paths: "/path/to/franka-lerobot-dataset"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_franka_dagger"

Then run:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh franka_dagger_sft_openpi

The SFT checkpoint is the student initialization for the online stage. For more
OpenPI SFT details, see :doc:`sft_openpi`.

4. Run Async HG-DAgger on the Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml`` to match
your cluster, cameras, target pose, and checkpoints:

.. code-block:: yaml

   cluster:
     num_nodes: 2
     node_groups:
       - label: "train"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   runner:
     ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

   algorithm:
     dagger:
       init_beta: 1.0
       beta_schedule: "exponential"
       beta_decay: 0.99
       only_save_expert: True

   env:
     train:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
     eval:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

   rollout:
     model:
       model_path: "/path/to/pi0-model"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_franka_dagger"

Launch HG-DAgger from the Ray head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

Visualization and Monitoring
----------------------------

**1. TensorBoard Logs**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. Useful Monitoring Metrics**

- ``train/dagger/actor_loss``: Supervised HG-DAgger loss on buffered intervention samples.
- ``train/replay_buffer/num_trajectories``: Number of stored trajectories.
- ``train/replay_buffer/total_samples``: Number of available replay samples.
- ``train/actor/lr``: Learning rate.
- ``train/actor/grad_norm``: Gradient norm.
