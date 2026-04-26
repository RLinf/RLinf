Real-World RL with XSquare Turtle2
====================================

This document provides a comprehensive guide to launching real-world reinforcement
learning training on the **XSquare Turtle2** dual-arm robot platform using the
RLinf framework.

The primary objective is to train a ResNet-based CNN policy from scratch for robotic
manipulation tasks on a real robot by:

1. **Visual Understanding**: Processing RGB images from up to three onboard cameras.
2. **Action Generation**: Producing precise delta end-effector actions (position, rotation, and gripper) for one or two arms.
3. **Reinforcement Learning**: Optimizing the policy via SAC with real-environment feedback.

Environment
-----------

**Real-World Environment**

- **Robot**: XSquare Turtle2 – a dual-arm tabletop robot with up to 2 arms (left arm ID ``0``, right arm ID ``1``) and up to 3 RGB cameras (IDs ``0``, ``1``, ``2``).
- **Task**: Currently we support the **button-pressing** task (``ButtonEnv``):

  - The robot end-effector moves downward to press a button located at a target pose.
  - Random resets add ±5 cm position noise and ±20° orientation noise to increase difficulty.
  - The task description string: *"Press the button with the end-effector."*

- **Observation**:

  - RGB images (128 × 128) from one or more cameras, returned as ``frames/wrist_<k>``.
  - TCP pose: position (xyz) + quaternion (xyzw) per active arm, concatenated as a flat vector.

    - Single arm: ``[batch_size, 7]``
    - Dual arm: ``[batch_size, 14]``

- **Action Space**: 7-dimensional continuous action per arm, stacked for dual-arm use:

  - 3D delta position (Δx, Δy, Δz)
  - 3D delta orientation (Δroll, Δpitch, Δyaw)
  - Gripper width command (open/close)

  Single arm: ``(7,)`` — Dual arm: ``(14,)``; values normalized to ``[-1, 1]``.

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 128, 128, 3]``
- **Actions**: Normalized continuous values ``[-1, 1]`` per dimension
- **Rewards**: ``1.0`` on success (all active arms reach target within threshold), ``0.0`` otherwise; optionally a dense exponential reward


Algorithm
---------

**Core Algorithm Components**

1. **SAC (Soft Actor-Critic)**

   - Learns Q-values via Bellman backups with entropy regularization.
   - Learns a policy that maximizes the entropy-regularized Q objective.
   - Automatically tunes the temperature parameter (``alpha``) for exploration–exploitation balance.

2. **Cross-Q** (optional)

   - A SAC variant that removes the target Q-network.
   - Concatenates current and next observations in one batch with BatchNorm for stable Q learning.

3. **RLPD (Reinforcement Learning with Prior Data)** (optional)

   - Augments online SAC with offline demonstration data.
   - High update-to-data ratio exploits collected data efficiently.

4. **CNN Policy Network**

   - ResNet-based visual encoder for RGB input.
   - MLP layers fuse image features with proprioceptive state to produce actions.
   - Separate Q-heads for the critic.


Hardware Setup
--------------

The real-world setup requires:

- **Robot**: XSquare Turtle2 dual-arm robot
- **Cameras**: Up to 3 RGB cameras mounted on the robot (IDs 0–2)
- **Training / Rollout Node**: A computer with GPU support for running the CNN policy
- **Robot Controller Node**: A small computer (GPU not required) connected to the robot in the same local network

.. warning::

  Ensure the training node and the robot controller node are in the **same local network**.


Dependency Installation
-----------------------

The controller node and the training/rollout node(s) require different software dependencies.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~

The XSquare Turtle2 platform ships with its own SDK and ROS-based controller stack. **Please ensure that you have entered the official Docker container of Xsquare before starting the following installation.**. Contact `XSquare <https://x2robot.com>`_
for the exact Docker image and startup instructions.

After entering the XSquare Docker container, clone the RLinf repository inside it:

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

Then install the RLinf Python dependencies for the embodied real-world setup:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag for better speed.
   bash requirements/install.sh embodied --env xsquare_turtle2
   source .venv/bin/activate

.. note::

   On the Turtle2 controller node, we recommend a **dual-container split** to reduce ROS ownership conflicts:

   - The **control container** (for example, the vendor-provided Turtle2 control container such as ``turtle2_release``) owns the ROS master, UI, low-level bring-up, and the camera / arm control nodes.
   - A **separate RLinf container** only runs RLinf and the Ray worker, and should not directly own ``roscore``, the UI, or ``run.sh``.

   The RLinf container should be built from an image configuration that is **compatible with the control container**, ideally sharing the same base image, system dependencies, ROS runtime, and required mounts, while only adding the RLinf Python environment and code on top. This reduces drift in ROS dependencies, message definitions, and device access.

   A common setup is to run both containers with ``host network`` so they share the controller node's network namespace. In that case, the RLinf container can connect to the existing ROS master instead of starting another control plane inside its own container. During training or evaluation, it is safer to open the UI only temporarily for calibration in the control container, then close it before running RLinf, to avoid multiple processes publishing to the same arm-control topics.

Training / Rollout Node
~~~~~~~~~~~~~~~~~~~~~~~

a. Clone RLinf Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

**Option 1: Docker Image**

.. code:: bash

   # use maniskill_libero image for training / rollout nodes
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

**Option 2: Custom Environment**

.. code:: bash

   # install openvla + maniskill_libero environment on training / rollout nodes
   # For mainland China users, you can add the `--use-mirror` flag for better speed.
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate


Model Download
--------------

Before starting training, download the pretrained ResNet CNN backbone:

.. code:: bash

   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Method 2: Using huggingface-hub
   # For mainland China users:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

After downloading, update the ``model_path`` field in the configuration YAML file.


Running the Experiment
-----------------------

Prerequisites
~~~~~~~~~~~~~

**Acquire the Target End-Effector Pose**

For each task, you need to record the target end-effector pose that triggers a success
signal. Move the robot arm(s) to the desired target pose manually via the XSquare control
interface, then read back the pose.

The pose is stored as Euler angles in the format
``[x, y, z, rz, ry, rx]`` (XSquare convention). Record this for both
arms if using dual-arm mode.

Cluster Setup
~~~~~~~~~~~~~

Before starting the experiment, set up the Ray cluster properly.

.. warning::

  This step is critical. Any misconfiguration may cause missing packages or failure
  to control the robot.

RLinf uses Ray for managing distributed environments. When ``ray start`` is run on
a node, the current Python interpreter and environment variables are recorded and
inherited by all subsequent Ray processes on that node.

We provide ``ray_utils/realworld/setup_before_ray.sh`` to help configure the environment
before starting Ray on each node. Modify and source it before ``ray start``.

The script sets up the following:

1. Source the correct virtual Python environment (see Dependency Installation above).
2. On the controller node: ensure the XSquare SDK packages are discoverable (this is
   handled automatically when using the XSquare official Docker image).
3. Set RLinf environment variables on all nodes:

.. code-block:: bash

   export PYTHONPATH=<path_to_your_RLinf_repo>:$PYTHONPATH
   export RLINF_NODE_RANK=<node_rank_of_this_node>
   export RLINF_COMM_NET_DEVICES=<network_device>  # Optional if only one NIC

``RLINF_NODE_RANK`` is set to ``0 ~ N-1`` for each of the ``N`` nodes.
``RLINF_COMM_NET_DEVICES`` is only needed if the machine has multiple network interfaces;
check with ``ifconfig`` or ``ip addr``.

Start Ray on each node after sourcing the script:

.. code-block:: bash

   # On the head node (node rank 0)
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

   # On worker nodes (node rank 1 ~ N-1)
   ray start --address='<head_node_ip_address>:6379'

Run ``ray status`` to verify the cluster is up.

Configuration File
~~~~~~~~~~~~~~~~~~

Modify ``examples/embodiment/config/realworld_button_turtle2_sac_cnn.yaml`` to match
your setup.

Key fields to update:

.. code-block:: yaml

  cluster:
    num_nodes: 2  # 1 training/rollout node + 1 controller node
    component_placement:
      actor:
        node_group: "gpu"
        placement: 0
      rollout:
        node_group: "gpu"
        placement: 0
      env:
        node_group: turtle2
        placement: 0
    node_groups:
      - label: "gpu"
        node_ranks: 0
      - label: turtle2
        node_ranks: 1
        hardware:
          type: Turtle2
          configs:
            - node_rank: 1

  env:
    train:
      override_cfg:
        is_dummy: False
        use_arm_ids: [1]          # 0=left arm, 1=right arm; use [0,1] for dual arm
        use_camera_ids: [2]       # camera IDs to use (0, 1, or 2)
        target_ee_pose:           # [[left_arm_pose], [right_arm_pose]], Euler [x,y,z,rz,ry,rx]
          - [0, 0, 0, 0, 0, 0]
          - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]

  actor:
    model:
      model_path: "/path/to/RLinf-ResNet10-pretrained"
      state_dim: 6    # 6 for single arm (xyz+euler); 12 for dual arm
      action_dim: 6   # 6 for single arm (xyz_delta+rpy_delta); 12 for dual arm

  rollout:
    model:
      model_path: "/path/to/RLinf-ResNet10-pretrained"

For the **button-pressing** task, the ``target_ee_pose`` defines both the success
threshold position and the reset position (arms reset to a pose slightly above the
target along the Z axis).

Testing the Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the full experiment, you can verify the setup using dummy mode:

Set ``is_dummy: True`` in both ``env.train.override_cfg`` and ``env.eval.override_cfg``
to enable dummy mode (no real robot required). This validates the cluster and model
pipeline without physical robot interaction.

Run on the head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_turtle2_sac_cnn

Running the Experiment
~~~~~~~~~~~~~~~~~~~~~~

After verifying the setup, start the real-world training experiment on the head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_button_turtle2_sac_cnn


Deployment
----------

For policy-only real-world rollout or evaluation, RLinf provides a generic
Turtle2 deployment environment registered as ``Turtle2DeployEnv-v1``. The env
fragment is:

.. code-block:: text

   examples/embodiment/config/env/realworld_turtle2_deploy.yaml

Use it as the ``env.eval`` entry in an eval-only config, for example by copying
``examples/embodiment/config/realworld_eval.yaml`` and replacing the env default:

.. code-block:: yaml

   defaults:
     - env/realworld_turtle2_deploy@env.eval
     - model/pi0@actor.model
     - training_backend/fsdp@actor.fsdp_config
     - weight_syncer/patch_syncer@weight_syncer
     - override hydra/job_logging: stdout

The Turtle2 deploy fragment sets ``init_params.id`` to ``Turtle2DeployEnv-v1``
and exposes ``action_mode`` at the env level:

.. code-block:: yaml

   env:
     eval:
       action_mode: absolute_pose
       override_cfg:
         task_description: "Describe your Turtle2 deployment task here."
         use_arm_ids: [0, 1]
         use_camera_ids: [0, 1, 2]
         reset_ee_pose:
           - [0.20, 0.00, 0.10, 0.00, 0.00, 0.00]
           - [0.20, 0.00, 0.10, 0.00, 0.00, 0.00]
         ee_pose_limit_min:
           - [-0.20, -0.60, -0.05, -3.20, -3.20, -3.20]
           - [-0.20, -0.60, -0.05, -3.20, -3.20, -3.20]
         ee_pose_limit_max:
           - [0.80, 0.60, 0.60, 3.20, 3.20, 3.20]
           - [0.80, 0.60, 0.60, 3.20, 3.20, 3.20]

``apply_dual_pose_action_wrappers`` attaches the deploy action wrapper stack:

- ``absolute_pose`` uses ``DualAbsolutePoseActionWrapper``. The policy emits
  one 7D absolute pose command per active arm:
  ``[x, y, z, roll, pitch, yaw, gripper]``.
- ``relative_pose`` uses ``DualRelativePoseActionWrapper``. By default it also
  applies ``DualRelativeFrame`` so end-effector-frame delta actions are
  transformed back to the base frame before reaching the robot env.
- Both modes finish with ``DualQuat2EulerWrapper`` so the policy observes
  Euler-format TCP state.

Run the eval-only deployment config from the Ray head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh <your_turtle2_eval_config_name>

Visualization and Results
--------------------------

**1. TensorBoard Logging**

On the Ray head node:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- **Environment Metrics**:

  - ``env/episode_len``: Number of environment steps elapsed in the episode (unit: step).
  - ``env/return``: Episode return.
  - ``env/reward``: Step-level reward.
  - ``env/success_once``: Whether the robot succeeded at least once in the episode (0 or 1).

- **Training Metrics**:

  - ``train/sac/critic_loss``: Q-function loss.
  - ``train/critic/grad_norm``: Q-function gradient norm.
  - ``train/sac/actor_loss``: Policy loss.
  - ``train/actor/entropy``: Policy entropy.
  - ``train/actor/grad_norm``: Policy gradient norm.
  - ``train/sac/alpha_loss``: Temperature parameter loss.
  - ``train/sac/alpha``: Temperature parameter value.
  - ``train/alpha/grad_norm``: Temperature gradient norm.
  - ``train/replay_buffer/size``: Current replay buffer size.
  - ``train/replay_buffer/max_reward``: Maximum reward in replay buffer.
  - ``train/replay_buffer/min_reward``: Minimum reward in replay buffer.
  - ``train/replay_buffer/mean_reward``: Mean reward in replay buffer.
  - ``train/replay_buffer/utilization``: Replay buffer utilization rate.
