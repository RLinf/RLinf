Real-World flowGRPO on YAM with Beaker GPUs
==============================================

This guide walks through running **flowGRPO** — a Flow Matching policy trained
with the GRPO (Group Relative Policy Optimization) algorithm — on the **YAM
bimanual robot**, using a Beaker GPU node for actor training and rollout
inference, and a separate robot controller node for environment interaction.

Overview
--------

**What is flowGRPO?**

flowGRPO combines two components:

- **Flow Matching Policy** (``flow_policy``): A transformer-based generative
  model that iteratively denoises Gaussian noise into robot actions using a
  learned velocity field.  It uses the ``JaxFlowTActor`` architecture with
  cross-attention over the observation.
- **GRPO** (``adv_type: grpo``): An on-policy advantage estimator that
  normalises rewards within a group of trajectories collected for the same
  task.  Unlike SAC-Flow, GRPO has no replay buffer and no critic network.

**Why GRPO instead of SAC for real-world flow policies?**

+-------------------+--------------------------+---------------------------+
| Property          | SAC-Flow                 | flowGRPO                  |
+===================+==========================+===========================+
| Data efficiency   | High (replay buffer)     | Lower (on-policy)         |
+-------------------+--------------------------+---------------------------+
| Implementation    | Complex (dual Q-network) | Simple (no critic)        |
+-------------------+--------------------------+---------------------------+
| Stability         | Needs entropy tuning     | Clip ratio controls range |
+-------------------+--------------------------+---------------------------+
| Reward shaping    | Not required             | Sparse rewards work well  |
+-------------------+--------------------------+---------------------------+

For tasks where sparse binary success/failure rewards are natural and you
prefer a simpler training loop, flowGRPO is a good starting point.

**YAM robot**

The YAM is a 7-DOF-per-arm bimanual platform.  Key facts:

- Arms connect via **CAN bus** (``DMChainCanInterface``), not TCP/IP.
- ``left_ip`` / ``right_ip`` in the hardware config are used only for a
  network-reachability pre-check; they do not establish the robot connection.
- Cameras attach as ``yam_realtime`` ``CameraNode`` objects (RealSense or
  OpenCV).

**Beaker GPU node**

Beaker (AI2's compute platform) sets ``CUDA_VISIBLE_DEVICES`` to GPU UUID
strings instead of integer indices.  RLinf handles this automatically in
``NvidiaGPUManager.get_visible_devices()``; no manual workaround is needed.

Environment
-----------

- **Robot**: YAM bimanual arm (2 × 7 DOF)
- **Observation**:

  - ``states``: 14-dimensional joint-angle vector (7 per arm)
  - ``main_images``: RGB image (224 × 224) from the primary camera

- **Action space**: 14-dimensional continuous joint-position targets, clipped
  to ``[-1, 1]``
- **Control rate**: 10 Hz (configurable via ``control_rate_hz``)

Hardware Setup
--------------

You need two machines on the same network:

+------------------+------------------------------------+--------------------+
| Node             | Role                               | GPU required?      |
+==================+====================================+====================+
| Node 0 (Beaker)  | Actor training + rollout inference | Yes                |
+------------------+------------------------------------+--------------------+
| Node 1 (YAM PC)  | Env worker / robot controller      | No                 |
+------------------+------------------------------------+--------------------+

.. warning::

   Both nodes must be on the same local network.  The robot controller node
   does **not** need internet access after installation.

Dependency Installation
-----------------------

Robot Controller Node (Node 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

Install ``yam_realtime`` by following the instructions in your YAM hardware
package.  The ``yam_realtime`` Python package must be importable before
starting Ray on this node.

GPU / Beaker Node (Node 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model flow_policy
   source .venv/bin/activate

Model Download
--------------

The flow policy uses a pretrained ResNet-10 vision encoder.  Download it
before starting training:

.. code-block:: bash

   pip install huggingface-hub
   # For mainland China users: export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-ResNet10-pretrained \
       --local-dir /path/to/RLinf-ResNet10-pretrained

Set ``actor.model.encoder_config.ckpt_name`` to the filename
(``resnet10_pretrained.pt``) and ``actor.model.model_path`` to the directory
containing it.

If you have a pretrained flow policy checkpoint (e.g., from simulation
pre-training), set ``actor.model.model_path`` and
``rollout.model.model_path`` to that checkpoint directory.  Otherwise, leave
``model_path`` empty to train from random initialisation.

Cluster Setup
-------------

Set the following on **every node** before starting Ray:

.. code-block:: bash

   export PYTHONPATH=/path/to/RLinf:$PYTHONPATH
   export RLINF_NODE_RANK=<0 for GPU node, 1 for YAM node>
   # Only if you have multiple network interfaces:
   export RLINF_COMM_NET_DEVICES=<interface_name>   # e.g. eth0

.. note::

   On Beaker, ``RLINF_NODE_RANK`` is typically set per-node via the Beaker
   job spec.  ``CUDA_VISIBLE_DEVICES`` may contain GPU UUIDs — RLinf remaps
   them to sequential indices automatically.

Start Ray:

.. code-block:: bash

   # GPU node (head, node rank 0)
   ray start --head --port=6379 --node-ip-address=<gpu_node_ip>

   # YAM controller node (worker, node rank 1)
   ray start --address='<gpu_node_ip>:6379'

   # Verify both nodes appear
   ray status

Configuration
-------------

The ready-to-use config is at
``examples/embodiment/config/yam_flowgrpo.yaml``.

Before launching, edit the following fields:

**1. Hardware — YAM IP addresses (reachability check only)**

.. code-block:: yaml

   cluster:
     node_groups:
       - label: yam
         node_ranks: 1
         hardware:
           type: YAM
           configs:
             - node_rank: 1
               left_ip: 192.168.1.10   # Replace with your left arm IP
               right_ip: 192.168.1.11  # Replace with your right arm IP

**2. Robot config paths (CAN bus)**

Add ``robot_cfgs`` under ``env.train`` and ``env.eval``:

.. code-block:: yaml

   env:
     train:
       robot_cfgs:
         left:  ["/path/to/yam_realtime/robot_configs/yam/left.yaml"]
         right: ["/path/to/yam_realtime/robot_configs/yam/right.yaml"]

**3. Camera config (optional)**

.. code-block:: yaml

   env:
     train:
       camera_cfgs:
         cam_front:
           _target_: yam_realtime.sensors.cameras.camera.CameraNode
           camera:
             _target_: yam_realtime.sensors.cameras.opencv_camera.OpencvCamera
             device_path: "/dev/video0"
             camera_type: "realsense_camera"

.. note::

   ``YAMEnv.get_obs()`` asserts that ``camera_dict`` is not ``None``.  Pass
   an empty dict (``camera_cfgs: {}``) when running without cameras; the
   env will then return zero-filled images.

**4. Model paths**

.. code-block:: yaml

   rollout:
     model:
       model_path: "/path/to/flow_policy_checkpoint"  # or "" for scratch

   actor:
     model:
       model_path: "/path/to/flow_policy_checkpoint"

**5. Task description**

.. code-block:: yaml

   env:
     train:
       task_description: "bimanual pick and place"

Key Algorithm Parameters
------------------------

.. code-block:: yaml

   algorithm:
     group_size: 4       # Trajectories per GRPO group.  Keep 2–4 for real HW.
     rollout_epoch: 4    # Should equal group_size
     update_epoch: 2     # Gradient passes per group
     adv_type: grpo      # Within-group normalised advantage
     loss_type: actor    # No critic for GRPO
     kl_beta: 0.01       # KL penalty to prevent collapse
     clip_ratio_high: 0.2
     gamma: 0.99

.. code-block:: yaml

   actor:
     model:
       denoising_steps: 4        # Flow matching refinement steps
       d_model: 256              # Transformer width
       n_head: 4
       n_layers: 2
       flow_actor_type: "JaxFlowTActor"
       noise_std_train: 0.3      # Exploration noise during data collection
       noise_std_rollout: 0.02   # Exploitation noise during rollout

.. tip::

   Increase ``group_size`` for better advantage estimates at the cost of more
   wall-clock time.  Start with 2 to verify the pipeline, then increase to 4.

Dry-Run in Dummy Mode
---------------------

Before connecting real hardware, validate the cluster setup with dummy mode
(returns zero observations, no robot movement):

.. code-block:: yaml

   env:
     train:
       is_dummy: True
     eval:
       is_dummy: True

Run:

.. code-block:: bash

   bash examples/embodiment/run_realworld.sh yam_flowgrpo

If training starts and logs appear in TensorBoard, the cluster and config are
correct.  Set ``is_dummy: False`` when you are ready for real hardware.

Running the Experiment
----------------------

On the **GPU node** (head node):

.. code-block:: bash

   bash examples/embodiment/run_realworld.sh yam_flowgrpo

The script sets ``EMBODIED_PATH`` and launches the Hydra training loop.  To
use the async runner (decoupled rollout and training):

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh yam_flowgrpo

Standalone Evaluation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash examples/embodiment/eval_embodiment.sh yam_flowgrpo

Beaker-Specific Notes
---------------------

1. **GPU UUID remapping**: RLinf detects non-integer ``CUDA_VISIBLE_DEVICES``
   values and remaps them to ``[0, 1, ...]``.  A warning is logged but no
   action is required.

2. **``torch.compile``**: Set ``rollout.enable_torch_compile: False`` (already
   the default in ``yam_flowgrpo.yaml``) if you encounter tracing errors with
   UUID-remapped GPUs on certain driver versions.

3. **``RLINF_NODE_RANK``**: Set this per-node in your Beaker job spec.  The
   GPU node must be rank 0 (head) and the YAM controller node must be rank 1.

4. **Shared filesystem**: Ensure ``runner.logger.log_path`` points to a path
   accessible from the GPU node.  Checkpoints and TensorBoard logs are written
   there.

Reward Design
-------------

``YAMEnv`` reports a zero reward by default; you must inject your task reward
signal.  Options:

- **Success flag**: Override ``YAMEnv.step()`` to return ``reward=1.0`` when
  the task succeeds and ``0.0`` otherwise.  GRPO works well with sparse
  binary rewards.
- **Dense reward**: Implement distance-to-goal shaping in the env subclass.
- **External reward model**: Set ``reward.use_reward_model: True`` and supply
  a reward worker.

Visualisation and Results
--------------------------

**TensorBoard**

.. code-block:: bash

   tensorboard --logdir ../results --port 6006

**Key metrics**

- ``env/success_once``: Fraction of episodes with at least one success.
  The primary metric to track.
- ``env/return``: Cumulative episode reward.
- ``train/actor/grad_norm``: Policy gradient norm; should stay below 10.
- ``train/actor/entropy``: Policy entropy; should not collapse to zero.

Troubleshooting
---------------

**Robot does not move / CAN bus error**

Check that the ``yam_realtime`` config YAML paths under ``robot_cfgs`` are
correct and that the CAN interface is up on the controller node before
starting Ray.

**Camera observation key error**

``YAMEnv`` reads image data via ``cam_data.get('images', {})`` and takes the
first value.  If your camera returns a different dict structure, verify the
``yam_realtime`` CameraNode ``read()`` output format on the controller node.

**CUDA_VISIBLE_DEVICES UUID warning**

This is expected on Beaker and is handled automatically.  The warning
``"CUDA_VISIBLE_DEVICES contains GPU UUIDs"`` in the log is informational.

**OOM on the GPU node**

Reduce ``actor.micro_batch_size`` and ``actor.global_batch_size``.  For a
single 24 GB card, ``micro_batch_size: 16`` is a safe starting point.

**GRPO advantage is NaN / all-same reward**

If all trajectories in a group receive the same reward, the within-group
standard deviation is zero and advantages become NaN.  Ensure your reward
function has meaningful variance across group members, or add a small
exploration bonus.

See Also
--------

- :doc:`sac_flow` — Flow Matching policy with SAC (replay buffer, critic)
- :doc:`franka` — Real-world RL setup with a Franka arm (SAC / RLPD)
- :doc:`../../tutorials/advance/hetero` — Heterogeneous cluster configuration
