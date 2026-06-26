SO101 Real-World Evaluation
===========================

Evaluate a post-SFT π₀ checkpoint on the SO101 6-DOF arm. The rollout
worker runs inference on the exported checkpoint, the env worker drives
the physical arm, and the :class:`~rlinf.runners.embodied_eval_runner.EmbodiedEvalRunner`
collects per-episode success metrics (return, success rate, episode length).

Related docs: :doc:`../../examples/embodied/so101` (data collection), :doc:`../../examples/embodied/so101_sft_openpi` (SFT recipe), :doc:`../../examples/embodied/sft_openpi` (generic OpenPI SFT).

Environment Setup
-----------------

**Hardware**

- SO101 6-DOF arm (assembled + calibrated), connected via USB (``/dev/ttyACM0`` / ``/dev/ttyACM1``).
- Cameras (optional — see :doc:`../../examples/embodied/so101`; blank ``camera_0`` placeholder is emitted when no camera is configured).
- GPU host with the post-SFT checkpoint. Can be the same machine as the arm (single-node) or a separate GPU node.

**Dependencies**

Single-node (GPU + arm on one host):

.. code-block:: bash

   bash requirements/install.sh embodied --env so101
   source .venv/bin/activate

Two-node (GPU on one host, arm on another — GPU node only needs
OpenPI, not the full data-collection stack):

.. code-block:: bash

   # GPU / rollout node
   bash requirements/install.sh embodied --model openpi --env so101
   source .venv/bin/activate

   # Robot control node
   bash requirements/install.sh embodied --env so101
   source .venv/bin/activate

**Node topology**

- **Single-node**: ``num_nodes: 1`` — both ``env`` and ``rollout`` run on the
  same host (the one with the SO101 arm connected). Default in
  ``realworld_so101_eval.yaml``.
- **Two-node**: ``num_nodes: 2`` — ``rollout`` on rank 0 (GPU), ``env`` on
  rank 1 (robot). Adjust ``component_placement`` and ``node_groups``
  accordingly (mirror ``realworld_pnp_eval.yaml``).

Example Config
---------------

The eval config is at ``evaluations/realworld/realworld_so101_eval.yaml``.
It uses the ``pi0_so101`` data config (``action_dim=6``,
``num_action_chunks=10``, two image slots) and enables random
perturbation on each reset for robustness.

Key fields for a first run:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Field
     - Location
     - Notes
   * - ``port``
     - ``cluster.node_groups[].hardware.configs``
     - Follower arm USB port (e.g. ``/dev/ttyACM0``)
   * - ``target_ee_pose``
     - ``env.eval.override_cfg``
     - Task EE target ``(x, y, z)`` in metres (set via Hydra override)
   * - ``ckpt_path``
     - ``runner``
     - Path to ``full_weights.pt`` (exported by SFT)
   * - ``model_path``
     - ``rollout.model``
     - π₀ base model directory (with ``so101_data/norm_stats.json``)
   * - ``config_name``
     - ``rollout.model.openpi``
     - Needs to be ``"pi0_so101"``

Pre-flight Checks
-----------------

1. **Arm connectivity** (on the robot host):

   .. code-block:: bash

      ls /dev/ttyACM*   # should show two devices (leader + follower)
      # Verify the follower port matches the YAML.

2. **Calibration**: the arm must already be calibrated with lerobot's
   calibration tool before evaluation. Set ``auto_calibrate: False`` in
   ``override_cfg``.

3. **EE target pose**: Position the arm at the desired task goal and run
   :ref:`compute_ee_pose`:

   .. code-block:: bash

      python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0

   Paste the output ``(x, y, z)`` into the ``target_ee_pose`` override.

4. **Ray cluster**:

   .. code-block:: bash

      ray status  # should show at least one node (two if multi-node)

5. **Dummy mode (optional)**: Set ``is_dummy: True`` in
   ``env.eval.override_cfg`` to verify cluster wiring and config parsing
   without moving the arm.

.. warning::

   Verify the arm's workspace is clear and no cables are tangled before
   launching evaluation. Use a small ``env.eval.rollout_epoch`` (e.g. 2)
   on the first run and keep the emergency off button within reach.

Starting the Ray Cluster
------------------------

Single-node:

.. code-block:: bash

   ray start --head --port=6379 --node-ip-address=127.0.0.1

Two-node (mirror ``realworld_pnp_eval.yaml``):

.. code-block:: bash

   # GPU node (rank 0, head)
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

   # SO101 node (rank 1)
   export RLINF_NODE_RANK=1
   ray start --address=<head_ip>:6379

End-to-End Workflow
-------------------

**Step 1: Export the checkpoint**

After SFT training (see :doc:`../../examples/embodied/so101_sft_openpi`),
locate the exported ``full_weights.pt``:

.. code-block:: text

   so101_sft_openpi/
   └── checkpoints/
       └── global_step_N/
           └── actor/
               └── model_state_dict/
                   └── full_weights.pt

**Step 2: Prepare the EE pose**

Run ``compute_ee_pose.py`` on the robot host to record the target pose.

**Step 3: Launch evaluation**

.. code-block:: bash

   # Via the launcher
   bash evaluations/run_eval.sh realworld realworld_so101_eval \
       runner.ckpt_path=/path/to/full_weights.pt \
       rollout.model.model_path=/path/to/pi0_base_so101 \
       cluster.node_groups.0.hardware.configs.0.port=/dev/ttyACM0 \
       env.eval.override_cfg.target_ee_pose=[0.35,0.0,0.15]

   # Short smoke test (2 episodes)
   bash evaluations/run_eval.sh realworld realworld_so101_eval \
       env.eval.rollout_epoch=2 \
       ...

**Step 4: Check results**

Per-episode metrics appear on the terminal (``eval/success_once``,
``eval/return``, ``eval/episode_len``). Logs go to
``logs/<timestamp>-realworld_so101_eval/eval_embodiment.log``.

What Reset Runs After Each Episode
------------------------------------

When the RL runner detects that the arm has terminated (reward ≥ 1.0
or timeout), it triggers an auto-reset before the next episode begins:

1. The env worker calls ``RealWorldEnv._handle_auto_reset()``, which
   invokes ``SO101PickEnv.reset()`` on the done env index.
2. ``reset()`` adds a small uniform perturbation to each arm joint
   (the eval config enables ``enable_random_reset: True`` with
   ``random_joint_noise_deg: 5.0`` — clipped to the joint limits).
3. ``go_to_rest()`` sends the noised reset pose via
   ``robot.send_action()``; the arm moves there and holds.
4. ``_update_state()`` reads the fresh joint positions + gripper.

The perturbation ensures each evaluation episode starts from a
**slightly different initial position**, testing robustness rather than
letting the policy memorise a single trajectory from one starting pose.
Reset is joint-space (not EE-space) and the arm returns to the vicinity
of ``reset_joint_qpos`` — the same rest pose used during data collection.

Evaluation Config Reference
---------------------------

Required ``env.eval`` fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Notes
   * - ``rollout_epoch``
     - Number of evaluation episodes; start with 2–5
   * - ``max_episode_steps``
     - Max steps per episode; default 200
   * - ``max_steps_per_rollout_epoch``
     - Steps per rollout round; **must be divisible by** ``num_action_chunks`` (10)
   * - ``override_cfg.is_dummy``
     - Must be ``False`` for real hardware
   * - ``override_cfg.auto_calibrate``
     - Must be ``False`` (arm is pre-calibrated)
   * - ``override_cfg.target_ee_pose``
     - EE target ``[x, y, z]`` in metres
   * - ``override_cfg.reward_threshold_m``
     - Success radius; default 0.03 (3 cm)
   * - ``override_cfg.enable_random_reset``
     - ``True`` → noisy restart (recommended)

Key ``rollout`` fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Notes
   * - ``model.model_path``
     - π₀ base model directory; must contain ``so101_data/norm_stats.json``
   * - ``model.action_dim``
     - Must be 6 for SO101
   * - ``model.num_action_chunks``
     - Must be 10 (match SFT)
   * - ``model.openpi.config_name``
     - Must be ``"pi0_so101"``
   * - ``model.openpi.num_images_in_input``
     - Set to 2 (front + extra-view camera)

FAQ
---

- **"No cameras configured" warning**: Normal — SO101 eval with
  ``camera_cfgs: {}`` emits blank placeholders. This is fine for
  joint-space eval; add cameras in the YAML for visual-conditioned
  evaluation.
- **Arm doesn't move on the first step**: Verify ``is_dummy: False`` in
  ``override_cfg`` and that the correct port is configured.
- **Reward stays at 0.0**: Check that ``target_ee_pose`` is set and
  ``reward_threshold_m`` is large enough (start generous, e.g. 0.05 m).
- **Step-count errors**: Ensure ``max_steps_per_rollout_epoch`` is an
  integer multiple of ``rollout.model.num_action_chunks`` (10).
- **Checkpoint not loading**: The SFT-exported ``full_weights.pt`` must
  be from a training run that used ``model.openpi.config_name: pi0_so101``.
- **Ray shows wrong number of nodes**: Check firewall and
  ``RLINF_COMM_NET_DEVICES`` on multi-NIC hosts.
