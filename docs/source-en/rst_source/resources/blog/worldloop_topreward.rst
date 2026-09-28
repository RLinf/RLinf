RLinf × WorldLoop: Extending WoVR with a Training-Free VLM Reward
=================================================================

Last updated: 09/21/2026.

This article explains how we reproduce WoVR's static-world-model RL stage on AMD GPUs and replace its task-specific reward with a training-free Qwen3-VL reward. World models let a VLA policy explore in generated video without continuously calling a physics simulator or a real robot during RL. `WoVR <https://arxiv.org/abs/2602.13977>`_ builds this pipeline on RLinf, using Wan to generate action-conditioned video and GRPO to update OpenVLA-OFT.

Both reward options described by WoVR require training: ResNet-18 uses labels derived from privileged simulator state, while its Qwen3-VL dense reward uses LoRA and an MLP reward head. We instead follow `TOPReward <https://arxiv.org/abs/2602.19313>`_: freeze Qwen3-VL-8B-Instruct and read the log probability of the ``" True"`` token as the success signal.

Across 500 MuJoCo evaluation episodes on LIBERO-Spatial, success rises from 0.448 to 0.560, within evaluation noise of the 0.574 result from a suite-specific ResNet reward. After fixing video sampling metadata, the same Qwen3-VL weights, threshold, and 16-frame window transfer to LIBERO-Object without recalibration. The final success rate is 0.348, equal to the ResNet arm; neither arm produces a distinguishable policy gain on this suite.

01 WoVR Still Requires a Task-Specific Reward Model
----------------------------------------------------

A physics simulator provides both dynamics and task completion signals. WoVR replaces the dynamics with an action-conditioned world model, but it still needs a separate reward model.

WoVR supports two reward designs:

- a ResNet-18 sparse reward that predicts success from the current frame using labels derived from privileged simulator state;
- a Qwen3-VL dense reward that consumes four frames and task text, then predicts progress from 0 to 10 through LoRA and an MLP head.

Both support policy optimization inside a world model, but both need new labels and reward training for a new task domain. WorldLoop keeps WoVR's static-world-model training stage while replacing this component with a training-free VLM.

The public Wan LIBERO checkpoints include the ResNet-18 reward used as our control. It has three transfer limitations:

1. LIBERO-Spatial and LIBERO-Object use different weights, each trained from suite-specific simulator labels;
2. the model sees only the current image, not the task instruction;
3. its output is highly concentrated: 83% of 16,384 frames score below 0.01, and rounding leaves 71% of samples without a GRPO gradient because every trajectory in the group receives the same reward.

02 TOPReward: Read Token Probability Instead of Generating Text
---------------------------------------------------------------

A common VLM reward design asks the model to generate a success verdict or a progress value. TOPReward performs no text generation and directly reads the log probability of the ``" True"`` token. On the same open-source VLM, the paper reports near-zero Value-Order Correlation for text output and 0.947 for token probability.

Before online training, we checked three conditions on saved frames:

- Qwen3-VL can read task semantics from Wan-generated video;
- successful and failed trajectories can be separated from the same initial state;
- one reward inference per action chunk adds limited overhead to the overall rollout.

The video window is the sensitive parameter. The initial experiment configured a 16-frame window, but without ``video_metadata`` the Qwen3-VL processor sampled only four of those frames. The public implementation fixes this so the model receives all 16 frames. With the fix, a threshold of 0.46 and a 16-frame window trigger on 0.212 of Spatial frame dumps and 0.205 of Object frame dumps, at the same quantile of both distributions.

03 Lightweight RLinf Integration
--------------------------------

An eight-action chunk is the atomic unit of the loop: OpenVLA-OFT emits actions, Wan converts them into future frames, and Qwen3-VL converts a 16-frame window plus the task instruction into a binary reward. Only OpenVLA-OFT is updated; Wan and Qwen3-VL remain frozen.

RLinf collocates actor, rollout, and environment Ray workers on every GPU. Qwen3-VL lives inside the environment worker and completes the environment step together with the world model.

.. code-block:: text

   OpenVLA-OFT ── 8 actions ──> Wan world model ── 8 frames ──> Qwen3-VL
        ^                         │                               │
        │                         └── last frame = next obs       │
       └──── GRPO update <──── P(" True") ≥ 0.46 → 0/1 ─────┘

Thresholding produces the same binary shape as the original ResNet reward after ``round()``. WoVR's reward differencing, termination handling, ``loss_mask`` truncation, and within-group GRPO normalization are reused without changing the RL algorithm.

In a physics simulator, reward and termination typically come from the same task predicate. A world model has no corresponding physical state, so the Qwen3-VL label serves as both immediate reward and ``terminations``. Running it inside the environment worker produces the value before the environment step returns and preserves RLinf's environment interface.

04 Results
----------

All policies are evaluated in MuJoCo on 10 tasks with 50 episodes each, for 500 episodes in total. We use LIBERO's official ``success_once`` metric. The standard error for the difference between two evaluation readings is approximately 3.1 percentage points. MuJoCo is not called during RL training.

LIBERO-Spatial
~~~~~~~~~~~~~~

The two arms differ only in the reward model.

.. list-table::
   :header-rows: 1
   :widths: 42 18 18 22

   * - Reward model
     - Steps
     - n=500
     - vs. base
   * - Spatial SFT base
     - 0
     - 0.448
     - —
   * - ResNet, suite-specific
     - 20
     - 0.574
     - +12.6
   * - Qwen3-VL, training-free
     - 10
     - 0.560
     - +11.2
   * - Qwen3-VL, training-free
     - 20
     - 0.532
     - +8.4

Qwen3-VL reaches 0.560 at step 10, within evaluation noise of ResNet's 0.574. Its 0.532 result at step 20 is 2.8 points lower than step 10, which is not enough evidence for a stable decline.

.. note::

   These Spatial policy results predate the video-metadata fix: the configured window contained 16 frames, but the processor sampled four. Offline rescoring moves Spatial's trigger rate by roughly one standard error, so the result still validates the training loop but is not a final reproduction number for the fixed 16-frame implementation. The policy run must be repeated before that stronger claim can be made.

LIBERO-Object Transfer Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 0.46 threshold and 16-frame window were selected on Spatial. Object uses the same constants and Qwen3-VL weights; only ``task_suite_name`` changes the instruction.

.. list-table::
   :header-rows: 1
   :widths: 34 26 13 13 14

   * - Reward model
     - Suite change
     - Base
     - Step 10
     - Step 20
   * - ResNet
     - Object-specific ``.pth``
     - 0.342
     - 0.368
     - 0.348
   * - Qwen3-VL, fixed 16-frame input
     - Task instruction
     - 0.342
     - 0.348
     - 0.348

Qwen3-VL and ResNet both score 0.348 at step 20, only 0.6 points above the base. The fixed reward configuration remains operational after transfer, but neither arm produces a policy gain beyond evaluation noise. Object therefore checks cross-suite operation, not the effectiveness of the current Object RL recipe.

Training-time reward readings cannot replace external evaluation. On Object, ResNet reports approximately 0.55 inside the world model against 0.348 in MuJoCo. The fixed Qwen3-VL reports approximately 0.35 inside the world model and 0.348 in MuJoCo. This agreement indicates calibration for this setting, not policy improvement.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/worldloop-results.png
   :alt: WorldLoop training-time and MuJoCo results on LIBERO-Spatial and LIBERO-Object
   :align: center
   :width: 100%

   Spatial Qwen3-VL data predates the video-sampling fix; Object Qwen3-VL data uses the fixed 16-frame implementation.

05 Public Implementation and Experiment Configuration
------------------------------------------------------

The implementation and bilingual usage documentation are under review in `RLinf #1554 <https://github.com/RLinf/RLinf/pull/1554>`_. The immutable experiment branch used for these measurements is `ZJLi2013/RLinf@0c7a3d5f <https://github.com/ZJLi2013/RLinf/tree/0c7a3d5f8d43a9c2ae242236a5699fb9d0a7243b>`_.

Experiments use one node with eight AMD MI300-series GPUs, 1 TB of host memory, and ROCm 6.4. Runtime overrides are ``total_num_envs=128``, ``rollout_epoch=2``, ``global_batch_size=2048``, ``lr=1e-5``, and ``group_size=8``. Each training step contains 8,192 samples, four optimizer updates, and 32 initial states. Each GPU hosts one world-model copy and one reward-model copy; measured memory usage is 84–116 GB out of 192 GB.

Qwen3-VL requires Transformers 4.57.1, while OpenVLA-OFT uses 4.40.1. The implementation therefore loads the newer dependency only inside environment-worker processes, leaving actor and rollout workers unchanged.

06 Next: Let the World Model Evolve with the Policy
----------------------------------------------------

This work reproduces WoVR's static-world-model stage and replaces its task-specific reward with frozen TOPReward. It is not a complete WoVR reproduction: Wan remains frozen during policy optimization, so PACE is not included.

After the first RL stage, the policy's action distribution moves away from the data used to train WM Base. PACE collects a limited number of trajectories from the evolved policy, refines the world model into WM Evo, and then runs the next policy-optimization stage. WorldLoop's next stage will close this loop: the policy evolves inside the world model, the evolved policy produces new trajectories, and those trajectories update the world model.

PACE still needs occasional access to the target environment. It replaces continuous online interaction with low-frequency updates rather than eliminating real-environment or simulator data. Follow-up work will also rerun Spatial with the fixed video input and test fixed TOPReward parameters on more task suites.

References
----------

- `TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics <https://arxiv.org/abs/2602.19313>`_
- `WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL <https://arxiv.org/abs/2602.13977>`_
- `RLinf <https://github.com/RLinf/RLinf>`_
- `RLinf-Wan-LIBERO-Spatial <https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial>`_
- `OpenVLA-OFT <https://github.com/moojink/openvla-oft>`_
- `Qwen3-VL <https://github.com/QwenLM/Qwen3-VL>`_
- `Wan2.2 <https://github.com/Wan-Video/Wan2.2>`_
- `LIBERO <https://github.com/Lifelong-Robot-Learning/LIBERO>`_

AI Assistance Disclosure
------------------------

The authors provided and verified the experiments, data, and conclusions. Cursor AI assisted with structure, bilingual editing, and wording.
