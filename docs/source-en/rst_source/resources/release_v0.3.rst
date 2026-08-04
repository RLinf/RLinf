Release Notes
=============

RLinf v0.3 Release
------------------

🎉 Introducing RLinf v0.3.

This release completes an end-to-end real-world training pipeline, adds new real-world RL components and algorithms, and brings more simulators and SOTA models to simulation RL. All supported examples have been strictly validated for correctness and reproducibility (see the test results at the end).

Embodied
^^^^^^^^^^

1. Models
"""""""""""

Continuing to expand the model ecosystem, 6 new embodied models are added, covering world models, VLA models, and system-level acceleration.

- Added **Dexbotic DM0** model support, with online RL fine-tuning using PPO on LIBERO. Link: `Dexbotic <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dexbotic.html>`__

- Added **DreamZero** model support: a VLA policy fine-tuned from the WAN2.1/2.2 video-generation world model, integrated into the SFT workflow, achieving nearly **4×** throughput improvement via FSDP2/CUDA Graph and other system-level acceleration. Link: `DreamZero SFT <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/sft_dreamzero.html>`__

- Added **GR00T-N1.6 / N1.7** model RL fine-tuning support. Link: `GR00T <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/gr00t.html>`__

- Added **ABot-M0** model support. Link: `ABot-M0 <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/abot_m0.html>`__

- Added **StarVLA** model support (GRPO on LIBERO). Link: `StarVLA <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/starvla.html>`__

- Added **LingBot-VLA** model support (RoboTwin environment SFT/RL). Link: `LingBot-VLA <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/lingbotvla.html>`__

2. Simulators
"""""""""""""""

Broadening simulation-RL scene coverage, 5 new simulators are added, with refined simulator-based training examples and results.

- Added **Genesis** simulator support. Link: `Genesis <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/genesis.html>`__

- Added **Polaris** simulator support. Link: `Polaris <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/polaris.html>`__

- Added **RoboVerse** simulator support. Link: `RoboVerse <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/roboverse.html>`__

- Improved **Behavior** environment support: added v3.7.1 / v3.7.2 patches, a π0.5 PPO config, and object/pose randomization. Link: `Behavior <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/behavior.html>`__

- Added **Libero+ / LiberoPro** variant environments support. Link: `LIBERO <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/libero.html>`__

- Added **Embodichain (CartPole)** environment support. Link: `Embodichain <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/embodichain.html>`__

- Added **IsaacLab π0.5 PPO fine-tuning** support. Link: `IsaacLab <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/isaaclab.html>`__

- Added **RoboCasa** close-drawer and other RL examples support. Link: `RoboCasa <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/robocasa.html>`__

3. Real World
"""""""""""""""

Fully connecting the data collection → SFT → RL → real-world deployment loop, adding 3 teleoperation methods, 3 real-world platforms, and 2 end-effectors; real-world operation capability is significantly strengthened.

**Data collection support:**

- Added **Spacemouse** teleoperation data collection support. Link: `Franka <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka.html>`__

- Added **VR teleoperation** data collection support. Link: `Franka VR <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_vr.html>`__

- Added **GELLO teleoperation** data collection support. Link: `Franka GELLO <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_gello.html>`__

**Training pipeline support:**

- Added **LeRobot-format data collection** support, for interop with the HuggingFace LeRobot ecosystem. Link: `Data Collection <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/data_collection.html>`__

- Added **Pi0 real-world SFT deployment** support, connecting the data collection → SFT → real-world deployment link. Link: `Franka Pi0 SFT Deploy <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_pi0_sft_deploy.html>`__

- Added **real-world reward model data collection** support (collecting labeled reward training data). Link: `Franka Reward Model <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_reward_model.html>`__

**Real-world platforms and end-effectors support:**

- Added **Dual-arm Franka** platform support (joint-space and TCP/rot6d control, data collection, SFT, deployment). Link: `Dual Franka <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dual_franka.html>`__

- Added **GimArm** real-world platform support. Link: `GimArm <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/gim_arm.html>`__

- Added **DOS-W1** real-world platform support. Link: `DOS-W1 <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dosw1.html>`__

- Added **Franka DexHand** dexterous hand end-effector support. Link: `Franka DexHand <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_dexhand.html>`__

- Added **Franka Robotiq** gripper backend support. Link: `Franka Robotiq <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_zed_robotiq.html>`__

- Added **Franka Robotiq and ZED / LUMOS V4L2** camera and gripper backend support. Link: `Franka ZED/Lumos <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_zed_robotiq.html>`__

4. Algorithms
"""""""""""""""

Major new algorithms across real-world RL, simulation RL, and human-in-the-loop learning, achieving SOTA real-world task success rates.

**Real-World RL algorithms:**

- Extended **DSRL** (Diffusion Steering via Reinforcement Learning) to the Pi0.5 model. Link: `DSRL <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dsrl.html>`__

- Added **RECAP** (offline advantage-based policy optimization) training pipeline support. Link: `RECAP <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/recap.html>`__

- Added **SAC-Flow** algorithm support, extended to DOS-W1 and other real-world scenarios. Link: `SAC-Flow <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/sac_flow.html>`__

**Simulation RL algorithms:**

- **Async PPO**: on top of v0.2, extended to support MLP and other new policies, with new async DSRL configs. Link: `Async PPO <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/reference/algorithms/async_ppo.html>`__

- Added **Pi-StepNFT** algorithm support.

- Added **D4RL offline IQL** training support (Antmaze / Kitchen-Adroit / MuJoCo, based on FSDPStrategy). Link: `IQL-D4RL <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/iql_d4rl.html>`__

**Human-in-the-loop learning:**

- Added **DAgger** online imitation learning algorithm support (LIBERO, ManiSkill, RoboTwin, real-world PnP scenarios). Link: `DAgger <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dagger.html>`__

- Added **HG-DAgger** (Human-Gated DAgger) real-world online training support. Link: `HG-DAgger <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/hg-dagger.html>`__

5. System
"""""""""""

System-level new performance optimization techniques, plus refined support for Ascend, AMD ROCm, and Musa accelerators; overall system robustness and scalability are greatly improved.

**New component support:**

- Added **Reward Model** component support: embodied reward worker + ResNet/VLM reward model, supporting standalone reward for realworld env. Link: `Reward Model <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/reward_model.html>`__

- Added **Value Model** component support: a general value model infrastructure supporting pipelines such as RECAP. Link: `RECAP <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/recap.html>`__

- Added **SGLang inference server** component support (HTTP server + router mode, usable as a reward service / rollout inference backend). Link: `SGLang Server <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/sglang_server.html>`__

- Added **Decoupled env mode** component support (decouples the one-to-one binding between Env Worker and Rollout Worker, improving rollout GPU utilization). Link: `Env Decoupled Mode <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/env_decoupled_mode.html>`__

**Performance & memory optimization support:**

- Added **torch.compile acceleration** for π0 / π0.5 predict.

- Added **rollout-training overlap** support (including bootstrap-training overlap and advantage normalization under the embodied pipeline mode).

- Added **weight synchronization upgrade**: broadcast-based weight sync, weight diff patch incremental sync, bucket sync, trainable-params-and-buffers-only sync, and async wait. Link: `Weight Syncer <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/weight_syncer.html>`__

- Added **FSDP full offload** support, and fixed checkpoint/SFT dataloader resume, actor offload state restoration, and GPU memory leak.

- Added **nsys trace, unified accelerator profiling, metrics logging file**, and other runtime & profiling support. Link: `GPU Profiling <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/profile.html>`__

**Domestic-card & cross-hardware support:**

- Added **Ascend (CANN / torch-npu)** end-to-end runnable support (``install.sh --platform ascend``, ``agentic-rlinf0.3-libero-cann9.0`` CANN Docker image). Link: `Ascend CANN <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/ascend_cann.html>`__

- Added **Musa** support for running world-model Wan RL on Musa devices.

- Added **AMD ROCm** end-to-end runnable support (``install.sh --platform amd``, auto-detects ROCm version and matches the ``+rocm`` wheel). Link: `AMD ROCm <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/amd_rocm.html>`__

**Configuration & scheduling:**

- Added **custom model registration** and **override cfgs** support, improving configuration flexibility and extensibility. Link: `New Model (FSDP) <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/new_model_fsdp.html>`__

- Added **Ray-cluster-based code sync** support (``RLINF_CODE_WORKING_DIR``, auto-distributing the ``rlinf/`` package when the filesystem is not shared).

- Added **SFT workflow refactor**: unified SFT loss/metrics API, and fixed SFT data-loading resume.

Agentic AI
^^^^^^^^^^^^

Provides a stronger training and evaluation foundation for agentic RL scenarios.

- Added **AgentLightning multiturn single-agent RL training** and Calc-X evaluation support. Link: `AgentLightning Calc-X <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/agentic/agentlightning_calc_x.html>`__

- Added **Megatron-Bridge actor backend** support (RL training and SFT for Megatron-mbridge models). Link: `Megatron-Bridge <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/mbridge.html>`__

- Refactored **SearchR1** into a multiturn interface, and added built-in sglang support for the WideSeek judge.

Papers
^^^^^^^^

2 papers are accepted to **OSDI 2026**:

- **RLinf: Flexible and Efficient Large-Scale Reinforcement Learning via Macro-to-Micro Flow Transformation** (OSDI 2026). Corresponds to the RLinf large-scale RL system. Doc: `RLinf System <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/resources/publications/rlinf_system.html>`__ ｜ Paper: `arXiv:2509.15965 <https://arxiv.org/abs/2509.15965>`__ ｜ `OSDI Talk <https://www.usenix.org/conference/osdi26/presentation/yu-chao>`__.

- **DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training** (OSDI 2026). Corresponds to RLinf's dynamic scheduling feature. Doc: `Dynamic Scheduling <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/dynamic_scheduling.html>`__ ｜ `OSDI Talk <https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing>`__.

2 more papers are accepted to **RSS 2026**:

- **USER: A Unified and Extensible System for Online Real-World Policy Learning in Embodied AI** (RSS 2026, i.e. RLinf-USER). Corresponds to the RLinf real-world online policy learning system. Doc: `RLinf-USER <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/resources/publications/rlinf_user.html>`__ ｜ Paper: `arXiv:2602.07837 <https://arxiv.org/abs/2602.07837>`__ ｜ `RSS Paper <https://roboticsconference.org/program/papers/37/>`__.

- **RLux-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models** (RSS 2026, i.e. RLinf-VLA). Corresponds to RLinf's unified VLA+RL framework. Doc: `RLinf-VLA <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/resources/publications/rlinf_vla.html>`__ ｜ Paper: `arXiv:2510.06710 <https://arxiv.org/abs/2510.06710>`__ ｜ `RSS Paper <https://roboticsconference.org/program/papers/89/>`__.

Important Fixes
^^^^^^^^^^^^^^^^^

v0.3 fixes several issues affecting training stability and data/collection correctness. **We recommend upgrading to the latest version to get these fixes.** Main fixes:

- Fixed behavior env issues including missing/blurry textures, assets loaded early during config validation, and asymmetric dump/load of TRO state.

- Fixed the openpi evaluation toolkit config-dict import error.

- Fixed the issue that the openpi model's gradient checkpointing had to be manually disabled.

- Fixed the incorrect return type when sending split trajectories to the actor.

- Unified the gripper action format, and fixed the wrong initial gripper open/close state during data collection.

- Fixed the maniskill stale offload video counter state issue.

- Fixed send_num misusing world size in the SAC actor worker.

- Fixed the issue that env did not correctly trigger offload after init, and that actor reserved memory was not released during rollout.

- Fixed system-side issues including CUDA IPC memory not being reclaimed after communication, broadcast not constrained to the same device, and AMD GPU visible-device env var configuration.

- Fixed the deadlock between weight sync and the actor barrier.

- Fixed FSDP checkpoint resume, actor offload state restoration, and GPU memory leak.

Contributors
^^^^^^^^^^^^^^

@andylin-hao @guozhen1997 @zhexuanxu @anHappyDog @Brunch-Life @thereAreDemonsNearby @yushuang20091011 @qurakchin @zanghz21 @F9rozen @FxxxxU @jx-qiu @Lin-xs @tiny-xie @lwbscu @QuanluZhang @kunni918 @Iron-Wph @secretsites @ligediaomao @ZhaoRunyi @duzhengye-droid @fy2462 @matthewmzy @chenkang455 @weiyunfei @XuS1994 @pikaxinge @drewzhao @WayneTimer @Matrix326 @pancake-w @lizuojun04 @MrHappa @HzfFrank @renq-mt @liuhaoyunBUPT @yxuan1234 @crabxiexy @MuggleZzzH @ppppppppppper @xb534 @zhigenzhao @wingAGI @aasivas @git-xuxin @LiuZhihao2022 @pyy233 @Dps799 @yangchen73 @jeis4wpi @NLC2004 @AIhuaYuan @zjk-prog @YimingZhou2002 @Walkism @slzhta @iamxjy @YifWRobotics @AlphaReimu @hongyuxiyohung @WinstonWmj @jzndd @Elessar123

RLinf v0.3 Test Results
^^^^^^^^^^^^^^^^^^^^^^^^^

We tested most configuration files to guarantee the correctness of the provided examples in this release.

.. list-table::
   :header-rows: 1
   :widths: 22 22 40

   * - Configuration file
     - Model name
     - Result curve
   * - maniskill_ppo_openpi.yaml
     - RLinf-Pi0-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openpi.png
          :alt: maniskill_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openpi_pi05.yaml
     - RLinf-Pi05-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openpi_pi05.png
          :alt: maniskill_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openvla.png
          :alt: maniskill_ppo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-ManiSkill-Base-Main
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openvlaoft.png
          :alt: maniskill_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_mlp.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_mlp.png
          :alt: maniskill_ppo_mlp.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_grpo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_grpo_openvla.png
          :alt: maniskill_grpo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_grpo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-ManiSkill-Base-Main
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_grpo_openvlaoft.png
          :alt: maniskill_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - libero_goal_ppo_openpi.yaml
     - RLinf-Pi0-LIBERO-130-fullshot-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_goal_ppo_openpi.png
          :alt: libero_goal_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - libero_goal_ppo_openpi_pi05.yaml
     - RLinf-Pi05-LIBERO-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_goal_ppo_openpi_pi05.png
          :alt: libero_goal_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_10_ppo_gr00t.yaml
     - RLinf-Gr00t-SFT-Long
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_10_ppo_gr00t.png
          :alt: libero_10_ppo_gr00t.yaml result curve
          :width: 95%
          :align: center
   * - calvin_abcd_d_ppo_openpi_pi05.yaml
     - RLinf-Pi05-CALVIN-ABC-D-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/calvin_abcd_d_ppo_openpi_pi05.png
          :alt: calvin_abcd_d_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_place_empty_cup_ppo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/robotwin_place_empty_cup_ppo_openvlaoft.png
          :alt: robotwin_place_empty_cup_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_beat_block_hammer_grpo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/robotwin_beat_block_hammer_grpo_openvlaoft.png
          :alt: robotwin_beat_block_hammer_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - isaaclab_franka_stack_cube_ppo_gr00t.yaml
     - RLinf-Gr00t-SFT-Stack-cube
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/isaaclab_franka_stack_cube_ppo_gr00t.png
          :alt: isaaclab_franka_stack_cube_ppo_gr00t.yaml result curve
          :width: 95%
          :align: center
   * - gsenv_ppo_openpi_pi05.yaml
     - RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/gsenv_ppo_openpi_pi05.png
          :alt: gsenv_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - frankasim_ppo_mlp.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/frankasim_ppo_mlp.png
          :alt: frankasim_ppo_mlp.yaml result curve
          :width: 95%
          :align: center
   * - frankasim_sac_cnn_async.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/frankasim_sac_cnn_async.png
          :alt: frankasim_sac_cnn_async.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openpi.yaml
     - RLinf-Pi0-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openpi.png
          :alt: maniskill_async_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openpi_pi05.yaml
     - RLinf-Pi05-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openpi_pi05.png
          :alt: maniskill_async_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openvla.png
          :alt: maniskill_async_ppo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-trajall
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openvlaoft.png
          :alt: maniskill_async_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_sac_mlp.png
          :alt: maniskill_sac_mlp.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_async_ppo_openpi.yaml
     - RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_async_ppo_openpi.png
          :alt: libero_spatial_async_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - libero_object_async_ppo_openpi_pi05.yaml
     - RLinf-Pi05-LIBERO-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_object_async_ppo_openpi_pi05.png
          :alt: libero_object_async_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_grpo_openpi_pi05.yaml
     - RLinf-Pi05-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_grpo_openpi_pi05.png
          :alt: libero_spatial_grpo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_10_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-traj1
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_10_grpo_openvlaoft.jpg
          :alt: libero_10_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - opensora_libero_spatial_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero-spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/opensora_libero_spatial_grpo_openvlaoft.jpg
          :alt: opensora_libero_spatial_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - wan_libero_spatial_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero-spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/wan_libero_spatial_grpo_openvlaoft.jpg
          :alt: wan_libero_spatial_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen2_5_vl_sft_vlm.yaml
     - Qwen/Qwen2.5-VL-3b-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen2_5_vl_sft_vlm.png
          :alt: examples/sft/config/qwen2_5_vl_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen3_vl_sft_vlm.yaml
     - Qwen/Qwen3-VL-4b-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen3_vl_sft_vlm.png
          :alt: examples/sft/config/qwen3_vl_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml
     - Qwen/Qwen2.5-1.5B-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen2.5-1.5b-ppo-megatron.png
          :alt: examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp_resnet_reward_async.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_sac_mlp_resnet_reward_async.png
          :alt: maniskill_sac_mlp_resnet_reward_async.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp_async_decoupled.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_sac_mlp_async_decoupled.png
          :alt: maniskill_sac_mlp_async_decoupled.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_grpo_starvla.yaml
     - StarVLA (Qwen2.5-VL-OFT-LIBERO-4in1)
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_grpo_starvla.jpg
          :alt: libero_spatial_grpo_starvla.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_ppo_gr00t_n1d6.yaml
     - GR00T-N1.6
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_ppo_gr00t_n1d6.png
          :alt: libero_spatial_ppo_gr00t_n1d6.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_dagger_openpi.yaml
     - RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_dagger_openpi.png
          :alt: libero_spatial_dagger_openpi.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_place_shoe_grpo_lingbotvla.yaml
     - LingBot-VLA-4B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/robotwin_place_shoe_grpo_lingbotvla.png
          :alt: robotwin_place_shoe_grpo_lingbotvla.yaml result curve
          :width: 95%
          :align: center
   * - genesis_cubepick_ppo_cnn.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/genesis_cubepick_ppo_cnn.png
          :alt: genesis_cubepick_ppo_cnn.yaml result curve
          :width: 95%
          :align: center
   * - d4rl_iql_kitchen_adroit.yaml
     - IQL
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/d4rl_iql_kitchen_adroit_1.png
          :alt: d4rl_iql_kitchen_adroit.yaml result curve
          :width: 95%
          :align: center

       .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/d4rl_iql_kitchen_adroit_2.png
          :alt: d4rl_iql_kitchen_adroit.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/libero_sft_dreamzero_5b.yaml
     - DreamZero-5B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_sft_dreamzero_5b.png
          :alt: examples/sft/config/libero_sft_dreamzero_5b.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen2_5_vl_megatron_sft_vlm.yaml
     - Qwen2.5-VL
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen2_5_vl_megatron_sft_vlm.png
          :alt: examples/sft/config/qwen2_5_vl_megatron_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn.yaml
     - Qwen2.5-1.5B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/agentlightning_qwen2.5-1.5b-enginehttp-multiturn.png
          :alt: examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn.yaml result curve
          :width: 95%
          :align: center
   * - examples/agent/searchr1/config/train_qwen2.5.yaml
     - Qwen2.5
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/searchr1_train_qwen2.5_1.png
          :alt: examples/agent/searchr1/config/train_qwen2.5.yaml result curve
          :width: 95%
          :align: center

       .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/searchr1_train_qwen2.5_2.png
          :alt: examples/agent/searchr1/config/train_qwen2.5.yaml result curve
          :width: 95%
          :align: center

Quick Start
^^^^^^^^^^^^^

- **Baidu AI Cloud**: `https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1 <https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1>`__
- **Infinigence AI**: `https://docs.neogpu.com/posts/rlinf-ppo-vla.html <https://docs.neogpu.com/posts/rlinf-ppo-vla.html>`__
