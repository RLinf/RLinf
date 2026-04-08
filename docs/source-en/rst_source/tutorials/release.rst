Release Notes
=============

RLinf v0.2 Release
------------------

🎉 Introducing RLinf v0.2.

RLinf v0.2 focuses on two major directions: **Real-World RL** and **Multi-agent RL** systems. To support these goals, RLinf now supports real-world platforms including **XSquare Turtle2 Arms** and the **Franka Arm**, while offering a richer set of embodied benchmarks, simulators, models, algorithms, together with **native asynchronous training** designed for high-throughput workloads. This release also strengthens real-world deployment, sim-to-real, and co-training workflows, alongside more robust data and replay infrastructure and improved training stability. For multi-agent training, RLinf introduces **native multi-agent support** for extensible multi-agent RL algorithms and unified data interfaces, lowering the barrier to developing and scaling multi-agent workloads while enabling rapid reproduction of advanced training solutions such as **WideSeek-R1**.

Embodied Intelligence
^^^^^^^^^^^^^^^^^^^^^

1. Core Capability Upgrades, highlighting Real-World Robotics RL and World Models

- Supported Real-World RL with :doc:`XSquare Turtle2 <../examples/embodied/xsquare_turtle2>`.

- Supported World Models as simulators for RL training, including :doc:`OpenSora <../examples/embodied/opensora>`, :doc:`Wan <../examples/embodied/wan>`, and :doc:`WoVR <../publications/wovr>`.

- Vision-Language Model Supervised Fine‑Tuning adds supervised fine‑tuning (SFT) capabilities for vision‑language models (VLMs), supporting efficient fine‑tuning on custom datasets. Verified on the Robo2VLM dataset, achieving approximately 95% reproduction accuracy for `PR 708 <https://github.com/RLinf/RLinf/pull/708>`_ and `PR 781 <https://github.com/RLinf/RLinf/pull/781>`_ models. See :doc:`SFT VLM <../examples/embodied/sft_vlm>`.

- Supported Real2Sim2Real RL training based on :doc:`GSEnv (ManiSkill-GS) <../examples/embodied/gsenv>`.

- Supported RL-based Sim-Real Co-Training of the π(0.5) model with :doc:`Co-training <../examples/embodied/co_training>`.

1. Model and Algorithm Ecosystem Expansion

- Supported Dexbotic models and RL training with :doc:`Dexbotic <../examples/embodied/dexbotic>`.

- Improved support for :doc:`IsaacLab <../examples/embodied/isaaclab>`, especially for GR00T+IsaacLab.

- Supported RL training of the openpi model family on :doc:`RoboTwin 2.0 <../examples/embodied/robotwin>`.

- Supported RL with the :doc:`CALVIN benchmark <../examples/embodied/calvin>`.

- Supported RL with the :doc:`RoboCasa benchmark <../examples/embodied/robocasa>`.

- Supported DSRL (Diffusion Steering via Reinforcement Learning) for pi0 with :doc:`DSRL <../examples/embodied/dsrl>`.

- Supported SAC training for flow matching policy with :doc:`SAC-Flow <../examples/embodied/sac_flow>`.

1. Training Infrastructure Enhancements

- Added a new wrapper in the data layer to support replay buffer collection of real-robot and simulator data as a standard module. Refer to :doc:`Data collection <components/data_collection>`.

- Async Training Support introduces asynchronous training as a first‑class capability for embodied models, providing asynchronous PPO workflows and usability improvements to boost training efficiency in high‑throughput scenarios. Refer to :doc:`Async PPO <rlalg/async_ppo>`.

- Data and Replay Pipeline Upgrade enhances data collection and replay pipelines, strengthens buffer preloading, updating, and checkpoint handling, and improves overall dataflow robustness. Refer to :doc:`Replay buffer API <../apis/replay_buffer>`.

- Runtime Performance Optimizations add runtime features such as CUDA Graph, torch.compile, environment offloading, and FSDP path optimizations to improve execution efficiency for embodied training. Refer to :doc:`YAML configuration <user/yaml>`.

1. Stability Improvements and Usability

- Applies multiple fixes to PPO/GRPO behavior and real‑world configuration handling, enhancing training stability and configuration correctness.

- Added huggingface model link in yaml configuration files for easy downloading

Agentic RL
^^^^^^^^^^

1. Core Capability Upgrades, highlighting Multi-Agent RL

- Native Multi‑Agent Training Support introduces extensible multi‑agent reinforcement learning algorithms and unified data interfaces, significantly reducing the entry barrier for multi‑agent tasks. Enables rapid reproduction of complex `PR 824 <https://github.com/RLinf/RLinf/pull/824>`_ such as :doc:`WideSeek-R1 <../examples/agentic/wideseek_r1/index>`.

- PPO Support for Reasoning Tasks: `PR 771 <https://github.com/RLinf/RLinf/pull/771>`_. Extends PPO algorithm support to reasoning tasks, further broadening RLinf’s applicability in complex reasoning and decision‑making scenarios with :doc:`Reasoning PPO <../examples/agentic/reasoning_ppo>`.

- The Megatron-LM backend now supports the FUSCO communication library: `PR 783 <https://github.com/RLinf/RLinf/pull/783>`_. Delivers significant performance and scalability improvements for All-to-All communication during MoE model training and inference with :doc:`FUSCO <../examples/system/fusco>`.

- Supported agentic reinforcement learning on :doc:`rStar2 <../examples/agentic/rstar2>` (`PR 522 <https://github.com/RLinf/RLinf/pull/522>`_) and :doc:`Search-R1 <../examples/agentic/searchr1>` (`PR 639 <https://github.com/RLinf/RLinf/pull/639>`_).

Other improvements and bug fixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- RLinf refactored init_worker and weight synchronization to improve performance, and added support for agents to compute rewards within the agent loop, eliminating the need for a separate reward worker: `PR 524 <https://github.com/RLinf/RLinf/pull/524>`_

- RLinf updated the FSDP backend to support dynamic batch size, data‑parallel load balancing, gradient scaling with fp16 via an unscale_patch, and multi‑bucket weight synchronization: `PR 553 <https://github.com/RLinf/RLinf/pull/553>`_

- Resolved dataset reading issues and added batch encoding during tokenizer loading to increase throughput: `PR 653 <https://github.com/RLinf/RLinf/pull/653>`_

- Supported using Qdrant as the Wiki server, enabling efficient vector search and storage for wiki documents: `PR 673 <https://github.com/RLinf/RLinf/pull/673>`_

- Refactored FSDP precision handling to support only fp16 and bf16, replacing the ambiguous AMP structure while preserving backward compatibility: `PR 715 <https://github.com/RLinf/RLinf/pull/715>`_

- Fixed an issue in reasoning tasks where batch counts across ranks became inconsistent due to improper splitting of rollout outputs: `PR 775 <https://github.com/RLinf/RLinf/pull/775>`_

- Fixed the epsilon configuration bug in pi models' training: `PR 623 <https://github.com/RLinf/RLinf/pull/623>`_

- Fixed an issue in the openpi model where gradient checkpointing previously had to be disabled manually before training: `PR 843 <https://github.com/RLinf/RLinf/pull/843>`_

- Fixed the Docker image for Franka Arm: `PR 862 <https://github.com/RLinf/RLinf/pull/862>`_

- Fixed send_num to use env world size instead of rollout world size in SAC actor worker: `PR 882 <https://github.com/RLinf/RLinf/pull/882>`_

- Fixed an issue where the second round of rollout would receive random reset_state_ids: `PR 886 <https://github.com/RLinf/RLinf/pull/886>`_

- Fixed a bug that caused environment offloading after initialization and ensured actor reserved memory is properly released during rollout: `PR 897 <https://github.com/RLinf/RLinf/pull/897>`_

Documentation
^^^^^^^^^^^^^

- Reorganized the structure of the :doc:`examples index <../examples/index>`, classified examples into embodied scenarios, agentic scenarios, and system-level optimizations.

- Added the :doc:`FAQ <../faq>` document for breakpoint debugging.

- Added awesome work and adoption section in README

Contributors
~~~~~~~~~~~~

@Hao Lin @qurakchin @guozhen1997 @zanghz21 @Bo Dai @FxxxxU @Elessar123 @LiuYiwei @Xzxuan @ysa @jzndd @zlock @xusi @Louis-J @WinstonWmj @shiletong @xzxuan @liyanghao @Iron_Wph @chenkang455 @shengyz @yimingzhou2002 @Florielle @xuxin @Yinuo Chen @nufukim @Lin-xs @zhangruize @Iron-Wph @Hongyi Zhu @red0orange @chenkang @hongzhi @thereAreDemonsNearby @Zoran Zhu @Tziy @Yimingzhou2002 @Nan Yang @AIhuaYuan @AIhuayuan @xuxin @MacBook-M3-Pro @wangxiangyuan @slzhta @Iron-Wph @fy2462 @Ning Xu @weimingjie @zlockewtg @smallcracker @gongyue teng @cc @Xin Xu @xiebin @yuyingyinya @Yun Liu @Tao Liu @renqian @Wheels Wu @Wheeeeeeeeels @Felix Zhang @pyy233 @LiuZhihao2022

RLinf v0.2 test results
~~~~~~~~~~~~~~~~~~~~~~~

We tested most configuration files to guarantee the correctness of our provided examples in this release.

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - Configuration file
     - Model name
     - Result curve
   * - maniskill_ppo_openpi.yaml
     - RLinf-Pi0-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_ppo_openpi.png
          :alt: maniskill_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openpi_pi05.yaml
     - RLinf-Pi05-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_ppo_openpi_pi05.png
          :alt: maniskill_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_ppo_openvla.png
          :alt: maniskill_ppo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-trajall (LORA: RLinf-OpenVLAOFT-ManiSkill-Base-Lora)
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_ppo_openvlaoft.png
          :alt: maniskill_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_mlp.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_ppo_mlp.png
          :alt: maniskill_ppo_mlp.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_grpo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_grpo_openvla.png
          :alt: maniskill_grpo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-trajall (LORA: RLinf-OpenVLAOFT-ManiSkill-Base-Lora)
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_grpo_openvlaoft.png
          :alt: maniskill_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - libero_goal_ppo_openpi.yaml
     - RLinf-Pi0-LIBERO-130-fullshot-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/libero_goal_ppo_openpi.png
          :alt: libero_goal_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - libero_goal_ppo_openpi_pi05.yaml
     - RLinf-Pi05-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/libero_goal_ppo_openpi_pi05.png
          :alt: libero_goal_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - calvin_abcd_d_ppo_openpi_pi05.yaml
     - RLinf-Pi05-CALVIN-ABC-D-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/calvin_abcd_d_ppo_openpi_pi05.png
          :alt: calvin_abcd_d_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_place_empty_cup_ppo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/robotwin_place_empty_cup_ppo_openvlaoft.png
          :alt: robotwin_place_empty_cup_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_beat_block_hammer_grpo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/robotwin_beat_block_hammer_grpo_openvlaoft.png
          :alt: robotwin_beat_block_hammer_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - isaaclab_franka_stack_cube_ppo_gr00t.yaml
     - RLinf-Gr00t-SFT-Stack-cube
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/isaaclab_franka_stack_cube_ppo_gr00t.png
          :alt: isaaclab_franka_stack_cube_ppo_gr00t.yaml result curve
          :width: 95%
          :align: center
   * - gsenv_ppo_openpi_pi05.yaml
     - RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/gsenv_ppo_openpi_pi05.png
          :alt: gsenv_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - frankasim_ppo_mlp.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/frankasim_ppo_mlp.jpeg
          :alt: frankasim_ppo_mlp.yaml result curve
          :width: 95%
          :align: center
   * - frankasim_sac_cnn_async.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/frankasim_sac_cnn_async.jpeg
          :alt: frankasim_sac_cnn_async.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openpi.yaml
     - RLinf-Pi0-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_async_ppo_openpi.png
          :alt: maniskill_async_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openpi_pi05.yaml
     - RLinf-Pi05-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_async_ppo_openpi_pi05.png
          :alt: maniskill_async_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_async_ppo_openvla.png
          :alt: maniskill_async_ppo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-trajall
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_async_ppo_openvlaoft.png
          :alt: maniskill_async_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/maniskill_sac_mlp.png
          :alt: maniskill_sac_mlp.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_async_ppo_openpi.yaml
     - RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/libero_spatial_async_ppo_openpi.png
          :alt: libero_spatial_async_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - libero_object_async_ppo_openpi_pi05.yaml
     - RLinf-Pi05-LIBERO-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/libero_object_async_ppo_openpi_pi05.png
          :alt: libero_object_async_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_grpo_openpi_pi05.yaml
     - RLinf-Pi05-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/libero_spatial_grpo_openpi_pi05.png
          :alt: libero_spatial_grpo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_10_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-traj1
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/libero_10_grpo_openvlaoft.png
          :alt: libero_10_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - opensora_libero_spatial_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero-spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/opensora_libero_spatial_grpo_openvlaoft.png
          :alt: opensora_libero_spatial_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - wan_libero_spatial_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero-spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/wan_libero_spatial_grpo_openvlaoft.png
          :alt: wan_libero_spatial_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen2_5_sft_vlm.yaml
     - Qwen/Qwen2.5-VL-3b-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/qwen2_5_sft_vlm.png
          :alt: examples/sft/config/qwen2_5_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen3_sft_vlm.yaml
     - Qwen/Qwen3-VL-4b-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/qwen3_sft_vlm.png
          :alt: examples/sft/config/qwen3_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml
     - Qwen/Qwen2.5-1.5B-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.2/qwen2.5-1.5b-ppo-megatron.png
          :alt: examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml result curve
          :width: 95%
          :align: center

RLinf v0.1 Release
------------------

🎉 Introducing RLinf v0.1.

Built on robust system-level scheduling and communication components, RLinf is a scalable and flexible framework for post-training via reinforcement learning in embodiment, reasoning, and agent scenarios. The framework has been validated on popular models and tasks and achieves state-of-the-art model performance and training throughput, showcasing its extensibility, versatility, and efficiency in diverse scenarios.

Embodied Intelligence
^^^^^^^^^^^^^^^^^^^^^

- Supported end-to-end embodied RL training on multiple mainstream simulators (e.g., ManiSkill, Libero, MetaWorld, CALVIN), achieving state-of-the-art performance with multiple VLA models (e.g., OpenVLA, OpenVLA-OFT, π₀, π₀.₅, GR00T) and algorithms (GRPO, PPO), reaching a success rate of up to 99%.

- Up to 143.4% faster training (2.434× throughput) compared to existing frameworks, with flexibly allocated, decoupled, and hybrid execution modes, scaling effortlessly to thousands of GPUs.

- The effectiveness of RL training has been verified in ManiSkill, LIBERO, MetaWorld, and CALVIN and reproducible best-practice scripts are provided in our embodiment examples.

Agent & Reasoning RL
^^^^^^^^^^^^^^^^^^^^

- With 1.5B and 7B models, RLinf achieves state-of-the-art results on AIME 24, AIME 25, and GPQA-Diamond benchmarks.

- Through pipeline parallelism (20%+) and automatic scheduling (30%+), RLinf demonstrates significant efficiency improvements and strong reasoning capabilities.

- Supported auto scheduling and scaling of Megatron-based training. SGLang/vLLM and Megatron can automatically scale down/up during training to achieve maximum throughput, delivering 40%+ speedup compared to static placement.

- Introduced the first open-source 1.5B online RL agent, boosting code completion accuracy by 50%+, outperforming even 32B-scale models.
