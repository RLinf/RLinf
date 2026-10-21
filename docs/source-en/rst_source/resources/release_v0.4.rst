Release Notes
=============

RLinf v0.4 Release
------------------

🚀 RLinf v0.4 is here. Reinforcement learning steps out of simulation and into agile control of the physical world.

v0.3 connected the path from simulation to real robots. v0.4 is the step toward deploying embodied RL at real-world scale. The release rebuilds the stack around three pillars: a modular real-robot system, high-efficiency real-robot algorithms, and full-modality models on a heterogeneous compute base.

- 🧩 **Modular real-robot system** (``rlinf.robotics``): arms, dexterous hands, cameras, mobile bases, and teleoperation devices become composable blocks. Hardware drivers are fully decoupled from task logic, so changing a robot body, a configuration, or a cross-node deployment does not require changes to task or policy code.
- ⚡ **Real-robot SOTA post-training**: representation-decoupled online RL (RLT), label-free self-supervised offline RL (STEAM), on-policy distillation (OPD) that reaches 96%+ success in 30 steps, and asynchronous real-time control (RTC) that compresses inference wait to about 2 ms. Together they address expensive real-robot samples, costly labels, and inference stutter.
- 🌐 **Full-modality models**: a PyTorch-native OpenPI (π₀ / π₀.₅) rewritten to match official JAX numerics operator by operator; NVIDIA Cosmos3, FastWAM, π₀-FAST, Evo-1, and MolmoAct2; and the first extension of RL to SD3 / Wan2.2 visual generation.
- 🤝 **Eight accelerator backends and the open ecosystem**: Kunlunxin, Biren, MetaX, and Intel XPU join the stack. Huawei Ascend and Moore Threads reach up to 2.78× throughput, with convergence aligned to A100. RLinf joins the PyTorch Ecosystem, ships as the RL training engine in NVIDIA Isaac Lab v3.0, and publishes joint practices with SGLang and AMD.

Training and evaluation recipes covered by this release are validated for correctness and reproducibility (see the test results at the end).

Embodied
^^^^^^^^

1. Real World
"""""""""""""

A unified robot interface decouples hardware from tasks. Arms, end-effectors, cameras, and bases compose freely. The release adds three real-robot platforms and a full dual-arm / single-arm collect–train–deploy toolchain.

**Robot architecture:**

- **Modular real-robot interface** (``rlinf.robotics``): drivers, task definitions, and environment wrappers are decoupled. Arms, dexterous hands / grippers, multi-view cameras, and mobile bases are addressed by one hierarchy. Swapping a driver or a cross-node topology does not require task or policy changes. Links: :doc:`Robot Interface <../concepts/robotics>` ｜ :doc:`Robot Architecture <../concepts/robotics_architecture>` ｜ :doc:`Add a Robot <../extending/new_robot>`

**Platforms and training pipelines:**

- **Dual-arm Franka**: PICO VR bimanual teleoperation, per-arm HG-DAgger, and native PyTorch OpenPI π₀.₅ dual-arm SFT and deployment. Links: :doc:`Dual Franka PICO DAgger <../examples/embodied/dual_franka_pico_dagger>` ｜ :doc:`Dual Franka OpenPI <../examples/embodied/dual_franka_openpi_pytorch>`
- **Single-arm Franka**: a Qwen-VL real-world reward model, depth capture, smooth intervention, online DAgger writes into LeRobot episodes, and LeRobot dataset merge / filter tools. Link: :doc:`Real-World Reward Model <../examples/embodied/franka_reward_model>`
- **New platforms**: dual-arm YAM (PICO VR teleoperation and LeRobot collection), AgileX Piper, and SO-101. Links: :doc:`AgileX Piper <../examples/embodied/piper>` ｜ :doc:`SO-101 <../examples/embodied/so101>`

2. Algorithms
"""""""""""""

New SOTA algorithms cover online real-robot RL, label-free offline policy optimization, on-policy distillation, and high-frequency real-time control. They cut real-robot sample cost and raise both success rate and deployment smoothness.

**Real-world and simulation RL:**

- **RLT (RL Token)**: two-stage online RL. High-dimensional visual representation is decoupled from control. Stage 1 jointly fine-tunes the VLA and the RLT token extractor; Stage 2 trains a lightweight off-policy actor-critic (including a TD3 variant) on the frozen compact representation. It converges efficiently on ManiSkill and real Franka peg insertion. Link: :doc:`RLT <../examples/embodied/rlt>`
- **STEAM**: self-supervised offline advantage modeling. No human labels, hand-crafted rewards, or online interaction. A deep ensemble of progress critics (worst-of-N) is trained on the temporal order of demonstration frame pairs, then drives classifier-free guidance policy optimization. It clearly surpasses RECAP on real tasks such as bimanual towel folding, checkout, and restocking. Link: :doc:`STEAM <../examples/embodied/steam>`

**On-policy distillation and real-time control:**

- **OPD (On-Policy Distillation)**: the student rolls out, and the teacher scores action tokens densely, then offloads immediately. On LIBERO-Spatial, 30 steps raise an OpenVLA-OFT student from 67% to 96%+. Link: :doc:`OPD <../examples/embodied/opd>`
- **RTC (Real-Time Control)**: inference of the next action chunk overlaps execution of the current chunk, with soft / hard masks smoothing chunk boundaries. Rollout inference wait drops from seconds to about 2 ms, and episode time shrinks by up to 7.4 seconds. Supported on LIBERO and real Franka. Link: :doc:`RTC <../guides/rtc>`

3. Models
"""""""""

Five new world-action and VLA models join the embodied stack, and OpenPI is rebuilt as native PyTorch. The coverage spans SFT, online RL, and high-throughput closed-loop evaluation.

- **OpenPI (π₀ / π₀.₅), native PyTorch**: numerically aligned with the official JAX reference, removing the drift and convergence issues of the previous community PyTorch port. Full SFT, LoRA, and RL now run on LIBERO, ManiSkill, BEHAVIOR-1K, RoboTwin, and single- / dual-arm Franka. Links: :doc:`OpenPI SFT <../examples/embodied/sft_openpi>` ｜ :doc:`π₀ / π₀.₅ RL <../examples/embodied/pi0>`
- **NVIDIA Cosmos3**: FSDP2 SFT of the 16B mixture-of-transformers world-action model on LIBERO, with convergence aligned to the official run. SGLang multi-batch and pipeline-parallel evaluation raises throughput by 3.33× over a direct integration. Links: :doc:`Cosmos3 SFT <../examples/embodied/sft_cosmos3>` ｜ :doc:`Cosmos3 SGLang Evaluation <../evaluations/guides/cosmos3_sglang>`
- **FastWAM**: FSDP SFT and batched closed-loop evaluation of the video-diffusion world-action model on the four LIBERO suites. Link: :doc:`FastWAM <../examples/embodied/sft_fastwam>`
- **π₀-FAST**: the LeRobot autoregressive action-token policy, with deterministic evaluation and token-level GRPO on LIBERO-Long. Link: :doc:`π₀-FAST <../examples/embodied/pi0_fast>`
- **Evo-1**: an about-1B flow-matching VLA (InternVL3-1B plus a DiT action head), with full-parameter SFT and GRPO on LIBERO. Link: :doc:`Evo-1 <../examples/embodied/evo1>`
- **MolmoAct2**: AllenAI's open VLA, with parallel closed-loop evaluation on LIBERO Spatial / Object / Goal / Long. Link: :doc:`MolmoAct2 <../examples/embodied/molmoact2>`

4. Simulators and Evaluation
""""""""""""""""""""""""""""

The release adds a large-scale household benchmark, a high-fidelity real-robot link simulator, and a high-throughput evaluation path for world-action models and VLAs.

- **RoboCasa365**: large-scale household manipulation, with an OpenPI PPO example. Link: :doc:`RoboCasa365 <../examples/embodied/robocasa365>`
- **Real-world simulator**: injects measured sensor delay and cross-cluster / cloud-edge bandwidth and latency, so real-robot algorithms can be regression-tested before they touch hardware. Link: :doc:`Real-World Simulator <../examples/embodied/real_simulator>`
- **SGLang and ApxInf evaluation backends**: SGLang evaluation for world-action models such as Cosmos3 and DreamZero, plus ApxInf, an in-process Rust/CUDA backend for low-latency π₀.₅ evaluation on LIBERO. Links: :doc:`SGLang Embodied Evaluation <../extending/sglang_embodied_model>` ｜ :doc:`DreamZero SGLang Evaluation <../evaluations/guides/dreamzero_sglang>` ｜ :doc:`ApxInf <../guides/apxinf>`
- **Existing simulators**: LIBERO gains camera calibration, arbitrary-resolution rendering, and skipped intermediate frames, and aligns LIBERO-Pro initial perturbations. RoboTwin gains a new DAgger path and memory reclamation. EmbodiChain is updated to 0.3.0.

5. Systems and Heterogeneous Hardware
"""""""""""""""""""""""""""""""""""""

Image and tensor transport, weight resharding, and training observability are upgraded. Four new domestic / heterogeneous accelerators are added, with large gains on Huawei Ascend and Moore Threads.

**Performance and communication:**

- **Lossless observation compression**: XOR-delta plus LZ4/Zstd on Env → Rollout images. The payload falls to 36%–41% of the original size and still restores bit for bit. Link: :doc:`Observation Compression <../guides/obs_compression>`
- **Communication and weight sync**: tensor compression for collective communication, flexible trajectory transfer, and a rebuilt weight-reshard path. Async runners can safely overlap rollout and weight sync.
- **Training stack and observability**: the default runtime moves to PyTorch 2.11, with newer SGLang / vLLM. FSDP module-level mixed precision, the SGLang VLM trend reward (QwenTrend), and end-to-end distributed tracing / profiling are supported. Link: :doc:`Engine Versions <../guides/version>`

**Domestic and cross-vendor hardware:**

- **Four new accelerators**: Kunlunxin XPU, Biren SUPA, MetaX MACA, and Intel XPU. One training stack now runs on NVIDIA, AMD, and six mainstream domestic / heterogeneous accelerators.
- **Huawei Ascend (CANN)**: training support for GR00T (N1.5 / N1.6 / N1.7), StarVLA, DreamZero, and Wan. Attention-metadata caching, fused NPU operators, on-device batched preprocessing, and FSDP forward/backward prefetch overlap raise GR00T N1.5/N1.6 action-inference throughput by 195%–278%, four-card training throughput by 37%–42%, and StarVLA training throughput by 186%–240%. Ten mainstream embodied models reach more than 75% of A100 end-to-end efficiency on Ascend 910B3.
- **Moore Threads (MUSA)**: installation, Docker images, and end-to-end tests are in place. WoVR world-model RL is reproduced on S5000; the 191-step success curve correlates with A100 at 0.976 (LIBERO evaluation 74.6% vs. 73.4% on A100). Two-node hybrid sharding cuts step time by 15.3% and weight-sync time by 66%.

Generative Models
^^^^^^^^^^^^^^^^^

Reinforcement learning now extends beyond embodied policies to a standalone image and video diffusion pipeline.

- **Diffusion-NFT**: generation is treated as the environment. SD3 and Wan2.2 (including TI2V-5B) can be fine-tuned online, with image OCR and frame-level video OCR rewards, plus custom media scorers. Link: :doc:`RL for Video Generation <../examples/video_models>`

Agentic AI
^^^^^^^^^^

Large MoE reasoning training and a multi-agent FSDP backend are added.

- **Moonlight-16B-A3B GRPO**: Megatron-Bridge (TP=8, EP=8) and SGLang expert parallelism train MLA + MoE models, including Moonlight-16B-A3B and DeepSeek-V3 / Kimi-K2 architectures, on math reasoning. Link: :doc:`Moonlight-16B GRPO <../examples/agentic/math_reasoning/moonlight>`
- **MA-FSDP**: multi-agent RL can use either FSDP or Megatron. SearchR1 gains an FSDP recipe.

Ecosystem and Research
^^^^^^^^^^^^^^^^^^^^^^

- **PyTorch Ecosystem**: RLinf is now a PyTorch Ecosystem project. Blog: `PyTorch Ecosystem Landscape Q3 Update <https://pytorch.org/blog/pytorch-ecosystem-landscape-q3-update/>`__
- **NVIDIA Isaac Lab v3.0.0**: Isaac Lab adopts RLinf as the infrastructure for VLA post-training and reinforcement learning. Docs: :doc:`RLinf in Isaac Lab <../examples/embodied/isaaclab>`
- **RLinf × SGLang**: SFT and pipeline-parallel SGLang evaluation for world-action models such as Cosmos3 and DreamZero, with 3.33× evaluation throughput. Blog: :doc:`RLinf × SGLang: Cosmos3 <blog/cosmos3_sft_and_sglang_evaluate>`
- **RLinf × AMD**: WoVR static world-model RL is reproduced on AMD MI300 (ROCm). A frozen Qwen3-VL reward (TOPReward) replaces a task-specific reward model and lifts LIBERO-Spatial success from 44.8% to 56.0%, in line with a dedicated ResNet reward. Blog: :doc:`RLinf × WorldLoop <blog/worldloop_topreward>`
- **NSDI 2027**: *FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion*, the MoE transform-communication fusion library, is accepted to NSDI 2027. Docs: :doc:`FUSCO <../examples/system/fusco>` ｜ Paper: `arXiv:2512.22036 <https://arxiv.org/abs/2512.22036>`__
- **STEAM**: *STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning*. Docs: :doc:`STEAM <publications/steam>` ｜ Paper: `arXiv:2606.29834 <https://arxiv.org/abs/2606.29834>`__

Important Fixes
^^^^^^^^^^^^^^^

v0.4 fixes issues that affect numerical correctness, distributed communication, and resume. Upgrading is recommended.

- **Algorithms and numerics**: PPO actor metrics are normalized by valid mask width; entropy is no longer mis-scaled by micro-batch size; the ReinForce++ baseline no longer crashes on a single-sample group; DAgger bootstrap masks align; co-training applies the SFT loss; critic explained variance is aggregated globally.
- **Models and backends**: the IQL critic runs through the FSDP root; FSDP gradient norms reduce across all shards; FSDP2 local-shard restore uses the correct dtype; DCP checkpoints save per-rank RNG state.
- **Scheduling and weight sync**: the overlap race between rollout and weight sync is fixed; BucketWeightSyncer sorts parameter names so collectives no longer hang; the sender waits between initial sync buckets; FSDP collectives honor ``RLINF_TIMEOUT``.
- **Resume and environments**: auto-resume skips incomplete checkpoints. ManiSkill offload reclaims memory safely, MetaWorld EGL device mapping is correct under multi-GPU scheduling, and LIBERO-10 GRPO samples by initial state.

Upgrade Notes
^^^^^^^^^^^^^

Moving from v0.3 to v0.4 requires the following changes:

- **Environment**: the default stack is PyTorch 2.11 and newer SGLang / vLLM. Rebuild with ``requirements/install.sh`` or pull an ``rlinf0.4`` Docker image. Do not reuse a v0.3 virtual environment.
- **Real-robot API**: ``rlinf.envs.realworld`` and ``rlinf.scheduler.hardware.robots`` move to ``rlinf.robotics`` and ``rlinf.envs.real``. See :doc:`Add a Robot <../extending/new_robot>`.
- **OpenPI checkpoints**: ``model_type: openpi`` now points at the JAX-aligned PyTorch implementation. Convert older OpenPI-PyTorch checkpoints with the :doc:`checkpoint conversion tool <../guides/convertor>` before loading them.

Quick Start
^^^^^^^^^^^

- **Install**: rebuild the environment with this release's ``requirements/install.sh``. Do not reuse a v0.3 virtual environment. Docs: :doc:`Installation <../start/installation>`
- **Baidu AI Cloud**: `https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1 <https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1>`__
- **Infinigence AI**: `https://docs.neogpu.com/posts/rlinf-ppo-vla.html <https://docs.neogpu.com/posts/rlinf-ppo-vla.html>`__

RLinf v0.4 Test Results
^^^^^^^^^^^^^^^^^^^^^^^

The table below lists every v0.4 test result. Each configuration includes its result curve.

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - Configuration
     - Description
     - Result Curve
   * - ``examples/sft/config/behavior_sft_openpi_pi05_rlinf.yaml``
     - OpenPI_RLinf π₀.₅ · BEHAVIOR SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/behavior_sft_openpi_pi05_rlinf.png
          :alt: OpenPI_RLinf π₀.₅ · BEHAVIOR SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/calvin_abcd_d_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · CALVIN ABC-D PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/calvin_abcd_d_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · CALVIN ABC-D PPO result curve
          :width: 95%
          :align: center
   * - ``examples/offline_rl/config/cfg_rl_openpi.yaml``
     - CFG-RL · OpenPI
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/cfg_rl_openpi.png
          :alt: CFG-RL · OpenPI result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/frankasim_ppo_mlp.yaml``
     - MLP · FrankaSim PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/frankasim_ppo_mlp.png
          :alt: MLP · FrankaSim PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/frankasim_sac_cnn_async.yaml``
     - CNN · FrankaSim SAC Async
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/frankasim_sac_cnn_async.png
          :alt: CNN · FrankaSim SAC Async result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · GSEnv PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/gsenv_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · GSEnv PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - GR00T · IsaacLab Stack Cube PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/isaaclab_franka_stack_cube_ppo_gr00t.png
          :alt: GR00T · IsaacLab Stack Cube PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``
     - OpenVLA-OFT · LIBERO-10 GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_10_grpo_openvlaoft.png
          :alt: OpenVLA-OFT · LIBERO-10 GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_10_opd_openvlaoft.yaml``
     - OPD · OpenVLA-OFT · LIBERO-10
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_10_opd_openvlaoft.png
          :alt: OPD · OpenVLA-OFT · LIBERO-10 result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_10_ppo_gr00t.yaml``
     - GR00T · LIBERO-10 PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_10_ppo_gr00t.png
          :alt: GR00T · LIBERO-10 PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_goal_ppo_openpi.yaml``
     - OpenPI π₀ · LIBERO-Goal PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_goal_ppo_openpi.png
          :alt: OpenPI π₀ · LIBERO-Goal PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_goal_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · LIBERO-Goal PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_goal_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · LIBERO-Goal PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_object_async_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · LIBERO-Object Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_object_async_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · LIBERO-Object Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/libero_sft_cosmos3.yaml``
     - Cosmos3 · LIBERO SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_sft_cosmos3.png
          :alt: Cosmos3 · LIBERO SFT result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/libero_sft_dreamzero_5b.yaml``
     - DreamZero-5B · LIBERO SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_sft_dreamzero_5b.png
          :alt: DreamZero-5B · LIBERO SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_async_ppo_openpi.yaml``
     - OpenPI π₀ · LIBERO-Spatial Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_async_ppo_openpi.png
          :alt: OpenPI π₀ · LIBERO-Spatial Async PPO result curve
          :width: 95%
          :align: center
   * - ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``
     - Cosmos3 · SGLang Evaluation
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_cosmos3_eval_sglang.png
          :alt: Cosmos3 · SGLang Evaluation result curve
          :width: 95%
          :align: center
   * - ``evaluations/libero/libero_spatial_eval_pi05_RTC.yaml``
     - RTC · π₀.₅ · LIBERO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_eval_pi05_RTC.png
          :alt: RTC · π₀.₅ · LIBERO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_grpo_evo1.yaml``
     - Evo-1 · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_grpo_evo1.png
          :alt: Evo-1 · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_grpo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_grpo_starvla.yaml``
     - StarVLA · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_grpo_starvla.png
          :alt: StarVLA · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_opd_openvlaoft.yaml``
     - OPD · OpenVLA-OFT · LIBERO-Spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_opd_openvlaoft.png
          :alt: OPD · OpenVLA-OFT · LIBERO-Spatial result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_ppo_gr00t_n1d6.yaml``
     - GR00T-N1.6 · LIBERO-Spatial PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_ppo_gr00t_n1d6.png
          :alt: GR00T-N1.6 · LIBERO-Spatial PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/libero_spatial_ppo_openpi_rlinf.yaml``
     - OpenPI_RLinf · LIBERO-Spatial PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/libero_spatial_ppo_openpi_rlinf.png
          :alt: OpenPI_RLinf · LIBERO-Spatial PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openpi.yaml``
     - OpenPI π₀ · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openpi.png
          :alt: OpenPI π₀ · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml``
     - OpenVLA · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openvla.png
          :alt: OpenVLA · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_async_ppo_openvlaoft.yaml``
     - OpenVLA-OFT · ManiSkill Async PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_async_ppo_openvlaoft.png
          :alt: OpenVLA-OFT · ManiSkill Async PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_grpo_openvla.yaml``
     - OpenVLA · ManiSkill GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_grpo_openvla.png
          :alt: OpenVLA · ManiSkill GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_grpo_openvlaoft.yaml``
     - OpenVLA-OFT · ManiSkill GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_grpo_openvlaoft.png
          :alt: OpenVLA-OFT · ManiSkill GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_mlp.yaml``
     - MLP · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_mlp.png
          :alt: MLP · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_mlp_vlm_trend_reward_sglang.yaml``
     - SGLang VLM Trend reward
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_mlp_vlm_trend_reward_sglang.png
          :alt: SGLang VLM Trend reward result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openpi.yaml``
     - OpenPI π₀ · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openpi.png
          :alt: OpenPI π₀ · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml``
     - OpenPI π₀.₅ · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openpi_pi05.png
          :alt: OpenPI π₀.₅ · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openvla.yaml``
     - OpenVLA · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openvla.png
          :alt: OpenVLA · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_ppo_openvlaoft.yaml``
     - OpenVLA-OFT · ManiSkill PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_ppo_openvlaoft.png
          :alt: OpenVLA-OFT · ManiSkill PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``
     - RLT Stage 1 · ManiSkill
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_rlt_stage1_sft_openpi_pi05.png
          :alt: RLT Stage 1 · ManiSkill result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``
     - RLT Stage 2 AC · ManiSkill
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_rlt_stage2_ac_mlp.png
          :alt: RLT Stage 2 AC · ManiSkill result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml``
     - RLT Stage 2 TD3 · ManiSkill
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_rlt_stage2_td3_mlp.png
          :alt: RLT Stage 2 TD3 · ManiSkill result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_sac_mlp.yaml``
     - MLP · ManiSkill SAC
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_sac_mlp.png
          :alt: MLP · ManiSkill SAC result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_sac_mlp_async_decoupled.yaml``
     - MLP · ManiSkill SAC Async Decoupled
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_sac_mlp_async_decoupled.png
          :alt: MLP · ManiSkill SAC Async Decoupled result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
     - ResNet reward · ManiSkill SAC Async
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/maniskill_sac_mlp_resnet_reward_async.png
          :alt: ResNet reward · ManiSkill SAC Async result curve
          :width: 95%
          :align: center
   * - ``examples/reasoning/config/math/moonlight-16b-grpo-megatron.yaml``
     - Moonlight-16B · GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/moonlight-16b-grpo-megatron.png
          :alt: Moonlight-16B · GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/opensora_libero_spatial_grpo_openvlaoft.yaml``
     - OpenSora · OpenVLA-OFT · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/opensora_libero_spatial_grpo_openvlaoft.png
          :alt: OpenSora · OpenVLA-OFT · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn.yaml``
     - AgentLightning · Qwen2.5-1.5B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2.5-1.5b-enginehttp-multiturn.png
          :alt: AgentLightning · Qwen2.5-1.5B result curve
          :width: 95%
          :align: center
   * - ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``
     - Qwen2.5-1.5B · PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2.5-1.5b-ppo-megatron.png
          :alt: Qwen2.5-1.5B · PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/qwen2_5_vl_megatron_sft_vlm.yaml``
     - Qwen2.5-VL · Megatron SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2_5_vl_megatron_sft_vlm.png
          :alt: Qwen2.5-VL · Megatron SFT result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/qwen2_5_vl_sft_vlm.yaml``
     - Qwen2.5-VL · SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen2_5_vl_sft_vlm.png
          :alt: Qwen2.5-VL · SFT result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/qwen3_vl_sft_vlm.yaml``
     - Qwen3-VL · SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/qwen3_vl_sft_vlm.png
          :alt: Qwen3-VL · SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/realsimulator_robotwin_adjust_bottle_dagger_openpi.yaml``
     - OpenPI · RoboTwin DAgger
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/realsimulator_robotwin_adjust_bottle_dagger_openpi.png
          :alt: OpenPI · RoboTwin DAgger result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``
     - RLT Stage 1 · Franka
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/realworld_rlt_stage1_sft_openpi_pi05.png
          :alt: RLT Stage 1 · Franka result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml``
     - OpenPI PyTorch · Dual-Arm Franka SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/realworld_sft_openpi_pytorch_dual_franka_tcp_rot6d.png
          :alt: OpenPI PyTorch · Dual-Arm Franka SFT result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml``
     - OpenPI · RoboCasa365 PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robocasa365_opendrawer_ppo_openpi.png
          :alt: OpenPI · RoboCasa365 PPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/robotwin_beat_block_hammer_grpo_openvlaoft.yaml``
     - OpenVLA-OFT · RoboTwin GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robotwin_beat_block_hammer_grpo_openvlaoft.png
          :alt: OpenVLA-OFT · RoboTwin GRPO result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/robotwin_place_empty_cup_ppo_openvlaoft.yaml``
     - OpenVLA-OFT · RoboTwin PPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robotwin_place_empty_cup_ppo_openvlaoft.png
          :alt: OpenVLA-OFT · RoboTwin PPO result curve
          :width: 95%
          :align: center
   * - ``examples/sft/config/robotwin_sft_openpi_rlinf.yaml``
     - OpenPI_RLinf · RoboTwin SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/robotwin_sft_openpi_rlinf.png
          :alt: OpenPI_RLinf · RoboTwin SFT result curve
          :width: 95%
          :align: center
   * - ``examples/diffusion/config/sd3_nft_ocr.yaml``
     - Diffusion-NFT · SD3 · OCR
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/sd3_nft_ocr.png
          :alt: Diffusion-NFT · SD3 · OCR result curve
          :width: 95%
          :align: center
   * - ``examples/offline_rl/config/steam_compute_advantages_ensemble.yaml``
     - STEAM advantage ensemble
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/steam_compute_advantages_ensemble.png
          :alt: STEAM advantage ensemble result curve
          :width: 95%
          :align: center
   * - ``examples/offline_rl/config/steam_value_model_sft.yaml``
     - STEAM value model SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/steam_value_model_sft.png
          :alt: STEAM value model SFT result curve
          :width: 95%
          :align: center
   * - ``examples/agent/searchr1/config/train_qwen2.5.yaml``
     - SearchR1 · Qwen2.5
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/train_qwen2.5.png
          :alt: SearchR1 · Qwen2.5 result curve
          :width: 95%
          :align: center
   * - ``examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml``
     - SearchR1 · Qwen2.5 FSDP
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/train_qwen2.5_fsdp.png
          :alt: SearchR1 · Qwen2.5 FSDP result curve
          :width: 95%
          :align: center
   * - ``examples/diffusion/config/wan22_ti2v_5b_nft_ocr.yaml``
     - Diffusion-NFT · Wan2.2 · OCR
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/wan22_ti2v_5b_nft_ocr.png
          :alt: Diffusion-NFT · Wan2.2 · OCR result curve
          :width: 95%
          :align: center
   * - ``examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml``
     - Wan · OpenVLA-OFT · LIBERO-Spatial GRPO
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.4/wan_libero_spatial_grpo_openvlaoft.png
          :alt: Wan · OpenVLA-OFT · LIBERO-Spatial GRPO result curve
          :width: 95%
          :align: center
