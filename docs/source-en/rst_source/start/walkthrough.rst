RLinf Walkthrough
=================

Build a working mental model of RLinf in five minutes: what the framework connects, what happens after you launch a training script, how runs scale from one GPU to clusters and real robots, and where key files live in the repository.

The Big Picture
---------------

RLinf pairs a unified distributed runtime with pluggable models, environments, robots, and RL algorithms. Every task—simulation VLA training, real-world robot learning, reasoning LLMs, or diffusion post-training—runs through the same launcher and Hydra YAML workflow.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 348" width="100%" role="img" aria-label="Layered overview of RLinf: workloads, unified YAML interface, distributed runtime, and hardware backends" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rlinf-down-en" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="6" markerHeight="6" orient="auto">
           <path d="M 1.5 0 L 5 8.5 L 8.5 0 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Layer 1: Workloads -->
       <text x="16" y="20" font-size="11.5" font-weight="700" letter-spacing="0.04em" fill="var(--pst-color-text-muted, #64748b)">1. WORKLOADS YOU RUN</text>
       <rect x="16" y="30" width="212" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="30" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Embodied Simulation</text>
       <text x="30" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">LIBERO · ManiSkill · RoboTwin</text>
       <text x="30" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">OpenPI · OpenVLA · GR00T</text>

       <rect x="242" y="30" width="212" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="256" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Real-World Robots</text>
       <text x="256" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Franka · SO-101 · ARX X5 · R2</text>
       <text x="256" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Teleop · Data · SFT · Online RL</text>

       <rect x="468" y="30" width="212" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="482" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Agentic &amp; Reasoning</text>
       <text x="482" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Qwen · Search-R1 · WideSeek</text>
       <text x="482" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Math · Tool Use · Coding RL</text>

       <rect x="694" y="30" width="210" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="708" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Image &amp; Video Diffusion</text>
       <text x="708" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">SD3.5 Medium · Wan2.2</text>
       <text x="708" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Diffusion-NFT · ART · ReFL</text>

       <!-- Connector bar: Unified Interface -->
       <line x1="460" y1="108" x2="460" y2="128" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rlinf-down-en)"/>
       <rect x="16" y="132" width="888" height="36" rx="9" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4"/>
       <text x="460" y="155" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">One Interface:  run_*.sh  +  Hydra YAML Config  (override any key from CLI)</text>
       <line x1="460" y1="168" x2="460" y2="188" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rlinf-down-en)"/>

       <!-- Layer 2: Unified Runtime -->
       <text x="16" y="198" font-size="11.5" font-weight="700" letter-spacing="0.04em" fill="var(--pst-color-text-muted, #64748b)">2. UNIFIED RL RUNTIME</text>
       <rect x="16" y="208" width="286" height="74" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="32" y="232" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Interaction &amp; Rollout</text>
       <text x="32" y="252" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Env / Robot / Agent workers</text>
       <text x="32" y="270" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Hugging Face · SGLang · vLLM</text>

       <rect x="317" y="208" width="286" height="74" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="333" y="232" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Algorithms &amp; Data</text>
       <text x="333" y="252" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">PPO · GRPO · SAC · DAPO · SFT</text>
       <text x="333" y="270" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Rewards · Replay Buffer · LeRobot</text>

       <rect x="618" y="208" width="286" height="74" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="634" y="232" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Distributed Training &amp; Scheduler</text>
       <text x="634" y="252" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">FSDP / FSDP2 · Megatron-Core</text>
       <text x="634" y="270" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Ray Collocated · Disaggregated · Hybrid</text>

       <!-- Layer 3: Hardware -->
       <rect x="16" y="300" width="888" height="38" rx="9" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="460" y="324" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">3. Hardware Backends:   NVIDIA CUDA · AMD ROCm · Huawei Ascend · Moore Threads MUSA · Intel / Kunlunxin / MetaX / Biren</text>
     </svg>
   </div>

How a Training Run Works
------------------------

When you run a launch script such as ``bash examples/embodiment/run_embodiment.sh <config>``, RLinf reads your YAML file, starts Ray worker groups on the requested hardware, and repeats four stages until training finishes:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/core_dataflow.svg
   :align: center
   :width: 100%
   :alt: RLinf core training loop connecting Env, Rollout, Reward and Buffer, Actor, and Weight Sync

   Data flow inside each training iteration and weight synchronization back to rollout.

Each box in the diagram maps to one top-level section in your YAML config:

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Stage
     - YAML Section
     - What It Does
   * - Environment / Agent
     - ``env.*`` or ``agentloop.*``
     - Steps simulators, real-world robots, or tool sandboxes to collect observations and rewards.
   * - Rollout
     - ``rollout.*``
     - Runs policy inference (Hugging Face, SGLang, or vLLM) to produce actions or responses.
   * - Reward & Algorithm
     - ``algorithm.*``, ``reward.*``
     - Scores trajectories, estimates PPO / GRPO advantages, or samples off-policy replay batches.
   * - Actor Training
     - ``actor.*``
     - Updates policy weights with FSDP or Megatron, syncs weights back to rollout, and writes checkpoints.

.. tip::

   Set ``actor.model.model_path`` once when switching checkpoints. Rollout inherits that path automatically so you keep one source of truth.

From One GPU to Clusters and Real Robots
----------------------------------------

You keep the same training script and algorithm code as your hardware grows. Changing ``cluster.component_placement`` in YAML switches how workers occupy GPUs and hosts:

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/deployment_modes.svg
   :align: center
   :width: 100%
   :alt: Comparison of Collocated, Disaggregated, and Hybrid placement modes across GPUs and hosts

   Three placement modes for the same worker groups: collocated, disaggregated, and hybrid.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Mode
     - Hardware Scale
     - When to Use It
   * - Collocated
     - 1–8 GPUs
     - Default for single-GPU and single-node runs; components share GPUs with automatic offload.
   * - Disaggregated
     - 8+ GPUs
     - Gives each component dedicated GPUs for pipelining and asynchronous RL without offload stalls.
   * - Hybrid / Real-World
     - Multi-node or Robot + GPU
     - Combines shared GPU workers with separate CPU nodes or a physical robot control computer.

Repository Map: From Command to Output
--------------------------------------

The diagram below traces how a launch command moves through the repository—from configs and launchers into the core runtime, training outputs, and helper utilities.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 310" width="100%" role="img" aria-label="RLinf repository workflow from launch scripts and configs to runtime, logs, and toolkits" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rlinf-arrow-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Column 1: Launch & Configs -->
       <rect x="16" y="20" width="256" height="270" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8"/>
       <rect x="16" y="20" width="256" height="38" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="144" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1. Launch &amp; Configs</text>

       <rect x="32" y="74" width="224" height="88" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="46" y="96" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">examples/</text>
       <text x="46" y="116" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">run_*.sh launchers &amp; Hydra YAMLs</text>
       <text x="46" y="134" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">embodiment · sft · reasoning · agent</text>
       <text x="46" y="152" font-size="11" fill="var(--pst-color-primary, #6d28d9)">Override any key on the CLI</text>

       <rect x="32" y="178" width="224" height="94" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="46" y="200" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">evaluations/</text>
       <text x="46" y="220" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Unified run_eval.sh entrypoint</text>
       <text x="46" y="238" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">LIBERO · RoboTwin · BEHAVIOR · Real</text>
       <text x="46" y="256" font-size="11" fill="var(--pst-color-primary, #6d28d9)">Standalone checkpoint benchmarks</text>

       <!-- Arrow 1 -> 2 -->
       <line x1="272" y1="155" x2="314" y2="155" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-en)"/>

       <!-- Column 2: Core Framework -->
       <rect x="322" y="20" width="276" height="270" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.8"/>
       <rect x="322" y="20" width="276" height="38" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="460" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">2. Core Framework (rlinf/)</text>

       <rect x="338" y="74" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="92" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">scheduler/ · workers/ · runners/</text>
       <text x="352" y="108" font-size="11" fill="var(--pst-color-text-muted, #475569)">Ray cluster, placement &amp; training loops</text>

       <rect x="338" y="126" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="144" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">models/ · hybrid_engines/</text>
       <text x="352" y="160" font-size="11" fill="var(--pst-color-text-muted, #475569)">VLA/LLM policies, FSDP, Megatron, SGLang</text>

       <rect x="338" y="178" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="196" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">envs/ · robotics/</text>
       <text x="352" y="212" font-size="11" fill="var(--pst-color-text-muted, #475569)">Simulators, real robots &amp; teleoperation</text>

       <rect x="338" y="230" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="248" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">algorithms/ · data/</text>
       <text x="352" y="264" font-size="11" fill="var(--pst-color-text-muted, #475569)">PPO, GRPO, SAC, rewards &amp; datasets</text>

       <!-- Arrows 2 -> 3 -->
       <line x1="598" y1="96" x2="640" y2="96" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-en)"/>
       <line x1="598" y1="218" x2="640" y2="218" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-en)"/>

       <!-- Column 3 Top: Run Outputs -->
       <rect x="648" y="20" width="256" height="126" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8"/>
       <rect x="648" y="20" width="256" height="36" rx="12" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="776" y="43" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">3. Run Outputs (logs/)</text>
       <text x="664" y="78" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• TensorBoard / W&amp;B / SwanLab</text>
       <text x="664" y="102" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Checkpoints &amp; rollout videos</text>
       <text x="664" y="126" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• LeRobot &amp; replay-buffer shards</text>

       <!-- Column 3 Bottom: Setup & Toolkits -->
       <rect x="648" y="162" width="256" height="128" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.8"/>
       <rect x="648" y="162" width="256" height="36" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="776" y="185" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">4. Setup &amp; Toolkits</text>
       <text x="664" y="220" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• requirements/ &amp; docker/</text>
       <text x="664" y="244" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• toolkits/ (data, check, auto-place)</text>
       <text x="664" y="268" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• tests/ (unit, e2e, robot_mocks)</text>
     </svg>
   </div>

Click any area below to jump directly to its guide or reference:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Training Scripts & Configs (``examples/``)
      :link: ../reference/configuration
      :link-type: doc

      Launch scripts and Hydra YAMLs grouped by task: embodied RL, SFT, offline RL, reasoning, agents, and diffusion.

   .. grid-item-card:: Standalone Evaluation (``evaluations/``)
      :link: ../evaluations/index
      :link-type: doc

      Unified ``run_eval.sh`` launcher and ready-to-run benchmark configs (LIBERO, RoboTwin, BEHAVIOR, ManiSkill, real robots).

   .. grid-item-card:: Helper Utilities (``toolkits/``)

      Standalone tools for LeRobot datasets, real-robot hardware checks, replay buffer inspection, auto-placement, and checkpoint conversion.

   .. grid-item-card:: Core Framework (rlinf/)
      :link: ../concepts/overview
      :link-type: doc

      Distributed runtime, workers, runners, models, simulator and real-robot environments, robotics abstractions, and algorithm registries.

   .. grid-item-card:: Installers & Images (``requirements/`` & ``docker/``)
      :link: installation
      :link-type: doc

      One-line ``install.sh`` environment builder and Dockerfiles for NVIDIA, AMD, Ascend, and MUSA.

   .. grid-item-card:: Tests & Extensions (``tests/`` & ``docs/``)
      :link: ../extending/index
      :link-type: doc

      Unit tests, end-to-end configs, fake vendor SDKs for real-robot CI, and English/Chinese docs.

.. dropdown:: Inside ``toolkits/`` — What Tools Are Included?

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Directory
        - Purpose
      * - ``toolkits/lerobot/``
        - Compute ``norm_stats.json``, merge LeRobot datasets, and export episode MP4 previews.
      * - ``toolkits/realworld_check/``
        - Interactive hardware checks for arms, grippers, cameras, GELLO, and PICO before real-robot training.
      * - ``toolkits/replay_buffer/``
        - Inspect, visualize, merge, or split saved trajectory replay buffers.
      * - ``toolkits/auto_placement/``
        - Search optimal collocated vs disaggregated GPU layouts from measured iteration timings.
      * - ``toolkits/cosmos3/``, ``dual_franka/``, ``standalone_eval_scripts/``
        - Model- and robot-specific checkpoint converters, dataset cleaners, and standalone eval wrappers.

Where to Go Next
----------------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: Training Configuration
      :link: ../reference/configuration
      :link-type: doc

      Read an annotated minimal YAML and look up every config directory.

   .. grid-item-card:: Embodied Training
      :link: ../examples/index
      :link-type: doc

      Pick a simulator, VLA model, SFT recipe, or world model.

   .. grid-item-card:: Real-World Robots
      :link: ../examples/real_world_index
      :link-type: doc

      Follow the hardware, teleoperation, data collection, and online RL workflow.

   .. grid-item-card:: Agentic & Generative
      :link: ../examples/agentic/index
      :link-type: doc

      Train math reasoning LLMs, multi-turn agents, and diffusion models.
