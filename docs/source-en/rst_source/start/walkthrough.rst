RLinf Walkthrough
=================

Build a working mental model of RLinf in five minutes: what the framework connects, what happens after you launch a training script, how runs scale from one GPU to clusters and real robots, and where key files live in the repository.

The Big Picture
---------------

RLinf pairs a unified distributed runtime with pluggable models, environments, robots, and RL algorithms. Every task—simulation VLA training, real-world robot learning, reasoning LLMs, or diffusion post-training—runs through the same launcher and Hydra YAML workflow.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/overview.svg
   :align: center
   :width: 100%
   :alt: Overview of RLinf covering embodied simulation, real-world robots, agentic reasoning, diffusion models, and distributed backends

   High-level view of RLinf's workloads, training and rollout engines, and supported accelerators.

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
     <svg viewBox="0 0 920 330" width="100%" role="img" aria-label="RLinf repository workflow from launch scripts and configs to runtime, logs, and toolkits" style="display:block;max-width:920px;margin:0 auto;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rlinf-arrow-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>
       <rect x="2" y="2" width="916" height="326" rx="14" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.5"/>

       <!-- Column 1: Launch & Configs -->
       <rect x="24" y="34" width="248" height="262" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8"/>
       <rect x="24" y="34" width="248" height="38" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="148" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1. Launch &amp; Configs</text>

       <rect x="40" y="88" width="216" height="86" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="54" y="110" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">examples/</text>
       <text x="54" y="130" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">run_*.sh launchers &amp; Hydra YAMLs</text>
       <text x="54" y="148" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">embodiment · sft · reasoning · agent</text>
       <text x="54" y="164" font-size="11" fill="var(--pst-color-primary, #6d28d9)">Override any key on the CLI</text>

       <rect x="40" y="190" width="216" height="88" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="54" y="212" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">evaluations/</text>
       <text x="54" y="232" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Unified run_eval.sh entrypoint</text>
       <text x="54" y="250" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">LIBERO · RoboTwin · BEHAVIOR · Real</text>
       <text x="54" y="266" font-size="11" fill="var(--pst-color-primary, #6d28d9)">Standalone checkpoint benchmarks</text>

       <!-- Arrow 1 -> 2 -->
       <line x1="272" y1="165" x2="314" y2="165" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-en)"/>

       <!-- Column 2: Core Framework -->
       <rect x="322" y="34" width="276" height="262" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.8"/>
       <rect x="322" y="34" width="276" height="38" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="460" y="58" text-anchor="middle" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">2. Core Framework (rlinf/)</text>

       <rect x="338" y="88" width="244" height="40" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="106" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">scheduler/ · workers/ · runners/</text>
       <text x="352" y="121" font-size="11" fill="var(--pst-color-text-muted, #475569)">Ray cluster, placement &amp; training loops</text>

       <rect x="338" y="138" width="244" height="40" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="156" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">models/ · hybrid_engines/</text>
       <text x="352" y="171" font-size="11" fill="var(--pst-color-text-muted, #475569)">VLA/LLM policies, FSDP, Megatron, SGLang</text>

       <rect x="338" y="188" width="244" height="40" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="206" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">envs/ · robotics/</text>
       <text x="352" y="221" font-size="11" fill="var(--pst-color-text-muted, #475569)">Simulators, real robots &amp; teleoperation</text>

       <rect x="338" y="238" width="244" height="40" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="256" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">algorithms/ · data/</text>
       <text x="352" y="271" font-size="11" fill="var(--pst-color-text-muted, #475569)">PPO, GRPO, SAC, rewards &amp; datasets</text>

       <!-- Arrows 2 -> 3 -->
       <line x1="598" y1="108" x2="640" y2="108" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-en)"/>
       <line x1="598" y1="226" x2="640" y2="226" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-en)"/>

       <!-- Column 3 Top: Run Outputs -->
       <rect x="648" y="34" width="248" height="122" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8"/>
       <rect x="648" y="34" width="248" height="36" rx="12" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="772" y="57" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">3. Run Outputs (logs/)</text>
       <text x="664" y="92" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• TensorBoard / W&amp;B / SwanLab</text>
       <text x="664" y="114" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Checkpoints &amp; rollout videos</text>
       <text x="664" y="136" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• LeRobot &amp; replay-buffer shards</text>

       <!-- Column 3 Bottom: Setup & Toolkits -->
       <rect x="648" y="172" width="248" height="124" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.8"/>
       <rect x="648" y="172" width="248" height="36" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="772" y="195" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">4. Setup &amp; Toolkits</text>
       <text x="664" y="228" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• requirements/ &amp; docker/</text>
       <text x="664" y="250" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• toolkits/ (data, check, auto-place)</text>
       <text x="664" y="272" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• tests/ (unit, e2e, robot_mocks)</text>
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
