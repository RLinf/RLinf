Tour of RLinf
=============

Start from the launch command you just ran and work inward in three steps: where files live in the repository, how a training run executes, and the core abstractions that let RLinf scale across GPUs, accelerators, robots, and models.

1. From a Launch Command to the Repository
------------------------------------------

Every RLinf run follows the same pattern: pick a launcher and a YAML config name, optionally override keys on the CLI, and inspect results under ``logs/``.

.. raw:: html

   <div style="margin: 1rem 0 1.25rem 0;">
     <svg viewBox="0 0 920 198" width="100%" role="img" aria-label="Anatomy of an RLinf launch command mapped to examples/, rlinf/, logs/, and toolkits/" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="w3-pur-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Command bar -->
       <rect x="16" y="10" width="888" height="44" rx="10" fill="var(--rlinf-code-bg, #f8fafc)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.5"/>
       <text x="36" y="37" font-family="var(--pst-font-family-monospace,monospace)" font-size="13" fill="var(--pst-color-text-muted, #64748b)">bash</text>
       <rect x="78" y="18" width="322" height="28" rx="6" fill="rgba(124,58,237,0.12)"/>
       <text x="239" y="37" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="12.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">examples/embodiment/run_embodiment.sh</text>
       <rect x="410" y="18" width="166" height="28" rx="6" fill="rgba(59,130,246,0.12)"/>
       <text x="493" y="37" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="12.5" font-weight="700" fill="#3b82f6">maniskill_ppo_mlp</text>
       <rect x="586" y="18" width="300" height="28" rx="6" fill="rgba(16,185,129,0.12)"/>
       <text x="736" y="37" text-anchor="middle" font-family="var(--pst-font-family-monospace,monospace)" font-size="12.5" font-weight="700" fill="#10b981">actor.model.model_path=...</text>

       <!-- Command part captions -->
       <text x="239" y="72" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">▲ Launcher script</text>
       <text x="493" y="72" text-anchor="middle" font-size="10.5" font-weight="600" fill="#3b82f6">▲ YAML config name</text>
       <text x="736" y="72" text-anchor="middle" font-size="10.5" font-weight="600" fill="#10b981">▲ Hydra CLI override</text>

       <!-- 4 Folder cards -->
       <rect x="16" y="96" width="206" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="32" y="124" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">examples/ · evaluations/</text>
       <text x="32" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">Launchers &amp; Hydra YAMLs</text>
       <text x="32" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">Train &amp; eval configs</text>

       <line x1="222" y1="139" x2="244" y2="139" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#w3-pur-en)"/>

       <rect x="248" y="96" width="206" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="1.6"/>
       <text x="264" y="124" font-size="13" font-weight="700" fill="#3b82f6">rlinf/</text>
       <text x="264" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">Runners · Workers · Models</text>
       <text x="264" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">Envs · Robotics · Algorithms</text>

       <line x1="454" y1="139" x2="476" y2="139" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#w3-pur-en)"/>

       <rect x="480" y="96" width="200" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="1.6"/>
       <text x="496" y="124" font-size="13" font-weight="700" fill="#10b981">logs/</text>
       <text x="496" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">TensorBoard · Checkpoints</text>
       <text x="496" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">Videos · Dataset shards</text>

       <rect x="698" y="96" width="206" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="714" y="124" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">toolkits/ · requirements/</text>
       <text x="714" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">install.sh · Dockerfiles</text>
       <text x="714" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">Data · Checks · Auto-place</text>
     </svg>
   </div>

.. dropdown:: Inside ``toolkits/`` — What Helper Scripts Are Included?

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

Full config directory paths and an annotated minimal YAML live in :doc:`../reference/configuration`.

2. Inside a Run: Runner, Workers, and Channels
----------------------------------------------

Once launched, the entry script starts the Ray cluster and each Worker group, then hands the training loop to a **Runner**:

- **Runner (driver loop)**: runs on the driver process and defines each iteration—syncing weights, triggering rollout, computing rewards and advantages, updating the actor, and running periodic eval/checkpointing. **When adding a new training workflow to RLinf, the Runner is the main logic you edit or write.**

- **Workers (execution roles)**: Env, Rollout, Reward, and Actor each run as an independent process group mapped to one YAML section.

- **Channels (data queues)**: observations, actions, and trajectories stream directly between workers over async queues without routing through the driver; weights broadcast via hot weight sync.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 276" width="100%" role="img" aria-label="RLinf Control Plane (Runner driver loop: Sync Weights, Rollout, Reward and Advantage, Update Actor, Eval and Save) and Data Plane (Workers connected by async Channel queues and hot weight sync)" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="w1-pur-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
         <marker id="w1-grn-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#10b981"/>
         </marker>
       </defs>

       <!-- Legend top right -->
       <line x1="654" y1="20" x2="682" y2="20" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3"/>
       <text x="690" y="24" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Control</text>
       <line x1="756" y1="20" x2="784" y2="20" stroke="#10b981" stroke-width="2"/>
       <text x="792" y="24" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">Data (Channel)</text>

       <!-- Control Plane: Runner loop bar -->
       <rect x="24" y="8" width="606" height="48" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.10))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5"/>
       <text x="42" y="27" font-size="12" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Runner (Driver Loop · Custom Workflow Entry Point)</text>
       <text x="42" y="45" font-size="10.8" font-weight="500" fill="var(--pst-color-text-base, #0f172a)">1. Sync Weights  →  2. Rollout  →  3. Reward &amp; Advantage  →  4. Update Actor  →  5. Eval &amp; Save</text>

       <!-- Dashed control bus -->
       <path d="M 327 56 L 327 76 M 102 76 L 818 76" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3"/>
       <line x1="102" y1="76" x2="102" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-en)"/>
       <line x1="340" y1="76" x2="340" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-en)"/>
       <line x1="580" y1="76" x2="580" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-en)"/>
       <line x1="818" y1="76" x2="818" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-en)"/>

       <!-- Worker 1: Env (Blue) -->
       <rect x="24" y="106" width="156" height="86" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="2"/>
       <circle cx="54" cy="138" r="14" fill="rgba(59,130,246,0.14)"/>
       <rect x="47" y="131" width="14" height="14" rx="3" fill="none" stroke="#3b82f6" stroke-width="1.8"/>
       <circle cx="51.5" cy="136.5" r="1.4" fill="#3b82f6"/>
       <circle cx="56.5" cy="136.5" r="1.4" fill="#3b82f6"/>
       <text x="78" y="136" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Env</text>
       <text x="78" y="153" font-size="11" fill="var(--pst-color-text-muted, #64748b)">Sim · Robot</text>
       <rect x="44" y="166" width="116" height="18" rx="9" fill="rgba(59,130,246,0.12)"/>
       <text x="102" y="178.5" text-anchor="middle" font-size="10.5" font-weight="600" fill="#3b82f6">env.*</text>

       <!-- Channel pipe 1 -> 2 (FIFO slots) -->
       <line x1="180" y1="149" x2="200" y2="149" stroke="#10b981" stroke-width="2"/>
       <rect x="200" y="138" width="42" height="22" rx="6" fill="rgba(16,185,129,0.1)" stroke="#10b981" stroke-width="1.4"/>
       <rect x="205" y="143" width="8" height="12" rx="2" fill="#10b981"/>
       <rect x="217" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.65"/>
       <rect x="229" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.28"/>
       <line x1="242" y1="149" x2="258" y2="149" stroke="#10b981" stroke-width="2" marker-end="url(#w1-grn-en)"/>
       <text x="221" y="174" text-anchor="middle" font-size="10" font-weight="600" fill="#10b981">Channel</text>

       <!-- Worker 2: Rollout (Purple) -->
       <rect x="262" y="106" width="156" height="86" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <circle cx="292" cy="138" r="14" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.14))"/>
       <polygon points="294,128 286,140 292,140 290,148 298,136 292,136" fill="var(--pst-color-primary, #6d28d9)"/>
       <text x="316" y="136" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout</text>
       <text x="316" y="153" font-size="11" fill="var(--pst-color-text-muted, #64748b)">HF · SGLang</text>
       <rect x="282" y="166" width="116" height="18" rx="9" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.12))"/>
       <text x="340" y="178.5" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">rollout.*</text>

       <!-- Channel pipe 2 -> 3 -->
       <line x1="418" y1="149" x2="439" y2="149" stroke="#10b981" stroke-width="2"/>
       <rect x="439" y="138" width="42" height="22" rx="6" fill="rgba(16,185,129,0.1)" stroke="#10b981" stroke-width="1.4"/>
       <rect x="444" y="143" width="8" height="12" rx="2" fill="#10b981"/>
       <rect x="456" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.65"/>
       <rect x="468" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.28"/>
       <line x1="481" y1="149" x2="498" y2="149" stroke="#10b981" stroke-width="2" marker-end="url(#w1-grn-en)"/>
       <text x="460" y="174" text-anchor="middle" font-size="10" font-weight="600" fill="#10b981">Channel</text>

       <!-- Worker 3: Reward & Buffer (Amber) -->
       <rect x="502" y="106" width="156" height="86" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#d97706" stroke-width="2"/>
       <circle cx="532" cy="138" r="14" fill="rgba(217,119,6,0.14)"/>
       <circle cx="532" cy="138" r="7" fill="none" stroke="#d97706" stroke-width="1.8"/>
       <circle cx="532" cy="138" r="2.5" fill="#d97706"/>
       <text x="556" y="136" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Reward</text>
       <text x="556" y="153" font-size="11" fill="var(--pst-color-text-muted, #64748b)">Adv · Buffer</text>
       <rect x="522" y="166" width="116" height="18" rx="9" fill="rgba(217,119,6,0.12)"/>
       <text x="580" y="178.5" text-anchor="middle" font-size="10.5" font-weight="600" fill="#d97706">algorithm.*</text>

       <!-- Channel pipe 3 -> 4 -->
       <line x1="658" y1="149" x2="678" y2="149" stroke="#10b981" stroke-width="2"/>
       <rect x="678" y="138" width="42" height="22" rx="6" fill="rgba(16,185,129,0.1)" stroke="#10b981" stroke-width="1.4"/>
       <rect x="683" y="143" width="8" height="12" rx="2" fill="#10b981"/>
       <rect x="695" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.65"/>
       <rect x="707" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.28"/>
       <line x1="720" y1="149" x2="736" y2="149" stroke="#10b981" stroke-width="2" marker-end="url(#w1-grn-en)"/>
       <text x="699" y="174" text-anchor="middle" font-size="10" font-weight="600" fill="#10b981">Channel</text>

       <!-- Worker 4: Actor (Emerald) -->
       <rect x="740" y="106" width="156" height="86" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="2"/>
       <circle cx="770" cy="138" r="14" fill="rgba(16,185,129,0.14)"/>
       <polygon points="770,130 779,135 770,140 761,135" fill="none" stroke="#10b981" stroke-width="1.6"/>
       <polyline points="761,140 770,145 779,140" fill="none" stroke="#10b981" stroke-width="1.6"/>
       <text x="794" y="136" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Actor</text>
       <text x="794" y="153" font-size="11" fill="var(--pst-color-text-muted, #64748b)">FSDP · Megatron</text>
       <rect x="760" y="166" width="116" height="18" rx="9" fill="rgba(16,185,129,0.12)"/>
       <text x="818" y="178.5" text-anchor="middle" font-size="10.5" font-weight="600" fill="#10b981">actor.*</text>

       <!-- Bottom Return Loop: Weight Sync -->
       <path d="M 818 192 L 818 236 Q 818 250 804 250 L 354 250 Q 340 250 340 236 L 340 196" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2.2" marker-end="url(#w1-pur-en)"/>
       <rect x="498" y="236" width="164" height="28" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="580" y="254" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Hot Weight Sync</text>
     </svg>
   </div>

RLinf ships built-in runners for sync/async embodied RL, reasoning, agents, SFT, offline RL, and standalone evaluation; pipelined and async runners overlap the same workers over Channels without changing worker code. Set ``actor.model.model_path`` once and rollout inherits it automatically. Deep dive: :doc:`../concepts/execution_flow`, :doc:`../concepts/worker`, and :doc:`../concepts/channel`.

3. Core Abstractions: Hardware, Robotics & Registries
-----------------------------------------------------

The same color-coded workers map onto any hardware topology through ``cluster.component_placement``—from one shared GPU to eight accelerator vendors and physical robot hosts. On real robots, modular parts snap into one Robot object with a single read/write API shared across self-check, data collection, online RL, and real-world evaluation.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 308" width="100%" role="img" aria-label="Left: Hardware placement across GPUs, 8 accelerators, and robot hosts. Right: Modular robotics parts snapping into one Robot API." style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="w2-pur-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- LEFT PANEL: Hardware & Placement -->
       <text x="16" y="22" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Hardware &amp; Placement</text>
       <circle cx="242" cy="18" r="4.5" fill="#3b82f6"/>
       <text x="251" y="22" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Env</text>
       <circle cx="292" cy="18" r="4.5" fill="#7c3aed"/>
       <text x="301" y="22" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Rollout</text>
       <circle cx="362" cy="18" r="4.5" fill="#10b981"/>
       <text x="371" y="22" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Actor</text>

       <!-- Row 1: Collocated -->
       <text x="16" y="64" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Collocated</text>
       <text x="16" y="79" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">1–8 GPUs</text>
       <rect x="116" y="44" width="166" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="128" y="61" font-size="10" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">GPU 0</text>
       <rect x="128" y="66" width="44" height="14" rx="7" fill="#3b82f6"/>
       <rect x="177" y="66" width="44" height="14" rx="7" fill="#7c3aed"/>
       <rect x="226" y="66" width="44" height="14" rx="7" fill="#10b981"/>
       <text x="296" y="70" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Shared + offload</text>

       <!-- Row 2: Disaggregated -->
       <text x="16" y="122" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Disaggregated</text>
       <text x="16" y="137" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">8+ GPUs</text>
       <rect x="116" y="102" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="#7c3aed" stroke-width="1.6"/>
       <text x="128" y="119" font-size="10" font-weight="700" fill="#7c3aed">GPU 0–3</text>
       <rect x="128" y="124" width="44" height="14" rx="7" fill="#7c3aed"/>
       <rect x="178" y="124" width="44" height="14" rx="7" fill="#7c3aed" opacity="0.65"/>
       <line x1="234" y1="124" x2="262" y2="124" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#w2-pur-en)"/>
       <rect x="268" y="102" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="1.6"/>
       <text x="280" y="119" font-size="10" font-weight="700" fill="#10b981">GPU 4–7</text>
       <rect x="280" y="124" width="44" height="14" rx="7" fill="#10b981"/>
       <rect x="330" y="124" width="44" height="14" rx="7" fill="#10b981" opacity="0.65"/>

       <!-- Row 3: Hybrid / Real-World -->
       <text x="16" y="180" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Hybrid / Real</text>
       <text x="16" y="195" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Robot + GPU</text>
       <rect x="116" y="160" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="1.6"/>
       <text x="128" y="177" font-size="10" font-weight="700" fill="#3b82f6">Robot PC</text>
       <rect x="128" y="182" width="94" height="14" rx="7" fill="#3b82f6"/>
       <line x1="234" y1="182" x2="262" y2="182" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-start="url(#w2-pur-en)" marker-end="url(#w2-pur-en)"/>
       <rect x="268" y="160" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="280" y="177" font-size="10" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">GPU Server</text>
       <rect x="280" y="182" width="44" height="14" rx="7" fill="#7c3aed"/>
       <rect x="330" y="182" width="44" height="14" rx="7" fill="#10b981"/>

       <!-- Hardware Abstraction Rail -->
       <rect x="16" y="220" width="416" height="24" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.12))"/>
       <text x="224" y="236" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Hardware Layer  ·  8 Accelerator Backends</text>

       <!-- 8 Vendor Pills (2 rows of 4) -->
       <rect x="16" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="65" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">NVIDIA CUDA</text>
       <rect x="122" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="171" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">AMD ROCm</text>
       <rect x="228" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="277" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Ascend CANN</text>
       <rect x="334" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="383" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">MUSA</text>

       <rect x="16" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="65" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Intel XPU</text>
       <rect x="122" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="171" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Kunlunxin</text>
       <rect x="228" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="277" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">MetaX</text>
       <rect x="334" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="383" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Biren</text>

       <!-- Center Divider -->
       <line x1="460" y1="16" x2="460" y2="298" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.5" stroke-dasharray="4 4"/>

       <!-- RIGHT PANEL: Robotics Composable Bricks -->
       <text x="488" y="22" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Robotics Composable Interface</text>

       <!-- 4 Lego bricks on left -->
       <rect x="502" y="42" width="16" height="5" rx="2" fill="#3b82f6"/>
       <rect x="530" y="42" width="16" height="5" rx="2" fill="#3b82f6"/>
       <rect x="488" y="47" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="1.7"/>
       <text x="540" y="66" text-anchor="middle" font-size="11.5" font-weight="700" fill="#3b82f6">Arm</text>

       <rect x="502" y="86" width="16" height="5" rx="2" fill="#7c3aed"/>
       <rect x="530" y="86" width="16" height="5" rx="2" fill="#7c3aed"/>
       <rect x="488" y="91" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#7c3aed" stroke-width="1.7"/>
       <text x="540" y="110" text-anchor="middle" font-size="11.5" font-weight="700" fill="#7c3aed">End-Effector</text>

       <rect x="502" y="130" width="16" height="5" rx="2" fill="#d97706"/>
       <rect x="530" y="130" width="16" height="5" rx="2" fill="#d97706"/>
       <rect x="488" y="135" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#d97706" stroke-width="1.7"/>
       <text x="540" y="154" text-anchor="middle" font-size="11.5" font-weight="700" fill="#d97706">Cameras</text>

       <rect x="502" y="174" width="16" height="5" rx="2" fill="#10b981"/>
       <rect x="530" y="174" width="16" height="5" rx="2" fill="#10b981"/>
       <rect x="488" y="179" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="1.7"/>
       <text x="540" y="198" text-anchor="middle" font-size="11.5" font-weight="700" fill="#10b981">Teleop / Base</text>

       <!-- Converging lines into Robot Hub -->
       <path d="M 592 62 C 624 62, 624 126, 654 126" fill="none" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <path d="M 592 106 C 620 106, 622 126, 654 126" fill="none" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <path d="M 592 150 C 620 150, 622 126, 654 126" fill="none" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <path d="M 592 194 C 624 194, 624 126, 654 126" fill="none" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>

       <!-- Central Robot Hub -->
       <rect x="658" y="64" width="246" height="126" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="781" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Robot</text>
       <rect x="676" y="106" width="210" height="24" rx="12" fill="rgba(59,130,246,0.12)"/>
       <text x="781" y="122" text-anchor="middle" font-size="11" font-weight="700" fill="#3b82f6">◀  get_observation()</text>
       <rect x="676" y="138" width="210" height="24" rx="12" fill="rgba(16,185,129,0.12)"/>
       <text x="781" y="154" text-anchor="middle" font-size="11" font-weight="700" fill="#10b981">send_action()  ▶</text>
       <text x="781" y="180" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">Remote placement · Fault recovery</text>

       <!-- Bottom 4-Stage Pipeline -->
       <line x1="538" y1="268" x2="862" y2="268" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <circle cx="538" cy="268" r="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="538" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1</text>
       <text x="538" y="298" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Self-Check</text>

       <circle cx="646" cy="268" r="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="646" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">2</text>
       <text x="646" y="298" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Collect</text>

       <circle cx="754" cy="268" r="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="754" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">3</text>
       <text x="754" y="298" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Online RL</text>

       <circle cx="862" cy="268" r="14" fill="var(--pst-color-primary, #6d28d9)"/>
       <text x="862" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">4</text>
       <text x="862" y="298" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Eval &amp; Deploy</text>
     </svg>
   </div>

Models, engines, algorithms, and environments connect through registries—mix built-in blocks in YAML or register your own out-of-tree via ``RLINF_EXT_MODULE``.

.. raw:: html

   <div style="margin: 1rem 0 1.4rem 0;">
     <svg viewBox="0 0 920 156" width="100%" role="img" aria-label="Pluggable RLinf registry strips for Models, Engines, Algorithms, and Environments" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <!-- Row 1: Models -->
       <text x="16" y="24" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Models</text>
       <rect x="116" y="6" width="114" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="173" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">OpenPI π₀/π₀.₅</text>
       <rect x="238" y="6" width="118" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="297" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">OpenVLA / OFT</text>
       <rect x="364" y="6" width="76" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="402" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">GR00T</text>
       <rect x="448" y="6" width="112" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="504" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">StarVLA · MLP</text>
       <rect x="568" y="6" width="110" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="623" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">Qwen · MoE</text>
       <rect x="686" y="6" width="104" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="738" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">SD3.5 · Wan</text>
       <rect x="800" y="6" width="104" height="26" rx="13" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.3" stroke-dasharray="4 3"/>
       <text x="852" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">+ Custom</text>

       <!-- Row 2: Engines -->
       <text x="16" y="62" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Engines</text>
       <rect x="116" y="44" width="112" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="172" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">FSDP / FSDP2</text>
       <rect x="236" y="44" width="116" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="294" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">Megatron-Core</text>
       <rect x="360" y="44" width="110" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="415" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">Hugging Face</text>
       <rect x="478" y="44" width="84" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="520" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">SGLang</text>
       <rect x="570" y="44" width="70" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="605" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">vLLM</text>
       <rect x="648" y="44" width="76" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="686" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">PhyAI</text>

       <!-- Row 3: Algorithms -->
       <text x="16" y="100" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Algorithms</text>
       <rect x="116" y="82" width="60" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="146" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">PPO</text>
       <rect x="184" y="82" width="68" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="218" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">GRPO</text>
       <rect x="260" y="82" width="68" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="294" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">DAPO</text>
       <rect x="336" y="82" width="96" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="384" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">SAC / RLPD</text>
       <rect x="440" y="82" width="110" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="495" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">SFT · DAgger</text>
       <rect x="558" y="82" width="106" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="611" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">IQL · RECAP</text>
       <rect x="672" y="82" width="120" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="732" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">NFT · ART · ReFL</text>
       <rect x="800" y="82" width="104" height="26" rx="13" fill="none" stroke="#d97706" stroke-width="1.3" stroke-dasharray="4 3"/>
       <text x="852" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">+ Custom</text>

       <!-- Row 4: Envs -->
       <text x="16" y="138" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Envs</text>
       <rect x="116" y="120" width="92" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="162" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">ManiSkill3</text>
       <rect x="216" y="120" width="78" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="255" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">LIBERO</text>
       <rect x="302" y="120" width="90" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="347" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">RoboTwin</text>
       <rect x="400" y="120" width="124" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="462" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">BEHAVIOR · Isaac</text>
       <rect x="532" y="120" width="102" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="583" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">11+ Robots</text>
       <rect x="642" y="120" width="150" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="717" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">World · Math · Agents</text>
       <rect x="800" y="120" width="104" height="26" rx="13" fill="none" stroke="#10b981" stroke-width="1.3" stroke-dasharray="4 3"/>
       <text x="852" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">+ Custom</text>
     </svg>
   </div>

See :doc:`../concepts/placement` for hardware layout syntax and :doc:`../concepts/robotics` for real-robot assembly.

Where to Go Next
----------------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: Training Configuration
      :link: ../reference/configuration
      :link-type: doc

      Read an annotated minimal YAML and look up every config directory.

   .. grid-item-card:: Core Concepts
      :link: ../concepts/index
      :link-type: doc

      Dive deeper into Workers, Channels, Placement, and Robotics.

   .. grid-item-card:: Embodied & Real-World
      :link: ../examples/index
      :link-type: doc

      Pick a simulator, VLA model, SFT recipe, or real robot platform.

   .. grid-item-card:: Extending RLinf
      :link: ../extending/index
      :link-type: doc

      Plug in your own model, environment, robot, algorithm, or accelerator.
