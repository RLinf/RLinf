RLinf Walkthrough
=================

Build a clear mental model of RLinf in five minutes: the four architectural ideas that set RLinf apart—decoupled Workers and Channels, unified Hardware abstraction, a composable Robotics interface, and pluggable models and algorithms—plus a map of where everything lives in the repository.

1. Control vs. Data: Workers and Channels
-----------------------------------------

Most RL setups tie simulation, inference, and training into one fixed loop. RLinf separates **execution control** from **data movement**: each role runs as a Worker group scheduled by a Runner, while trajectories and weights flow through asynchronous Channel queues and collective weight sync.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 330" width="100%" role="img" aria-label="RLinf Control Plane (Runner and WorkerGroups) decoupled from Data Plane (async Channel queues and weight sync)" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="wf-purple-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
         <marker id="wf-teal-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--rlinf-admon-tip-border, #10b981)"/>
         </marker>
       </defs>

       <!-- Control Plane Banner -->
       <rect x="16" y="12" width="888" height="46" rx="10" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="34" y="32" font-size="12" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">CONTROL PLANE  ·  Runner &amp; Cluster</text>
       <text x="34" y="48" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Orchestrates lifecycle, stage scheduling, and hardware placement across WorkerGroups — one loop drives sync, pipelined, or async runs</text>

       <!-- Control arrows down to 4 WorkerGroups -->
       <line x1="122" y1="58" x2="122" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-en)"/>
       <line x1="347" y1="58" x2="347" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-en)"/>
       <line x1="572" y1="58" x2="572" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-en)"/>
       <line x1="797" y1="58" x2="797" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-en)"/>

       <!-- 4 WorkerGroup boxes -->
       <rect x="16" y="92" width="212" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="32" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Env / Agent Workers</text>
       <text x="32" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Simulators, real robots,</text>
       <text x="32" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">or multi-turn tool sandboxes</text>
       <text x="32" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">YAML: env.* / agentloop.*</text>

       <rect x="241" y="92" width="212" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="257" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout Workers</text>
       <text x="257" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Policy inference via HF,</text>
       <text x="257" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">SGLang, vLLM, or PhyAI</text>
       <text x="257" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">YAML: rollout.*</text>

       <rect x="466" y="92" width="212" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="482" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Reward &amp; Buffer</text>
       <text x="482" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Rule / model rewards, GAE,</text>
       <text x="482" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">GRPO, or replay sampling</text>
       <text x="482" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">YAML: algorithm.* / reward.*</text>

       <rect x="691" y="92" width="213" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="707" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Actor Workers</text>
       <text x="707" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">FSDP / Megatron updates</text>
       <text x="707" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">&amp; checkpoint saving</text>
       <text x="707" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">YAML: actor.*</text>

       <!-- Data arrows down to Channel bus -->
       <line x1="122" y1="184" x2="122" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-start="url(#wf-teal-en)" marker-end="url(#wf-teal-en)"/>
       <line x1="347" y1="184" x2="347" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-start="url(#wf-teal-en)" marker-end="url(#wf-teal-en)"/>
       <line x1="572" y1="184" x2="572" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-start="url(#wf-teal-en)" marker-end="url(#wf-teal-en)"/>
       <line x1="797" y1="184" x2="797" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-end="url(#wf-teal-en)"/>

       <!-- Data Plane: Channel Bus -->
       <rect x="16" y="220" width="888" height="46" rx="10" fill="var(--rlinf-admon-tip-bg, #ecfdf5)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <text x="34" y="240" font-size="12" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">DATA PLANE  ·  Named Channels (Async Queues) &amp; P2P Transfers</text>
       <text x="34" y="256" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Producers put() observations, actions, and trajectories; consumers get() / get_batch() with backpressure and key-based routing</text>

       <!-- Weight Sync return loop -->
       <path d="M 797 266 L 797 298 L 347 298 L 347 268" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#wf-purple-en)"/>
       <rect x="464" y="284" width="216" height="28" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4"/>
       <text x="572" y="302" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Hot Weight Sync (Actor → Rollout)</text>
     </svg>
   </div>

Because control and data are decoupled, changing execution from single-GPU sequential runs to fine-grained pipelining or async RL takes a config change instead of rewriting workers. See :doc:`../concepts/worker` and :doc:`../concepts/channel` for the full programming model.

.. tip::

   Set ``actor.model.model_path`` once when pointing to a checkpoint. Rollout inherits that path automatically so you maintain one model path per config.

2. Unified Hardware Abstraction & Placement
-------------------------------------------

RLinf treats **accelerators, CPU nodes, and physical robots** as first-class hardware resources managed by one scheduling layer. Training and inference code never hardcode ``torch.cuda``: the hardware layer resolves the PyTorch backend, device visibility, and collective communication for eight accelerator families automatically.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 286" width="100%" role="img" aria-label="RLinf Hardware Abstraction covering 8 accelerator families and physical robots mapped to Collocated, Disaggregated, and Hybrid placement" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="hw-arrow-en" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="6" markerHeight="6" orient="auto">
           <path d="M 1.5 0 L 5 8.5 L 8.5 0 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Top: 3 Placement Modes -->
       <text x="16" y="18" font-size="11.5" font-weight="700" letter-spacing="0.04em" fill="var(--pst-color-text-muted, #64748b)">SAME WORKER CODE  ·  SWITCH LAYOUT WITH  cluster.component_placement</text>

       <rect x="16" y="28" width="286" height="96" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="32" y="51" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Collocated (1–8 GPUs)</text>
       <text x="32" y="71" font-size="11.5" fill="var(--pst-color-text-base, #0f172a)">All workers share the same GPUs</text>
       <text x="32" y="89" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Automatic CPU offload between stages</text>
       <text x="32" y="111" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #475569)">Best for quickstarts &amp; single-node runs</text>

       <rect x="317" y="28" width="286" height="96" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="333" y="51" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Disaggregated (8+ GPUs)</text>
       <text x="333" y="71" font-size="11.5" fill="var(--pst-color-text-base, #0f172a)">Dedicated GPUs per component</text>
       <text x="333" y="89" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Zero offload stalls; pipelined or async</text>
       <text x="333" y="111" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #475569)">Best for high-throughput cluster runs</text>

       <rect x="618" y="28" width="286" height="96" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <text x="634" y="51" font-size="13" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">Hybrid / Real-World</text>
       <text x="634" y="71" font-size="11.5" fill="var(--pst-color-text-base, #0f172a)">Heterogeneous node groups</text>
       <text x="634" y="89" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Mix GPU servers + CPU or robot hosts</text>
       <text x="634" y="111" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #475569)">Best for real robots &amp; mixed clusters</text>

       <!-- Connector arrows -->
       <line x1="159" y1="124" x2="159" y2="148" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" marker-end="url(#hw-arrow-en)"/>
       <line x1="460" y1="124" x2="460" y2="148" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" marker-end="url(#hw-arrow-en)"/>
       <line x1="761" y1="124" x2="761" y2="148" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" marker-end="url(#hw-arrow-en)"/>

       <!-- Middle: Hardware Abstraction Layer -->
       <rect x="16" y="152" width="888" height="44" rx="10" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5"/>
       <text x="460" y="172" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Unified Hardware &amp; Scheduler Layer</text>
       <text x="460" y="188" text-anchor="middle" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Auto-discovers devices · Isolates visibility · Maps PyTorch platform &amp; CCL · Routes node_groups to physical machines</text>

       <!-- Bottom: Physical Resource Pools -->
       <rect x="16" y="212" width="587" height="64" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="32" y="236" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">8 Accelerator Backends  (install.sh --platform &lt;vendor&gt;)</text>
       <text x="32" y="258" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">NVIDIA CUDA · AMD ROCm · Huawei Ascend · Moore Threads MUSA · Intel · Kunlunxin · MetaX · Biren</text>

       <rect x="618" y="212" width="286" height="64" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="634" y="236" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Non-GPU &amp; Robot Hosts</text>
       <text x="634" y="258" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">CPU simulation nodes · Robot control PCs</text>
     </svg>
   </div>

See :doc:`../concepts/placement` for placement syntax and :doc:`../extending/new_accelerator` for how new accelerator backends plug in.

3. Composable Robotics Interface
--------------------------------

Physical robots vary widely—some put arm and gripper on separate network endpoints (Franka), others chain them on one servo bus (SO-101), and multi-arm or wheeled humanoids add cameras, torsos, and teleoperation devices. RLinf's Robotics layer separates **logical robot structure** from **hardware connections**, giving every task one clean read/write contract.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 236" width="100%" role="img" aria-label="RLinf Composable Robotics Interface: assemble parts, handle safety and remote placement, reuse across self-check, teleop, SFT, and online RL" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rb-arrow-en" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Left: Composable Parts -->
       <rect x="16" y="16" width="278" height="204" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <rect x="16" y="16" width="278" height="36" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="155" y="39" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1. Assemble Robot Parts</text>
       <text x="32" y="76" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">• Arm  ·  End-Effector  ·  Mobile Base</text>
       <text x="32" y="98" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">• Cameras  ·  GELLO / PICO / SpaceMouse</text>
       <rect x="32" y="114" width="246" height="88" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="44" y="135" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">One Path-Based Dict Contract</text>
       <text x="44" y="155" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">obs = robot.get_observation()</text>
       <text x="44" y="173" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">robot.send_action({"arm": ..., ...})</text>
       <text x="44" y="191" font-size="11" fill="var(--pst-color-primary, #6d28d9)">Independent links or shared bus, same API</text>

       <!-- Arrow 1 -> 2 -->
       <line x1="294" y1="118" x2="328" y2="118" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rb-arrow-en)"/>

       <!-- Middle: Runtime & Safety -->
       <rect x="334" y="16" width="272" height="204" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <rect x="334" y="16" width="272" height="36" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="470" y="39" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">2. Connection &amp; Safety Layer</text>
       <text x="350" y="78" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Lazy SDK imports (inspect offline)</text>
       <text x="350" y="106" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Remote placement on robot PCs</text>
       <text x="350" y="134" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Smooth motion &amp; batched actions</text>
       <text x="350" y="162" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Health checks &amp; fault recovery</text>
       <text x="350" y="190" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• Clean rollback on disconnect</text>

       <!-- Arrow 2 -> 3 -->
       <line x1="606" y1="118" x2="640" y2="118" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rb-arrow-en)"/>

       <!-- Right: End-to-End Reuse -->
       <rect x="646" y="16" width="258" height="204" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <rect x="646" y="16" width="258" height="36" rx="12" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="775" y="39" text-anchor="middle" font-size="13" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">3. Reuse Across Every Stage</text>
       <text x="662" y="78" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">① Hardware Self-Check</text>
       <text x="662" y="106" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">② Teleop &amp; Data Collection</text>
       <text x="662" y="134" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">③ SFT &amp; Policy Deployment</text>
       <text x="662" y="162" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">④ Online Real-World RL</text>
       <text x="662" y="194" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">11+ robot platforms share one API</text>
     </svg>
   </div>

Read :doc:`../concepts/robotics` for the full API walkthrough or jump into :doc:`../examples/real_world_index` to pick a robot platform.

4. Pluggable Models, Engines, and Algorithms
--------------------------------------------

Every layer in RLinf uses explicit registries rather than tightly coupled forks. Pick any supported combination in YAML—or register your own in-tree or out-of-tree via ``RLINF_EXT_MODULE``.

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 248" width="100%" role="img" aria-label="Pluggable RLinf ecosystem: Models, Training and Rollout Engines, Algorithms, and Environments" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <!-- 4 columns -->
       <rect x="16" y="12" width="212" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <rect x="16" y="12" width="212" height="34" rx="10" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="122" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Models</text>
       <text x="30" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Embodied VLA / WAM</text>
       <text x="30" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">OpenPI π₀/π₀.₅ · OpenVLA</text>
       <text x="30" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">GR00T · StarVLA · MLP</text>
       <text x="30" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">LLM &amp; Diffusion</text>
       <text x="30" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">Qwen2.5/3 · Moonlight MoE</text>
       <text x="30" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">SD3.5 Medium · Wan2.2</text>

       <rect x="242" y="12" width="212" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <rect x="242" y="12" width="212" height="34" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="348" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Engines</text>
       <text x="256" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Training Backends</text>
       <text x="256" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">PyTorch FSDP / FSDP2</text>
       <text x="256" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">Megatron-Core + MBridge</text>
       <text x="256" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout Backends</text>
       <text x="256" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">Hugging Face · PhyAI</text>
       <text x="256" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">SGLang · vLLM</text>

       <rect x="468" y="12" width="212" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <rect x="468" y="12" width="212" height="34" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="574" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Algorithms</text>
       <text x="482" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Online RL</text>
       <text x="482" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">PPO · GRPO · DAPO · Reinforce++</text>
       <text x="482" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">SAC / RLPD · SAC-Flow · DSRL</text>
       <text x="482" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">SFT, Offline &amp; Diffusion</text>
       <text x="482" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">SFT · DAgger · IQL · RECAP</text>
       <text x="482" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">Diffusion-NFT · ART · ReFL</text>

       <rect x="694" y="12" width="210" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <rect x="694" y="12" width="210" height="34" rx="10" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="799" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">Envs &amp; Tasks</text>
       <text x="708" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Embodied &amp; Real</text>
       <text x="708" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">ManiSkill · LIBERO · RoboTwin</text>
       <text x="708" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">BEHAVIOR · IsaacLab · 11+ Robots</text>
       <text x="708" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">World &amp; Agentic</text>
       <text x="708" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">OpenSora · Wan · BWM</text>
       <text x="708" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">Math · Search · Coding Sandboxes</text>

       <!-- Bottom extension bar -->
       <rect x="16" y="202" width="888" height="36" rx="9" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="460" y="225" text-anchor="middle" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Extend any column via registries — in-tree or out-of-tree without modifying RLinf (RLINF_EXT_MODULE)</text>
     </svg>
   </div>

5. Repository Map: From Command to Output
-----------------------------------------

The diagram below maps those concepts onto the six directories you interact with day to day.

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
       <text x="352" y="108" font-size="11" fill="var(--pst-color-text-muted, #475569)">Cluster, Workers, Channels &amp; Hardware</text>

       <rect x="338" y="126" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="144" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">models/ · hybrid_engines/</text>
       <text x="352" y="160" font-size="11" fill="var(--pst-color-text-muted, #475569)">VLA/LLM policies, FSDP, Megatron, SGLang</text>

       <rect x="338" y="178" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="196" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">envs/ · robotics/</text>
       <text x="352" y="212" font-size="11" fill="var(--pst-color-text-muted, #475569)">Simulators, composable robots &amp; teleop</text>

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

      Workers, Channels, Hardware abstraction, runners, models, simulators, composable robotics, and algorithm registries.

   .. grid-item-card:: Installers & Images (``requirements/`` & ``docker/``)
      :link: installation
      :link-type: doc

      One-line ``install.sh`` environment builder and Dockerfiles across eight accelerator backends.

   .. grid-item-card:: Tests & Extensions (``tests/`` & ``docs/``)
      :link: ../extending/index
      :link-type: doc

      Unit tests, end-to-end configs, fake vendor SDKs for real-robot testing, and English/Chinese docs.

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

   .. grid-item-card:: Core Concepts
      :link: ../concepts/index
      :link-type: doc

      Dive deeper into Workers, Channels, Placement, and Robotics architecture.

   .. grid-item-card:: Embodied & Real-World
      :link: ../examples/index
      :link-type: doc

      Pick a simulator, VLA model, SFT recipe, or real robot platform.

   .. grid-item-card:: Extending RLinf
      :link: ../extending/index
      :link-type: doc

      Plug in your own model, environment, robot, algorithm, or accelerator.
