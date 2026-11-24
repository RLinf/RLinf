框架导览
========

顺着刚才跑通的启动命令由浅入深，分三步建立对 RLinf 的认知：先看仓库目录与产出位置，再看一次训练如何在内部运转，最后了解支撑多卡扩展、真机与多模型生态的核心设计。

1. 从启动命令看仓库结构
-----------------------

RLinf 的日常使用统一遵循同一种模式：用启动脚本指定一份 Hydra YAML 配置，按需在命令行覆盖参数，结果统一落盘在 ``logs/`` 下。

.. raw:: html

   <div style="margin: 1rem 0 1.25rem 0;">
     <svg viewBox="0 0 920 198" width="100%" role="img" aria-label="RLinf 启动命令与仓库目录对应关系：examples/evaluations、rlinf、logs 与 toolkits/requirements" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="w3-pur-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
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
       <text x="239" y="72" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">▲ 启动脚本</text>
       <text x="493" y="72" text-anchor="middle" font-size="10.5" font-weight="600" fill="#3b82f6">▲ YAML 配置名</text>
       <text x="736" y="72" text-anchor="middle" font-size="10.5" font-weight="600" fill="#10b981">▲ 命令行覆盖参数</text>

       <!-- 4 Folder cards -->
       <rect x="16" y="96" width="206" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="32" y="124" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">examples/ · evaluations/</text>
       <text x="32" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">启动脚本与 Hydra YAML</text>
       <text x="32" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">训练配方与独立评测</text>

       <line x1="222" y1="139" x2="244" y2="139" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#w3-pur-zh)"/>

       <rect x="248" y="96" width="206" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="1.6"/>
       <text x="264" y="124" font-size="13" font-weight="700" fill="#3b82f6">rlinf/</text>
       <text x="264" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">Runner · Worker · 模型</text>
       <text x="264" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">环境 · 真机接口 · 算法</text>

       <line x1="454" y1="139" x2="476" y2="139" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#w3-pur-zh)"/>

       <rect x="480" y="96" width="200" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="1.6"/>
       <text x="496" y="124" font-size="13" font-weight="700" fill="#10b981">logs/</text>
       <text x="496" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">TensorBoard · Checkpoint</text>
       <text x="496" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">视频与数据分片产出</text>

       <rect x="698" y="96" width="206" height="86" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="714" y="124" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">toolkits/ · requirements/</text>
       <text x="714" y="146" font-size="11.5" fill="var(--pst-color-text-muted, #64748b)">install.sh 与 Dockerfile</text>
       <text x="714" y="166" font-size="11" fill="var(--pst-color-text-muted, #64748b)">数据 · 自检 · 自动放置</text>
     </svg>
   </div>

.. dropdown:: 查看 ``toolkits/`` 常用工具速查

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - 子目录
        - 用途
      * - ``toolkits/lerobot/``
        - 计算 ``norm_stats.json``、合并多个 LeRobot 数据集、导出 MP4 视频预览。
      * - ``toolkits/realworld_check/``
        - 真机训练前的交互式硬件自检（机械臂、夹爪、相机、GELLO、PICO）。
      * - ``toolkits/replay_buffer/``
        - 查看、可视化、合并或拆分已保存的轨迹 Replay Buffer。
      * - ``toolkits/auto_placement/``
        - 根据实测单步耗时自动搜索最优共置/分离 GPU 放置方案。
      * - ``toolkits/cosmos3/``、``dual_franka/``、``standalone_eval_scripts/``
        - 特定模型与机型的权重转换、数据清理与独立评测脚本。

各任务配置文件的完整路径与带注释最小 YAML 见 :doc:`../reference/configuration`。

2. 一次训练如何运转：Runner、Worker 与 Channel
----------------------------------------------

命令启动后，入口脚本先拉起 Ray 集群和各角色 Worker，再把训练主循环交给 **Runner** 接管：

- **Runner（主控循环）**：运行在主控进程中，定义每步迭代顺序——同步权重、触发采样、计算 Reward / Advantage、更新 Actor 以及定时评测和保存。**新增训练范式或自定义工作流时，核心改动就在 Runner。**

- **Worker（执行角色）**：Env、Rollout、Reward、Actor 作为独立进程组运行并对应一块 YAML 配置，只负责本职计算。

- **Channel（数据队列）**：观测、动作与轨迹在 Worker 间经异步队列直传，不经过主控进程；参数则经热权重同步广播。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 276" width="100%" role="img" aria-label="RLinf 控制面（Runner 主循环：同步权重、采样交互、Reward/Advantage、更新 Actor、评测保存）与数据面（四个 Worker 组经异步 Channel 队列与热权重同步连接）" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="w1-pur-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
         <marker id="w1-grn-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#10b981"/>
         </marker>
       </defs>

       <!-- Legend top right -->
       <line x1="658" y1="20" x2="686" y2="20" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3"/>
       <text x="694" y="24" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">控制面</text>
       <line x1="748" y1="20" x2="776" y2="20" stroke="#10b981" stroke-width="2"/>
       <text x="784" y="24" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #64748b)">数据面（Channel）</text>

       <!-- Control Plane: Runner loop bar -->
       <rect x="24" y="8" width="606" height="48" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.10))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5"/>
       <text x="42" y="27" font-size="12" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Runner（主控循环 · 自定义训练流程入口）</text>
       <text x="42" y="45" font-size="10.8" font-weight="500" fill="var(--pst-color-text-base, #0f172a)">1. 同步权重  →  2. 采样交互  →  3. 计算 Reward / Advantage  →  4. 更新 Actor  →  5. 评测与保存</text>

       <!-- Dashed control bus -->
       <path d="M 327 56 L 327 76 M 102 76 L 818 76" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3"/>
       <line x1="102" y1="76" x2="102" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-zh)"/>
       <line x1="340" y1="76" x2="340" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-zh)"/>
       <line x1="580" y1="76" x2="580" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-zh)"/>
       <line x1="818" y1="76" x2="818" y2="102" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#w1-pur-zh)"/>

       <!-- Worker 1: Env (Blue) -->
       <rect x="24" y="106" width="156" height="86" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="2"/>
       <circle cx="54" cy="138" r="14" fill="rgba(59,130,246,0.14)"/>
       <rect x="47" y="131" width="14" height="14" rx="3" fill="none" stroke="#3b82f6" stroke-width="1.8"/>
       <circle cx="51.5" cy="136.5" r="1.4" fill="#3b82f6"/>
       <circle cx="56.5" cy="136.5" r="1.4" fill="#3b82f6"/>
       <text x="78" y="136" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Env</text>
       <text x="78" y="153" font-size="11" fill="var(--pst-color-text-muted, #64748b)">仿真 · 真机</text>
       <rect x="44" y="166" width="116" height="18" rx="9" fill="rgba(59,130,246,0.12)"/>
       <text x="102" y="178.5" text-anchor="middle" font-size="10.5" font-weight="600" fill="#3b82f6">env.*</text>

       <!-- Channel pipe 1 -> 2 (FIFO slots) -->
       <line x1="180" y1="149" x2="200" y2="149" stroke="#10b981" stroke-width="2"/>
       <rect x="200" y="138" width="42" height="22" rx="6" fill="rgba(16,185,129,0.1)" stroke="#10b981" stroke-width="1.4"/>
       <rect x="205" y="143" width="8" height="12" rx="2" fill="#10b981"/>
       <rect x="217" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.65"/>
       <rect x="229" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.28"/>
       <line x1="242" y1="149" x2="258" y2="149" stroke="#10b981" stroke-width="2" marker-end="url(#w1-grn-zh)"/>
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
       <line x1="481" y1="149" x2="498" y2="149" stroke="#10b981" stroke-width="2" marker-end="url(#w1-grn-zh)"/>
       <text x="460" y="174" text-anchor="middle" font-size="10" font-weight="600" fill="#10b981">Channel</text>

       <!-- Worker 3: Reward & Buffer (Amber) -->
       <rect x="502" y="106" width="156" height="86" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="#d97706" stroke-width="2"/>
       <circle cx="532" cy="138" r="14" fill="rgba(217,119,6,0.14)"/>
       <circle cx="532" cy="138" r="7" fill="none" stroke="#d97706" stroke-width="1.8"/>
       <circle cx="532" cy="138" r="2.5" fill="#d97706"/>
       <text x="556" y="136" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Reward</text>
       <text x="556" y="153" font-size="11" fill="var(--pst-color-text-muted, #64748b)">打分 · Buffer</text>
       <rect x="522" y="166" width="116" height="18" rx="9" fill="rgba(217,119,6,0.12)"/>
       <text x="580" y="178.5" text-anchor="middle" font-size="10.5" font-weight="600" fill="#d97706">algorithm.*</text>

       <!-- Channel pipe 3 -> 4 -->
       <line x1="658" y1="149" x2="678" y2="149" stroke="#10b981" stroke-width="2"/>
       <rect x="678" y="138" width="42" height="22" rx="6" fill="rgba(16,185,129,0.1)" stroke="#10b981" stroke-width="1.4"/>
       <rect x="683" y="143" width="8" height="12" rx="2" fill="#10b981"/>
       <rect x="695" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.65"/>
       <rect x="707" y="143" width="8" height="12" rx="2" fill="#10b981" opacity="0.28"/>
       <line x1="720" y1="149" x2="736" y2="149" stroke="#10b981" stroke-width="2" marker-end="url(#w1-grn-zh)"/>
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
       <path d="M 818 192 L 818 236 Q 818 250 804 250 L 354 250 Q 340 250 340 236 L 340 196" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2.2" marker-end="url(#w1-pur-zh)"/>
       <rect x="506" y="236" width="148" height="28" rx="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="580" y="254" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">热权重同步</text>
     </svg>
   </div>

RLinf 为具身同步/异步 RL、推理、Agent、SFT、离线 RL 与独立评测提供了内置 Runner；在流水线或异步 Runner 中，各 Worker 通过同一套 Channel 并发重叠执行，而无需改动 Worker 本身。切换模型权重时只需设置 ``actor.model.model_path``\ ，rollout 会自动沿用。深入阅读见 :doc:`../concepts/execution_flow`、:doc:`../concepts/worker` 与 :doc:`../concepts/channel`。

3. 核心设计：硬件抽象、真机积木与可插拔生态
---------------------------------------------

通过调整 ``cluster.component_placement``，可将同一组 Worker 部署到单卡、多机集群、各类加速卡后端或真机主机；真机侧由模块化零部件组合出统一 Robot 接口，复用于自检、数据采集、在线 RL 与评测部署。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 308" width="100%" role="img" aria-label="左侧：硬件抽象与共置/分离/异构真机放置；右侧：积木式零部件组装为统一 Robot 接口并复用在四个阶段" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="w2-pur-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- LEFT PANEL: Hardware & Placement -->
       <text x="16" y="22" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">硬件抽象与部署</text>
       <circle cx="242" cy="18" r="4.5" fill="#3b82f6"/>
       <text x="251" y="22" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Env</text>
       <circle cx="292" cy="18" r="4.5" fill="#7c3aed"/>
       <text x="301" y="22" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Rollout</text>
       <circle cx="362" cy="18" r="4.5" fill="#10b981"/>
       <text x="371" y="22" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">Actor</text>

       <!-- Row 1: Collocated -->
       <text x="16" y="64" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">共置</text>
       <text x="16" y="79" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">1–8 卡</text>
       <rect x="116" y="44" width="166" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="128" y="61" font-size="10" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">GPU 0</text>
       <rect x="128" y="66" width="44" height="14" rx="7" fill="#3b82f6"/>
       <rect x="177" y="66" width="44" height="14" rx="7" fill="#7c3aed"/>
       <rect x="226" y="66" width="44" height="14" rx="7" fill="#10b981"/>
       <text x="296" y="70" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">共享 + offload</text>

       <!-- Row 2: Disaggregated -->
       <text x="16" y="122" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">分离</text>
       <text x="16" y="137" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">8+ 卡</text>
       <rect x="116" y="102" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="#7c3aed" stroke-width="1.6"/>
       <text x="128" y="119" font-size="10" font-weight="700" fill="#7c3aed">GPU 0–3</text>
       <rect x="128" y="124" width="44" height="14" rx="7" fill="#7c3aed"/>
       <rect x="178" y="124" width="44" height="14" rx="7" fill="#7c3aed" opacity="0.65"/>
       <line x1="234" y1="124" x2="262" y2="124" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#w2-pur-zh)"/>
       <rect x="268" y="102" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="1.6"/>
       <text x="280" y="119" font-size="10" font-weight="700" fill="#10b981">GPU 4–7</text>
       <rect x="280" y="124" width="44" height="14" rx="7" fill="#10b981"/>
       <rect x="330" y="124" width="44" height="14" rx="7" fill="#10b981" opacity="0.65"/>

       <!-- Row 3: Hybrid / Real-World -->
       <text x="16" y="180" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">异构 / 真机</text>
       <text x="16" y="195" font-size="10.5" fill="var(--pst-color-text-muted, #64748b)">真机 + GPU</text>
       <rect x="116" y="160" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="1.6"/>
       <text x="128" y="177" font-size="10" font-weight="700" fill="#3b82f6">真机控制机</text>
       <rect x="128" y="182" width="94" height="14" rx="7" fill="#3b82f6"/>
       <line x1="234" y1="182" x2="262" y2="182" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-start="url(#w2-pur-zh)" marker-end="url(#w2-pur-zh)"/>
       <rect x="268" y="160" width="118" height="44" rx="9" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="280" y="177" font-size="10" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">GPU 节点</text>
       <rect x="280" y="182" width="44" height="14" rx="7" fill="#7c3aed"/>
       <rect x="330" y="182" width="44" height="14" rx="7" fill="#10b981"/>

       <!-- Hardware Abstraction Rail -->
       <rect x="16" y="220" width="416" height="24" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.12))"/>
       <text x="224" y="236" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">统一硬件抽象层 · 适配八大加速卡后端</text>

       <!-- 8 Vendor Pills (2 rows of 4) -->
       <rect x="16" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="65" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">NVIDIA CUDA</text>
       <rect x="122" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="171" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">AMD ROCm</text>
       <rect x="228" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="277" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">昇腾 CANN</text>
       <rect x="334" y="252" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="383" y="267" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">摩尔 MUSA</text>

       <rect x="16" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="65" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">Intel XPU</text>
       <rect x="122" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="171" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">昆仑芯</text>
       <rect x="228" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="277" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">沐曦</text>
       <rect x="334" y="280" width="98" height="22" rx="6" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="383" y="295" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">壁仞</text>

       <!-- Center Divider -->
       <line x1="460" y1="16" x2="460" y2="298" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.5" stroke-dasharray="4 4"/>

       <!-- RIGHT PANEL: Robotics Composable Bricks -->
       <text x="488" y="22" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Robotics 积木式真机接口</text>

       <!-- 4 Lego bricks on left -->
       <rect x="502" y="42" width="16" height="5" rx="2" fill="#3b82f6"/>
       <rect x="530" y="42" width="16" height="5" rx="2" fill="#3b82f6"/>
       <rect x="488" y="47" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#3b82f6" stroke-width="1.7"/>
       <text x="540" y="66" text-anchor="middle" font-size="11.5" font-weight="700" fill="#3b82f6">机械臂</text>

       <rect x="502" y="86" width="16" height="5" rx="2" fill="#7c3aed"/>
       <rect x="530" y="86" width="16" height="5" rx="2" fill="#7c3aed"/>
       <rect x="488" y="91" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#7c3aed" stroke-width="1.7"/>
       <text x="540" y="110" text-anchor="middle" font-size="11.5" font-weight="700" fill="#7c3aed">夹爪 / 灵巧手</text>

       <rect x="502" y="130" width="16" height="5" rx="2" fill="#d97706"/>
       <rect x="530" y="130" width="16" height="5" rx="2" fill="#d97706"/>
       <rect x="488" y="135" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#d97706" stroke-width="1.7"/>
       <text x="540" y="154" text-anchor="middle" font-size="11.5" font-weight="700" fill="#d97706">多路相机</text>

       <rect x="502" y="174" width="16" height="5" rx="2" fill="#10b981"/>
       <rect x="530" y="174" width="16" height="5" rx="2" fill="#10b981"/>
       <rect x="488" y="179" width="104" height="30" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="#10b981" stroke-width="1.7"/>
       <text x="540" y="198" text-anchor="middle" font-size="11.5" font-weight="700" fill="#10b981">遥操作 / 底盘</text>

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
       <text x="781" y="180" text-anchor="middle" font-size="10" fill="var(--pst-color-text-muted, #64748b)">远程部署 · 安全容错 · 11+ 真机</text>

       <!-- Bottom 4-Stage Pipeline -->
       <line x1="538" y1="268" x2="862" y2="268" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <circle cx="538" cy="268" r="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="538" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1</text>
       <text x="538" y="298" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">硬件自检</text>

       <circle cx="646" cy="268" r="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="646" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">2</text>
       <text x="646" y="298" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">数据采集</text>

       <circle cx="754" cy="268" r="14" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2"/>
       <text x="754" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">3</text>
       <text x="754" y="298" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">在线 RL</text>

       <circle cx="862" cy="268" r="14" fill="var(--pst-color-primary, #6d28d9)"/>
       <text x="862" y="272" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">4</text>
       <text x="862" y="298" text-anchor="middle" font-size="11" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">评测部署</text>
     </svg>
   </div>

模型、引擎、算法与环境都通过注册表自由组合，既可直接选用内置配方，也能通过 ``RLINF_EXT_MODULE`` 在外部仓库零侵入接入。

.. raw:: html

   <div style="margin: 1rem 0 1.4rem 0;">
     <svg viewBox="0 0 920 156" width="100%" role="img" aria-label="RLinf 可插拔注册表标签：模型、引擎、算法与环境" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <!-- Row 1: Models -->
       <text x="16" y="24" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">模型</text>
       <rect x="92" y="6" width="118" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="151" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">OpenPI π₀/π₀.₅</text>
       <rect x="218" y="6" width="120" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="278" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">OpenVLA / OFT</text>
       <rect x="346" y="6" width="76" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="384" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">GR00T</text>
       <rect x="430" y="6" width="116" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="488" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">StarVLA · MLP</text>
       <rect x="554" y="6" width="114" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="611" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">Qwen · MoE</text>
       <rect x="676" y="6" width="108" height="26" rx="13" fill="rgba(124,58,237,0.12)"/>
       <text x="730" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">SD3.5 · Wan</text>
       <rect x="794" y="6" width="110" height="26" rx="13" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.3" stroke-dasharray="4 3"/>
       <text x="849" y="23" text-anchor="middle" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">+ 自定义注册</text>

       <!-- Row 2: Engines -->
       <text x="16" y="62" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">引擎</text>
       <rect x="92" y="44" width="114" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="149" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">FSDP / FSDP2</text>
       <rect x="214" y="44" width="120" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="274" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">Megatron-Core</text>
       <rect x="342" y="44" width="112" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="398" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">Hugging Face</text>
       <rect x="462" y="44" width="84" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="504" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">SGLang</text>
       <rect x="554" y="44" width="70" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="589" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">vLLM</text>
       <rect x="632" y="44" width="76" height="26" rx="13" fill="rgba(59,130,246,0.12)"/>
       <text x="670" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="#3b82f6">PhyAI</text>

       <!-- Row 3: Algorithms -->
       <text x="16" y="100" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">算法</text>
       <rect x="92" y="82" width="60" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="122" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">PPO</text>
       <rect x="160" y="82" width="68" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="194" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">GRPO</text>
       <rect x="236" y="82" width="68" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="270" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">DAPO</text>
       <rect x="312" y="82" width="100" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="362" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">SAC / RLPD</text>
       <rect x="420" y="82" width="114" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="477" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">SFT · DAgger</text>
       <rect x="542" y="82" width="110" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="597" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">IQL · RECAP</text>
       <rect x="660" y="82" width="124" height="26" rx="13" fill="rgba(217,119,6,0.12)"/>
       <text x="722" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">NFT · ART · ReFL</text>
       <rect x="794" y="82" width="110" height="26" rx="13" fill="none" stroke="#d97706" stroke-width="1.3" stroke-dasharray="4 3"/>
       <text x="849" y="99" text-anchor="middle" font-size="11" font-weight="600" fill="#d97706">+ 自定义注册</text>

       <!-- Row 4: Envs -->
       <text x="16" y="138" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">环境</text>
       <rect x="92" y="120" width="94" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="139" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">ManiSkill3</text>
       <rect x="194" y="120" width="80" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="234" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">LIBERO</text>
       <rect x="282" y="120" width="92" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="328" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">RoboTwin</text>
       <rect x="382" y="120" width="130" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="447" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">BEHAVIOR · Isaac</text>
       <rect x="520" y="120" width="98" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="569" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">11+ 真机</text>
       <rect x="626" y="120" width="158" height="26" rx="13" fill="rgba(16,185,129,0.12)"/>
       <text x="705" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">世界模型 · 推理 · 智能体</text>
       <rect x="794" y="120" width="110" height="26" rx="13" fill="none" stroke="#10b981" stroke-width="1.3" stroke-dasharray="4 3"/>
       <text x="849" y="137" text-anchor="middle" font-size="11" font-weight="600" fill="#10b981">+ 自定义注册</text>
     </svg>
   </div>

放置语法见 :doc:`../concepts/placement`\ ，真机组装与接口详见 :doc:`../concepts/robotics`。

下一步去哪里
------------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: 训练配置地图
      :link: ../reference/configuration
      :link-type: doc

      对照带注释的最小 YAML，查清每个任务的配置目录。

   .. grid-item-card:: 核心概念
      :link: ../concepts/index
      :link-type: doc

      深入了解 Worker、Channel、Placement 与 Robotics 架构。

   .. grid-item-card:: 具身与真机示例
      :link: ../examples/index
      :link-type: doc

      按模拟器、VLA 模型、SFT 或真机平台选择训练配方。

   .. grid-item-card:: 扩展开发
      :link: ../extending/index
      :link-type: doc

      接入自定义模型、环境、机器人、算法或加速卡后端。
