快速熟悉 RLinf
==============

用五分钟看懂 RLinf 与众不同的四个核心设计：控制与数据解耦的 Worker 与 Channel、统一的硬件抽象与多形态部署、积木式真机 Robotics 接口、可插拔的模型与算法生态，以及它们在仓库中的对应目录。

1. 控制与数据解耦：Worker 与 Channel
------------------------------------

传统强化学习代码往往把仿真、推理和训练写死在同一个循环里，换部署方式就得重写代码。RLinf 将\ **控制流**\ 与\ **数据流**\ 拆开：每个角色作为独立的 Worker 组由 Runner 编排，轨迹和权重则通过异步 Channel 队列与集合通信流转。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 330" width="100%" role="img" aria-label="RLinf 控制面（Runner 与 Worker 组）和数据面（异步 Channel 队列与权重同步）解耦架构图" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="wf-purple-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
         <marker id="wf-teal-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--rlinf-admon-tip-border, #10b981)"/>
         </marker>
       </defs>

       <!-- Control Plane Banner -->
       <rect x="16" y="12" width="888" height="46" rx="10" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="34" y="32" font-size="12" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">控制面（Control Plane） ·  Runner 与 Cluster</text>
       <text x="34" y="48" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">统一编排各 Worker 组的生命周期、阶段切换与硬件放置——同一套逻辑驱动同步、流水线或异步训练</text>

       <!-- Control arrows down to 4 WorkerGroups -->
       <line x1="122" y1="58" x2="122" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-zh)"/>
       <line x1="347" y1="58" x2="347" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-zh)"/>
       <line x1="572" y1="58" x2="572" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-zh)"/>
       <line x1="797" y1="58" x2="797" y2="88" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#wf-purple-zh)"/>

       <!-- 4 WorkerGroup boxes -->
       <rect x="16" y="92" width="212" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="32" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Env / Agent Worker</text>
       <text x="32" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">驱动仿真器、真机机器人</text>
       <text x="32" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">或多轮工具调用沙箱</text>
       <text x="32" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">配置块：env.* / agentloop.*</text>

       <rect x="241" y="92" width="212" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="257" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout Worker</text>
       <text x="257" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">通过 HF、SGLang、vLLM</text>
       <text x="257" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">或 PhyAI 执行 policy 推理</text>
       <text x="257" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">配置块：rollout.*</text>

       <rect x="466" y="92" width="212" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="482" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Reward 与 Buffer</text>
       <text x="482" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">规则/模型打分、GAE/GRPO</text>
       <text x="482" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">优势估计或离策略采样</text>
       <text x="482" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">配置块：algorithm.* / reward.*</text>

       <rect x="691" y="92" width="213" height="92" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.7"/>
       <text x="707" y="116" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Actor Worker</text>
       <text x="707" y="135" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">使用 FSDP 或 Megatron</text>
       <text x="707" y="151" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">更新权重并保存 checkpoint</text>
       <text x="707" y="173" font-size="11" font-weight="600" fill="var(--pst-color-primary, #6d28d9)">配置块：actor.*</text>

       <!-- Data arrows down to Channel bus -->
       <line x1="122" y1="184" x2="122" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-start="url(#wf-teal-zh)" marker-end="url(#wf-teal-zh)"/>
       <line x1="347" y1="184" x2="347" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-start="url(#wf-teal-zh)" marker-end="url(#wf-teal-zh)"/>
       <line x1="572" y1="184" x2="572" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-start="url(#wf-teal-zh)" marker-end="url(#wf-teal-zh)"/>
       <line x1="797" y1="184" x2="797" y2="216" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8" marker-end="url(#wf-teal-zh)"/>

       <!-- Data Plane: Channel Bus -->
       <rect x="16" y="220" width="888" height="46" rx="10" fill="var(--rlinf-admon-tip-bg, #ecfdf5)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <text x="34" y="240" font-size="12" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">数据面（Data Plane） ·  命名 Channel 异步队列与点对点通信</text>
       <text x="34" y="256" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">生产者 put() 写入观测、动作与轨迹，消费者 get() / get_batch() 按 key 与背压策略异步消费</text>

       <!-- Weight Sync return loop -->
       <path d="M 797 266 L 797 298 L 347 298 L 347 268" fill="none" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#wf-purple-zh)"/>
       <rect x="458" y="284" width="228" height="28" rx="7" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4"/>
       <text x="572" y="302" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">热权重同步（Actor → Rollout）</text>
     </svg>
   </div>

控制面与数据面解耦后，从单卡顺序运行切换到多卡细粒度流水线或异步 RL 只需改配置，无需改动 worker 代码。完整编程模型见 :doc:`../concepts/worker` 与 :doc:`../concepts/channel`。

.. tip::

   切换模型权重时只需设置 ``actor.model.model_path``\ ，rollout 会自动沿用这一路径，无需在两处重复填写。

2. 统一硬件抽象与多形态部署
---------------------------

RLinf 把\ **各类加速卡、CPU 节点与真机机器人**\ 统一视为集群里的一等硬件资源。训练和推理代码不写死 ``torch.cuda``\ ，底层硬件层自动完成设备发现、可见性隔离、PyTorch 后端映射与集合通信，支持八大加速卡后端。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 286" width="100%" role="img" aria-label="RLinf 硬件抽象与部署模式：一套代码经 cluster.component_placement 切换共置、分离、混合部署并适配八大加速卡与真机节点" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="hw-arrow-zh" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="6" markerHeight="6" orient="auto">
           <path d="M 1.5 0 L 5 8.5 L 8.5 0 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Top: 3 Placement Modes -->
       <text x="16" y="18" font-size="12" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">同一套 Worker 代码 · 只需改 cluster.component_placement 即可切换三种部署形态</text>

       <rect x="16" y="28" width="286" height="96" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="32" y="51" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">Collocated 共置（1–8 卡）</text>
       <text x="32" y="71" font-size="11.5" fill="var(--pst-color-text-base, #0f172a)">所有组件共享同一批 GPU</text>
       <text x="32" y="89" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">各阶段之间自动 offload 省显存</text>
       <text x="32" y="111" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #475569)">适合快速上手与单机训练</text>

       <rect x="317" y="28" width="286" height="96" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="333" y="51" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Disaggregated 分离（8+ 卡）</text>
       <text x="333" y="71" font-size="11.5" fill="var(--pst-color-text-base, #0f172a)">每个组件独占对应 GPU 池</text>
       <text x="333" y="89" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">消除 offload 停顿，支持流水线与异步</text>
       <text x="333" y="111" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #475569)">适合大规模高吞吐集群训练</text>

       <rect x="618" y="28" width="286" height="96" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <text x="634" y="51" font-size="13" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">Hybrid 混合 / 真机异构</text>
       <text x="634" y="71" font-size="11.5" fill="var(--pst-color-text-base, #0f172a)">按 node_groups 划分异构节点</text>
       <text x="634" y="89" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">灵活组合 GPU 节点、CPU 与真机控制机</text>
       <text x="634" y="111" font-size="11" font-weight="600" fill="var(--pst-color-text-muted, #475569)">适合真机 RL 与异构多机集群</text>

       <!-- Connector arrows -->
       <line x1="159" y1="124" x2="159" y2="148" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" marker-end="url(#hw-arrow-zh)"/>
       <line x1="460" y1="124" x2="460" y2="148" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" marker-end="url(#hw-arrow-zh)"/>
       <line x1="761" y1="124" x2="761" y2="148" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6" marker-end="url(#hw-arrow-zh)"/>

       <!-- Middle: Hardware Abstraction Layer -->
       <rect x="16" y="152" width="888" height="44" rx="10" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.5"/>
       <text x="460" y="172" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">统一硬件抽象与调度层（Hardware &amp; Scheduler）</text>
       <text x="460" y="188" text-anchor="middle" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">自动探测设备 · 隔离可见卡号 · 统一 PyTorch 平台与集合通信 · 将 node_groups 精准路由到物理主机</text>

       <!-- Bottom: Physical Resource Pools -->
       <rect x="16" y="212" width="587" height="64" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="32" y="236" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">八大加速卡后端（安装时指定 install.sh --platform &lt;vendor&gt;）</text>
       <text x="32" y="258" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">NVIDIA CUDA · AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA · Intel · 昆仑芯 · 沐曦 · 壁仞</text>

       <rect x="618" y="212" width="286" height="64" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="634" y="236" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">非 GPU 与真机节点</text>
       <text x="634" y="258" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">CPU 仿真节点 · 真机机器人控制主机</text>
     </svg>
   </div>

放置语法详见 :doc:`../concepts/placement`\ ，新增加速卡适配方式见 :doc:`../extending/new_accelerator`。

3. 积木式真机接口（Robotics）
-----------------------------

不同真机的硬件拓扑差异很大：Franka 的机械臂与夹爪走独立网口，SO-101 串在同一条舵机总线上，双臂和轮式人形还会组合多路相机、躯干与遥操作设备。RLinf 的 Robotics 层把\ **逻辑零部件结构**\ 和\ **底层硬件连接**\ 分开管理，让所有真机对外呈现同一套读写契约。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 236" width="100%" role="img" aria-label="RLinf 模块化 Robotics 接口：积木式组装零部件、连接与安全容错层、贯通自检/遥操作/SFT/在线RL" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rb-arrow-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Left: Composable Parts -->
       <rect x="16" y="16" width="278" height="204" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <rect x="16" y="16" width="278" height="36" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="155" y="39" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1. 积木式组装零部件</text>
       <text x="32" y="76" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">• 机械臂 · 夹爪/灵巧手 · 移动底盘</text>
       <text x="32" y="98" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">• 多路相机 · GELLO / PICO / SpaceMouse</text>
       <rect x="32" y="114" width="246" height="88" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="44" y="135" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">统一按路径读写的字典契约</text>
       <text x="44" y="155" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">obs = robot.get_observation()</text>
       <text x="44" y="173" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">robot.send_action({"arm": ..., ...})</text>
       <text x="44" y="191" font-size="11" fill="var(--pst-color-primary, #6d28d9)">独立连接或共享总线，上层调用完全一致</text>

       <!-- Arrow 1 -> 2 -->
       <line x1="294" y1="118" x2="328" y2="118" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rb-arrow-zh)"/>

       <!-- Middle: Runtime & Safety -->
       <rect x="334" y="16" width="272" height="204" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <rect x="334" y="16" width="272" height="36" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="470" y="39" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">2. 连接管理与安全容错</text>
       <text x="350" y="78" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• 延迟加载厂商 SDK（无硬件可先自检）</text>
       <text x="350" y="106" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• 声明节点后自动远程部署到控制机</text>
       <text x="350" y="134" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• 运动平滑插值与多零部件合并下发</text>
       <text x="350" y="162" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• 运行期健康诊断与分级故障自动恢复</text>
       <text x="350" y="190" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• 断开时按逆序安全回滚释放资源</text>

       <!-- Arrow 2 -> 3 -->
       <line x1="606" y1="118" x2="640" y2="118" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rb-arrow-zh)"/>

       <!-- Right: End-to-End Reuse -->
       <rect x="646" y="16" width="258" height="204" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <rect x="646" y="16" width="258" height="36" rx="12" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="775" y="39" text-anchor="middle" font-size="13" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">3. 一次接入，贯穿全流程</text>
       <text x="662" y="78" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">① 交互式硬件自检</text>
       <text x="662" y="106" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">② 遥操作与数据采集</text>
       <text x="662" y="134" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">③ SFT 与 policy 部署</text>
       <text x="662" y="162" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">④ 真机在线强化学习</text>
       <text x="662" y="194" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">11+ 真机平台共用同一套接口</text>
     </svg>
   </div>

接口用法详见 :doc:`../concepts/robotics`\ ，各机型实操指南见 :doc:`../examples/real_world_index`。

4. 可插拔的模型、引擎与算法生态
-------------------------------

RLinf 用显式注册表连接模型、训练/推理引擎、算法与环境。你可以直接在 YAML 里自由组合现有能力，也可以通过 ``RLINF_EXT_MODULE`` 在仓库外零侵入接入自己的组件。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 248" width="100%" role="img" aria-label="RLinf 可插拔生态：模型、训练与推理引擎、强化学习算法、环境与任务" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <!-- 4 columns -->
       <rect x="16" y="12" width="212" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <rect x="16" y="12" width="212" height="34" rx="10" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="122" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">模型（Models）</text>
       <text x="30" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">具身 VLA / WAM</text>
       <text x="30" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">OpenPI π₀/π₀.₅ · OpenVLA</text>
       <text x="30" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">GR00T · StarVLA · MLP</text>
       <text x="30" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">大模型与扩散生成</text>
       <text x="30" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">Qwen2.5/3 · Moonlight MoE</text>
       <text x="30" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">SD3.5 Medium · Wan2.2</text>

       <rect x="242" y="12" width="212" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <rect x="242" y="12" width="212" height="34" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="348" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">训练与推理引擎</text>
       <text x="256" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">训练后端</text>
       <text x="256" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">PyTorch FSDP / FSDP2</text>
       <text x="256" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">Megatron-Core + MBridge</text>
       <text x="256" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">Rollout 后端</text>
       <text x="256" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">Hugging Face · PhyAI</text>
       <text x="256" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">SGLang · vLLM</text>

       <rect x="468" y="12" width="212" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <rect x="468" y="12" width="212" height="34" rx="10" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="574" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">算法（Algorithms）</text>
       <text x="482" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">在线强化学习</text>
       <text x="482" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">PPO · GRPO · DAPO · Reinforce++</text>
       <text x="482" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">SAC / RLPD · SAC-Flow · DSRL</text>
       <text x="482" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">SFT、离线与扩散</text>
       <text x="482" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">SFT · DAgger · IQL · RECAP</text>
       <text x="482" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">Diffusion-NFT · ART · ReFL</text>

       <rect x="694" y="12" width="210" height="176" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.6"/>
       <rect x="694" y="12" width="210" height="34" rx="10" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="799" y="34" text-anchor="middle" font-size="13" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">环境与任务</text>
       <text x="708" y="70" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">仿真与真机</text>
       <text x="708" y="88" font-size="11" fill="var(--pst-color-text-muted, #475569)">ManiSkill · LIBERO · RoboTwin</text>
       <text x="708" y="104" font-size="11" fill="var(--pst-color-text-muted, #475569)">BEHAVIOR · IsaacLab · 11+ 真机</text>
       <text x="708" y="130" font-size="11.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">世界模型与智能体</text>
       <text x="708" y="148" font-size="11" fill="var(--pst-color-text-muted, #475569)">OpenSora · Wan · BWM</text>
       <text x="708" y="164" font-size="11" fill="var(--pst-color-text-muted, #475569)">数学推理 · 检索 · 代码沙箱</text>

       <!-- Bottom extension bar -->
       <rect x="16" y="202" width="888" height="36" rx="9" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="460" y="225" text-anchor="middle" font-size="12" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">四列均通过注册表插拔扩展——支持仓库内开发，也支持经 RLINF_EXT_MODULE 在外部仓库零侵入接入</text>
     </svg>
   </div>

5. 仓库结构：从启动命令到训练产出
---------------------------------

最后把上面的设计落回实际代码仓库，日常会用到的配置、脚本与工具都在以下六个核心目录中。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 310" width="100%" role="img" aria-label="RLinf 仓库结构与工作流：从启动脚本与配置到核心框架、产出日志与工具箱" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rlinf-arrow-zh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
           <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Column 1: Launch & Configs -->
       <rect x="16" y="20" width="256" height="270" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8"/>
       <rect x="16" y="20" width="256" height="38" rx="12" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))"/>
       <text x="144" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">1. 启动脚本与配置</text>

       <rect x="32" y="74" width="224" height="88" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="46" y="96" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">examples/</text>
       <text x="46" y="116" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">run_*.sh 启动脚本与 Hydra YAML</text>
       <text x="46" y="134" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">具身 · SFT · 推理 · 智能体 · 扩散</text>
       <text x="46" y="152" font-size="11" fill="var(--pst-color-primary, #6d28d9)">支持命令行直接覆盖任意配置项</text>

       <rect x="32" y="178" width="224" height="94" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="46" y="200" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">evaluations/</text>
       <text x="46" y="220" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">统一评测入口 run_eval.sh</text>
       <text x="46" y="238" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">LIBERO · RoboTwin · BEHAVIOR · 真机</text>
       <text x="46" y="256" font-size="11" fill="var(--pst-color-primary, #6d28d9)">独立评估任意训练好的权重</text>

       <!-- Arrow 1 -> 2 -->
       <line x1="272" y1="155" x2="314" y2="155" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-zh)"/>

       <!-- Column 2: Core Framework -->
       <rect x="322" y="20" width="276" height="270" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.8"/>
       <rect x="322" y="20" width="276" height="38" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="460" y="44" text-anchor="middle" font-size="14" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">2. 核心框架（rlinf/）</text>

       <rect x="338" y="74" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="92" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">scheduler/ · workers/ · runners/</text>
       <text x="352" y="108" font-size="11" fill="var(--pst-color-text-muted, #475569)">Cluster、Worker、Channel 与硬件抽象</text>

       <rect x="338" y="126" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="144" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">models/ · hybrid_engines/</text>
       <text x="352" y="160" font-size="11" fill="var(--pst-color-text-muted, #475569)">VLA/LLM 模型、FSDP、Megatron 与 SGLang</text>

       <rect x="338" y="178" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="196" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">envs/ · robotics/</text>
       <text x="352" y="212" font-size="11" fill="var(--pst-color-text-muted, #475569)">仿真器、积木式真机接口与遥操作</text>

       <rect x="338" y="230" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="248" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">algorithms/ · data/</text>
       <text x="352" y="264" font-size="11" fill="var(--pst-color-text-muted, #475569)">PPO、GRPO、SAC、奖励与数据集加载</text>

       <!-- Arrows 2 -> 3 -->
       <line x1="598" y1="96" x2="640" y2="96" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-zh)"/>
       <line x1="598" y1="218" x2="640" y2="218" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="2" marker-end="url(#rlinf-arrow-zh)"/>

       <!-- Column 3 Top: Run Outputs -->
       <rect x="648" y="20" width="256" height="126" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-admon-tip-border, #10b981)" stroke-width="1.8"/>
       <rect x="648" y="20" width="256" height="36" rx="12" fill="var(--rlinf-admon-tip-bg, #ecfdf5)"/>
       <text x="776" y="43" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--rlinf-admon-tip-border, #10b981)">3. 运行产出（logs/）</text>
       <text x="664" y="78" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• TensorBoard / W&amp;B / SwanLab 曲线</text>
       <text x="664" y="102" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• 模型 checkpoint 与 rollout 视频</text>
       <text x="664" y="126" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• LeRobot 与 Replay Buffer 分片</text>

       <!-- Column 3 Bottom: Setup & Toolkits -->
       <rect x="648" y="162" width="256" height="128" rx="12" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.8"/>
       <rect x="648" y="162" width="256" height="36" rx="12" fill="var(--rlinf-surface-muted, #f1f5f9)"/>
       <text x="776" y="185" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">4. 环境与实用工具</text>
       <text x="664" y="220" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• requirements/ 与 docker/</text>
       <text x="664" y="244" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• toolkits/（数据、自检、自动放置）</text>
       <text x="664" y="268" font-size="12" fill="var(--pst-color-text-base, #0f172a)">• tests/（单测、端到端与真机假驱动）</text>
     </svg>
   </div>

按需要点击下方卡片，直接前往对应的文档或参考页：

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 训练入口与配置（``examples/``）
      :link: ../reference/configuration
      :link-type: doc

      按任务分目录存放启动脚本与 Hydra YAML 配置：具身 RL、SFT、离线 RL、推理、智能体与扩散模型。

   .. grid-item-card:: 独立评测（``evaluations/``）
      :link: ../evaluations/index
      :link-type: doc

      统一评测入口 ``run_eval.sh`` 与各基准评测配置（LIBERO、RoboTwin、BEHAVIOR、ManiSkill、真机等）。

   .. grid-item-card:: 实用工具箱（``toolkits/``）

      按需调用的辅助脚本：LeRobot 数据处理、真机硬件自检、Replay Buffer 可视化、自动 Placement 与权重转换。

   .. grid-item-card:: 核心框架（rlinf/）
      :link: ../concepts/overview
      :link-type: doc

      Worker 与 Channel 调度、硬件抽象层、runner 循环、模型、仿真环境、积木式 Robotics 接口及算法注册表。

   .. grid-item-card:: 安装脚本与镜像（``requirements/`` 与 ``docker/``）
      :link: installation
      :link-type: doc

      一键环境构建脚本 ``install.sh`` 及覆盖八大加速卡后端的 Dockerfile。

   .. grid-item-card:: 测试与扩展（``tests/`` 与 ``docs/``）
      :link: ../extending/index
      :link-type: doc

      单元测试、端到端配置、真机假驱动以及中英文文档源码。

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
