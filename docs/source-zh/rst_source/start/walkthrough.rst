快速熟悉 RLinf
==============

用五分钟通过四张图建立对 RLinf 的整体认知：框架连接了哪些能力、一次训练如何运转、怎样从单卡扩展到多机与真机，以及日常会用到的配置和工具都在仓库的哪里。

一图看懂 RLinf 全貌
-------------------

RLinf 用一套统一的分布式运行时，对接仿真具身、真机机器人、推理智能体与图像/视频生成任务。无论切换模型、模拟器还是强化学习算法，都沿用相同的启动脚本与 Hydra YAML 配置方式。

.. raw:: html

   <div style="margin: 1.25rem 0;">
     <svg viewBox="0 0 920 348" width="100%" role="img" aria-label="RLinf 分层结构总览：任务方向、统一 YAML 入口、分布式运行时与硬件后端" style="display:block;max-width:920px;margin:0 auto;background:transparent;font-family:var(--pst-font-family-base,Inter,sans-serif);">
       <defs>
         <marker id="rlinf-down-zh" viewBox="0 0 10 10" refX="5" refY="8" markerWidth="6" markerHeight="6" orient="auto">
           <path d="M 1.5 0 L 5 8.5 L 8.5 0 z" fill="var(--pst-color-primary, #6d28d9)"/>
         </marker>
       </defs>

       <!-- Layer 1: Workloads -->
       <text x="16" y="20" font-size="12" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">1. 你要运行的任务</text>
       <rect x="16" y="30" width="212" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="30" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">仿真具身 RL / SFT</text>
       <text x="30" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">LIBERO · ManiSkill · RoboTwin</text>
       <text x="30" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">OpenPI · OpenVLA · GR00T</text>

       <rect x="242" y="30" width="212" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="256" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">真机机器人</text>
       <text x="256" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Franka · SO-101 · ARX X5 · R2</text>
       <text x="256" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">遥操作 · 数据采集 · SFT · 在线 RL</text>

       <rect x="468" y="30" width="212" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="482" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">智能体与推理 RL</text>
       <text x="482" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Qwen · Search-R1 · WideSeek</text>
       <text x="482" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">数学推理 · 工具调用 · 代码 RL</text>

       <rect x="694" y="30" width="210" height="78" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.6"/>
       <text x="708" y="54" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">图像与视频生成</text>
       <text x="708" y="74" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">SD3.5 Medium · Wan2.2</text>
       <text x="708" y="92" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Diffusion-NFT · ART · ReFL</text>

       <!-- Connector bar: Unified Interface -->
       <line x1="460" y1="108" x2="460" y2="128" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rlinf-down-zh)"/>
       <rect x="16" y="132" width="888" height="36" rx="9" fill="var(--rlinf-nav-active-bg, rgba(109,40,217,0.1))" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.4"/>
       <text x="460" y="155" text-anchor="middle" font-size="13" font-weight="700" fill="var(--pst-color-primary, #6d28d9)">统一使用入口：  run_*.sh  +  Hydra YAML 配置（命令行直接覆盖任意参数）</text>
       <line x1="460" y1="168" x2="460" y2="188" stroke="var(--pst-color-primary, #6d28d9)" stroke-width="1.8" marker-end="url(#rlinf-down-zh)"/>

       <!-- Layer 2: Unified Runtime -->
       <text x="16" y="198" font-size="12" font-weight="700" fill="var(--pst-color-text-muted, #64748b)">2. RLinf 统一分布式运行时</text>
       <rect x="16" y="208" width="286" height="74" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="32" y="232" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">交互与采样（Env &amp; Rollout）</text>
       <text x="32" y="252" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">仿真器 · 真机控制 · Agent 沙箱</text>
       <text x="32" y="270" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">Hugging Face · SGLang · vLLM</text>

       <rect x="317" y="208" width="286" height="74" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="333" y="232" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">算法与数据（Algorithms &amp; Data）</text>
       <text x="333" y="252" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">PPO · GRPO · SAC · DAPO · SFT</text>
       <text x="333" y="270" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">奖励模块 · Replay Buffer · LeRobot</text>

       <rect x="618" y="208" width="286" height="74" rx="10" fill="var(--rlinf-surface, #ffffff)" stroke="var(--rlinf-border-strong, #cbd5e1)" stroke-width="1.6"/>
       <text x="634" y="232" font-size="13" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">分布式训练与调度（Actor &amp; Ray）</text>
       <text x="634" y="252" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">FSDP / FSDP2 · Megatron-Core</text>
       <text x="634" y="270" font-size="11.5" fill="var(--pst-color-text-muted, #475569)">共置 · 分离 · 混合/真机异构放置</text>

       <!-- Layer 3: Hardware -->
       <rect x="16" y="300" width="888" height="38" rx="9" fill="var(--rlinf-surface-muted, #f1f5f9)" stroke="var(--rlinf-border, #e2e8f0)" stroke-width="1.4"/>
       <text x="460" y="324" text-anchor="middle" font-size="12.5" font-weight="600" fill="var(--pst-color-text-base, #0f172a)">3. 硬件后端：  NVIDIA CUDA · AMD ROCm · 华为昇腾 CANN · 摩尔线程 MUSA · Intel / 昆仑芯 / 沐曦 / 壁仞</text>
     </svg>
   </div>

一次训练是如何运转的
--------------------

执行 ``bash examples/embodiment/run_embodiment.sh <配置名>`` 后，RLinf 读取对应的 YAML 配置，在指定硬件上拉起 Ray worker 组，循环执行采样、打分、更新与权重同步四个环节：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/core_dataflow_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 核心训练闭环：环境、Rollout、奖励与缓冲区、Actor 训练及权重同步

   单次训练迭代的数据闭环与权重同步流向。

图里的每个模块都直接对应 YAML 配置文件里的一个顶层字段块：

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - 环节
     - 对应配置块
     - 负责内容
   * - 环境 / Agent
     - ``env.*`` 或 ``agentloop.*``
     - 驱动仿真器、真机机器人或工具沙箱，产出观测、奖励与结束信号。
   * - Rollout
     - ``rollout.*``
     - 用 Hugging Face、SGLang 或 vLLM 执行 policy 推理，生成动作或文本回复。
   * - 奖励与算法
     - ``algorithm.*``、``reward.*``
     - 计算规则或模型奖励、估计 PPO / GRPO 优势，或从 Replay Buffer 采样批次。
   * - Actor 训练
     - ``actor.*``
     - 用 FSDP 或 Megatron 更新权重，同步回 Rollout，并保存 checkpoint。

.. tip::

   切换模型权重时只需设置 ``actor.model.model_path``\ ，rollout 会自动沿用这一路径，无需在两处重复填写。

从单卡扩展到多机与真机
----------------------

硬件规模变化时无需改动训练脚本或算法代码，只要调整 YAML 中的 ``cluster.component_placement``\ ，就能把同一套 worker 切换到三种部署形态：

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/deployment_modes_zh.svg
   :align: center
   :width: 100%
   :alt: RLinf 三种部署模式对比：Collocated 共置、Disaggregated 分离与 Hybrid 异构部署

   同一套 worker 的三种硬件放置方式：共置、分离与混合/真机异构部署。

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - 模式
     - 常见规模
     - 适用场景
   * - Collocated 共置
     - 1–8 卡
     - 单卡与单机默认模式，各组件共享 GPU 并配合 offload 运行。
   * - Disaggregated 分离
     - 8 卡以上
     - 各组件独占 GPU，消除 offload 停顿，适合流水线并行与异步 RL。
   * - Hybrid 混合 / 真机
     - 多节点或真机 + GPU
     - 把共享 GPU 的训练组件与独立 CPU 节点或真机控制机灵活组合。

仓库结构：从启动命令到训练产出
------------------------------

下图按日常工作流串起 RLinf 的主要目录：从启动脚本和 YAML 配置进入核心框架，再连接到训练日志、checkpoint 与各类辅助工具。

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
       <text x="352" y="108" font-size="11" fill="var(--pst-color-text-muted, #475569)">Ray 集群调度、placement 与训练循环</text>

       <rect x="338" y="126" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="144" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">models/ · hybrid_engines/</text>
       <text x="352" y="160" font-size="11" fill="var(--pst-color-text-muted, #475569)">VLA/LLM 模型、FSDP、Megatron 与 SGLang</text>

       <rect x="338" y="178" width="244" height="42" rx="8" fill="var(--rlinf-surface-subtle, #f8fafc)" stroke="var(--rlinf-border, #e2e8f0)"/>
       <text x="352" y="196" font-size="12" font-weight="700" fill="var(--pst-color-text-base, #0f172a)">envs/ · robotics/</text>
       <text x="352" y="212" font-size="11" fill="var(--pst-color-text-muted, #475569)">各类仿真器、真机硬件抽象与遥操作</text>

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

      分布式调度与 worker、runner 循环、模型、仿真与真机环境、机器人硬件抽象及算法注册表。

   .. grid-item-card:: 安装脚本与镜像（``requirements/`` 与 ``docker/``）
      :link: installation
      :link-type: doc

      一键环境构建脚本 ``install.sh`` 及各硬件平台 Dockerfile。

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

   .. grid-item-card:: 具身训练
      :link: ../examples/index
      :link-type: doc

      按模拟器、VLA 模型、SFT 或世界模型选择训练配方。

   .. grid-item-card:: 真机机器人
      :link: ../examples/real_world_index
      :link-type: doc

      从硬件配置、遥操作、数据采集走到 SFT 与在线 RL。

   .. grid-item-card:: 智能体与生成
      :link: ../examples/agentic/index
      :link-type: doc

      训练数学推理大模型、多轮工具智能体与扩散模型。
