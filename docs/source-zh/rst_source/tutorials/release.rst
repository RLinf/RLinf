版本说明
========

RLinf v0.2 版本发布
-------------------

🎉 我们发布 RLinf v0.2。

RLinf v0.2 聚焦两个核心方向：**真实世界强化学习（Real-World RL）**与**多智能体强化学习系统（Multi-agent RL）**。围绕这一目标，RLinf 现已支持包括 **XSquare Turtle2 Arms** 与 **Franka Arm** 在内的真实平台，并提供更丰富的具身基准、模拟器、模型与算法组合，以及面向高吞吐场景的**原生异步训练能力**。本次发布还进一步增强了 real-world、sim-to-real、协同训练流程，以及数据与 replay 基础设施的鲁棒性和训练稳定性。面向多智能体训练，RLinf 提供了**原生且可扩展的多智能体算法与统一数据接口**，降低了开发与扩展门槛，并可快速复现如 **WideSeek-R1** 这类先进训练方案。

具身智能
^^^^^^^^

1. 核心能力升级（重点：真实机器人 RL 与世界模型）

- 支持基于 :doc:`XSquare Turtle2 <../examples/embodied/xsquare_turtle2>` Arms 的真实世界强化学习

- 支持将世界模型作为 RL 训练模拟器，包括 :doc:`OpenSora <../examples/embodied/opensora>`、:doc:`Wan <../examples/embodied/wan>`、:doc:`WoVR <../publications/wovr>`

- :doc:`SFT VLM <../examples/embodied/sft_vlm>` 能力支持在自定义数据集上高效微调，并在 Robo2VLM 数据集上验证 Qwen2.5‑VL 与 Qwen3‑VL 约 95% 的复现精度，详见 `PR 708 <https://github.com/RLinf/RLinf/pull/708>`_ 与 `PR 781 <https://github.com/RLinf/RLinf/pull/781>`_

- 支持基于 :doc:`GSEnv <../examples/embodied/gsenv>` 的 Real2Sim2Real 训练（ManiSkill-GS）

- 支持 π(0.5) 模型的 :doc:`Co-training <../examples/embodied/co_training>` sim-real 协同训练

1. 模型与算法生态扩展

- 支持 :doc:`Dexbotic <../examples/embodied/dexbotic>` 模型及其 RL 训练

- 增强 :doc:`IsaacLab <../examples/embodied/isaaclab>` 支持，尤其是 gr00t+isaaclab

- 支持 openpi 模型家族在 :doc:`RoboTwin <../examples/embodied/robotwin>` 2.0 上训练

- 支持 :doc:`CALVIN <../examples/embodied/calvin>` 基准

- 支持 :doc:`RoboCasa <../examples/embodied/robocasa>` 基准

- 支持 Pi0 的 :doc:`DSRL <../examples/embodied/dsrl>`，即 Diffusion Steering via Reinforcement Learning

- 支持 flow matching policy 的 :doc:`SAC-Flow <../examples/embodied/sac_flow>` 训练

1. 训练基础设施增强

- 在数据层新增 wrapper，支持真实机器人与模拟器数据的标准化 replay buffer 采集（见 :doc:`Data collection <components/data_collection>`）

- 异步训练支持：将异步训练作为具身模型的一等能力，提供 :doc:`Async PPO <rlalg/async_ppo>` 工作流与易用性改进，提升高吞吐场景效率

- 数据与 Replay 流水线升级：增强 buffer 预加载、更新与 checkpoint 处理能力，提升整体鲁棒性（见 :doc:`Replay buffer API <../apis/replay_buffer>`）

- 运行时性能优化：引入 CUDA Graph、torch.compile、环境 offload 与 FSDP 路径优化，提升执行效率（见 :doc:`YAML configuration <user/yaml>`）

1. 稳定性改进与可用性提升

- 对 PPO/GRPO 行为与真实世界配置处理进行了多项修复，增强训练稳定性与配置正确性。

- 在 YAML 配置中补充 HuggingFace 模型链接，便于下载。

Agentic 与 Reasoning RL
^^^^^^^^^^^^^^^^^^^^^^^

1. 核心能力升级（重点：多智能体 RL）

- 原生多智能体训练支持：引入可扩展多智能体算法与统一数据接口，显著降低任务门槛，可快速复现复杂方案（如 :doc:`WideSeek-R1 <../examples/agentic/wideseek_r1/index>`），详见 `PR 824 <https://github.com/RLinf/RLinf/pull/824>`_

- 推理任务 PPO 支持：将 PPO 能力扩展到推理场景（见 :doc:`Reasoning PPO <../examples/agentic/reasoning_ppo>`），详见 `PR 771 <https://github.com/RLinf/RLinf/pull/771>`_

- Megatron-LM 后端支持 FUSCO 通信库，提升 MoE 训练/推理阶段 All-to-All 通信性能（见 :doc:`FUSCO <../examples/system/fusco>`），详见 `PR 783 <https://github.com/RLinf/RLinf/pull/783>`_

- 支持 :doc:`rStar2 <../examples/agentic/rstar2>` 与 :doc:`Search-R1 <../examples/agentic/searchr1>` 的 agentic RL，详见 `PR 522 <https://github.com/RLinf/RLinf/pull/522>`_ 与 `PR 639 <https://github.com/RLinf/RLinf/pull/639>`_

其他改进与缺陷修复
^^^^^^^^^^^^^^^^^^

- 重构 init_worker 与权重同步流程，并支持在 agent 循环内计算奖励，减少独立 reward worker 依赖：`PR 524 <https://github.com/RLinf/RLinf/pull/524>`_

- FSDP 后端支持动态 batch size、数据并行负载均衡、fp16 梯度缩放（unscale_patch）与多桶权重同步：`PR 553 <https://github.com/RLinf/RLinf/pull/553>`_

- 修复数据集读取问题并加入 tokenizer 批量编码，提升吞吐：`PR 653 <https://github.com/RLinf/RLinf/pull/653>`_

- 支持将 Qdrant 作为 Wiki server，提升向量检索与文档存储能力：`PR 673 <https://github.com/RLinf/RLinf/pull/673>`_

- 重构 FSDP 精度处理，仅保留 fp16/bf16，替代模糊 AMP 结构并保持兼容：`PR 715 <https://github.com/RLinf/RLinf/pull/715>`_

- 修复推理任务中 rollout 拆分导致的跨 rank batch 不一致问题：`PR 775 <https://github.com/RLinf/RLinf/pull/775>`_

- 修复 pi 模型训练中的 epsilon 配置错误：`PR 623 <https://github.com/RLinf/RLinf/pull/623>`_

- 修复 openpi 模型需手动关闭梯度检查点的问题：`PR 843 <https://github.com/RLinf/RLinf/pull/843>`_

- 修复 Franka Arm Docker 镜像：`PR 862 <https://github.com/RLinf/RLinf/pull/862>`_

- 修复 SAC actor worker 中 send_num 使用错误 world size 的问题：`PR 882 <https://github.com/RLinf/RLinf/pull/882>`_

- 修复第二轮 rollout 收到随机 reset_state_ids 的问题：`PR 886 <https://github.com/RLinf/RLinf/pull/886>`_

- 修复环境初始化后 offloading 行为与 rollout 中 actor 预留显存释放问题：`PR 897 <https://github.com/RLinf/RLinf/pull/897>`_

文档更新
^^^^^^^^

- 重构 :doc:`examples index <../examples/index>` 结构，按 embodied/agentic/system 分类

- 在 :doc:`FAQ <../faq>` 中新增断点调试文档

- 在 README 新增 awesome work 与 adoption 板块。

贡献者
~~~~~~

@Hao Lin @qurakchin @guozhen1997 @zanghz21 @Bo Dai @FxxxxU @Elessar123 @LiuYiwei @Xzxuan @ysa @jzndd @zlock @xusi @Louis-J @WinstonWmj @shiletong @xzxuan @liyanghao @Iron_Wph @chenkang455 @shengyz @yimingzhou2002 @Florielle @xuxin @Yinuo Chen @nufukim @Lin-xs @zhangruize @Iron-Wph @Hongyi Zhu @red0orange @chenkang @hongzhi @thereAreDemonsNearby @Zoran Zhu @Tziy @Yimingzhou2002 @Nan Yang @AIhuaYuan @AIhuayuan @xuxin @MacBook-M3-Pro @wangxiangyuan @slzhta @Iron-Wph @fy2462 @Ning Xu @weimingjie @zlockewtg @smallcracker @gongyue teng @cc @Xin Xu @xiebin @yuyingyinya @Yun Liu @Tao Liu @renqian @Wheels Wu @Wheeeeeeeeels @Felix Zhang @pyy233 @LiuZhihao2022

RLinf v0.2 测试结果
~~~~~~~~~~~~~~~~~~~

我们测试了大多数配置文件，以保证本次发布中所提供示例的正确性。

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - 配置文件
     - 模型名称
     - 结果曲线
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


RLinf v0.1 版本发布
-------------------

🎉 我们发布 RLinf v0.1。

RLinf 构建于稳健的系统级调度与通信组件之上，是一个面向具身、推理与 agent 场景的可扩展、高灵活强化学习后训练框架。框架已在多种主流模型与任务上完成验证，并在模型效果与训练吞吐上达到业界领先水平，体现了其在多场景下的可扩展性、通用性与效率优势。

具身智能
^^^^^^^^

- 支持在多个主流模拟器（如 ManiSkill、Libero、MetaWorld、CALVIN）上开展端到端具身 RL 训练，结合多种 VLA 模型（如 OpenVLA、OpenVLA-OFT、π₀、π₀.₅、GR00T）与算法（GRPO、PPO）可达最高 99% 的成功率。

- 相较现有框架，训练速度最高提升 143.4%（2.434× 吞吐），并支持灵活的分配式、解耦式与混合执行模式，可平滑扩展至上千 GPU。

- 已在 ManiSkill、LIBERO、MetaWorld 与 CALVIN 上验证训练有效性，并在具身示例中提供可复现的最佳实践脚本。

Agent 与推理 RL
^^^^^^^^^^^^^^^

- 在 1.5B 与 7B 模型规模下，RLinf 在 AIME 24、AIME 25 与 GPQA-Diamond 基准上取得了领先结果。

- 通过流水线并行（20%+）与自动调度（30%+），RLinf 在效率与推理能力方面均取得显著提升。

- 支持基于 Megatron 的训练自动调度与弹性扩缩；SGLang/vLLM 与 Megatron 可在训练中自动缩放以达到最大吞吐，相比静态放置可带来 40%+ 加速。

- 发布首个开源 1.5B 在线 RL Agent，代码补全准确率提升 50%+，性能超过部分 32B 规模模型。
