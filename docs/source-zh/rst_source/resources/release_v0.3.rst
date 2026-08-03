版本说明
==========

RLinf v0.3 版本发布
---------------------

🎉 我们发布 RLinf v0.3。

本版本打通完善的真机训练全流程、新增真机强化学习组件及算法，仿真强化学习新增更多的仿真器和 SOTA 模型。该版本对已支持的各类examples进行了严格的正确性与可复现性验证（见release note末尾）。

具身
^^^^^^

1. 模型
"""""""""

继续扩展模型生态，新增 6 款具身模型支持，涵盖世界模型、VLA 模型及系统级加速

- 新增 **Dexbotic DM0** 模型支持，在 LIBERO 上用 PPO 进行在线 RL 微调。链接：`Dexbotic <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dexbotic.html>`__

- 新增 **DreamZero** 模型支持：基于 WAN2.1/2.2 视频生成世界模型微调的 VLA 策略，集成进 SFT 工作流，通过 FSDP2/CUDA Graph 等系统级加速取得近 **4×** 吞吐提升。链接：`DreamZero SFT <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/sft_dreamzero.html>`__

- 新增 **GR00T-N1.6 / N1.7** 模型 RL 微调支持。链接：`GR00T <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/gr00t.html>`__

- 新增 **ABot-M0** 模型支持。链接：`ABot-M0 <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/abot_m0.html>`__

- 新增 **StarVLA** 模型支持（GRPO on LIBERO）。链接：`StarVLA <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/starvla.html>`__

- 新增 **LingBot-VLA** 模型支持（RoboTwin 环境 SFT/RL）。链接：`LingBot-VLA <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/lingbotvla.html>`__

2. 仿真器
"""""""""""

提升仿真强化学习的场景覆盖，新增 5 种仿真器，完善基于仿真器的训练示例与效果

- 新增 **Genesis** 仿真器支持。链接：`Genesis <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/genesis.html>`__

- 新增 **Polaris** 仿真器支持。链接：`Polaris <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/polaris.html>`__

- 新增 **RoboVerse** 仿真器支持。链接：`RoboVerse <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/roboverse.html>`__

- 完善 **Behavior** 环境支持：新增 v3.7.1 / v3.7.2 版本补丁、π0.5 PPO 配置与 object/pose randomization。链接：`Behavior <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/behavior.html>`__

- 新增 **Libero+ / LiberoPro** 变体环境支持。链接：`LIBERO <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/libero.html>`__

- 新增 **Embodichain（CartPole）** 环境支持。链接：`Embodichain <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/embodichain.html>`__

- 新增 **IsaacLab 上 π0.5 PPO finetuning** 支持。链接：`IsaacLab <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/isaaclab.html>`__

- 新增 **RoboCasa** close-drawer 等 RL 示例支持。链接：`RoboCasa <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/robocasa.html>`__

3. 真机
"""""""""

全面打通数据采集 → SFT → RL → 真机部署的闭环链路，新增 3 种遥操作方式、3 款真机平台、2 款末端执行器，真机实操能力显著增强

**数据采集支持：**

- 新增 **空间鼠标（Spacemouse）** 遥操作数据采集支持。链接：`Franka <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka.html>`__

- 新增 **VR 遥操作** 数据采集支持。链接：`Franka VR <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_vr.html>`__

- 新增 **GELLO 遥操作** 数据采集支持。链接：`Franka GELLO <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_gello.html>`__

**训练链路支持：**

- 新增 **LeRobot 格式数据采集** 支持，便于与 HuggingFace LeRobot 生态互通。链接：`Data Collection <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/data_collection.html>`__

- 新增 **Pi0 真机 SFT 部署** 支持，打通数据采集 → SFT → 真机部署链路。链接：`Franka Pi0 SFT Deploy <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_pi0_sft_deploy.html>`__

- 新增 **真机 reward model 数据采集** 支持（采集带标注 reward 训练数据）。链接：`Franka Reward Model <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_reward_model.html>`__

**真机平台与末端支持：**

- 新增 **双臂 Franka** 平台支持（关节空间与 TCP/rot6d 控制、数据采集、SFT、部署）。链接：`Dual Franka <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dual_franka.html>`__

- 新增 **GimArm** 真机平台支持。链接：`GimArm <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/gim_arm.html>`__

- 新增 **DOS-W1** 真机平台支持。链接：`DOS-W1 <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dosw1.html>`__

- 新增 **Franka DexHand 灵巧手** 末端执行器支持。链接：`Franka DexHand <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_dexhand.html>`__

- 新增 **Franka Robotiq** 夹爪后端支持。链接：`Franka Robotiq <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_zed_robotiq.html>`__

- 新增 **Franka Robotiq 及 ZED / LUMOS V4L2** 相机与夹爪后端支持。链接：`Franka ZED/Lumos <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/franka_zed_robotiq.html>`__

4. 算法
"""""""""

在真机 RL、仿真 RL 和人在环学习三个方向均有重要算法新增，表现出SOTA的真机任务成功率

**真机强化学习算法：**

- 将 **DSRL** （Diffusion Steering via Reinforcement Learning）扩展至 Pi0.5 模型。链接：`DSRL <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dsrl.html>`__

- 新增 **RECAP** （基于离线优势估计的策略优化）训练流水线支持。链接：`RECAP <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/recap.html>`__

- 新增 **SAC-Flow** 算法支持，并扩展到 DOS-W1 等真机场景。链接：`SAC-Flow <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/sac_flow.html>`__

**仿真强化学习算法：**

- **Async PPO**：在 v0.2 基础上扩展支持 MLP 等新策略，并新增 async DSRL 配置。链接：`Async PPO <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/reference/algorithms/async_ppo.html>`__

- 新增 **Pi-StepNFT** 算法支持。

- 新增 **D4RL 离线 IQL** 训练支持（Antmaze / Kitchen-Adroit / MuJoCo，基于 FSDPStrategy）。链接：`IQL-D4RL <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/iql_d4rl.html>`__

**人在环学习：**

- 新增 **DAgger** 在线模仿学习算法支持（LIBERO、ManiSkill、RoboTwin、真机 PnP 多场景）。链接：`DAgger <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/dagger.html>`__

- 新增 **HG-DAgger** （Human-Gated DAgger）真机在线训练支持。链接：`HG-DAgger <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/hg-dagger.html>`__

5. 系统
"""""""""

系统层面新增多项性能优化技术，完善昇腾、AMD ROCm、Musa 等加速芯片的支持，整体系统健壮性和可扩展性大幅提升

**新增组件支持：**

- 新增 **Reward Model** 组件支持：embodied reward worker + ResNet/VLM reward model，支持 standalone reward 用于 realworld env。链接：`Reward Model <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/reward_model.html>`__

- 新增 **Value Model** 组件支持：通用 value model 基础设施，支撑 RECAP 等流水线。链接：`RECAP <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/recap.html>`__

- 新增 **SGLang 推理 server 化** 组件支持（HTTP server + router 模式，可作 reward 服务/rollout 推理后端）。链接：`SGLang Server <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/sglang_server.html>`__

- 新增 **Decoupled env 模式** 组件支持（解除 Env Worker 与 Rollout Worker 一对一绑定，提升 rollout GPU 利用率）。链接：`Env Decoupled Mode <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/env_decoupled_mode.html>`__

**性能与显存优化支持：**

- 新增 **π0 / π0.5 predict 的 torch.compile** 加速支持。

- 新增 **rollout 与训练 overlap** 支持（含 bootstrap-training overlap 与 embodied pipeline 下的 advantage normalization）。

- 新增 **权重同步升级**：基于 broadcast 的 weight sync、weight diff patch 增量同步、分桶（bucket）同步、仅同步 trainable params 与 buffers、async wait。链接：`Weight Syncer <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/weight_syncer.html>`__

- 新增 **FSDP 全 offload** 支持，并修复 checkpoint/SFT dataloader resume、actor offload 状态恢复与 GPU 显存泄漏。

- 新增 **nsys trace、统一 accelerator profiling、metrics logging file** 等运行时与 profiling 支持。链接：`GPU Profiling <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/profile.html>`__

**国产卡与跨硬件支持：**

- 新增 **昇腾 Ascend（CANN / torch-npu）** 端到端可运行支持（``install.sh --platform ascend``、``agentic-rlinf0.3-libero-cann9.0`` CANN Docker 镜像）。链接：`Ascend CANN <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/ascend_cann.html>`__

- 新增 **Musa** 设备上运行世界模型 Wan RL 的支持。

- 新增 **AMD ROCm** 端到端可运行支持（``install.sh --platform amd``，自动探测 ROCm 版本并匹配 ``+rocm`` wheel）。链接：`AMD ROCm <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/amd_rocm.html>`__

**配置与调度：**

- 新增 **自定义模型注册** 与 **override cfgs** 支持，提升配置灵活性与可扩展性。链接：`New Model (FSDP) <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/new_model_fsdp.html>`__

- 新增 **基于 Ray cluster 的代码同步** 支持（``RLINF_CODE_WORKING_DIR``，不共享文件系统时自动下发 ``rlinf/`` 包）。

- 新增 **SFT 工作流重构**：统一 SFT loss/metrics API，修复 SFT 数据加载 resume。

Agentic AI
^^^^^^^^^^^^

为智能体 RL 场景提供了更强大的训练和评测基础

- 新增 **AgentLightning 多轮单智能体 RL 训练** 与 Calc-X 评测支持。链接：`AgentLightning Calc-X <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/agentic/agentlightning_calc_x.html>`__

- 新增 **Megatron-Bridge actor 后端** 支持（基于 Megatron-mbridge 模型的 RL 训练与 SFT）。链接：`Megatron-Bridge <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/extending/mbridge.html>`__

- 将 **SearchR1** 重构为多轮接口，并新增 WideSeek judge 的内置 sglang 支持。

论文
^^^^^^

有 2 篇论文被 **OSDI 2026** 收录：

- **RLinf: Flexible and Efficient Large-Scale Reinforcement Learning via Macro-to-Micro Flow Transformation** （OSDI 2026）。对应 RLinf 大规模 RL 系统。文档：`RLinf 系统 <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/resources/publications/rlinf_system.html>`__ ｜ 论文：`arXiv:2509.15965 <https://arxiv.org/abs/2509.15965>`__ ｜ `OSDI 报告 <https://www.usenix.org/conference/osdi26/presentation/yu-chao>`__。

- **DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training** （OSDI 2026）。对应 RLinf 的 dynamic scheduling 功能。文档：`Dynamic Scheduling <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/guides/dynamic_scheduling.html>`__ ｜ `OSDI 报告 <https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing>`__。

另有 2 篇论文被 **RSS 2026** 收录：

- **USER: A Unified and Extensible System for Online Real-World Policy Learning in Embodied AI** （RSS 2026，即 RLinf-USER）。对应 RLinf 真机在线策略学习系统。文档：`RLinf-USER <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/resources/publications/rlinf_user.html>`__ ｜ 论文：`arXiv:2602.07837 <https://arxiv.org/abs/2602.07837>`__ ｜ `RSS 论文页 <https://roboticsconference.org/program/papers/37/>`__。

- **RLux-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models** （RSS 2026，即 RLinf-VLA）。对应 RLinf 的 VLA+RL 统一框架。文档：`RLinf-VLA <https://rlinf.readthedocs.io/en/release-v0.3/rst_source/resources/publications/rlinf_vla.html>`__ ｜ 论文：`arXiv:2510.06710 <https://arxiv.org/abs/2510.06710>`__ ｜ `RSS 论文页 <https://roboticsconference.org/program/papers/89/>`__。

重要修复
^^^^^^^^^^

v0.3 修复了若干影响训练稳定性与数据/采集正确性的问题，**建议大家升级到最新版以获取这些修复**。主要包括：

- 修复 behavior env 中纹理缺失/模糊、资产在 config validation 阶段提前加载、TRO state 的 dump/load 不对称等问题。

- 修复 openpi 评测工具链在读取配置字典时的导入错误。

- 修复 openpi 模型 gradient checkpointing 需手动关闭的问题。

- 修复轨迹拆分后发往 actor 时的返回类型错误。

- 统一夹爪动作（gripper action）格式，并修复数据采集时夹爪初始开合状态错误。

- 修复 maniskill 在环境 offload 后视频计数器状态过期的问题。

- 修复 SAC actor worker 中 send_num 误用 world size 的问题。

- 修复 env 初始化后未正确触发 offload 的问题，以及 rollout 时 actor 预留显存未释放的问题。

- 修复 CUDA IPC memory 通信后未回收、broadcast 未约束同 device、AMD GPU 可见设备环境变量配置等系统侧问题。

- 修复 weight sync 与 actor barrier 间的死锁。

- 修复 FSDP 的 checkpoint resume、actor offload 状态恢复与 GPU 显存泄漏等问题。

贡献者
^^^^^^^^

@andylin-hao @guozhen1997 @zhexuanxu @anHappyDog @Brunch-Life @thereAreDemonsNearby @yushuang20091011 @qurakchin @zanghz21 @F9rozen @FxxxxU @jx-qiu @Lin-xs @tiny-xie @lwbscu @QuanluZhang @kunni918 @Iron-Wph @secretsites @ligediaomao @ZhaoRunyi @duzhengye-droid @fy2462 @matthewmzy @chenkang455 @weiyunfei @XuS1994 @pikaxinge @drewzhao @WayneTimer @Matrix326 @pancake-w @lizuojun04 @MrHappa @HzfFrank @renq-mt @liuhaoyunBUPT @yxuan1234 @crabxiexy @MuggleZzzH @ppppppppppper @xb534 @zhigenzhao @wingAGI @aasivas @git-xuxin @LiuZhihao2022 @pyy233 @Dps799 @yangchen73 @jeis4wpi @NLC2004 @AIhuaYuan @zjk-prog @YimingZhou2002 @Walkism @slzhta @iamxjy @YifWRobotics @AlphaReimu @hongyuxiyohung @WinstonWmj @jzndd @Elessar123

RLinf v0.3 测试结果
^^^^^^^^^^^^^^^^^^^^^

我们测试了大多数配置文件，以保证本次发布中所提供示例的正确性。

.. list-table::
   :header-rows: 1
   :widths: 22 22 40

   * - 配置文件
     - 模型名称
     - 结果曲线
   * - maniskill_ppo_openpi.yaml
     - RLinf-Pi0-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openpi.png
          :alt: maniskill_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openpi_pi05.yaml
     - RLinf-Pi05-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openpi_pi05.png
          :alt: maniskill_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openvla.png
          :alt: maniskill_ppo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-ManiSkill-Base-Main
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_openvlaoft.png
          :alt: maniskill_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_ppo_mlp.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_ppo_mlp.png
          :alt: maniskill_ppo_mlp.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_grpo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_grpo_openvla.png
          :alt: maniskill_grpo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_grpo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-ManiSkill-Base-Main
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_grpo_openvlaoft.png
          :alt: maniskill_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - libero_goal_ppo_openpi.yaml
     - RLinf-Pi0-LIBERO-130-fullshot-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_goal_ppo_openpi.png
          :alt: libero_goal_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - libero_goal_ppo_openpi_pi05.yaml
     - RLinf-Pi05-LIBERO-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_goal_ppo_openpi_pi05.png
          :alt: libero_goal_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_10_ppo_gr00t.yaml
     - RLinf-Gr00t-SFT-Long
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_10_ppo_gr00t.png
          :alt: libero_10_ppo_gr00t.yaml result curve
          :width: 95%
          :align: center
   * - calvin_abcd_d_ppo_openpi_pi05.yaml
     - RLinf-Pi05-CALVIN-ABC-D-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/calvin_abcd_d_ppo_openpi_pi05.png
          :alt: calvin_abcd_d_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_place_empty_cup_ppo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/robotwin_place_empty_cup_ppo_openvlaoft.png
          :alt: robotwin_place_empty_cup_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_beat_block_hammer_grpo_openvlaoft.yaml
     - RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/robotwin_beat_block_hammer_grpo_openvlaoft.png
          :alt: robotwin_beat_block_hammer_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - isaaclab_franka_stack_cube_ppo_gr00t.yaml
     - RLinf-Gr00t-SFT-Stack-cube
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/isaaclab_franka_stack_cube_ppo_gr00t.png
          :alt: isaaclab_franka_stack_cube_ppo_gr00t.yaml result curve
          :width: 95%
          :align: center
   * - gsenv_ppo_openpi_pi05.yaml
     - RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/gsenv_ppo_openpi_pi05.png
          :alt: gsenv_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - frankasim_ppo_mlp.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/frankasim_ppo_mlp.png
          :alt: frankasim_ppo_mlp.yaml result curve
          :width: 95%
          :align: center
   * - frankasim_sac_cnn_async.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/frankasim_sac_cnn_async.png
          :alt: frankasim_sac_cnn_async.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openpi.yaml
     - RLinf-Pi0-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openpi.png
          :alt: maniskill_async_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openpi_pi05.yaml
     - RLinf-Pi05-ManiSkill-25Main-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openpi_pi05.png
          :alt: maniskill_async_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openvla.yaml
     - openvla-7b
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openvla.png
          :alt: maniskill_async_ppo_openvla.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_async_ppo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-trajall
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_async_ppo_openvlaoft.png
          :alt: maniskill_async_ppo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_sac_mlp.png
          :alt: maniskill_sac_mlp.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_async_ppo_openpi.yaml
     - RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_async_ppo_openpi.png
          :alt: libero_spatial_async_ppo_openpi.yaml result curve
          :width: 95%
          :align: center
   * - libero_object_async_ppo_openpi_pi05.yaml
     - RLinf-Pi05-LIBERO-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_object_async_ppo_openpi_pi05.png
          :alt: libero_object_async_ppo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_grpo_openpi_pi05.yaml
     - RLinf-Pi05-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_grpo_openpi_pi05.png
          :alt: libero_spatial_grpo_openpi_pi05.yaml result curve
          :width: 95%
          :align: center
   * - libero_10_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero10-traj1
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_10_grpo_openvlaoft.jpg
          :alt: libero_10_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - opensora_libero_spatial_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero-spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/opensora_libero_spatial_grpo_openvlaoft.jpg
          :alt: opensora_libero_spatial_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - wan_libero_spatial_grpo_openvlaoft.yaml
     - Openvla-oft-SFT-libero-spatial
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/wan_libero_spatial_grpo_openvlaoft.jpg
          :alt: wan_libero_spatial_grpo_openvlaoft.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen2_5_vl_sft_vlm.yaml
     - Qwen/Qwen2.5-VL-3b-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen2_5_vl_sft_vlm.png
          :alt: examples/sft/config/qwen2_5_vl_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen3_vl_sft_vlm.yaml
     - Qwen/Qwen3-VL-4b-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen3_vl_sft_vlm.png
          :alt: examples/sft/config/qwen3_vl_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml
     - Qwen/Qwen2.5-1.5B-Instruct
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen2.5-1.5b-ppo-megatron.png
          :alt: examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp_resnet_reward_async.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_sac_mlp_resnet_reward_async.png
          :alt: maniskill_sac_mlp_resnet_reward_async.yaml result curve
          :width: 95%
          :align: center
   * - maniskill_sac_mlp_async_decoupled.yaml
     - None
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/maniskill_sac_mlp_async_decoupled.png
          :alt: maniskill_sac_mlp_async_decoupled.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_grpo_starvla.yaml
     - StarVLA (Qwen2.5-VL-OFT-LIBERO-4in1)
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_grpo_starvla.jpg
          :alt: libero_spatial_grpo_starvla.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_ppo_gr00t_n1d6.yaml
     - GR00T-N1.6
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_ppo_gr00t_n1d6.png
          :alt: libero_spatial_ppo_gr00t_n1d6.yaml result curve
          :width: 95%
          :align: center
   * - libero_spatial_dagger_openpi.yaml
     - RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_spatial_dagger_openpi.png
          :alt: libero_spatial_dagger_openpi.yaml result curve
          :width: 95%
          :align: center
   * - robotwin_place_shoe_grpo_lingbotvla.yaml
     - LingBot-VLA-4B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/robotwin_place_shoe_grpo_lingbotvla.png
          :alt: robotwin_place_shoe_grpo_lingbotvla.yaml result curve
          :width: 95%
          :align: center
   * - genesis_cubepick_ppo_cnn.yaml
     - RLinf-ResNet10-pretrained
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/genesis_cubepick_ppo_cnn.png
          :alt: genesis_cubepick_ppo_cnn.yaml result curve
          :width: 95%
          :align: center
   * - d4rl_iql_kitchen_adroit.yaml
     - IQL
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/d4rl_iql_kitchen_adroit_1.png
          :alt: d4rl_iql_kitchen_adroit.yaml result curve
          :width: 95%
          :align: center

       .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/d4rl_iql_kitchen_adroit_2.png
          :alt: d4rl_iql_kitchen_adroit.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/libero_sft_dreamzero_5b.yaml
     - DreamZero-5B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/libero_sft_dreamzero_5b.png
          :alt: examples/sft/config/libero_sft_dreamzero_5b.yaml result curve
          :width: 95%
          :align: center
   * - examples/sft/config/qwen2_5_vl_megatron_sft_vlm.yaml
     - Qwen2.5-VL
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/qwen2_5_vl_megatron_sft_vlm.png
          :alt: examples/sft/config/qwen2_5_vl_megatron_sft_vlm.yaml result curve
          :width: 95%
          :align: center
   * - examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn.yaml
     - Qwen2.5-1.5B
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/agentlightning_qwen2.5-1.5b-enginehttp-multiturn.png
          :alt: examples/agent/agentlightning/calc_x/config/qwen2.5-1.5b-enginehttp-multiturn.yaml result curve
          :width: 95%
          :align: center
   * - examples/agent/searchr1/config/train_qwen2.5.yaml
     - Qwen2.5
     - .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/searchr1_train_qwen2.5_1.png
          :alt: examples/agent/searchr1/config/train_qwen2.5.yaml result curve
          :width: 95%
          :align: center

       .. image:: https://github.com/RLinf/misc/raw/main/pic/release_0.3/searchr1_train_qwen2.5_2.png
          :alt: examples/agent/searchr1/config/train_qwen2.5.yaml result curve
          :width: 95%
          :align: center

快速开始
^^^^^^^^^^

- **百度智能云**：`https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1 <https://cloud.baidu.com/doc/AIHC/s/fmrenj9u1>`__
- **无问芯穹**：`https://docs.neogpu.com/posts/rlinf-ppo-vla.html <https://docs.neogpu.com/posts/rlinf-ppo-vla.html>`__
