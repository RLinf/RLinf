RLinf × WorldLoop：用零训练 VLM 奖励扩展 WoVR 世界模型强化学习
================================================================

最后更新：09/21/2026。

这篇文章记录如何在 AMD GPU 上复现 WoVR 的静态世界模型强化学习阶段，并用无需训练的 Qwen3-VL 奖励替换任务专用 reward model。世界模型让 VLA policy 在生成的视频中探索，无需在强化学习阶段持续调用物理仿真器或真实机器人；`WoVR <https://arxiv.org/abs/2602.13977>`_ 基于 RLinf 打通了这条链路，用 Wan 生成动作条件视频，再通过 GRPO 更新 OpenVLA-OFT。

WoVR 提供的两种奖励都需要训练：ResNet-18 使用仿真器特权状态生成的标签，Qwen3-VL dense reward 则需要 LoRA 和 MLP reward head。这里改用 `TOPReward <https://arxiv.org/abs/2602.19313>`_ 的读取方式，冻结 Qwen3-VL-8B-Instruct，直接取 ``" True"`` token 的 log probability 作为成功信号。

在 LIBERO-Spatial 的 500 个 MuJoCo 评测 episode 上，成功率从 0.448 提升到 0.560，与套件专用 ResNet 奖励模型的 0.574 处于同一波动区间。修复视频采样元数据后，同一份 Qwen3-VL 权重、阈值和 16 帧窗口在 LIBERO-Object 上无需重新标定，最终成功率为 0.348，与 ResNet 臂的 0.348 相同；两条臂在该套件上都没有取得可分辨的 policy 增益。

01 WoVR 仍需训练任务相关的奖励模型
------------------------------------

物理仿真器在机器人强化学习中同时提供两项能力：一是执行动作后的环境变化，即动力学；二是任务是否完成，即奖励与终止判定。WoVR 用动作条件世界模型替代前者，但仍需要单独的奖励模型提供后者。

WoVR 支持两种奖励建模方式：

- ResNet-18 sparse reward：从当前帧预测任务是否成功，训练标签来自 ``is_obj_placed`` 一类仿真器特权状态；
- Qwen3-VL dense reward：输入连续四帧与任务文本，通过 LoRA 和 MLP head 预测 0–10 的任务进度。

两种方案都能支持世界模型内的 policy 优化，但换到新任务域时仍需要重新生成标签并训练奖励模型。WorldLoop 本轮工作的目标，是保留 WoVR 静态世界模型阶段的训练流程，将奖励组件替换为无需训练的通用视觉语言模型。

公开的 Wan LIBERO 权重配套了 ResNet-18 奖励模型，本文将其作为对照。它有三个迁移限制：

1. LIBERO-Spatial 与 LIBERO-Object 使用不同权重，每份权重都依赖对应套件的仿真器标签；
2. 输入只有当前图像，不包含语言指令，无法区分同一场景中的不同任务；
3. 输出分布高度集中：16,384 帧中有 83% 的分数低于 0.01，取整后 71% 的样本因组内奖励相同而不产生 GRPO 梯度。

02 TOPReward：读取 token 概率，不生成文本
------------------------------------------

常见的 VLM 奖励方案会要求模型生成“成功/失败”或任务进度文本。TOPReward 不执行文本生成，直接读取 ``" True"`` token 的 log probability。其论文报告，在同一个开源 VLM 上，文本输出的 Value-Order Correlation 接近 0，而 token probability 可达到 0.947。

接入在线训练前，我们使用已有帧数据验证了三项条件：

- Qwen3-VL 能够识别 Wan 生成帧中的任务语义；
- 相同初始状态下，成功与失败轨迹的分数可以分离；
- 每个动作 chunk 执行一次奖励推理，对整体 rollout 增加的开销有限。

视频窗口是影响判定质量的关键参数。初版实验虽然设置了 16 帧窗口，但因缺少 ``video_metadata``，Qwen3-VL processor 实际只采样了其中 4 帧。公开实现已修复这一问题，使模型真正接收 16 帧。修复后，固定阈值 0.46 与 16 帧窗口在 Spatial 和 Object 帧 dump 上的触发率分别为 0.212 和 0.205，落在两个分布的相同分位。

03 轻量接入 RLinf，复用 WoVR 的静态世界模型阶段
--------------------------------------------------

一个 chunk 包含 8 步动作，是训练链路的最小执行单元：OpenVLA-OFT 生成动作，Wan 将动作转换为后续画面，Qwen3-VL 再将 16 帧视频与任务指令转换为 0/1 奖励。当前实验只更新 OpenVLA-OFT，Wan 与 Qwen3-VL 全程冻结。

RLinf 将 actor、rollout 和 env 三类 Ray worker collocate 到每张 GPU。Qwen3-VL 集成在 env worker 内，与世界模型共同完成环境 step。

.. code-block:: text

   OpenVLA-OFT ── 8 步动作 ──> Wan world model ── 8 帧画面 ──> Qwen3-VL
        ^                         │                                  │
        │                         └── 末帧作为下一步观测             │
        └────── GRPO 更新 <────── log P(" True") ≥ 0.46 → 0/1 ──────┘

Qwen3-VL 的连续分数经过阈值处理后输出 0/1，与 ResNet 奖励经过 ``round()`` 后的数据形状一致。WoVR 原有的奖励差分、轨迹终止、 ``loss_mask`` 截断和 GRPO 组内标准化逻辑可以直接复用，无需修改强化学习算法。

在物理仿真器中，reward 与 termination 通常来自同一个任务谓词。世界模型没有对应的物理状态，因此 Qwen3-VL 的 0/1 判定同时作为即时奖励和 ``terminations``。将其放在 env worker 内，可以在环境 step 返回前完成判定，并保持 RLinf 环境接口不变。

04 实验结果：零训练奖励达到专用模型的同等水平
------------------------------------------------

所有 policy 均在 MuJoCo 中评测：10 个任务，每个任务 50 个 episode，共 500 个 episode。指标采用 LIBERO 官方的 ``success_once``。两个评测读数之差的标准误约为 3.1 个百分点；强化学习训练阶段不调用 MuJoCo。

LIBERO-Spatial
~~~~~~~~~~~~~~

除奖励模型外，两组实验配置保持一致。

.. list-table::
   :header-rows: 1
   :widths: 42 18 18 22

   * - 奖励模型
     - 训练步
     - n=500
     - 相对起点
   * - 起点（Spatial SFT）
     - 0
     - 0.448
     - —
   * - ResNet（套件专用）
     - 20
     - 0.574
     - +12.6
   * - Qwen3-VL（零训练）
     - 10
     - 0.560
     - +11.2
   * - Qwen3-VL（零训练）
     - 20
     - 0.532
     - +8.4

Qwen3-VL 在 step 10 达到 0.560，与 ResNet 的 0.574 处于评测波动范围内。step 20 的 0.532 与 step 10 相差 2.8 个百分点，当前结果不足以判断已出现稳定回落。

.. note::

   这组 Spatial policy 结果来自视频采样元数据修复前的实现：配置窗口为 16 帧，但 processor 实际从该窗口采样 4 帧。离线重扫显示修复前后 Spatial 触发率只相差约一个标准误，因此该结果仍能说明训练链路有效，但尚不能作为修复后 16 帧实现的最终复现数字。对应 policy 重跑完成前，这一点属于数据边界。

LIBERO-Object：跨套件 sanity check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

阈值 0.46 与 16 帧窗口均在 Spatial 数据上确定。迁移到 Object 时，这两个参数和 Qwen3-VL 权重保持不变，仅通过 ``task_suite_name`` 切换任务指令。

.. list-table::
   :header-rows: 1
   :widths: 34 26 13 13 14

   * - 奖励模型
     - 换套件改了什么
     - base
     - step 10
     - step 20
   * - ResNet
     - 更换 Object 专用 ``.pth``
     - 0.342
     - 0.368
     - 0.348
   * - Qwen3-VL（16 帧修复后）
     - 更换任务指令
     - 0.342
     - 0.348
     - 0.348

Qwen3-VL 与 ResNet 在 step 20 均为 0.348，只比起点高 0.6 个百分点。同一组 Qwen3-VL 奖励配置迁移后没有失效，但两条臂都没有获得超过评测波动的 policy 增益。因此 Object 只验证固定奖励参数能够跨套件运行，不用于证明当前世界模型 RL 配方能在 Object 上带来收益。

训练期奖励也不能替代外部评测：Object 实验中，ResNet 给出的世界模型内成功率约为 0.55，而 MuJoCo 结果为 0.348；修复后的 Qwen3-VL 内部成功率约为 0.35，MuJoCo 结果同为 0.348。该对齐只说明当前判分器完成了标定，最终 policy 能力仍需独立评测。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/worldloop-results.png
   :alt: WorldLoop 在 LIBERO-Spatial 与 LIBERO-Object 上的训练期和 MuJoCo 评测结果
   :align: center
   :width: 100%

   Spatial 的 Qwen3-VL 数据来自视频采样修复前实现；Object 的 Qwen3-VL 数据来自修复后的 16 帧实现。

05 公开实现与实验配置
----------------------

实现与双语使用文档正在 `RLinf #1554 <https://github.com/RLinf/RLinf/pull/1554>`_ 审阅。生成本文读数的固定实验分支为 `ZJLi2013/RLinf@0c7a3d5f <https://github.com/ZJLi2013/RLinf/tree/0c7a3d5f8d43a9c2ae242236a5699fb9d0a7243b>`_。

实验使用单节点 8 张 AMD MI300 系列 GPU、1 TB 主机内存和 ROCm 6.4。运行时配置为 ``total_num_envs=128``、 ``rollout_epoch=2``、 ``global_batch_size=2048``、 ``lr=1e-5``、 ``group_size=8``。每个训练 step 包含 8,192 个样本、4 次优化器更新和 32 个初始状态。世界模型与奖励模型在每张 GPU 上各部署一份，实测显存占用为 84–116 GB / 192 GB。

Qwen3-VL 需要 Transformers 4.57.1，而 OpenVLA-OFT 使用 4.40.1。实现因此只在 env worker 进程内加载新版依赖，避免改变 actor 与 rollout worker 的运行环境。

06 下一步：让 world model 跟随 policy 一起进化
-----------------------------------------------

本轮工作复现了 WoVR 的静态世界模型强化学习阶段，并将任务专用奖励替换为冻结的 TOPReward。这还不是完整的 WoVR：当前实验在 policy 优化期间冻结 Wan，没有复现 WoVR 的 PACE。

当 policy 经过第一阶段强化学习后，其动作分布会逐渐偏离训练 WM Base 的数据分布。PACE 使用进化后 policy 新采集的少量轨迹低频更新 world model，得到 WM Evo，再进入下一阶段 policy 优化。WorldLoop 的下一阶段将沿这条闭环继续：policy 在 world model 中进化，进化后的 policy 产生新轨迹，world model 再用这些轨迹更新。

PACE 仍需阶段性访问目标环境采集轨迹，只是把持续在线交互变成低频更新，并未完全消除真实环境或仿真器数据。后续验证还将补齐修复后的 Spatial policy 结果，并在更多任务套件上检验 TOPReward 的固定参数迁移。

参考资料
--------

- `TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics <https://arxiv.org/abs/2602.19313>`_
- `WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL <https://arxiv.org/abs/2602.13977>`_
- `RLinf <https://github.com/RLinf/RLinf>`_
- `RLinf-Wan-LIBERO-Spatial <https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial>`_
- `OpenVLA-OFT <https://github.com/moojink/openvla-oft>`_
- `Qwen3-VL <https://github.com/QwenLM/Qwen3-VL>`_
- `Wan2.2 <https://github.com/Wan-Video/Wan2.2>`_
- `LIBERO <https://github.com/Lifelong-Robot-Learning/LIBERO>`_

AI 使用说明
-----------

本文实验、数据与结论由作者提供并核验；Cursor AI 参与了文章结构调整、双语改写和措辞精简。
