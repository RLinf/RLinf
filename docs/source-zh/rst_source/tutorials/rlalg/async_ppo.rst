异步近端策略优化 (Async PPO)
==================================

1. 引言
---------------

RLinf 中的 Async PPO 是面向具身任务的异步训练实现。它保留了 PPO 的核心优化思想，但不再采用“采样一轮，再训练一轮”的严格同步执行方式，而是将环境交互、策略推理和 actor 训练解耦为长期运行的并发流水线。

这套实现的目标不是改变 PPO 的理论目标，而是解决具身训练中的系统瓶颈，尤其是以下问题：

- 环境 step 较慢，actor 长时间等待 rollout 数据。
- 视觉策略或大模型策略推理较重，GPU 在同步 barrier 上空转。
- 单轮 rollout 较长，同步 PPO 的端到端吞吐偏低。

Async PPO 适合“系统吞吐是主要瓶颈”的场景。如果你的任务规模较小，或者同步 PPO 已经可以稳定高效训练，那么优先使用同步方案通常更简单。


2. 适用范围
----------------------

当前 RLinf 中的 Async PPO 只用于具身任务，相关入口和核心组件如下：

- 训练入口：``examples/embodiment/train_async.py``
- Runner：``rlinf/runners/async_ppo_embodied_runner.py``
- Actor Worker：``rlinf/workers/actor/async_ppo_fsdp_worker.py``
- Rollout Worker：``rlinf/workers/rollout/hf/async_huggingface_worker.py``
- Env Worker：``rlinf/workers/env/async_env_worker.py``

目前实现有以下边界条件：

- 仅支持具身任务，不支持 reasoning 或 agent 任务。
- Async PPO 分支要求 ``algorithm.loss_type: decoupled_actor_critic``。
- actor 模型必须带 value head，即 ``actor.model.add_value_head: True``。
- ``AsyncEnvWorker`` 和 ``AsyncMultiStepRolloutWorker`` 不支持 offload。
- 当 ``rollout.recompute_logprobs=True`` 时，actor 权重 offload 不受支持。
- ``AsyncPPOEmbodiedRunner`` 当前不支持验证流程；若配置 ``runner.val_check_interval > 0``，系统会给出 warning 并跳过验证。


3. 为什么需要 Async PPO
------------------------

同步 PPO 的典型流程如下：

1. 采样一批轨迹。
2. 停止采样。
3. 计算优势并进行若干轮更新。
4. 同步新权重。
5. 进入下一轮采样。

这种执行模式有一个明显特点：全局 barrier 很强。只要其中一个阶段慢，其余组件就会等待。

在具身训练中，这个问题尤其明显：

- 环境可能受物理仿真、渲染、重置逻辑影响而较慢。
- rollout 可能依赖较重的视觉编码或大模型推理。
- actor 训练阶段通常需要较大的 batch 和较多 GPU 资源。

Async PPO 的核心思路是把这三类工作拆开并并发执行：

- 环境持续产生观测。
- rollout 持续消费观测并输出动作，同时累计轨迹。
- actor 持续消费已经完成的 rollout batch 并做参数更新。

这样可以提升整体资源利用率，但代价是：actor 拿到的样本不一定来自最新策略，因此必须显式处理样本陈旧性。


4. 系统架构
-----------------

Async PPO 在 RLinf 中的逻辑数据流可以概括为：

.. code-block:: text

   AsyncEnvWorker
       |  env_channel
       v
   AsyncMultiStepRolloutWorker
       |  actor_channel
       v
   AsyncPPOEmbodiedFSDPActor
       |  weight sync
       v
   AsyncMultiStepRolloutWorker

其中还存在一条反向动作流：

- rollout 从环境观测中推理动作。
- rollout 通过 ``rollout_channel`` 将动作发送回 env。
- env 根据动作继续 step，并把新的观测再次写入 ``env_channel``。

从职责划分上看：

- ``AsyncEnvWorker`` 负责长期运行的环境交互循环。
- ``AsyncMultiStepRolloutWorker`` 负责策略推理、轨迹收集和版本记录。
- ``AsyncPPOEmbodiedRunner`` 负责训练节奏控制、权重同步和指标汇总。
- ``AsyncPPOEmbodiedFSDPActor`` 负责优势计算、近端 logprob 计算和 PPO 更新。

这套设计的关键不是“异步调用”，而是“长期存活的工作流分离”。env 和 rollout 并不会在每个训练 step 被重新启动，而是在后台持续运行。


5. 执行流程
-----------------

一次 Async PPO 训练更新在系统层面的执行顺序如下：

1. runner 初始化 worker，并先将 actor 权重同步到 rollout。
2. env 调用 ``bootstrap_step()``，生成初始观测。
3. rollout 持续执行 ``generate_one_epoch()``：
   - 从 ``env_channel`` 读取环境输出。
   - 基于当前 rollout 权重推理动作。
   - 记录 ``prev_logprobs``、``prev_values``、``forward_inputs``。
   - 为当前样本打上策略版本 ``versions``。
   - 将动作发送回 ``rollout_channel``，供 env 继续执行。
4. 一个 rollout epoch 结束后，rollout 将轨迹切分并发送到 ``actor_channel``。
5. actor 从 ``actor_channel`` 接收轨迹，执行：
   - 可选重算 ``proximal_logprobs``；
   - 计算 advantages 和 returns；
   - 展平 ``[T, B, ...]`` 维度并打乱；
   - 按 ``global_batch_size`` 和 ``micro_batch_size`` 训练。
6. actor 完成一次更新后，runner 增加 ``global_step``，并再次将新权重同步给 rollout。
7. rollout 继续用新的版本采样；如果 rollout 跑得过快，系统会通过陈旧度控制逻辑限制其继续前进。

这里要特别注意：Async PPO 是“采样与训练重叠”，但不是“完全无序并发”。权重同步、版本推进和样本准入都有明确规则。


6. 版本控制与样本陈旧性
------------------------

异步训练最核心的问题是样本陈旧性。RLinf 的 Async PPO 通过两个层面控制它。

6.1 样本版本标记
^^^^^^^^^^^^^^^^^^^^

rollout 在生成每个 step 的数据时，会将当前 rollout 策略版本写入 ``versions``。这个字段表示该样本是由哪一版策略产生的行为数据。

随后 actor 在训练时还会维护自己的当前版本号。这样，一批样本就同时关联了：

- 行为策略版本：样本真正产生时使用的策略版本。
- 近端策略版本：本轮 PPO 更新所依赖的 anchor 版本。
- 当前训练版本：正在优化的目标版本。

这使得系统能够显式感知“这批数据到底老了多少”。

6.2 rollout 侧陈旧度限流
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rollout worker 通过 ``staleness_threshold`` 控制最多允许领先 actor 多少个版本。

如果 rollout 已经积累了过多由旧策略产生的 episode，而 actor 尚未跟上，rollout 会在 ``wait_if_stale()`` 中暂停，直到版本差回到可接受范围。

这个机制的作用是：

- 防止 rollout 无限超前，生成大量过时样本。
- 在系统吞吐和样本新鲜度之间做显式权衡。

经验上：

- ``staleness_threshold`` 越小，样本越新鲜，训练通常更稳。
- ``staleness_threshold`` 越大，系统吞吐通常更高，但旧样本比例也更高。


7. 算法机制
-----------------

7.1 三个策略视角
^^^^^^^^^^^^^^^^^^^^

为了理解 RLinf 中的 Async PPO，需要区分三种策略：

- 行为策略 :math:`\pi_{b}`：实际生成样本的 rollout 策略。
- 近端策略 :math:`\pi_{p}`：当前 PPO 更新所参考的 anchor 策略。
- 当前策略 :math:`\pi_{\theta}`：正在反向传播中被优化的策略。

同步 PPO 中，这三者通常非常接近，甚至可以近似看作只涉及“旧策略”和“新策略”。但在异步场景里，行为策略可能明显落后于当前训练策略，因此需要单独处理。

7.2 解耦 actor-critic 损失
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf 的 Async PPO 使用 ``decoupled_actor_critic`` 损失。它不是直接对行为策略做标准 PPO clip，而是把“近端约束”和“行为分布修正”拆开处理。

记

.. math::

   r_{\mathrm{prox}} =
   \frac{\pi_{\theta}(a_t \mid s_t)}
        {\pi_{p}(a_t \mid s_t)},
   \qquad
   w_{\mathrm{behav}} =
   \frac{\pi_{p}(a_t \mid s_t)}
        {\pi_{b}(a_t \mid s_t)}.

则 actor 部分可以理解为：先围绕近端策略 :math:`\pi_p` 做 PPO clip，再用 :math:`w_{\mathrm{behav}}` 修正行为策略带来的分布偏移。

其工程意义很直接：

- PPO clip 负责控制“当前策略相对近端策略”的更新幅度。
- 行为权重负责控制“旧样本相对近端策略”的偏移影响。

7.3 dual-clip 与行为样本裁剪
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了进一步提高稳定性，RLinf 还叠加了两层保护：

- ``clip_ratio_c``：dual-clip，上界限制极端优势值对梯度的冲击。
- ``behave_weight_threshold``：当行为权重过大时，直接将这部分样本屏蔽。

如果训练中出现明显震荡，优先检查这两个参数和样本陈旧度指标，而不是直接增大学习率或 batch size。

7.4 GAE 与 value head
^^^^^^^^^^^^^^^^^^^^^^^^

优势函数仍然使用标准 GAE：

.. math::

   \hat{A}_t^{\mathrm{GAE}}
   =
   \sum_{l=0}^{\infty} (\gamma \lambda)^l
   \left(r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\right).

这意味着 Async PPO 仍然需要价值估计，因此 actor 模型必须带 value head。配置上至少需要满足：

- ``algorithm.adv_type: gae``
- ``algorithm.loss_type: decoupled_actor_critic``
- ``actor.model.add_value_head: True``


8. proximal logprob 的两种来源
-------------------------------

Async PPO 里的关键量之一是 ``proximal_logprobs``。它有两种来源：

1. 显式重算

   当 ``rollout.recompute_logprobs=True`` 时，actor 会在训练前基于当前权重重新前向，显式得到 proximal logprob。

2. 近似插值

   当 ``rollout.recompute_logprobs=False`` 时，系统会根据 ``versions`` 将行为 logprob 和当前 logprob 进行插值，近似构造 proximal anchor。

两种方式的取舍如下：

- 显式重算更稳，适合作为默认配置。
- 近似插值吞吐更高，但对陈旧样本更敏感。

如果你正在排查训练不稳定问题，优先使用显式重算。


9. 配置说明
-----------------

下面是一份 Async PPO 的最小配置骨架，和 ``examples/embodiment/config/maniskill_async_ppo_openvla.yaml`` 的核心语义保持一致：

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0-3
       env: 0-1
       rollout: 2-3

   runner:
     task_type: embodied
     max_epochs: 1000
     val_check_interval: -1
     save_interval: 40

   algorithm:
     adv_type: gae
     loss_type: decoupled_actor_critic
     normalize_advantages: True
     staleness_threshold: 1
     behave_weight_threshold: 2.0
     clip_ratio_high: 0.3
     clip_ratio_low: 0.3
     clip_ratio_c: 3.0
     value_clip: 0.2
     gamma: 0.99
     gae_lambda: 0.95
     entropy_bonus: 0.0
     rollout_epoch: 1

   env:
     train:
       total_num_envs: 16
       max_episode_steps: 80
       max_steps_per_rollout_epoch: 80

   rollout:
     backend: huggingface
     recompute_logprobs: True
     pipeline_stage_num: 1

   actor:
     training_backend: fsdp
     micro_batch_size: 40
     global_batch_size: 320
     model:
       add_value_head: True

这些参数中，最需要重点理解的是以下几项：

- ``staleness_threshold``
  控制 rollout 最多允许领先 actor 多远。越小越稳，越大吞吐通常越高。

- ``behave_weight_threshold``
  控制旧样本的最大行为权重。越小越保守，越大对旧样本容忍度越高。

- ``rollout.recompute_logprobs``
  控制是否显式重算 proximal logprob。建议默认开启。

- ``actor.micro_batch_size`` 和 ``actor.global_batch_size``
  直接决定训练阶段的显存占用和吞吐。应优先保证训练稳定和不 OOM，再追求更大 batch。

另外，actor 训练阶段必须满足以下批量约束：

- ``global_batch_size`` 必须能被 ``micro_batch_size * actor_world_size`` 整除。
- rollout 展平后的样本数必须能被单卡训练 batch 整除。

如果这些条件不满足，系统会在训练时直接报错。


10. 启动方式与模型链接
-----------------------

推荐使用脚本启动：

.. code-block:: bash

   bash examples/embodiment/run_async.sh maniskill_async_ppo_openvla

运行前需要注意：

- 多机训练时，Ray 必须提前启动，且只在 head 节点运行训练入口。
- ``run_async.sh`` 会设置 ``MUJOCO_GL=egl`` 和 ``PYOPENGL_PLATFORM=egl``。
- ``ROBOT_PLATFORM`` 必须与机器人平台一致，否则动作维度和归一化逻辑可能不匹配。

目前examples/embodiment/config/ 目录下的 Async PPO 配置都已经过测试，可以直接用来跑具身任务。
以下是对应yaml所使用模型的下载链接:

- ``libero_goal_async_ppo_gr00t.yaml``: https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Goal
- ``libero_object_async_ppo_openpi_pi05.yaml``: https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT
- ``libero_spatial_async_ppo_openpi.yaml``: https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
- ``libero_spatial_async_ppo_openpi_pi05.yaml``: https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT
- ``maniskill_async_ppo_openpi.yaml``: https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-SFT
- ``maniskill_async_ppo_openpi_pi05.yaml``: https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT
- ``maniskill_async_ppo_openvla.yaml``: https://huggingface.co/gen-robot/openvla-7b-rlvla-warmup
- ``maniskill_async_ppo_openvlaoft.yaml``: https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-trajall

其中``maniskill_async_ppo_openvlaoft.yaml``中所使用的lora HF仓库链接为: https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora


11. 监控指标
-----------------

调试 Async PPO 时，不要只看 reward。至少需要同时观察以下几类指标：

系统侧：

- ``time/env/*``：环境交互是否成为瓶颈。
- ``time/rollout/*``：推理是否成为瓶颈。
- ``time/actor_training``：训练阶段是否过重。

策略更新侧：

- ``train/actor/proximal_approx_kl``：当前策略相对近端策略的偏移。
- ``train/actor/clip_fraction``：PPO clip 命中比例。
- ``train/actor/dual_clip_fraction``：dual-clip 命中比例。

样本陈旧性侧：

- ``train/actor/behav_approx_kl``：行为策略与近端策略的偏移。
- ``train/actor/behav_clip_fraction``：被行为权重裁掉的样本比例。
- ``train/actor/average_version``：训练样本的平均版本号。
- ``train/actor/current_version``：当前训练版本号。

如果系统吞吐很高，但 ``behav_approx_kl`` 和 ``behav_clip_fraction`` 长期偏高，通常说明 rollout 超前过多，训练已经开始受过时样本影响。


12. 调参建议
-----------------

建议按下面的顺序调参。

12.1 先求稳，再提吞吐
^^^^^^^^^^^^^^^^^^^^^^^^

推荐起点：

- ``staleness_threshold: 1``
- ``behave_weight_threshold: 2.0``
- ``rollout.recompute_logprobs: True``

先确认 reward、KL 和行为裁剪指标稳定，再尝试提高系统吞吐。

12.2 吞吐不足时怎么调
^^^^^^^^^^^^^^^^^^^^^^^^

优先顺序建议如下：

1. 调整 ``cluster.component_placement``，减少 env、rollout、actor 之间的资源争用。
2. 提高 ``env.train.total_num_envs``，增加并行环境数。
3. 适度增大 ``staleness_threshold``，例如从 ``1`` 调到 ``2``。
4. 最后才考虑关闭 ``recompute_logprobs``。

12.3 训练不稳时怎么调
^^^^^^^^^^^^^^^^^^^^^^^^

优先顺序建议如下：

1. 降低 ``staleness_threshold``。
2. 降低 ``behave_weight_threshold``。
3. 降低 actor 学习率。
4. 降低 ``clip_ratio_high`` 和 ``clip_ratio_low``。
5. 减少 ``update_epoch`` 或减小 batch。

12.4 OOM 时怎么调
^^^^^^^^^^^^^^^^^^^^^^^^

优先顺序建议如下：

1. 降低 ``actor.micro_batch_size``。
2. 开启或保留 ``gradient_checkpointing``。
3. 适度减少 ``env.train.total_num_envs``。


13. 与同步 PPO 的区别
----------------------

可以用一句话概括两者差异：

- 同步 PPO 以“样本新鲜度优先”为核心。
- Async PPO 以“系统吞吐优先，但通过版本和权重机制控制陈旧性”为核心。

因此，Async PPO 相比同步 PPO 的主要变化不是目标函数本身，而是训练系统的执行模型发生了改变。
