FlexRLT：冻结 VLA 特征与可插拔 Policy
=====================================

为 FlexRLT 增加特征模型、policy head 或更新规则。ManiSkill 的 checkpoint 下载、训练启动和 STEAM 校准见 :doc:`FlexRLT <../examples/embodied/flexrlt>`。下面的例子在这份 ManiSkill 配置上扩展。

这里的 prefix 指 VLA 生成动作之前计算出的图像、语言 hidden states。在线 RL 期间，VLA 保持冻结；另一个 actor 读取这些特征、机器人状态和 VLA 生成的 reference action，输出要执行的 action chunk。

概览
----

选择这条路线时，先考虑是否需要专门训练 RL 表示。设置 ``prefix.pool: masked_mean``，就可以直接对已有 base 或 SFT VLA 的有效 prefix token 求均值，得到 ``z_rl``。如果希望使用 RL Token 学到的表示，则设置 ``prefix.pool: rlt_token``，加载 Stage 1 训练过的 token transformer。ManiSkill 的两种表示共用后续的 actor-critic 训练流程；RLT 的两阶段训练见 :doc:`../examples/embodied/rlt`。

在线更新训练 actor 和 critic head。replay 保存 ``{z_rl, proprio, ref_chunk}``，learner 可以复用已提取的特征。冻结 VLA 在 rollout 中提取特征并生成 reference action。

各部分如何组合
----------------

一次 FlexRLT 运行由四个互相独立的选择构成。每一项都有自己的配置字段和注册位置，因此替换时只改你真正要换的那一层。

.. list-table::
   :header-rows: 1
   :widths: 20 26 22 32

   * - 选择
     - 配置字段
     - 注册方式
     - 代码位置
   * - 冻结特征模型
     - ``rollout.prefix_feature_model``
     - ``register_model(..., category="embodied")``
     - 任何实现 ``extract_prefix_obs`` 的模块
   * - 可训练 policy
     - ``actor.model.model_type``
     - ``register_model(..., obs_contract=ObsContract.PREFIX)``
     - ``rlinf/models/embodiment/mlp_policy/prefix_mlp_policy.py``
   * - actor 与 critic head
     - ``actor.model.actor_head.name``、``critic_head.name``
     - ``register_prefix_actor`` / ``register_prefix_critic``
     - ``rlinf/models/embodiment/mlp_policy/heads.py``
   * - 更新规则
     - ``algorithm.name``\ （配合 ``loss_type: prefix_off_policy``）
     - ``register_prefix_off_policy_algorithm``
     - ``rlinf/algorithms/prefix_off_policy/``

各部分负责以下内容：

* **特征模型** 决定原始观测如何变成 ``{z_rl, proprio, ref_chunk}``。在线更新时保持冻结。
* **policy** 决定这三个张量如何拼成 policy state 和 reference 输入。``prefix_policy`` 组合两个 head，并暴露 worker 所需的 ``BasePolicy`` forward 类型。
* **head** 拥有网络本身。actor 和 critic 是两个独立对象、两套独立注册表，因此可以把随机 actor 配 CrossQ critic，或只替换其中一侧。
* **算法** 拥有 loss 和更新调度。它读取 head 在 ``capabilities`` 中声明的能力，拒绝无法训练的搭配。

配置校验通过观测契约将 policy 与 ``prefix_off_policy`` 匹配，所以新 policy 以 ``obs_contract=ObsContract.PREFIX`` 注册后，无需修改 ``rlinf/config.py`` 就能通过校验。它仍需满足 :ref:`flexrlt-policy-contract` 中的 policy 契约。

如何扩展
^^^^^^^^

找到你要改的那一行，只改那一行：

* **换 VLA** —— 写一个提供 ``z_dim`` 和 ``extract_prefix_obs`` 的 adapter，按具身模型注册，再把 ``rollout.prefix_feature_model`` 指过去。继续复用 policy、head、算法和 replay 格式。见 :ref:`扩展 BaseVLA 特征模型 <flexrlt-feature-model>`。
* **换网络** —— 继承 ``PrefixActorHead`` 或 ``PrefixCriticHead``，声明 capabilities 后注册。继续复用 rollout、replay 格式和训练循环。见 :ref:`扩展 MLP Actor 与 Critic <flexrlt-heads>`。
* **换更新规则** —— 继承 ``PrefixOffPolicyAlgorithm``，实现 ``actor_loss`` 和 ``target_actions``，并声明 ``requirements``。chunk 折扣、BC target 和权重调度继续共享。见 :ref:`扩展 FlexRLT 算法 <flexrlt-algorithm>`。
* **在同一特征上换 policy** —— 例如换成循环（recurrent）actor。继承 ``BasePolicy``，满足下面的契约，以 ``obs_contract=ObsContract.PREFIX`` 注册，再设置 ``actor.model.model_type``。继续复用算法、replay 格式和 rollout。

.. _flexrlt-policy-contract:

Policy 契约
^^^^^^^^^^^

算法、FSDP worker 和 rollout 运行时会直接调用 policy，所以替换 ``prefix_policy`` 的 policy 必须提供它们用到的接口：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 要求
     - 使用方
   * - 接受 ``actor.model.actor_head`` 和 ``critic_head``，其 ``name`` 为已注册的名字
     - 配置校验和算法读取这两个 head 名字，检查 actor distribution、action noise 和 Q head 数量。policy 应构建这两个 head，至少要与它们声明的 capabilities 行为一致。
   * - ``forward(forward_type=ForwardType.SAC, obs=...)`` 返回 ``actions, logprobs, None``
     - actor loss 和 target action。``actions`` 是展平的 ``[B, num_action_chunks * action_dim]``；调用还会传入 ``deterministic``、``apply_action_noise``、``apply_reference_dropout`` 和 ``reference_dropout_prob``。
   * - ``ForwardType.SAC_Q`` 返回 ``[B, num_q_heads]``；``ForwardType.CROSSQ_Q`` 返回 ``current_q, next_q``
     - critic loss 和 actor loss 中的 Q 项。只有 critic head 声明 ``supports_cross_q`` 时才需要 ``CROSSQ_Q``。两者都接受 ``detach_encoder``。
   * - critic 参数放在 ``q_head`` 属性下
     - worker 把名字中含 ``q_head``、``encoder``、``encoders`` 或 ``state_proj`` 的参数放进 critic 优化器，其余放进 actor 优化器。actor 专属模块使用与这些 critic 参数不同的名称。
   * - ``predict_action_batch(env_obs, mode=..., return_obs=True)``
     - rollout。返回 ``[B, num_action_chunks, action_dim]`` 的动作，并在 ``forward_inputs`` 中放入 ``action`` 以及 ``z_rl``、``proprio``、``ref_chunk``，它们会成为 replay transition。

使用循环 actor 时，把状态放进特征模型返回的观测中。

Replay 与更新
-------------

路由可能选择 base、actor 或 expert 的动作，通过 ``ActionPayload`` 发送实际动作和可选反馈。gate 用 ``rollout_info_keys`` 声明反馈字段；启用 gate 的环境必须实现 ``RolloutInfoReceiver.set_rollout_infos``，env worker 会在启动时检查。

环境通过 ``TrajectoryInfoProvider.trajectory_info_spec`` 声明校准标签。这些标签和 gate 诊断信息存放在 ``Trajectory.trajectory_infos`` 中，与 ``forward_inputs`` 分开。trace writer 读取后，replay 会丢弃这些诊断数据，只记录实际发送给环境的动作及其前后特征：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的 action chunk
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

critic 按 ``gamma`` 对 chunk 内 reward 折扣，再用 ``gamma ** chunk_horizon`` bootstrap 下一状态的 Q。actor objective 为 ``-q_weight * Q + bc_weight * BC``。BC 通常以 ``ref_chunk`` 为目标；带 intervention 标记的 step 则使用实际执行的 intervention action。

冻结 VLA 如何生成 Prefix observation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rollout worker 将环境原始 observation 传给 ``extract_prefix_obs``。特征模型通过 ``z_dim`` 声明输出宽度，并返回一个 ``PrefixObs``。接口定义在 ``rlinf/models/embodiment/prefix_features.py``，与 ``base_policy.py`` 同层。它和模型架构无关，任何 VLA 都可以实现：

.. code:: python

   from typing import Any, Protocol
   from rlinf.models.embodiment.prefix_features import (
       PrefixFeatureModel,
       PrefixObs,
   )

   class PrefixFeatureModel(Protocol):
       @property
       def z_dim(self) -> int: ...

       def extract_prefix_obs(
           self, env_obs: dict[str, Any]
       ) -> PrefixObs: ...

``PrefixObs`` 包含三个 batch-first tensor。下表是特征模型的输出：

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - key
     - shape
     - 含义
   * - ``z_rl``
     - ``[B, z_dim]``
     - 冻结 VLA 的 Prefix 表示。
   * - ``proprio``
     - ``[B, proprio_dim]``
     - 当前机器人或仿真状态，作为 policy state 的一部分。
   * - ``ref_chunk``
     - ``[B, ref_num_action_chunks, action_dim]``
     - 冻结 VLA 生成的 reference action chunk，也作为 actor 的条件输入。

OpenPI 的 ``Pi0Eval`` 通过 ``OpenPiPrefixMixin`` 实现这个接口。它先转换环境 observation，计算一次图像和语言 prefix，再复用 cache 提取 ``z_rl``、生成 reference action；选定的机器人状态作为 ``proprio``。``PrefixRolloutRuntime`` 将特征模型设为 ``eval`` 并关闭梯度。

Prefix pooling
^^^^^^^^^^^^^^

先确定哪些 token 参与构造 ``z_rl``，再选择聚合方式。``prefix.image_only: true`` 只保留图像 token；设为 ``false`` 时保留图像和语言 token。``prefix.pool`` 与它同属 ``prefix`` 块，该块与模型配置中的 ``openpi`` 并列：

* ``masked_mean`` 对有效 token 求均值，是 ``openpi.use_rlt`` 为 false 时的默认值。
* ``mean`` 对选定的整个序列求均值，包括被 mask 的位置。
* ``last`` 读取最后一个有效 token；未提供 mask 时读取序列末端。
* ``rlt_token`` 运行 Stage 1 的 ``rlt_module.encode_flat``。

``pool_prefix`` 接收 shape 为 ``[B, T, D]`` 的 ``hidden`` 和形状为 ``[B, T]`` 或 ``[B, T, 1]`` 的可选 boolean mask，前三种 pooling 返回 ``[B, D]``。未设置 ``prefix.pool`` 时，``resolve_prefix_pool`` 会在 ``openpi.use_rlt`` 为 true 时选择 ``rlt_token``，否则选择 ``masked_mean``。``prefix.image_only`` 默认为 true。比较不同表示时，请显式设置这两个字段。

rollout 启动时检查特征模型的 ``z_dim`` 与 ``actor.model.z_dim`` 是否一致，每次提取特征时还会检查 ``z_rl`` 的实际宽度是否符合模型声明。

.. _flexrlt-feature-model:

扩展 BaseVLA 特征模型
---------------------

使用其他 VLA 时，将它的 observation 和输出适配为 ``PrefixObs`` 即可接入。这里的 BaseVLA 指供应商模型或基座模型。具体 adapter 要支持 ``eval``、``requires_grad_``、``to`` 和 ``extract_prefix_obs``；如果 route 还会调用 VLA 的动作 API，也需提供对应接口。

实现 adapter
^^^^^^^^^^^^

下面的骨架从一次 observation 提取 prefix 特征和 reference chunk。预处理、编码和采样方法仅作示意，接入时要换成 VLA 实际提供的 API：

.. code:: python

   import torch
   from rlinf.models.embodiment.prefix_features import pool_prefix

   class MyBaseVLAAdapter(torch.nn.Module):
       def __init__(self, base_vla):
           super().__init__()
           self.base_vla = base_vla
           self.z_dim = base_vla.prefix_width

       @torch.no_grad()
       def extract_prefix_obs(self, env_obs):
           observation = self.base_vla.encode_observation(env_obs)
           hidden, mask = self.base_vla.encode_prefix(observation)
           z_rl = pool_prefix(hidden, mask, mode="masked_mean")
           ref_chunk = self.base_vla.sample_action_chunk(observation)
           proprio = torch.as_tensor(env_obs["states"], device=z_rl.device)
           return {
               "z_rl": z_rl.float(),
               "proprio": proprio.float(),
               "ref_chunk": ref_chunk.float(),
           }

``extract_prefix_obs`` 接收环境 wrapper 提供的 observation 字典，返回 ``PREFIX_OBS_KEYS`` 对应的三个 float32 batch tensor。示例中，``hidden`` 按 mask 求均值得到 ``z_rl``，采样动作得到 ``ref_chunk``，状态字段作为 ``proprio``。检查 reference 的动作维度和长度是否符合环境要求，状态归一化方式也要与训练数据一致。

接入 model factory
^^^^^^^^^^^^^^^^^^

rollout worker 通过 ``get_model`` 创建 ``rollout.prefix_feature_model``。使用 ``rlinf.models`` 的 ``register_model`` 注册 builder，并设置 ``category="embodied"``。builder 接收 model config 和 ``torch_dtype``，返回 adapter；注册时也会将 model type 加入配置校验。

每个 worker 进程都要执行注册：仓库内的 builder 由 ``rlinf/models/__init__.py`` 导入，外部组件则通过 ``RLINF_EXT_MODULE`` 加载注册模块。完整流程见 :doc:`new_model_fsdp`。接入后可以保留现有 ``prefix_policy``、算法和 replay 格式，再通过 rollout 路径检查 adapter 的 batch shape、device、dtype、reset 行为和 action chunk。

.. _flexrlt-heads:

扩展 MLP Actor 与 Critic
------------------------

要在相同特征上比较不同网络，替换 head。actor 接收拼好的 state 和 reference；``forward`` 返回 flat action 和每维 log probability，``mean_action`` 返回 supervised forward 使用的无噪声预测。critic 接收 state 和 flat action，每个 Q head 返回一个值。joint CrossQ critic 还会返回 next-state Q。

将下面的 actor、critic 和算法代码保存到仓库根目录的 ``flexrlt_ext.py``。

注册 actor head
^^^^^^^^^^^^^^^

继承 ``PrefixActorHead``，用 ``ActorHeadCapabilities`` 声明行为。registry 将 ``state_dim``、``reference_dim``、``action_dim`` 和 head config 传给 builder。以下示例注册一个支持 reference dropout 和 action noise 的 deterministic MLP：

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.mlp_policy.heads import (
       ActorHeadCapabilities,
       PrefixActorHead,
       register_prefix_actor,
   )

   class CustomMLPActor(PrefixActorHead):
       capabilities = ActorHeadCapabilities(
           distribution="deterministic", supports_action_noise=True,
       )

       def __init__(self, state_dim, reference_dim, action_dim, cfg):
           super().__init__()
           hidden_dim = int(cfg.get("hidden_dim", 256))
           self.net = nn.Sequential(
               nn.Linear(state_dim + reference_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, action_dim),
           )

       def mean_action(self, state, reference):
           return self.net(torch.cat([state, reference], dim=-1)).clamp(-1, 1)

       def forward(
           self, state, reference, *, deterministic,
           apply_reference_dropout, reference_dropout_prob,
           apply_action_noise,
       ):
           if apply_reference_dropout:
               probability = float(reference_dropout_prob or 0.0)
               keep = torch.rand(reference.shape[0], 1, device=reference.device)
               reference = reference * (keep >= probability).to(reference.dtype)
           action = self.mean_action(state, reference)
           if apply_action_noise and not deterministic:
               action = (action + torch.randn_like(action) * 0.1).clamp(-1, 1)
           return action, torch.zeros_like(action)

   @register_prefix_actor(
       "custom_mlp", capabilities=CustomMLPActor.capabilities,
   )
   def make_custom_mlp_actor(state_dim, reference_dim, action_dim, cfg):
       return CustomMLPActor(state_dim, reference_dim, action_dim, cfg)

``register_prefix_actor`` 以 YAML 中的名称注册这个 actor，供 model builder 构建。``mean_action`` 为 policy 的 SFT forward 提供无噪声预测，``forward`` 按请求应用 dropout 和探索噪声，再返回动作及 log probability。内置的 ``fixed_std_mlp`` 用于 AC，``deterministic_mlp`` 用于 TD3；AC 要求 stochastic actor，TD3 要求支持 action noise 的 deterministic actor。

注册 critic head
^^^^^^^^^^^^^^^^

critic 需要声明 ``CriticHeadCapabilities`` 和 ``num_q_heads``。下面的实现将同一组 state-action 输入送入两个独立 Q 网络，并注册为 YAML 中的名称：

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.mlp_policy.heads import (
       CriticHeadCapabilities,
       PrefixCriticHead,
       register_prefix_critic,
   )

   class CustomTwinCritic(PrefixCriticHead):
       capabilities = CriticHeadCapabilities(fixed_num_q_heads=2)

       def __init__(self, state_dim, action_dim, cfg):
           super().__init__()
           self.num_q_heads = 2
           width = int(cfg.get("hidden_dim", 256))
           def q_network():
               return nn.Sequential(
                   nn.Linear(state_dim + action_dim, width),
                   nn.ReLU(),
                   nn.Linear(width, 1),
               )
           self.q1 = q_network()
           self.q2 = q_network()

       def forward(self, state, actions, *, next_state=None, next_actions=None):
           del next_state, next_actions
           inputs = torch.cat([state, actions], dim=-1)
           return torch.cat([self.q1(inputs), self.q2(inputs)], dim=-1)

   @register_prefix_critic(
       "custom_twin", capabilities=CustomTwinCritic.capabilities,
   )
   def make_custom_twin_critic(state_dim, action_dim, cfg):
       return CustomTwinCritic(state_dim, action_dim, cfg)

它的 ``forward`` 返回 ``[B, 2]``。``multi_q_mlp`` 基于共享的 ``MultiQHead`` 构建 ``num_q_heads`` 个独立 Q head，``cross_q_mlp`` 用 batch renormalization 联合计算当前和下一状态的 Q。TD3 的双 Q critic 就是 ``num_q_heads: 2`` 的 ``multi_q_mlp``。选择时要满足算法要求的 capabilities。

仓库内的 head 放在 ``rlinf/models/embodiment/mlp_policy/heads.py``，模块被导入时即完成注册；构建 policy 或算法时就会触发这次导入。仓库外的 head 在通过 ``RLINF_EXT_MODULE`` 加载的模块里注册，确保每个 worker 进程都执行到。然后在配置中选择注册名：

.. code:: yaml

   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

这会替换两个网络，保留特征模型和训练 worker。只要 Q head 数量满足要求，AC 和 TD3 都可以使用 ``multi_q_mlp``。

.. _flexrlt-algorithm:

扩展 FlexRLT Algorithm
----------------------

actor objective、critic target 或下一动作的生成规则发生变化时，扩展算法。三个入口分别是 ``critic_loss``、``actor_loss`` 和 ``target_actions``。基类的 ``critic_loss`` 已实现 chunk 折扣的 Bellman target 和 critic MSE；只有这些规则不同时才需要覆盖。用 ``requirements`` 声明所需的 head 行为，再注册子类。

下面的算法复用 BC/Q objective，以最小 Q 聚合，并用 target actor 生成 bootstrap 动作：

.. code:: python

   from rlinf.algorithms.prefix_off_policy import (
       PrefixAlgorithmRequirements, PrefixOffPolicyAlgorithm,
       register_prefix_off_policy_algorithm,
   )
   from rlinf.models.embodiment.base_policy import ForwardType

   @register_prefix_off_policy_algorithm("my_algorithm")
   class MyAlgorithm(PrefixOffPolicyAlgorithm):
       requirements = PrefixAlgorithmRequirements(
           actor_distribution="deterministic",
           requires_action_noise=True,
       )

       def actor_loss(self, context, batch):
           return self._actor_loss(
               context,
               batch,
               actor_agg_q="min",
               apply_reference_dropout=False,
               apply_action_noise=True,
           )

       def target_actions(self, context, next_obs):
           actions, _, _ = context.target_model(
               forward_type=ForwardType.SAC,
               obs=next_obs,
           )
           return actions

``actor_loss`` 返回共享 helper 的 loss、entropy 统计值和 metrics。``target_actions`` 将 ``next_obs`` 传给 worker 持有的 ``context.target_model``，返回 critic target 所需的 flat action。它和 ``context.model`` 都通过示例中的 FSDP 包装后 ``forward`` 调用。普通 Q 和 CrossQ 后端由 ``PrefixPolicy`` 选择。

``_actor_loss`` 复用 intervention-aware BC target、BC/Q 权重 schedule 和 Q 聚合。不同的目标可以直接实现 ``actor_loss``。通过 ``RLINF_EXT_MODULE`` 让 driver 和 worker 都载入注册。

配置并启动扩展
--------------

先按 :doc:`ManiSkill 示例 <../examples/embodied/flexrlt>` 准备环境和模型路径，再创建 ``examples/embodiment/config/maniskill_flexrlt_custom.yaml``：

.. code:: yaml

   defaults:
     - maniskill_prefix_stage2_td3_mlp_steam
     - _self_

   algorithm:
     name: my_algorithm
   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

从仓库根目录启动，所用环境和 checkpoint 继承自原 preset：

.. code:: bash

   export RLINF_EXT_MODULE=flexrlt_ext
   bash examples/embodiment/run_embodiment.sh maniskill_flexrlt_custom

只替换 head 时，保留原来的 ``algorithm.name``。替换特征模型时，注册 adapter 的 builder，并在同一个子配置中设置 ``rollout.prefix_feature_model.model_type`` 和模型路径；返回张量必须符合上面的观测契约。

内置算法的更新规则
------------------

内置 ``ac`` 用在线 actor 采样的下一步动作计算 bootstrap，并要求 ``algorithm.actor_agg_q: q1``。``td3`` 保留 `RLT 论文 <https://arxiv.org/html/2604.23073>`_ 第 IV-B 节公式（4）、（5）的固定方差高斯 policy：actor 更新默认采样动作，actor loss 中的 Q 项和 BC 项使用同一个动作样本。actor 优化 Q1，critic target 取前两个 Q head 的较小值。设置 ``algorithm.actor_update_action_noise: false`` 可改用均值动作更新 actor。

计算 target 动作时，先取 target actor 的均值，再加入一次截断高斯噪声，最后把动作限制在 [-1, 1]。``algorithm.target_policy_noise`` 默认跟随 actor head 的 ``noise_sigma``，``deterministic_mlp`` 的默认值是 0.1；``algorithm.target_noise_clip`` 默认是 0.5。preset 将两处采样标准差都设为 0.1。任一 target noise 参数设为 0 都会关闭 smoothing。

每完成 ``algorithm.critic_actor_ratio`` 次 critic 更新，actor 和两个 target 网络一起更新。默认比例为 2，首次更新在第二次 critic 更新后，更新比例遵循 RLT，target 更新时机遵循 `TD3 官方实现 <https://github.com/sfujim/TD3/blob/master/TD3.py>`_。显式设置的 ``algorithm.target_update_freq`` 必须等于 ``critic_actor_ratio``，``algorithm.target_update_type`` 必须为 ``all``。BC、reference dropout 和 BC/Q 权重 schedule 仍与 AC 复用。

两种算法的 critic target 都取前两个 Q head 的最小值；``multi_q_mlp`` 的其余 head 回归到同一个 target，因此应省略 ``algorithm.agg_q`` 字段或将它设为 ``min``。两者都使用 ``algorithm.entropy_tuning.alpha_type: fixed_alpha``。构建算法时会校验这些设置。

用冻结模型候选动作训练 EXPO
---------------------------

EXPO 从冻结 VLA 生成的候选动作及其小幅修正中选择执行动作。以 ``examples/embodiment/config/maniskill_prefix_expo_openpi_pi05_async.yaml`` 为起点：它使用现有 OpenPI Prefix 特征、随机 ``residual_mlp`` 编辑器和十个 ``multi_q_mlp`` critic，任务是 ManiSkill 关节动作控制的完整抓取插孔。

准备 OpenPI/ManiSkill 环境和两张 GPU，将 ``rollout.prefix_feature_model.model_path`` 与 ``rollout.prefix_feature_model.openpi_data.norm_stats_path`` 分别设为相互匹配的关节动作 SFT 检查点和归一化统计文件。在已激活的环境中运行：

.. code:: bash

   bash examples/embodiment/run_async.sh maniskill_prefix_expo_openpi_pi05_async

命令会启动异步采集和训练，记录 TensorBoard 指标并保存评测视频。启动前可在示例中调整硬件放置、环境数量，以及按 epoch 设置的评测和保存间隔。

候选选择与参数更新
^^^^^^^^^^^^^^^^^^

每个观测先由冻结模型采样 ``algorithm.num_candidates`` 个动作 chunk，再对前 ``algorithm.num_edits`` 个候选各做一次有界随机修正。示例将两个参数都设为八，因此共有十六个选项。选择器随机抽取两个 target critic，以二者的最小 Q 为每个选项打分，执行分数最高的 chunk。训练和评测都会随机修正候选动作，并随机抽取 critic 对。

critic target 将一个执行 chunk 内的奖励相加，下一状态价值只乘一次 ``gamma``。下一动作由在线编辑器和 target critic 选出，再独立抽取另一对 target critic 计算该动作的 bootstrap 值。真正终止的 transition 停止 bootstrap；``algorithm.bootstrap_time_limits: true`` 时，超时继续 bootstrap。EXPO 要求 ``backup_entropy: false``，所有在线 critic 回归到同一个 target。

actor 修正 replay 中记录的实际执行动作，并最小化 ``alpha * log_prob - mean(Q)``，其中 Q 取全部在线 critic 的均值。温度训练使用相同的 replay 动作作为编辑条件。``target_entropy: null`` 时，对整个执行 chunk 的每个坐标累加 ``log(edit_scale_i) - 0.5`` 作为目标熵。编辑动作的概率密度同时计入 tanh 和尺度变换的 Jacobian。

``ref_chunk`` 和 ``candidates`` 使用环境适配器接受的动作坐标。OpenPI 沿用已有的 output transform 解码每个候选。``actor_head.edit_scale`` 使用同一坐标系，可配置为正标量或沿展平 chunk 重复的向量。``0.02`` 表示动作坐标上的修正上限，具体物理含义取决于环境控制器。

采集、回放与检查点
^^^^^^^^^^^^^^^^^^

设置 ``rollout.policy_routing: full_task`` 可采集完整任务，同时关闭 routing gate，并移除 ``expert_model`` 配置。示例在收集至少 4,000 个 chunk transition 并完成 20,000 次 learner 预热更新期间，执行冻结模型的参考动作。沿用 Prefix 原有调度，从预热开始，每二十次 critic 更新就执行一次 actor 和温度更新。预热后 rollout 启用 EXPO 选择，replay 每新增一条 transition，learner 就增加二十次 critic 更新。更新次数与 Q head 数量无关。每轮追赶更新的上限由 ``rlt_schedule.max_updates_per_train_step`` 控制。

EXPO 将特征调用扩展为 ``extract_prefix_obs(env_obs, num_candidates=K)``。适配器除原有字段外，还需返回形状为 ``[B, K, L, A]`` 的 ``candidates``，首个候选必须等于 ``ref_chunk``。Rollout 保留实际执行长度，将当前与下一观测的候选都写入 replay，包括终止观测。EXPO 目标函数只依赖这些张量、奖励和终止标志；任务初始化、控制器解码和几何信息属于环境适配器。

Rollout 权重同步会一起发布在线编辑器和 target critic。检查点沿用已有 SAC 保存流程；恢复训练使用 ``runner.resume_dir``。单独评测前，从同一个 actor 检查点目录中读取在线模型和 target 模型，合并所需权重：

.. code:: python

   from pathlib import Path

   import torch

   from rlinf.algorithms.prefix_off_policy.base import rollout_actor_target_critic

   actor_dir = Path("/path/to/actor/checkpoint")
   online = torch.load(
       actor_dir / "model_state_dict/full_weights.pt",
       map_location="cpu",
       weights_only=True,
   )
   target = torch.load(
       actor_dir / "sac_components/target_model/checkpoint_rank_0.pt",
       map_location="cpu",
       weights_only=True,
   )
   torch.save(rollout_actor_target_critic(online, target), actor_dir / "expo_eval.pt")

运行 ``evaluations/eval_embodied_agent.py`` 时，设置 ``runner.only_eval: true``，并将 ``runner.ckpt_path`` 指向生成的 ``expo_eval.pt``。如果只加载在线模型权重，候选选择使用的 critic 就会与 rollout 不同。

其他算法若需要这些行为，可在 requirements 中声明 ``trains_entropy_temperature`` 或 ``rollout_uses_target_critic``，覆盖 ``alpha_loss`` 实现温度训练，覆盖 ``predict_action_batch`` 实现 rollout 选择器，覆盖 ``rollout_state_dict`` 组合所需权重。Replay、优化器、target 更新与权重同步仍由现有 Prefix worker 管理。

EEF 旋转输入
^^^^^^^^^^^^

使用归一化 EEF rot6d 动作时，注册的 head 应先撤销旋转归一化，再做施密特正交化后送入网络。内置的 ``eef_rot6d_residual_mlp`` 和 ``eef_rot6d_multi_q_mlp`` head 提供此转换，平移和夹爪输入保持归一化。编辑器将有界残差加到原始归一化参考动作上，熵在旋转投影前计算。``prefix.action_space: normalized_model`` 需与环境的 ``normalized_actions_stats_path`` 配套。任务和配置见 :ref:`maniskill-eef-preinsert`。
