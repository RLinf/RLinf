FlexRLT：冻结 VLA 特征与可插拔 Policy
=====================================

为 FlexRLT 增加特征模型、policy head 或更新规则。ManiSkill 的 checkpoint 下载、训练启动和 STEAM 校准见 :doc:`FlexRLT <../examples/embodied/flexrlt>`。本页从那份已经能运行的配置出发，说明观测约定以及注册新组件时要改的位置。

这里的 prefix 指 VLA 生成动作之前计算出的图像、语言 hidden states。在线 RL 期间，VLA 保持冻结；另一个 actor 读取这些特征、机器人状态和 VLA 生成的 reference action，输出要执行的 action chunk。

概览
----

选择这条路线时，先考虑是否需要专门训练 RL 表示。设置 ``prefix.pool: masked_mean``，就可以直接对已有 base 或 SFT VLA 的有效 prefix token 求均值，得到 ``z_rl``。如果希望使用 RL Token 学到的表示，则设置 ``prefix.pool: rlt_token``，加载 Stage 1 训练过的 token transformer。两种表示共用后续的 actor-critic 训练流程；RLT 的两阶段训练见 :doc:`../examples/embodied/rlt`。

在线更新只训练 actor 和 critic head，梯度不回传到 VLA。replay 保存 ``{z_rl, proprio, ref_chunk}`` 而不是图像，learner 可以复用采集时提取的特征。但 rollout 仍然要运行 VLA，提取特征并生成 reference action；冻结模型并不意味着省掉这部分推理。

与 RLT 专属的 AC、TD3 policy 和 worker 相比，FlexRLT 使用同一个 ``prefix_policy`` 模型和 ``prefix_off_policy`` worker，在配置中分别选择 actor head、critic head 和算法。chunk reward 折扣、BC target 和权重 schedule 由 AC、TD3 共用。比较新的特征或 head 时，可以保留训练循环，只替换对应组件。pooling 和 RLT token 的任务效果仍需在相同设置下比较。

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
     - ``algorithm.name``（配合 ``loss_type: prefix_off_policy``）
     - ``register_prefix_off_policy_algorithm``
     - ``rlinf/algorithms/prefix_off_policy/``

各层的职责边界按「谁拥有什么」划分：

* **特征模型**决定原始观测如何变成 ``{z_rl, proprio, ref_chunk}``。它保持冻结，不带优化器，也不接收梯度。
* **policy** 决定这三个张量如何拼成 policy state 和 reference 输入。``prefix_policy`` 自身不增加参数，只组合两个 head，并暴露 worker 所需的 ``BasePolicy`` forward 类型。
* **head** 拥有网络本身。actor 和 critic 是两个独立对象、两套独立注册表，因此可以把随机 actor 配 CrossQ critic，或只替换其中一侧。
* **算法**拥有 loss 和更新调度。它读取 head 在 ``capabilities`` 中声明的能力，拒绝无法训练的搭配。

配置校验按**观测契约**而不是按名字把 policy 与 ``prefix_off_policy`` 匹配，所以新 policy 以 ``obs_contract=ObsContract.PREFIX`` 注册后，无需修改 ``rlinf/config.py`` 就能通过校验。它仍需满足 :ref:`flexrlt-policy-contract` 中的 policy 契约。

如何扩展
^^^^^^^^

找到你要改的那一行，只改那一行：

* **换 VLA** —— 写一个提供 ``z_dim`` 和 ``extract_prefix_obs`` 的 adapter，按具身模型注册，再把 ``rollout.prefix_feature_model`` 指过去。policy、head、算法和 replay 格式都不变。见 :ref:`扩展 BaseVLA 特征模型 <flexrlt-feature-model>`。
* **换网络** —— 继承 ``PrefixActorHead`` 或 ``PrefixCriticHead``，声明 capabilities 后注册。rollout 运行时、replay 格式和训练循环都不受影响。见 :ref:`扩展 MLP Actor 与 Critic <flexrlt-heads>`。
* **换更新规则** —— 继承 ``PrefixOffPolicyAlgorithm``，实现 ``actor_loss`` 和 ``target_actions``，并声明 ``requirements``。chunk 折扣、BC target 和权重调度继续共享。见 :ref:`扩展 FlexRLT 算法 <flexrlt-algorithm>`。
* **在同一特征上换 policy** —— 例如换成循环（recurrent）actor。继承 ``BasePolicy``，满足下面的契约，以 ``obs_contract=ObsContract.PREFIX`` 注册，再设置 ``actor.model.model_type``。算法、replay 格式和 rollout 运行时保持不变。

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
     - worker 把名字中含 ``q_head``、``encoder``、``encoders`` 或 ``state_proj`` 的参数放进 critic 优化器，其余放进 actor 优化器。不要把只属于 actor 的模块命名为 ``encoder``。
   * - ``predict_action_batch(env_obs, mode=..., return_obs=True)``
     - rollout。返回 ``[B, num_action_chunks, action_dim]`` 的动作，并在 ``forward_inputs`` 中放入 ``action`` 以及 ``z_rl``、``proprio``、``ref_chunk``，它们会成为 replay transition。

有状态的 policy（例如循环 actor）没有按环境重置的 rollout 接口；应把状态放进特征模型返回的观测中。

运行示例
--------

ManiSkill 插接 preset、checkpoint 目录、路由设置和 STEAM 校准命令见 :doc:`FlexRLT <../examples/embodied/flexrlt>`。用那一页启动 ``maniskill_prefix_stage2_td3_mlp_steam``。下面说明这次运行保存的数据，以及增加组件时要修改的代码。

Replay 与更新
-------------

路由可能选择 base、actor 或 expert 的动作，通过 ``ActionPayload`` 发送实际动作和可选反馈。gate 用 ``rollout_info_keys`` 声明反馈字段；启用 gate 的环境必须实现 ``RolloutInfoReceiver.set_rollout_infos``，env worker 会在启动时检查。

环境通过 ``TrajectoryInfoProvider.trajectory_info_spec`` 声明校准标签。这些标签和 gate 诊断信息存放在 ``Trajectory.trajectory_infos`` 中，与 ``forward_inputs`` 分开。trace writer 读取后，replay 会丢弃这些诊断数据，只记录实际发送给环境的动作及其前后特征：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的 action chunk
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

critic 按 ``gamma`` 对 chunk 内 reward 折扣，再用 ``gamma ** chunk_horizon`` bootstrap 下一状态的 Q。actor objective 为 ``-q_weight * Q + bc_weight * BC``。BC 通常以 ``ref_chunk`` 为目标；带 intervention 标记的 step 则使用实际执行的 intervention action。

配置最小示例
------------

迁移到其他任务时，把冻结 VLA 放在 ``rollout.prefix_feature_model`` 下，把可训练 head 放在 ``actor.model`` 下。以下片段用 AC、stochastic MLP actor 和 twin-Q critic 展示这种配置关系。启动前，还需从完整 preset 补齐环境、placement、optimizer 和 rollout 设置。

.. code:: yaml

   algorithm:
     loss_type: prefix_off_policy
     name: ac
     gamma: 0.99
     q_weight: 1.0
     bc_weight: 1.0
     entropy_tuning:
       alpha_type: fixed_alpha
       initial_alpha: 0.0

   rollout:
     prefix_feature_model:
       model_type: openpi
       model_path: /path/to/base-or-sft-vla
       precision: bf16
       prefix:
         pool: masked_mean
         image_only: false
       openpi_data:
         repo_id: my_lerobot_dataset
         norm_stats_path: /path/to/norm_stats.json
       openpi:
         task: eval
         config_name: pi05_franka_state

   actor:
     model:
       model_type: prefix_policy
       z_dim: 2048
       proprio_dim: 19
       action_dim: 7
       num_action_chunks: 10
       ref_num_action_chunks: 20
       actor_head:
         name: fixed_std_mlp
         hidden_dims: [256, 256, 256]
         activation: tanh
         fixed_std: 0.002
       critic_head:
         name: multi_q_mlp
         hidden_dims: [256, 256, 256]
         num_q_heads: 2

``actor.model.z_dim`` 是特征模型 ``z_rl`` 输出的宽度。``proprio_dim`` 对应返回的机器人状态，``action_dim`` 对应环境的单步动作维度，``num_action_chunks`` 对应实际执行的 chunk 长度。reference 至少要覆盖这些动作；``ref_num_action_chunks`` 可以更长。

配置校验要求 ``loss_type: prefix_off_policy``、观测契约为 ``prefix`` 的 policy、两个已注册的 head name 和已注册的 ``algorithm.name``。actor distribution、action noise 支持和 Q head 数量不匹配时，会在 worker 启动前报错。``prefix_policy`` 是仓库内满足该契约的 policy；任何以 ``obs_contract=ObsContract.PREFIX`` 注册的模型都会被接受。

当前实现
--------

冻结 VLA 通过三个 tensor 向可训练 policy 提供 observation。下面沿着特征提取、actor 和 critic 调用的顺序解释它们；接入其他 VLA 或 head 时，保持这些 shape 一致。

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

OpenPI 的 ``Pi0Eval`` 通过 ``OpenPiPrefixMixin`` 实现这个接口，不增加参数或改变 checkpoint key。它先转换环境 observation，计算一次图像和语言 prefix，再复用 cache 提取 ``z_rl``、生成 reference action；选定的机器人状态作为 ``proprio``。``PrefixRolloutRuntime`` 将特征模型设为 ``eval`` 并关闭梯度。

Prefix pooling
^^^^^^^^^^^^^^

先确定哪些 token 参与构造 ``z_rl``，再选择聚合方式。``prefix.image_only: true`` 只保留图像 token；设为 ``false`` 时保留图像和语言 token。``prefix.pool`` 与它同属 ``prefix`` 块，该块与模型配置中的 ``openpi`` 并列：

* ``masked_mean`` 对有效 token 求均值，是 ``openpi.use_rlt`` 为 false 时的默认值。
* ``mean`` 对选定的整个序列求均值，包括被 mask 的位置。
* ``last`` 读取最后一个有效 token；未提供 mask 时读取序列末端。
* ``rlt_token`` 运行 Stage 1 的 ``rlt_module.encode_flat``，不做 pooling。

``pool_prefix`` 接收 shape 为 ``[B, T, D]`` 的 ``hidden`` 和可选 boolean mask，前三种 pooling 返回 ``[B, D]``。未设置 ``prefix.pool`` 时，``resolve_prefix_pool`` 会在 ``openpi.use_rlt`` 为 true 时选择 ``rlt_token``，否则选择 ``masked_mean``。``prefix.image_only`` 默认为 true。比较不同表示时，请显式设置这两个字段。

rollout 启动时检查特征模型的 ``z_dim`` 与 ``actor.model.z_dim`` 是否一致，每次提取特征时还会检查 ``z_rl`` 的实际宽度是否符合模型声明。

PrefixPolicy 的 forward contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PrefixPolicy`` 位于 ``rlinf/models/embodiment/mlp_policy/prefix_mlp_policy.py``。它把展平后的 ``z_rl`` 和 ``proprio`` 拼成 policy state，截取 ``ref_chunk`` 的前 ``num_action_chunks`` 个动作并展平，用作 actor 的 reference 输入和 BC target。它自身不增加参数：全部权重来自按配置构建的 actor 和 critic head。冻结 VLA、可选 expert 和 gate，以及 terminal 特征缓存由 ``PrefixRolloutRuntime`` 管理。FSDP worker 通过 ``q_head`` 属性识别 critic 参数。

它使用标准 ``ForwardType``，与 ``MLPPolicy`` 相同：

* ``ForwardType.SAC``：返回 actor head 的 ``actions, logprobs, None``。
* ``ForwardType.SAC_Q``：返回当前状态的 Q 值，按 head 拼接。
* ``ForwardType.CROSSQ_Q``：joint CrossQ critic 返回 ``current_q, next_q``。
* ``predict_action_batch``：将 action reshape 为 ``[B, num_action_chunks, action_dim]``，并在 ``forward_inputs`` 中返回 action 和特征 observation。

算法依据 critic head 的 ``supports_cross_q`` 能力选择 ``SAC_Q`` 或 ``CROSSQ_Q``，两种后端都落到同一个 policy 方法。所有训练调用都经过 FSDP 包装后的 ``forward``。

.. _flexrlt-feature-model:

扩展 BaseVLA 特征模型
---------------------

使用其他 VLA 时，将它的 observation 和输出适配为 ``PrefixObs`` 即可接入。现有 OpenPI adapter 是一种实现，policy head 不依赖 OpenPI。这里的 BaseVLA 指供应商模型或基座模型，不是 RLinf 的基类。具体 adapter 要支持 ``eval``、``requires_grad_``、``to`` 和 ``extract_prefix_obs``；如果 route 还会调用 VLA 的动作 API，也需提供对应接口。

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

``register_prefix_actor`` 让 model builder 可以通过 ``custom_mlp`` 找到这个 head。``mean_action`` 为 policy 的 SFT forward 提供无噪声预测，``forward`` 按请求应用 dropout 和探索噪声，再返回动作及 log probability。内置的 ``fixed_std_mlp`` 用于 AC，``deterministic_mlp`` 用于 TD3；AC 要求 stochastic actor，TD3 要求支持 action noise 的 deterministic actor。

注册 critic head
^^^^^^^^^^^^^^^^

critic 需要声明 ``CriticHeadCapabilities`` 和 ``num_q_heads``。下面的实现将同一组 state-action 输入送入两个独立 Q 网络，并注册为 ``custom_twin``：

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

仓库内的 head 放在 ``rlinf/models/embodiment/mlp_policy/heads.py``，模块被导入时即完成注册；构建 policy 或算法时就会触发这次导入。出树的 head 在通过 ``RLINF_EXT_MODULE`` 加载的模块里注册，确保每个 worker 进程都执行到。然后在配置中选择注册名：

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
           uses_target_actor=True,
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

``actor_loss`` 返回共享 helper 的 loss、entropy 统计值和 metrics。``target_actions`` 将 ``next_obs`` 传给 worker 持有的 ``context.target_model``，返回 critic target 所需的 flat action。它和 ``context.model`` 都是经过包装的训练模型；按示例通过 ``forward`` 调用，不要用 ``.module`` 绕过 FSDP。普通 Q 和 CrossQ 后端由 ``PrefixPolicy`` 选择。

可选的 ``_actor_loss`` helper 为 AC、TD3 复用 intervention-aware BC target、BC/Q weight schedule 和 Q 聚合。其他 objective 可以直接实现 ``actor_loss``。在 registry 的 ``_load_builtin_algorithms`` 中导入模块后，设置 ``algorithm.name: my_algorithm``，继续使用 ``loss_type: prefix_off_policy``。

内置 ``ac`` 用在线 actor 采样的下一步动作计算 bootstrap，并要求 ``algorithm.actor_agg_q: q1``。``td3`` 保留 `RLT 论文 <https://arxiv.org/html/2604.23073>`_ 第 IV-B 节公式（4）、（5）的固定方差高斯策略：actor 更新默认采样动作，Q 目标和 BC 使用同一个样本。actor 优化 Q1，critic target 取前两个 Q head 的较小值。设置 ``algorithm.actor_update_action_noise: false`` 可改用均值动作更新 actor。

计算 target 动作时，先取 target actor 的均值，再加入一次截断高斯噪声，最后把动作限制在 [-1, 1]。``algorithm.target_policy_noise`` 默认跟随 actor head 的 ``noise_sigma``，``deterministic_mlp`` 的默认值是 0.1；``algorithm.target_noise_clip`` 默认是 0.5。preset 将两处采样标准差都设为 0.1，论文只写了较小的固定标准差，没有给出数值。target actor head 不会再额外采样一份噪声。任一 target noise 参数设为 0 都会关闭 smoothing。

每完成 ``algorithm.critic_actor_ratio`` 次 critic 更新，actor 和两个 target 网络一起更新。默认比例为 2，首次更新在第二次 critic 更新后，更新比例遵循 RLT，target 更新时机遵循 `TD3 官方实现 <https://github.com/sfujim/TD3/blob/master/TD3.py>`_。禁止 actor 训练时，TD3 的两个 target 也保持不变。显式设置的 ``algorithm.target_update_freq`` 必须等于 ``critic_actor_ratio``，``algorithm.target_update_type`` 必须为 ``all``。BC、reference dropout 和 BC/Q 权重 schedule 仍与 AC 复用。

两种算法的 critic target 都取前两个 Q head 的最小值；``multi_q_mlp`` 的其余 head 回归到同一个 target，因此应省略 ``algorithm.agg_q`` 字段或将它设为 ``min``。两者都要求 ``algorithm.entropy_tuning.alpha_type: fixed_alpha``，不训练 entropy temperature。构建算法时会校验这些设置。

实现位置与检查点
----------------

需要从配置追踪到实现时，可以从以下文件开始：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 组件
     - 代码位置
   * - ``PrefixObs``、feature contract、pooling、policy 输入宽度
     - ``rlinf/models/embodiment/prefix_features.py``
   * - trainable policy
     - ``rlinf/models/embodiment/mlp_policy/prefix_mlp_policy.py``
   * - actor/critic heads 与 registry
     - ``rlinf/models/embodiment/mlp_policy/heads.py``
   * - 运行时配置解析
     - ``rlinf/algorithms/prefix_off_policy/config.py``
   * - off-policy algorithms
     - ``rlinf/algorithms/prefix_off_policy/``
   * - rollout 生命周期与 replay/scheduling
     - ``rlinf/algorithms/prefix_off_policy/rollout.py``、``replay.py``、``schedule.py``
   * - OpenPI 特征 mixin
     - ``rlinf/models/embodiment/openpi/prefix.py``
   * - FSDP actor worker
     - ``rlinf/workers/actor/fsdp_prefix_off_policy_worker.py``

遇到 shape 错误，先比较特征模型的 ``z_dim``、``actor.model.z_dim`` 和环境 flat action 宽度 ``num_action_chunks * action_dim``。动作尺度不对时，检查每个 checkpoint 使用的 OpenPI data config 和 ``norm_stats.json``。使用 RLT token 的特征模型还必须从 Stage 1 checkpoint 加载完整 ``rlt_module``，对应步骤见 :doc:`../examples/embodied/rlt`。
