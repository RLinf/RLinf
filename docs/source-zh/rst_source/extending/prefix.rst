Prefix Fine-Tune：冻结 VLA 特征与可插拔 Policy
==============================================

Prefix Fine-Tune 使用冻结 VLA 特征训练可插拔的轻量 policy。本页按 rollout、policy、algorithm 的调用顺序说明当前实现，再说明如何接入 VLA adapter、MLP head 和 off-policy algorithm。RL Token 的 Stage 1 token transformer 和任务运行流程见 :doc:`../examples/embodied/rlt`。

Prefix Fine-Tune 将 VLA 表示和在线 RL policy 分成两个生命周期。rollout worker 加载冻结的 VLA feature model，actor worker 只训练轻量的 actor 与 critic head。两个 worker 通过固定的 ``{z_rl, proprio, ref_chunk}`` contract 连接，因此 VLA 的视觉语言编码器、policy head 和更新算法可以分别替换。

概览
----

Prefix 的一次 policy step 经过以下路径：

1. 环境提供图像、语言、状态和任务元信息。
2. ``rollout.prefix_feature_model`` 调用 VLA 的 ``extract_prefix_obs``。
3. VLA 生成 ``z_rl``、``proprio`` 和 ``ref_chunk``。
4. rollout 侧的 ``prefix_policy`` 拼接 state，并调用注册的 actor head 生成 action chunk。
5. learner 侧的 ``prefix_off_policy`` worker 收取 transition、写入 replay buffer、计算 critic target 和 actor objective，再同步 rollout policy。

Prefix 的默认 ``z_rl`` 是 VLM prefix hidden 的 masked mean。``prefix.pool: mean``、``last`` 和 ``rlt_token`` 提供其他表示路径。``rlt_token`` 会读取 Stage 1 训练得到的 token transformer，相关训练步骤放在 :doc:`../examples/embodied/rlt`。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 冻结特征模型
      :text-align: center

      ``PrefixFeatureModel`` contract

   .. grid-item-card:: Prefix observation
      :text-align: center

      ``z_rl``、``proprio``、``ref_chunk``

   .. grid-item-card:: 可训练 policy
      :text-align: center

      ``PrefixPolicy`` 与注册的 heads

   .. grid-item-card:: 更新 runtime
      :text-align: center

      ``prefix_off_policy`` worker

当前实现
--------

冻结 VLA 如何生成 Prefix observation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.contracts`` 定义了 feature model 的最小接口：

.. code:: python

   from typing import Any, Protocol
   from rlinf.models.embodiment.prefix.types import PrefixObs

   class PrefixFeatureModel(Protocol):
       @property
       def z_dim(self) -> int: ...

       def extract_prefix_obs(
           self, env_obs: dict[str, Any]
       ) -> PrefixObs: ...

``PrefixObs`` 是一个 TypedDict，包含三个 batch-first tensor。下表中的 ``z_dim`` 指 feature model 的原始输出宽度；启用 state history 后，rollout 会在 ``z_rl`` 后追加历史 proprio。

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

OpenPI 的 ``Pi0Eval.extract_prefix_obs`` 是当前实现。它先将环境输入转换成 OpenPI observation，再由 ``build_prefix_cache`` 运行图像和语言 Prefix。``_encode_stage2_z`` 根据 ``prefix.pool`` 选择 pooling 或 RLT token，``_sample_actions_from_prefix_cache`` 生成 reference chunk，最后将原始状态映射为 ``proprio``。rollout worker 在初始化时将这个 model 置为 ``eval`` 并关闭梯度。

Prefix pooling
^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.pool.pool_prefix`` 接收 ``hidden``\ （``[B, T, D]``）和可选的 boolean mask，返回 ``[B, D]``：

* ``masked_mean`` 只平均有效 token，默认用于 Prefix Fine-Tune。
* ``mean`` 平均整个 token 序列。
* ``last`` 读取最后一个有效 token；没有 mask 时读取序列末端。
* ``rlt_token`` 是配置层的标记。OpenPI 会转到 ``rlt_module.encode_flat``，因此不会直接调用 ``pool_prefix``。

``resolve_prefix_pool`` 按 ``prefix.pool``、``stage2_z_source``、``use_rlt`` 和 ``rlt_use_mask`` 解析最终模式。建议在配置中显式写出 ``prefix.pool``，这样 Stage 1 和 Stage 2 的表示来源可以直接检查。

State history
^^^^^^^^^^^^^

``actor.model.state_history.enable: true`` 时，``StateHistoryBuffer`` 在 policy decision rate 保存最近 ``steps`` 个 proprio，并将展平结果追加到 ``z_rl``。``pad`` 支持 ``zero`` 和 ``repeat``。episode reset 会清理指定环境的 history。policy 输入宽度由 ``get_prefix_feature_dim`` 计算：

.. code:: text

   runtime_z_dim = z_dim + steps * proprio_dim  # state_history.enable=true
   runtime_z_dim = z_dim                         # state_history.enable=false

rollout 和 actor 使用同一个 ``actor.model.state_history`` 配置，避免 feature width 不一致。

PrefixPolicy 的 forward contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.policy.PrefixPolicy`` 继承 ``BasePolicy``，但它的训练入口是 off-policy forward：

* ``ForwardType.SAC`` 调用 actor head，返回 ``actions, logprobs, None``。
* ``ForwardType.SAC_Q`` 调用 critic head，返回每个 Q head 的值。
* ``ForwardType.CROSSQ_Q`` 在支持 CrossQ 的 critic 上同时评估当前和下一状态。
* ``predict_action_batch`` 将 flat action reshape 为 ``[B, num_action_chunks, action_dim]``，并在 ``forward_inputs`` 中返回 action 与 Prefix observation。

policy state 是 ``concat(flatten(z_rl), flatten(proprio))``。reference 先按 action dimension reshape，再截取前 ``num_action_chunks``，用于 actor 条件输入和 BC target。``ref_num_action_chunks`` 可以大于实际执行的 ``num_action_chunks``，feature model 因此可以提供更长的 reference window。

PrefixPolicy 不读取 VLA 参数。VLA 由 rollout worker 单独持有，PrefixPolicy 的参数只有 actor head 和 critic head，FSDP worker 通过 ``q_head`` 路径识别 critic 参数。

配置最小示例
------------

下面的片段展示一个使用 OpenPI masked-mean feature、stochastic MLP actor 和 twin-Q critic 的 Stage 2 配置。路径和数据字段需要按具体环境补全。

.. code:: yaml

   algorithm:
     loss_type: prefix_off_policy
     name: ac
     gamma: 0.99
     q_weight: 1.0
     bc_weight: 1.0

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
         name: twin_q_mlp
         hidden_dims: [256, 256, 256]

``actor.model.z_dim`` 必须等于 feature model 输出的 base ``z_rl`` 宽度。打开 state history 后，history width 由配置自动追加。``actor.model.proprio_dim``、``action_dim``、``num_action_chunks`` 和 feature model、环境 action contract 保持一致。

训练路径由 ``rlinf/config.py`` 校验。``loss_type: prefix_off_policy`` 要求 ``actor.model.model_type: prefix_policy``、两个 head name 和已注册的 ``algorithm.name``。不匹配的 actor distribution、action noise 能力或 Q head 数量会在 worker 启动前报错。

扩展 BaseVLA 特征模型
---------------------

新的 VLA 通过 feature model contract 接入。这里的 BaseVLA 指现有 VLA 基座或供应商模型；RLinf 当前没有名为 BaseVLA 的通用基类。Prefix runtime 使用 ``PrefixFeatureModel`` Protocol 约束冻结模型与 rollout worker 的边界。实际对象需要支持 ``eval``、``requires_grad_``、``to`` 和 ``extract_prefix_obs``；当 route 会执行 VLA reference 或复用 VLA 动作 API 时，还需满足对应 policy 的调用接口。

实现 adapter
^^^^^^^^^^^^

adapter 需要完成四步：将 ``env_obs`` 转成 VLA 输入，运行 Prefix encoder，生成 reference action chunk，将三项结果统一为 float32 batch tensor。下面的骨架使用示意性的预处理、编码和动作采样方法；接入时按供应商的真实 API 实现这三个操作。

.. code:: python

   import torch
   from rlinf.models.embodiment.prefix.pool import pool_prefix

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

``extract_prefix_obs`` 的输入由环境 wrapper 决定，返回 key 必须保持 ``PREFIX_OBS_KEYS``。``ref_chunk`` 的最后一维必须对应环境 action dimension，长度至少覆盖 ``num_action_chunks``。proprio 的语义和归一化方式需要与 ``actor.model.proprio_dim`` 以及训练数据保持一致。

接入 model factory
^^^^^^^^^^^^^^^^^^

rollout worker 通过 ``get_model`` 构造 ``rollout.prefix_feature_model``。自定义 adapter 需要在 ``rlinf/models/__init__.py`` 增加 lazy builder，在 ``rlinf/config.py`` 的 ``SupportedModel`` 注册 model type，并将其归入 embodied model。builder 返回的对象必须同时满足 rollout 的 PyTorch module 生命周期和 Prefix feature contract；需要执行 VLA base action 的 route 还要求对象支持对应的动作调用。

adapter 接入后可以复用现有 ``prefix_policy``、``prefix_off_policy`` worker 和 replay schema。测试至少覆盖 batch size、device/dtype、reset 后的状态语义和 action chunk shape。

扩展 MLP Actor 与 Critic
------------------------

Prefix head 接收已经拼好的 state 和 reference/action tensor。actor 的 ``forward`` 返回 flat action 与每维 log probability，``mean_action`` 返回 BC 使用的无噪声 action。critic 的 ``forward`` 返回 ``[..., num_q_heads]``，CrossQ critic 可以额外返回 next-state Q。

注册 actor head
^^^^^^^^^^^^^^^

继承 ``PrefixActorHead``，声明 ``ActorHeadCapabilities``，再使用 ``register_prefix_actor``。builder 参数依次是 ``state_dim``、``reference_dim``、``action_dim`` 和该 head 的 Hydra config。

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.prefix.heads.base import (
       ActorHeadCapabilities, PrefixActorHead,
   )
   from rlinf.models.embodiment.prefix.heads.registry import register_prefix_actor

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

内置 actor 名称是 ``fixed_std_mlp`` 和 ``deterministic_mlp``。AC 需要 stochastic actor，TD3 需要支持 action noise 的 deterministic actor。capability metadata 让配置校验和 algorithm 组合保持一致。

注册 critic head
^^^^^^^^^^^^^^^^

继承 ``PrefixCriticHead``，设置 ``CriticHeadCapabilities`` 和 ``num_q_heads``，再使用 ``register_prefix_critic``。``multi_q_mlp`` 支持可配置 Q 数量，``twin_q_mlp`` 固定两个独立 Q，``cross_q_mlp`` 额外支持 CrossQ 计算。

.. code:: python

   import torch
   from torch import nn
   from rlinf.models.embodiment.prefix.heads.base import (
       CriticHeadCapabilities, PrefixCriticHead,
   )
   from rlinf.models.embodiment.prefix.heads.registry import register_prefix_critic

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

新的 head 放入 ``rlinf.models.embodiment.prefix.heads`` 并在 registry 的 ``_load_builtin_heads`` 中导入。配置只写注册名：

.. code:: yaml

   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

扩展 Prefix Algorithm
---------------------

``PrefixOffPolicyAlgorithm`` 将 actor objective、critic target action 和 Q 聚合规则抽成组件。新 algorithm 继承该基类，实现 ``requirements``、``actor_forward_kwargs``、``target_actions`` 和 ``aggregate_actor_q``，再用 ``register_prefix_off_policy_algorithm`` 注册。

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

       def actor_forward_kwargs(self):
           return {"apply_action_noise": True}

       def target_actions(self, context, next_obs):
           actions, _, _ = context.target_model(
               forward_type=ForwardType.SAC,
               obs=next_obs,
           )
           return actions

       def aggregate_actor_q(self, all_q_values):
           return self._min_twin_q(all_q_values)

``PrefixOffPolicyAlgorithm`` 已实现 chunk reward 折扣、termination bootstrap、BC target、human intervention mask、BC/Q weight schedule 和 critic MSE。子类只覆盖算法语义。在 registry 的 ``_load_builtin_algorithms`` 中导入模块后，算法通过 ``algorithm.name: my_algorithm`` 选择，worker 仍使用 ``loss_type: prefix_off_policy``。

内置 ``ac`` 使用 stochastic actor 和 Q1 聚合，``td3`` 使用 deterministic actor、target actor、action noise，并支持 ``min``、``q1``、``mean`` 三种 actor Q 聚合。每个 algorithm 的 requirements 会和 head capabilities 一起校验。

EXPO edit actor 与 candidate selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

内置 ``expo`` algorithm 让 stochastic edit actor 围绕 VLA 的 reference chunk 学习。将 ``actor_head.name`` 设为 ``expo_edit_mlp`` 即可使用 ``ExpoEditMLPActor``。它拼接 policy state 和 reference action，采样 ``tanh`` edit，再输出裁剪到 action 范围内的 ``reference + edit_scale * edit``。actor 返回每个 action 维度的 transformed log probability，供 entropy objective 使用。``mean_action`` 在 supervised 调用中使用确定性的 mean edit。

``expo`` 要求 stochastic actor 和至少两个 Q estimate。``multi_q_mlp`` 通过 ``critic_head.num_q_heads`` 提供 Q ensemble。critic target 始终包含未修改的 reference action，并加入 ``algorithm.expo_num_edit_samples`` 个采样 edit。``algorithm.expo_num_min_qs`` 从 Q head 中随机选取子集计算保守的 candidate score，最高分 candidate 提供 chunk-level bootstrap value。chunk 内 reward 先按折扣聚合，再使用 ``gamma ** chunk_horizon`` 计算下一状态 bootstrap。

actor objective 为 ``alpha * log_prob(edit) - q_weight * Q(edit) + bc_weight * BC``，其中 ``Q(edit)`` 对 ensemble estimate 取平均。将 ``bc_weight`` 设为正数可以保留 reference 或 intervention BC 项。``entropy_tuning.alpha_type: softplus`` 开启自动 temperature training；``initial_alpha`` 和 ``target_entropy`` 分别定义初始值和目标值。``actor_tau`` 单独控制 EXPO actor target 的更新速率，critic target 继续使用 ``tau``。

rollout 可以在发送 action 前对 candidate set 打分。``actor.model.expo_candidate_samples`` 会在 reference candidate 外增加采样 edit，rollout policy 选择前两个 Q estimate 的较小值最高的 candidate。将该值设为零可以关闭 rollout-time selection，同时保留 EXPO training。

当前 Prefix 集成包含 edit MLP、candidate Q selection、Q ensemble target、chunk-level EXPO objective 和 temperature training。Real-Time EXPO 的 latency handling、异步 VLA inference 和 noise filtering 不在当前实现中。

训练与运行
----------

仓库提供 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml`` 作为 Prefix Fine-Tune 示例。当前配置使用 ``use_rlt: false`` 和 masked-mean VLM Prefix 特征、``expo_edit_mlp``、十个 head 的 ``multi_q_mlp`` 以及 ``algorithm.name: expo``。将 ``rollout.prefix_feature_model.model_path``、OpenPI ``repo_id``、``norm_stats_path`` 和环境资产改成当前机器的路径后运行：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

rollout 配置使用 ``stage2_z_source: vlm_prefix`` 和 ``prefix_pool: masked_mean``，因此 ``z_rl`` 来自 VLM pooling，同时关闭 RLT token module。STEAM routing 只负责 base VLA、Stage 2 actor 与 expert 的 rollout route，head 和 algorithm 仍由 Prefix runtime 负责。EXPO 的 candidate 和 temperature 设置也位于同一文件中。

Stage 2 replay 的 observation 固定为：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的 action chunk
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

critic 将 chunk 内 reward 按 ``gamma`` 折扣，并使用 ``gamma ** chunk_horizon`` bootstrap 下一状态 Q。actor 使用 ``-q_weight * Q + bc_weight * BC``；BC target 默认是 ``ref_chunk``，带 intervention flag 的 step 使用实际 human action。

实现位置与检查点
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 组件
     - 代码位置
   * - ``PrefixObs`` 与 feature contract
     - ``rlinf/models/embodiment/prefix/contracts.py``、``types.py``
   * - pooling、state history、配置解析
     - ``rlinf/models/embodiment/prefix/pool.py``、``history.py``、``config.py``
   * - trainable policy
     - ``rlinf/models/embodiment/prefix/policy.py``
   * - actor/critic heads 与 registry
     - ``rlinf/models/embodiment/prefix/heads/``
   * - off-policy algorithms
     - ``rlinf/algorithms/prefix_off_policy/``
   * - EXPO edit actor 和 algorithm
     - ``rlinf/models/embodiment/prefix/heads/mlp.py``、``rlinf/algorithms/prefix_off_policy/expo.py``
   * - FSDP actor worker
     - ``rlinf/workers/actor/fsdp_prefix_off_policy_worker.py``

排查时先比较三组维度：feature model 的 ``z_dim``、policy 的 ``z_dim + history.extra_dim``、环境 action chunk 的 ``num_action_chunks * action_dim``。再确认 Stage 1/Stage 2 使用同一个 OpenPI dataconfig 与 ``norm_stats.json``。RLT token feature 的 checkpoint 必须包含完整的 ``rlt_module``，对应的 Stage 1/Stage 2 配置和运行步骤见 :doc:`../examples/embodied/rlt`。
