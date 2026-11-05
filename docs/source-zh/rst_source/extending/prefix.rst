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

``prefix.pool`` 和 ``prefix.image_only`` 与 OpenPI 模型配置中的 ``openpi`` 块并列。未设置 ``prefix.pool`` 时，``resolve_prefix_pool`` 在 ``openpi.use_rlt`` 为 true 时返回 ``rlt_token``，否则返回 ``masked_mean``。``prefix.image_only`` 默认为 true，会在 pooling 或 RLT 编码前只保留图像 token。建议在配置中显式写出这两个字段，这样 Stage 1 和 Stage 2 的表示来源可以直接检查。rollout worker 启动时会检查特征模型的 ``z_dim`` 与 ``actor.model.z_dim`` 一致。每次调用 ``extract_prefix_obs`` 时，还会检查 ``z_rl`` 的最后一维与特征模型的 ``z_dim`` 一致。

State history
^^^^^^^^^^^^^

``actor.model.state_history.enable: true`` 时，``StateHistoryBuffer`` 在 policy decision rate 保存最近 ``steps`` 个 proprio，并将展平结果追加到 ``z_rl``。``pad`` 支持 ``zero`` 和 ``repeat``。episode reset 会清理指定环境的 history。policy 输入宽度由 ``get_prefix_feature_dim`` 计算：

.. code:: text

   runtime_z_dim = z_dim + steps * proprio_dim  # state_history.enable=true
   runtime_z_dim = z_dim                         # state_history.enable=false

rollout 和 actor 使用同一个 ``actor.model.state_history`` 配置，避免 feature width 不一致。

PrefixPolicy 的 forward contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rlinf.models.embodiment.prefix.policy.PrefixPolicy`` 继承 ``BasePolicy``，提供 actor 和 critic 的语义 forward 接口：

* ``PrefixForwardType.ACTOR`` 调用 actor head，返回 ``actions, logprobs, None``。
* ``PrefixForwardType.CRITIC`` 返回 ``current_q, next_q``，每个 Q head 对应一个值；非 joint critic 的 ``next_q`` 为 ``None``。
* ``predict_action_batch`` 将 flat action reshape 为 ``[B, num_action_chunks, action_dim]``，并在 ``forward_inputs`` 中返回 action 与 Prefix observation。

``PrefixPolicy`` 在内部选择普通 critic 或 joint CrossQ critic。algorithm 对两种后端使用同一套语义接口，训练调用均经过 FSDP 包装后的 ``forward``。

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

算法接口以 ``critic_loss``、``actor_loss`` 和 ``target_actions`` 为三个主要入口。新 algorithm 用 ``requirements`` 声明所需的 head capabilities，并通过 ``register_prefix_off_policy_algorithm`` 注册。基类已提供按 chunk 折扣的 Bellman critic loss；只有 target 或 critic objective 不同时才需要覆盖。子类实现自己的 ``actor_loss`` 和 ``target_actions``。

下面的示例使用 minimum-Q 聚合，并复用共享的 BC/Q actor objective。actor 请求通过 ``context.target_model(...)`` 发出，因此仍经过 worker 持有的 FSDP wrapper 及其 ``forward`` hooks。``PrefixPolicy`` 负责 actor 分发和普通 Q / CrossQ critic 的选择，算法不需要指定 SAC 或 CrossQ forward type。

.. code:: python

   from rlinf.algorithms.prefix_off_policy import (
       PrefixAlgorithmRequirements, PrefixOffPolicyAlgorithm,
       register_prefix_off_policy_algorithm,
   )
   from rlinf.models.embodiment.prefix import PrefixForwardType

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
               forward_type=PrefixForwardType.ACTOR,
               obs=next_obs,
           )
           return actions

``PrefixOffPolicyAlgorithm`` 在 ``critic_loss`` 中复用 chunk reward 折扣、termination bootstrap 和 critic MSE。可选的 ``_actor_loss`` helper 为 AC 和 TD3 共享 BC target、human intervention mask、BC/Q weight schedule 和 Q 聚合。其他 actor objective 可以直接实现 ``actor_loss``，无需使用该 helper。在 registry 的 ``_load_builtin_algorithms`` 中导入模块后，算法通过 ``algorithm.name: my_algorithm`` 选择，worker 仍使用 ``loss_type: prefix_off_policy``。

内置 ``ac`` 使用 stochastic actor 和 Q1 聚合，``algorithm.actor_agg_q`` 只能为 ``q1``。``td3`` 使用 deterministic actor、target actor、action noise，并支持 ``min``、``q1``、``mean`` 三种 actor Q 聚合。critic target 始终取前两个 Q head 的最小值，因此 ``algorithm.agg_q`` 只能为 ``min`` 或不设置；``multi_q_mlp`` 的其余 head 回归到同一个 target。Prefix algorithm 不训练 entropy temperature，要求 ``algorithm.entropy_tuning.alpha_type: fixed_alpha``。构建 algorithm 时会一起校验这些设置和 head capabilities。

训练与运行
----------

使用 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml`` 在 ManiSkill peg insertion 任务上运行 Prefix Fine-Tune。该 preset 使用 masked-mean VLM Prefix 特征、``deterministic_mlp``、``twin_q_mlp`` 和 ``algorithm.name: td3``。在仓库根目录操作，并激活 :doc:`../examples/embodied/rlt` 中的 OpenPI 与 ManiSkill 环境。

下载 Checkpoint
^^^^^^^^^^^^^^^

四个 checkpoint 路径和 gate 阈值集中在同一个 YAML 中。各字段对应的公开资源如下：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 资源
     - Hugging Face 仓库
     - Preset 字段
   * - Base / feature SFT，step 1000
     - `ChikaYokoyama/maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`_
     - ``rollout.prefix_feature_model.model_path``
   * - Expert SFT，step 3000
     - `ChikaYokoyama/maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`_
     - ``rollout.expert_model.model_path``
   * - STEAM value model，step 5000
     - `ChikaYokoyama/rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`_
     - ``rollout.routing_gate.model.model_path``
   * - Phase head 与 gate 推荐配置
     - `ChikaYokoyama/rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`_
     - ``rollout.routing_gate.actor_switch.phase_head_path``

选择本地 checkpoint 目录，下载推理权重、归一化统计量和 processor 文件。使用 hf-mirror 时，先执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. code:: bash

   CHECKPOINT_ROOT="$(pwd)/data/steam_peg"

   hf download ChikaYokoyama/maniskill_rlt_1000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/maniskill_rlt_3000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
     --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
     --exclude 'dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_phase_head \
     --local-dir "$CHECKPOINT_ROOT"

   hf download google/siglip-so400m-patch14-384 \
     --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
   hf download google/gemma-3-270m \
     --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

两份 SFT 仓库的文件路径已包含 ``global_step_1000/actor`` 和 ``global_step_3000/actor``。STEAM 仓库的根目录直接存放 actor 目录的内容，因此应按上述命令下载到本地 ``global_step_5000/actor`` 目录。这些命令保留推理所用的文件，跳过分布式训练状态。

将 preset 中四个 checkpoint 路径的 ``/path/to`` 替换为 ``CHECKPOINT_ROOT`` 展开后的绝对路径。保留 ``openpi_data.repo_id: maniskill_peginsertionside_joint`` 和 ``norm_stats_path: null``，每个 OpenPI 模型会从各自的 ``<model_path>/<repo_id>/norm_stats.json`` 加载归一化统计量。

下载的 STEAM ``config.json`` 也保存了本地 backbone 路径。编辑 ``$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json``，将 ``vision_repo_id`` 改为 ``$CHECKPOINT_ROOT/siglip-so400m-patch14-384`` 展开后的绝对路径，将 ``language_repo_id`` 改为 ``$CHECKPOINT_ROOT/gemma-3-270m`` 展开后的绝对路径。tokenizer 和 image processor 文件须与权重保存在同一 checkpoint 目录，gate 会从该目录加载它们。

.. warning::

   已发布的 phase head 使用 ``enter_threshold: 0.940951`` 和 ``patience_chunks: 1``。expert 候选阈值 ``-0.568753`` 在验证集上的 predicted takeover 数和 oracle-stall recall 均为 0，因此 preset 将 expert takeover 保持在 ``shadow`` 模式。切换到 ``active`` 前，应使用 actor 实际控制的 episode 重新校准并验证 expert 阈值。使用 ``steam_gate_recommendation.yaml`` 时，替换其中的源服务器 phase-head 路径，并保留 preset 的 shadow 模式。

运行训练
^^^^^^^^

设置本地路径后，启动训练 preset：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

冻结的特征模型显式设置 ``prefix.pool: masked_mean``、``prefix.image_only: true`` 和 ``use_rlt: false``。``z_rl`` 只对 VLM 图像 token 做 pooling；语言仍用于生成 reference action。STEAM 控制 base 到 actor 的切换，并在 shadow 模式下记录 expert takeover 决策。``rollout.routing_gate.cameras`` 将每个 STEAM processor camera key 映射到提供该图像的 env observation key。

采集校准数据
^^^^^^^^^^^^

要为自己的设置拟合 head，使用采集 preset。该 preset 清空 phase-head 路径，使用环境几何 gate 路由，并将 expert takeover 保持为 shadow 模式：

.. code:: bash

   STEPS=20 SAVE_INTER=-1 bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

preset 启用 ``rollout.routing_gate.actor_switch.collect_phase_features`` 输出 STEAM 特征，并启用 ``algorithm.rlt_gate_calibration.enable`` 保存特征、几何标签和 gate 诊断信息。每次启动都会在 ``logs/`` 下创建独立的运行目录，learner 将数据写入 ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``。保存目录由 ``algorithm.rlt_gate_calibration.save_path`` 配置。

20 轮 rollout 用于验证采集流程；正负几何标签都出现时，可以拟合初版 phase head。它们不足以完成 preset 的 30,000 次更新 warmup，部署用的 expert 阈值需要用后续 actor 实际控制的 episode 校准。将 ``TRACE_DIR`` 设为此次采集的 trace 目录，拟合初版 head 时不设置 minimum-version 过滤：

.. code:: bash

   TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
   python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
     --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
     --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

检查验证指标后，再将阈值复制到训练 preset，并将 phase-head 路径设为本地输出位置。校准工具在训练更新之外离线运行。正常运行使用训练 preset，两个采集开关保持 false。

Replay 与更新
^^^^^^^^^^^^^

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
   * - FSDP actor worker
     - ``rlinf/workers/actor/fsdp_prefix_off_policy_worker.py``

排查时先比较三组维度：feature model 的 ``z_dim``、policy 的 ``z_dim + history.extra_dim``、环境 action chunk 的 ``num_action_chunks * action_dim``。再确认 Stage 1/Stage 2 使用同一个 OpenPI dataconfig 与 ``norm_stats.json``。RLT token feature 的 checkpoint 必须包含完整的 ``rlt_module``，对应的 Stage 1/Stage 2 配置和运行步骤见 :doc:`../examples/embodied/rlt`。
