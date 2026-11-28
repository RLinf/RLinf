:orphan:

FlexRLT 可插拔组件
==================

无需修改 rollout 循环与 replay 链路，即可替换 :doc:`FlexRLT <../examples/embodied/flexrlt>` 中的冻结 VLA 特征提取器、Actor/Critic 网络头或更新规则。

扩展点速览
----------

FlexRLT 的四个层次各自对应独立的配置项与注册入口，按需替换其中一层即可：

.. list-table::
   :header-rows: 1
   :widths: 20 28 24 28

   * - 层次
     - 配置字段
     - 注册方式 / 契约
     - 参考代码
   * - 冻结特征模型
     - ``rollout.prefix_feature_model``
     - ``register_model(..., category="embodied")`` + ``PrefixFeatureModel``
     - ``rlinf/models/embodiment/prefix_features.py``
   * - 可训练 Policy
     - ``actor.model.model_type``
     - ``register_model(..., obs_contract=ObsContract.PREFIX)``
     - ``rlinf/models/embodiment/mlp_policy/prefix_mlp_policy.py``
   * - Actor / Critic 网络头
     - ``actor.model.actor_head.name``、``critic_head.name``
     - ``@register_prefix_actor`` / ``@register_prefix_critic``
     - ``rlinf/models/embodiment/mlp_policy/heads.py``
   * - 更新规则
     - ``algorithm.name``\ （配合 ``loss_type: prefix_off_policy``）
     - ``@register_prefix_off_policy_algorithm``
     - ``rlinf/algorithms/prefix_off_policy/``

Rollout 期间，冻结 VLA 把原始观测编码为 ``{z_rl, proprio, ref_chunk}``；replay 保存这些特征与实际执行的动作块，在线更新只训练轻量 Actor 与 Critic。

实现自定义扩展
--------------

按需要替换的组件切换标签页（仓库外扩展可将示例代码保存在根目录的 ``flexrlt_ext.py``）：

.. tab-set::

   .. tab-item:: 1. 特征模型（VLA）
      :sync: flexrlt-ext-vla

      为任意冻结 VLA 编写适配器，暴露 ``z_dim`` 与 ``extract_prefix_obs(env_obs) -> PrefixObs``，再通过 ``register_model(..., category="embodied")`` 注册（注册流程见 :doc:`new_model_fsdp`）：

      .. code-block:: python

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

      - 形状约定：``z_rl`` 为 ``[B, z_dim]``，``proprio`` 为 ``[B, proprio_dim]``，``ref_chunk`` 为 ``[B, ref_num_action_chunks, action_dim]``。
      - 启动时 rollout 会自动校验 ``z_dim`` 与 ``actor.model.z_dim`` 一致。

   .. tab-item:: 2. Actor / Critic 网络头
      :sync: flexrlt-ext-heads

      继承 ``PrefixActorHead`` 或 ``PrefixCriticHead``，声明能力集合并注册 YAML 名称。内置网络头包括 ``fixed_std_mlp``\ （AC）、``deterministic_mlp``\ （TD3）、``multi_q_mlp`` 与 ``cross_q_mlp``。

      .. code-block:: python

         import torch
         from torch import nn
         from rlinf.models.embodiment.mlp_policy.heads import (
             ActorHeadCapabilities, CriticHeadCapabilities,
             PrefixActorHead, PrefixCriticHead,
             register_prefix_actor, register_prefix_critic,
         )

         class CustomMLPActor(PrefixActorHead):
             capabilities = ActorHeadCapabilities(
                 distribution="deterministic", supports_action_noise=True,
             )

             def __init__(self, state_dim, reference_dim, action_dim, cfg):
                 super().__init__()
                 hidden = int(cfg.get("hidden_dim", 256))
                 self.net = nn.Sequential(
                     nn.Linear(state_dim + reference_dim, hidden), nn.ReLU(),
                     nn.Linear(hidden, action_dim),
                 )

             def mean_action(self, state, reference):
                 return self.net(torch.cat([state, reference], dim=-1)).clamp(-1, 1)

             def forward(self, state, reference, *, deterministic,
                         apply_reference_dropout, reference_dropout_prob, apply_action_noise):
                 if apply_reference_dropout:
                     keep = torch.rand(reference.shape[0], 1, device=reference.device)
                     reference = reference * (keep >= float(reference_dropout_prob or 0.0)).to(reference.dtype)
                 action = self.mean_action(state, reference)
                 if apply_action_noise and not deterministic:
                     action = (action + torch.randn_like(action) * 0.1).clamp(-1, 1)
                 return action, torch.zeros_like(action)

         @register_prefix_actor("custom_mlp", capabilities=CustomMLPActor.capabilities)
         def make_custom_mlp_actor(state_dim, reference_dim, action_dim, cfg):
             return CustomMLPActor(state_dim, reference_dim, action_dim, cfg)

         class CustomTwinCritic(PrefixCriticHead):
             capabilities = CriticHeadCapabilities(fixed_num_q_heads=2)

             def __init__(self, state_dim, action_dim, cfg):
                 super().__init__()
                 self.num_q_heads = 2
                 width = int(cfg.get("hidden_dim", 256))
                 self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, width), nn.ReLU(), nn.Linear(width, 1))
                 self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, width), nn.ReLU(), nn.Linear(width, 1))

             def forward(self, state, actions, *, next_state=None, next_actions=None):
                 x = torch.cat([state, actions], dim=-1)
                 return torch.cat([self.q1(x), self.q2(x)], dim=-1)

         @register_prefix_critic("custom_twin", capabilities=CustomTwinCritic.capabilities)
         def make_custom_twin_critic(state_dim, action_dim, cfg):
             return CustomTwinCritic(state_dim, action_dim, cfg)

   .. tab-item:: 3. 更新规则
      :sync: flexrlt-ext-algo

      继承 ``PrefixOffPolicyAlgorithm``，通过 ``requirements`` 声明兼容的网络头能力，并按需覆写 ``actor_loss``、``target_actions`` 或 ``critic_loss``\ （基类 ``critic_loss`` 已处理 chunk 折扣 Bellman MSE）：

      .. code-block:: python

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
                     context, batch,
                     actor_agg_q="min",
                     apply_reference_dropout=False,
                     apply_action_noise=True,
                 )

             def target_actions(self, context, next_obs):
                 actions, _, _ = context.target_model(forward_type=ForwardType.SAC, obs=next_obs)
                 return actions

配置与启动
----------

在 ``examples/embodiment/config/`` 下新建子配置 ``my_flexrlt_custom.yaml``，继承 :doc:`FlexRLT <../examples/embodied/flexrlt>` 的 ManiSkill 预设：

.. code-block:: yaml

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

通过 ``RLINF_EXT_MODULE`` 让所有 Ray worker 加载自定义注册后启动：

.. code-block:: bash

   export RLINF_EXT_MODULE=flexrlt_ext
   bash examples/embodiment/run_embodiment.sh my_flexrlt_custom

参考细则
--------

.. dropdown:: 自定义 Policy 契约（``ObsContract.PREFIX``）

   如需整体替换 ``prefix_policy``\ （例如改用循环网络 Actor），需注册继承 ``BasePolicy`` 且声明 ``obs_contract=ObsContract.PREFIX`` 的类，并满足：

   .. list-table::
      :header-rows: 1
      :widths: 36 64

      * - 要求
        - 调用方
      * - 接受 ``actor.model.actor_head`` 与 ``critic_head`` 名称
        - 配置校验与算法能力匹配。
      * - ``forward(ForwardType.SAC, obs=...) -> (actions, logprobs, None)``
        - Actor 损失与 target 动作（展平形状 ``[B, num_action_chunks * action_dim]``）。
      * - ``ForwardType.SAC_Q``\ （支持 CrossQ 时提供 ``CROSSQ_Q``）
        - Critic 损失与 Actor 损失的 Q 项。
      * - Critic 参数挂在 ``q_head`` 下
        - FSDP worker 将 ``q_head`` / ``encoder`` / ``state_proj`` 归入 Critic 优化器，其余归入 Actor 优化器。
      * - ``predict_action_batch(..., return_obs=True)``
        - Rollout：返回 ``[B, num_action_chunks, action_dim]`` 动作，并在 ``forward_inputs`` 中写入 ``action``、``z_rl``、``proprio``、``ref_chunk``。

.. dropdown:: Prefix 池化方式与内置 AC / TD3 规则

   - **Prefix 池化**\ （与 ``openpi`` 同级配置）：``prefix.image_only: true``\ （默认）仅选图像 token；``prefix.pool`` 支持 ``masked_mean``\ （无 RLT 默认）、``mean``、``last`` 与 ``rlt_token``\ （Stage 1 token transformer）。
   - **内置 ``ac``**：随机 Actor（``fixed_std_mlp``）、在线下一步动作 bootstrap、``algorithm.actor_agg_q: q1``。
   - **内置 ``td3``**：确定性带噪 Actor（``deterministic_mlp``）、目标策略平滑（``target_policy_noise`` 默认 ``0.1``，``target_noise_clip`` 默认 ``0.5``），每 ``critic_actor_ratio``\ （默认 ``2``）次 Critic 更新同步更新一次 Actor 与目标网络。
   - 两者均取前两个 Q 头的最小值作为 Critic 目标（``algorithm.agg_q: min``），并使用 ``algorithm.entropy_tuning.alpha_type: fixed_alpha``。

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
