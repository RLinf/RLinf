Prefix Fine-Tune：冻结 VLA 特征与可插拔 Policy
==============================================

用 Prefix Fine-Tune，可以从已有 VLA checkpoint 出发，只训练小型 off-policy actor-critic。本页先说明它与 RL Token 路线的区别，再给出 ManiSkill 示例的运行步骤，最后介绍如何替换特征模型、policy head 和更新算法。

这里的 prefix 指 VLA 生成动作之前计算出的图像、语言 hidden states。在线 RL 期间，VLA 保持冻结；另一个 actor 读取这些特征、机器人状态和 VLA 生成的 reference action，输出要执行的 action chunk。

概览
----

选择这条路线时，先考虑是否需要专门训练 RL 表示。设置 ``prefix.pool: masked_mean``，就可以直接对已有 base 或 SFT VLA 的有效 prefix token 求均值，得到 ``z_rl``。如果希望使用 RL Token 学到的表示，则设置 ``prefix.pool: rlt_token``，加载 Stage 1 训练过的 token transformer。两种表示共用后续的 actor-critic 训练流程；RLT 的两阶段训练见 :doc:`../examples/embodied/rlt`。

在线更新只训练 actor 和 critic head，梯度不回传到 VLA。replay 保存 ``{z_rl, proprio, ref_chunk}`` 而不是图像，learner 可以复用采集时提取的特征。但 rollout 仍然要运行 VLA，提取特征并生成 reference action；冻结模型并不意味着省掉这部分推理。

与 RLT 专属的 AC、TD3 policy 和 worker 相比，Prefix 使用同一个 ``prefix_policy`` 模型和 ``prefix_off_policy`` worker，在配置中分别选择 actor head、critic head 和算法。chunk reward 折扣、BC target 和权重 schedule 由 AC、TD3 共用。比较新的特征或 head 时，可以保留训练循环，只替换对应组件。pooling 和 RLT token 的任务效果仍需在相同设置下比较。

训练与运行
----------

ManiSkill peg insertion 的完整配置是 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml``：冻结 VLA，使用 masked-mean 特征，以 ``deterministic_mlp`` 和 ``twin_q_mlp`` 训练 TD3。独立的 STEAM value model 判断何时从 base VLA 切换到 actor。先按 :doc:`../examples/embodied/rlt` 激活 OpenPI 与 ManiSkill 环境，以下命令均在仓库根目录运行。

下载 Checkpoint
^^^^^^^^^^^^^^^

这个 preset 使用 base VLA、expert VLA、STEAM value model 和校准后的 phase head。下载资源后，将本地路径填入对应配置字段：

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

选择一个本地目录存放权重、归一化统计量和 processor 文件。使用 hf-mirror 时，先执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

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

前两个仓库的文件路径已经包含 ``global_step_1000/actor`` 和 ``global_step_3000/actor``。STEAM 仓库根目录直接存放 actor 目录内的文件，因此下载目标要包含 ``global_step_5000/actor``。上述命令跳过分布式训练状态，保留推理所需文件。

将 preset 四个 checkpoint 路径中的 ``/path/to`` 替换为 ``CHECKPOINT_ROOT`` 展开后的绝对路径。保留 ``openpi_data.repo_id: maniskill_peginsertionside_joint`` 和 ``norm_stats_path: null``，让每个 OpenPI checkpoint 读取各自的 ``<model_path>/<repo_id>/norm_stats.json``。

STEAM checkpoint 还记录了本地 backbone 路径。编辑 ``$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json``，将 ``vision_repo_id`` 设为 ``$CHECKPOINT_ROOT/siglip-so400m-patch14-384`` 展开后的绝对路径，``language_repo_id`` 设为 ``$CHECKPOINT_ROOT/gemma-3-270m`` 展开后的绝对路径。tokenizer 和 image processor 文件要与 STEAM 权重放在同一目录，gate 会从这里加载。

.. warning::

   preset 的 phase/expert 阈值、patience 和 expert warmup 保持 ``null``，即可使用当前 phase head 的 metadata。已发布 head 建议使用 ``enter_threshold: 0.940951`` 和 ``patience_chunks: 1``；显式填写的值会覆盖 metadata，不一致时会记录 warning。expert 候选阈值 ``-0.568753`` 在验证集上的 predicted takeover 数和 oracle-stall recall 均为 0。请保留 ``shadow`` 模式，只记录判断，不执行 expert 接管。切换到 ``active`` 前，需要用 actor 实际控制的 episode 重新校准并验证 expert 阈值。如果使用 ``steam_gate_recommendation.yaml``，替换其中的源服务器 phase-head 路径，并保留 shadow 模式。

运行训练
^^^^^^^^

更新好路径后，启动 preset：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

命令会启动 actor、rollout 和 env worker。特征模型显式设置 ``prefix.pool: masked_mean``、``prefix.image_only: true`` 和 ``use_rlt: false``；只有图像 token 参与 ``z_rl`` 的 pooling，语言仍用于生成 reference action。STEAM 决定 base 到 actor 的切换，expert takeover 保持 shadow 模式。``rollout.routing_gate.cameras`` 把 STEAM processor 的 camera key 映射到提供图像的环境 observation key。

采集校准数据
^^^^^^^^^^^^

要为自己的设置拟合 phase head，使用采集 preset。它设置 ``rollout.routing_gate.actor_switch.collect_phase_features: true``，让 gate 返回 STEAM 特征；同时设置 ``algorithm.rlt_gate_calibration.enable: true``，将特征、几何标签和 gate 诊断信息写入文件。采集时清空 phase-head 路径，改用环境几何 gate 路由，expert takeover 仍为 shadow 模式。

.. code:: bash

   STEPS=20 SAVE_INTER=-1 bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

learner 将 ``trace_*.pt`` 写入 ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/``。每次启动会在 ``logs/`` 下创建运行目录；需要另存时，修改 ``algorithm.rlt_gate_calibration.save_path``。这 20 轮用于检查采集流程；若几何标签的正负样本都已出现，也可拟合初版 phase head，但不能完成 preset 的 30,000 次更新 warmup，也不能据此验证 expert 接管。

将 ``TRACE_DIR`` 指向本次采集的 trace 目录，拟合初版 head 时不设置 minimum-version 过滤：

.. code:: bash

   TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
   python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
     --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
     --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

工具在训练循环之外生成 ``steam_phase_head.pt`` 和 ``steam_gate_recommendation.yaml``。检查验证指标后，把训练 preset 的 phase-head 路径改为新输出，gate 设置保持 ``null``，即可读取该 checkpoint 的建议值。部署用的 expert 阈值还需要用后续 actor 实际控制的 episode 校准。正常训练使用训练 preset，其中两个采集开关均为 false。

Replay 与更新
^^^^^^^^^^^^^

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
         name: twin_q_mlp
         hidden_dims: [256, 256, 256]

``actor.model.z_dim`` 是追加 state history 之前的特征宽度。``proprio_dim`` 对应返回的机器人状态，``action_dim`` 对应环境的单步动作维度，``num_action_chunks`` 对应实际执行的 chunk 长度。reference 至少要覆盖这些动作；``ref_num_action_chunks`` 可以更长。

配置校验要求 ``loss_type: prefix_off_policy``、``actor.model.model_type: prefix_policy``、两个已注册的 head name 和已注册的 ``algorithm.name``。actor distribution、action noise 支持和 Q head 数量不匹配时，会在 worker 启动前报错。

当前实现
--------

冻结 VLA 通过三个 tensor 向可训练 policy 提供 observation。下面沿着特征提取、actor 和 critic 调用的顺序解释它们；接入其他 VLA 或 head 时，保持这些 shape 一致。

冻结 VLA 如何生成 Prefix observation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rollout worker 将环境原始 observation 传给 ``extract_prefix_obs``。特征模型通过 ``z_dim`` 声明输出宽度，并返回一个 ``PrefixObs``。接口定义在 ``rlinf.models.embodiment.prefix.contracts``：

.. code:: python

   from typing import Any, Protocol
   from rlinf.models.embodiment.prefix.types import PrefixObs

   class PrefixFeatureModel(Protocol):
       @property
       def z_dim(self) -> int: ...

       def extract_prefix_obs(
           self, env_obs: dict[str, Any]
       ) -> PrefixObs: ...

``PrefixObs`` 包含三个 batch-first tensor。下表是特征模型的原始输出，尚未追加 state history：

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

State history
^^^^^^^^^^^^^

如果 policy 还需要最近的运动状态，设置 ``actor.model.state_history.enable: true``。``StateHistoryBuffer`` 按 policy decision rate 保存最近 ``steps`` 个 proprio，展平后追加到 ``z_rl``。``pad`` 可选 ``zero`` 或 ``repeat``；episode reset 会清理对应环境的历史。

.. code:: text

   runtime_z_dim = z_dim + steps * proprio_dim  # state_history.enable=true
   runtime_z_dim = z_dim                         # state_history.enable=false

``get_prefix_feature_dim`` 计算追加后的宽度。rollout 和 actor 共用 ``actor.model.state_history``，因此 ``actor.model.z_dim`` 仍填写特征模型的原始输出宽度。

历史按 mode、环境 rank、pipeline stage 和 source-row offset 索引。terminal observation 会在当前 observation 重置或推进历史前读取上一 episode 的历史，且不提交修改。evaluation 的历史与训练分开保存。 STEAM gate 的图像历史和 routing latch 也按同一 source 身份保存，请求拆分或行顺序变化不会重置其他环境。

PrefixPolicy 的 forward contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PrefixPolicy`` 将展平后的 ``z_rl`` 和 ``proprio`` 拼成 policy state，再截取 ``ref_chunk`` 的前 ``num_action_chunks`` 个动作，展平后用作 actor 的 reference 输入和 BC target。它只持有 actor、critic head；冻结 VLA、可选 expert 和 gate，以及 train/eval 历史由 ``PrefixRolloutRuntime`` 管理。FSDP worker 通过 ``q_head`` 路径识别 critic 参数。

它通过 ``BasePolicy`` 接口提供以下调用：

* ``PrefixForwardType.ACTOR``：返回 actor head 的 ``actions, logprobs, None``。
* ``PrefixForwardType.CRITIC``：返回 ``current_q, next_q``，每个 Q head 对应一个值；非 joint critic 的 ``next_q`` 为 ``None``。
* ``predict_action_batch``：将 action reshape 为 ``[B, num_action_chunks, action_dim]``，并在 ``forward_inputs`` 中返回 action 和特征 observation。

``PrefixPolicy`` 在内部选择普通 Q 或 joint CrossQ critic。算法对两种后端发起相同的 actor、critic 请求，所有训练调用都经过 FSDP 包装后的 ``forward``。

扩展 BaseVLA 特征模型
---------------------

使用其他 VLA 时，将它的 observation 和输出适配为 ``PrefixObs`` 即可接入。现有 OpenPI adapter 是一种实现，policy head 不依赖 OpenPI。这里的 BaseVLA 指供应商模型或基座模型，不是 RLinf 的基类。具体 adapter 要支持 ``eval``、``requires_grad_``、``to`` 和 ``extract_prefix_obs``；如果 route 还会调用 VLA 的动作 API，也需提供对应接口。

实现 adapter
^^^^^^^^^^^^

下面的骨架从一次 observation 提取 prefix 特征和 reference chunk。预处理、编码和采样方法仅作示意，接入时要换成 VLA 实际提供的 API：

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

``extract_prefix_obs`` 接收环境 wrapper 提供的 observation 字典，返回 ``PREFIX_OBS_KEYS`` 对应的三个 float32 batch tensor。示例中，``hidden`` 按 mask 求均值得到 ``z_rl``，采样动作得到 ``ref_chunk``，状态字段作为 ``proprio``。检查 reference 的动作维度和长度是否符合环境要求，状态归一化方式也要与训练数据一致。

接入 model factory
^^^^^^^^^^^^^^^^^^

rollout worker 通过 ``get_model`` 创建 ``rollout.prefix_feature_model``。使用 ``rlinf.models`` 的 ``register_model`` 注册 builder，并设置 ``category="embodied"``。builder 接收 model config 和 ``torch_dtype``，返回 adapter；注册时也会将 model type 加入配置校验。

每个 worker 进程都要执行注册：仓库内的 builder 由 ``rlinf/models/__init__.py`` 导入，外部组件则通过 ``RLINF_EXT_MODULE`` 加载注册模块。完整流程见 :doc:`new_model_fsdp`。接入后可以保留现有 ``prefix_policy``、算法和 replay 格式，再通过 rollout 路径检查 adapter 的 batch shape、device、dtype、reset 行为和 action chunk。

扩展 MLP Actor 与 Critic
------------------------

要在相同特征上比较不同网络，替换 head。actor 接收拼好的 state 和 reference；``forward`` 返回 flat action 和每维 log probability，``mean_action`` 返回 supervised forward 使用的无噪声预测。critic 接收 state 和 flat action，每个 Q head 返回一个值。joint CrossQ critic 还会返回 next-state Q。

注册 actor head
^^^^^^^^^^^^^^^

继承 ``PrefixActorHead``，用 ``ActorHeadCapabilities`` 声明行为。registry 将 ``state_dim``、``reference_dim``、``action_dim`` 和 head config 传给 builder。以下示例注册一个支持 reference dropout 和 action noise 的 deterministic MLP：

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

``register_prefix_actor`` 让 model builder 可以通过 ``custom_mlp`` 找到这个 head。``mean_action`` 为 policy 的 SFT forward 提供无噪声预测，``forward`` 按请求应用 dropout 和探索噪声，再返回动作及 log probability。内置的 ``fixed_std_mlp`` 用于 AC，``deterministic_mlp`` 用于 TD3；AC 要求 stochastic actor，TD3 要求支持 action noise 的 deterministic actor。

注册 critic head
^^^^^^^^^^^^^^^^

critic 需要声明 ``CriticHeadCapabilities`` 和 ``num_q_heads``。下面的实现将同一组 state-action 输入送入两个独立 Q 网络，并注册为 ``custom_twin``：

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

它的 ``forward`` 返回 ``[B, 2]``。内置 ``multi_q_mlp`` 可以配置 head 数量，``twin_q_mlp`` 固定为两个，``cross_q_mlp`` 支持联合计算当前和下一状态的 Q。选择时要满足算法要求的 capabilities。

在 registry 的 ``_load_builtin_heads`` 中导入仓库内新增的 head，确保每个 worker 进程都完成注册，再在配置中选择注册名：

.. code:: yaml

   actor:
     model:
       actor_head:
         name: custom_mlp
       critic_head:
         name: custom_twin

这会替换两个网络，保留特征模型和训练 worker。只要 Q head 数量满足要求，AC 和 TD3 都可以使用 ``multi_q_mlp``。

扩展 Prefix Algorithm
---------------------

actor objective、critic target 或下一动作的生成规则发生变化时，扩展算法。三个入口分别是 ``critic_loss``、``actor_loss`` 和 ``target_actions``。基类的 ``critic_loss`` 已实现 chunk 折扣的 Bellman target 和 critic MSE；只有这些规则不同时才需要覆盖。用 ``requirements`` 声明所需的 head 行为，再注册子类。

下面的算法复用 BC/Q objective，以最小 Q 聚合，并用 target actor 生成 bootstrap 动作：

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

``actor_loss`` 返回共享 helper 的 loss、entropy 统计值和 metrics。``target_actions`` 将 ``next_obs`` 传给 worker 持有的 ``context.target_model``，返回 critic target 所需的 flat action。它和 ``context.model`` 都是经过包装的训练模型；按示例通过 ``forward`` 调用，不要用 ``.module`` 绕过 FSDP。普通 Q 和 CrossQ 后端由 ``PrefixPolicy`` 选择。

可选的 ``_actor_loss`` helper 为 AC、TD3 复用 intervention-aware BC target、BC/Q weight schedule 和 Q 聚合。其他 objective 可以直接实现 ``actor_loss``。在 registry 的 ``_load_builtin_algorithms`` 中导入模块后，设置 ``algorithm.name: my_algorithm``，继续使用 ``loss_type: prefix_off_policy``。

内置 ``ac`` 用在线 stochastic actor 生成下一动作，要求 ``algorithm.actor_agg_q: q1``。``td3`` 的 target action 不加噪声，也不使用 target policy smoothing；rollout 保留探索噪声，actor 更新时是否加噪声由 ``algorithm.actor_update_action_noise`` 单独控制。actor Q 聚合支持 ``min``、``q1``、``mean``。两种算法的 critic target 都取前两个 Q head 的最小值；``multi_q_mlp`` 的其余 head 回归到同一个 target，因此应省略 ``algorithm.agg_q`` 字段或将它设为 ``min``。两者都要求 ``algorithm.entropy_tuning.alpha_type: fixed_alpha``，不训练 entropy temperature。构建算法时会校验这些设置。

实现位置与检查点
----------------

需要从配置追踪到实现时，可以从以下文件开始：

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
   * - rollout 生命周期与 replay/scheduling
     - ``rlinf/algorithms/prefix_off_policy/rollout.py``、``replay.py``、``schedule.py``
   * - OpenPI 特征 mixin
     - ``rlinf/models/embodiment/openpi/prefix.py``
   * - FSDP actor worker
     - ``rlinf/workers/actor/fsdp_prefix_off_policy_worker.py``

遇到 shape 错误，先比较特征模型的 ``z_dim``、policy 的运行时宽度 ``z_dim + history.extra_dim`` 和环境 flat action 宽度 ``num_action_chunks * action_dim``。动作尺度不对时，检查每个 checkpoint 使用的 OpenPI data config 和 ``norm_stats.json``。使用 RLT token 的特征模型还必须从 Stage 1 checkpoint 加载完整 ``rlt_module``，对应步骤见 :doc:`../examples/embodied/rlt`。
