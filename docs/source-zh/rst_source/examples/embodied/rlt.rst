RL Token
========

**RL Token: Bootstrapping Online RL with Vision-Language-Action Models** 在冻结的 VLA 特征模型之上训练一个轻量级强化学习策略。在 RLinf 的配置和代码中，这套流程简称为 **RLT**。整个流程分为两个阶段：

1. 在示范数据上联合训练 VLA 检查点和 RLT token transformer。
2. 冻结第一阶段得到的特征模型，用提取出的 RLT 状态训练一个轻量级 off-policy actor-critic。

当前仓库中的示例配置面向 Franka peg insertion 和 ManiSkill ``PegInsertionSideWideClearance-v1`` joint-control 仿真。pipeline 本身不绑定具体任务；只要示范数据、环境配置、动作维度、状态语义和 OpenPI dataconfig 对齐，就可以复用相同的两阶段结构。

官方项目页：`Precise Manipulation with Efficient Online RL <https://www.pi.website/research/rlt>`_。

ManiSkill 的 pooled prefix 特征和 STEAM 路由训练见 :doc:`FlexRLT <flexrlt>`。

概览
----

RLT 将表示学习和在线 RL 控制拆开。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      RLT（SFT + Off-Policy Actor-Critic / TD3）

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI π₀.₅ + RLT MLP

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      Franka 真机 · ManiSkill

   .. grid-item-card:: 训练
      :text-align: center

      两阶段（离线 SFT → 在线 RL）

| **你将完成：** 准备示范数据 -> 训练 Stage 1 -> 在 Stage 2 中加载 Stage 1 检查点 -> 启动 actor-critic 训练 -> 观察 replay buffer 与任务成功率指标。
| **前置条件：** 准备好 `OpenPI π₀.₅ <https://huggingface.co/lerobot/pi05_base>`__ 基座模型，并按所选示例配置 :doc:`Franka 真机环境 <../embodied/franka>` 或 :doc:`ManiSkill 仿真环境 <../embodied/maniskill>` （二选一）。

提供的配置文件
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - 阶段
     - 配置
     - 作用
   * - Franka Stage 1
     - ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``
     - 在 Franka 示范数据上联合 SFT π₀.₅ 和 RLT token transformer。
   * - Franka Stage 2
     - ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml``
     - 使用冻结的 Stage 1 特征模型，在真机上训练 RLT actor-critic。
   * - 双臂 Franka Stage 2
     - ``examples/embodiment/config/realworld_dual_franka_rlt_stage2_ac_mlp.yaml``
     - 在同一台主机上驱动双臂，用 ``pi05_dualfranka_tcp_rot6d`` 的 Stage 1 checkpoint 训练 Stage 2 MLP。
   * - ManiSkill Stage 1
     - ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``
     - 联合训练 ManiSkill OpenPI 基座和 RLT token transformer。
   * - ManiSkill Stage 2
     - ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``
     - 使用自动 ``rlt_policy_switch`` 和 transition replay 训练仿真 RLT actor-critic。
   * - ManiSkill Stage 2（TD3）
     - ``examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml``
     - 使用相同的冻结 Stage 1 特征模型和 replay 路径运行仿真 TD3-MLP 变体。

安装
----

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: OpenPI（LIBERO / ManiSkill）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

下面的 RLT 配置使用 ``model_type: openpi``。安装命令仍使用 ``--model openpi``，因为 vendored PyTorch 模型复用 OpenPI 运行环境，并且 RLT Stage 1 dataloader 为了保持 ManiSkill 和真机行为一致，仍使用 OpenPI 数据管线。

RLT 如何工作
------------

Stage 1：学习 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 从一个 VLA 检查点开始，在同一批示范数据上优化两个目标：

1. 普通 VLA 动作预测目标，日志中记为 ``vla_loss``。
2. RLT token 目标，日志中记为 ``rlt_loss``。

Stage 1 的总损失为：

.. code-block:: text

   total_loss = rlt_loss + rlt_alpha * vla_loss

OpenPI 模型会暴露 VLA prefix hidden states。RLT token transformer 读取这些 prefix states，并输出紧凑向量 ``z_rl``。Stage 2 使用 ``z_rl`` 作为学习到的 RL 表示，而不是直接在图像观测上训练 actor-critic。

``repo_id``、``config_name`` 和归一化统计需要与 Stage 2 特征模型配置保持一致。``openpi_data`` 应写在 ``actor.model`` 下。

.. dropdown:: 进阶配置（可选）

   ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml`` 中的 Stage 1 默认字段：

   .. code-block:: yaml

      data:
        train_data_paths: "/path/to/data"

      actor:
        model:
          openpi_data:
            repo_id: "realworld_peg_insertion_rlt_stage1"
          model_type: "openpi"
          is_lora: False
          model_path: "/path/to/pi05_base_openpi"
          num_action_chunks: 20
          action_dim: 7
          num_steps: 4
          add_value_head: False
          openpi:
            task: sft
            config_name: "pi05_franka_state"
            num_images_in_input: 1
            # 网络预测 horizon（Pi0Config.action_horizon），区别于环境执行窗口 num_action_chunks。
            action_horizon: 20
            action_chunk: ${actor.model.num_action_chunks}
            action_env_dim: ${actor.model.action_dim}
            num_steps: ${actor.model.num_steps}
            model_action_dim: 32
            paligemma_variant: "gemma_2b"
            action_expert_variant: "gemma_300m"
            max_token_len: 200
            discrete_state_input: True
            use_rlt: True
            rlt_alpha: 1.0
            rlt_prefix_seq_len: 1024
            rlt_image_only: False
            rlt_use_mask: True

.. note::

   ManiSkill joint 示例不依赖 ``state_indices`` 手工切片。``pi05_rlt_maniskill_joint`` dataconfig 将 LeRobot 数据中的 ``state`` 映射到 OpenPI ``observation.state``；Stage 2 rollout 也使用 OpenPI 处理后的 ``observation.state`` 作为 ``proprio``。这样 Stage 1、Stage 2 和 OpenPI normalization 看到的是同一种状态语义。

Stage 2：训练 Actor-Critic 策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 2 冻结 Stage 1 特征模型，只训练轻量级 RLT MLP actor 和 critic。

.. note::

   当前 Stage 2 实现不是标准 maximum-entropy SAC。

rollout 时：

1. 环境返回原始观测和任务元信息。
2. ``rollout.rlt_feature_model`` 运行冻结的 Stage 1 模型，将原始观测转换为：

   - ``z_rl``：紧凑的 RLT 表示。
   - ``proprio``：选中的机器人或仿真状态。
   - ``ref_chunk``：VLA reference action chunk。

3. Stage 2 actor 的输入是 ``ref_chunk``、``z_rl`` 和 ``proprio``。
4. replay buffer 存储 RLT transition：

.. code-block:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的动作
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

Franka 真机配置中，``keyboard_reward_wrapper: rlt_policy_switch`` 控制阶段切换：操作员按下 ``b`` 前执行 VLA 的 ``ref_chunk``，按下 ``b`` 后切换到 Stage 2 actor。

ManiSkill joint 配置中，``env.*.rlt_policy_switch`` 根据任务进展在整 chunk 粒度自动进入 critical phase（从 VLA ``ref_chunk`` 切换到 Stage 2 actor），并在启用 expert takeover 时请求专家接管。

critic 使用 chunked rewards 上的 TD target。一个 action chunk 内的奖励会先按折扣累计，然后用下一状态 Q 值 bootstrap：

.. code-block:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

如果 episode 已终止，并且 ``bootstrap_type`` 为 ``standard``，则不会加入 bootstrap 项。

.. dropdown:: 进阶配置（可选）

   ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml`` 中的 Stage 2 默认字段：

   .. code-block:: yaml

      algorithm:
        loss_type: rlt_ac
        q_weight: 0.1
        bc_weight: 5
        reference_dropout_prob: 0.5
        gamma: 0.96
        entropy_tuning:
          alpha_type: fixed_alpha
          initial_alpha: 0.0

      rollout:
        collect_transitions: True
        rlt_feature_model:
          model_type: "openpi"
          precision: bf16
          is_lora: False
          num_action_chunks: 20
          action_dim: 7
          num_steps: 4
          add_value_head: False
          model_path: "/path/to/stage1/checkpoint/actor"
          openpi_data:
            repo_id: "realworld_peg_insertion_rlt_stage1"
            norm_stats_path: /path/to/lerobot_dataset/norm_stats.json
          openpi:
            task: eval
            config_name: "pi05_franka_state"
            num_images_in_input: 1
            action_horizon: 20
            action_chunk: ${actor.model.ref_num_action_chunks}
            action_env_dim: ${rollout.rlt_feature_model.action_dim}
            num_steps: ${rollout.rlt_feature_model.num_steps}
            model_action_dim: 32
            paligemma_variant: "gemma_2b"
            action_expert_variant: "gemma_300m"
            max_token_len: 200
            discrete_state_input: True
            state_indices: []      # 保留完整 raw state；例如 19D state
            use_rlt: True
            rlt_prefix_seq_len: 1024
            rlt_image_only: False
            rlt_use_mask: True

      actor:
        model:
          model_type: "rlt_mlp_policy"
          precision: fp32
          add_value_head: False
          add_q_head: True
          q_head_type: "default"
          fixed_std: 0.002
          is_lora: False
          z_dim: 2048
          proprio_dim: 19
          action_dim: 7
          num_action_chunks: 10
          ref_num_action_chunks: 20

      env:
        train:
          keyboard_reward_wrapper: rlt_policy_switch

Stage 2 的 actor loss 为：

.. code-block:: text

   actor_loss = -q_weight * Q(obs, pi(obs)) + bc_weight * BC(pi(obs), target_action)

普通 policy step 中，BC target 是 VLA reference action。如果某一步记录了人类干预动作，该步的 BC target 会切换成人类动作。ManiSkill 默认关闭 ``expert_takeover``，因此默认路线只使用 VLA reference 和 actor action。

运行当前 Franka 示例
--------------------

数据：采集 Franka 示范并计算归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 需要 LeRobot 格式的 Franka 示范数据，数据目录应直接包含 ``data/`` 和 ``meta/``。在控制节点上按照 :doc:`Franka 真机文档 <../embodied/franka>` 的数据采集流程准备机器人和目标位姿，并在采集配置中导出 LeRobot 格式数据：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       export_format: lerobot_dataset

然后启动采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

LeRobot shard 位于 ``logs/<timestamp>/lerobot_dataset/rank_0/id_0/``，其中直接包含 ``data/`` 和 ``meta/``。采集完成后，将该 shard 放到训练节点，并为当前 RLT OpenPI dataconfig 计算归一化统计。``repo_id`` 需要与 Stage 1 / Stage 2 配置中的 ``actor.openpi_data.repo_id`` 和 ``rollout.rlt_feature_model.openpi_data.repo_id`` 保持一致：

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state \
       --repo-id realworld_peg_insertion_rlt_stage1

后续将 Stage 1 配置中的 ``data.train_data_paths`` 指向该 LeRobot 数据集目录。若 checkpoint 未包含 norm stats，可在 ``openpi_data`` 中设置 ``norm_stats_path`` 指向 ``norm_stats.json``。

Stage 1：训练 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml`` 中填写数据集与基础模型路径：

.. code-block:: yaml

   data:
     train_data_paths: /path/to/lerobot_dataset

   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"

启动 SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05

保存出的检查点目录通常形如：

.. code-block:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Stage 2 中需要将这个 ``actor`` 目录填到 ``rollout.rlt_feature_model.model_path``。不要把 Stage 1 checkpoint 填到 ``actor.model.model_path``，该字段不负责加载 Stage 1 特征模型。

Stage 2：运行 RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml`` 中填写 Stage 1 检查点路径和目标位姿：

.. code-block:: yaml

   rollout:
     rlt_feature_model:
       model_path: /path/to/stage1/checkpoint/actor
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"

   env:
     train:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]

``realworld_rlt_stage2_ac_mlp.yaml`` 默认使用 **2 节点**\ （``cluster.num_nodes: 2``：GPU 训练节点与 Franka 控制节点）。按你的机器修改 ``cluster.node_groups`` 中的节点 rank 和 ``robot_ip``，然后在 head 节点启动异步训练：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_rlt_stage2_ac_mlp

当前默认键盘模块实现了 RLT 的关键阶段切换：按 ``b`` 进入 Stage 2 actor 控制阶段。其他功能可在真机 policy-switch wrapper 中按任务定制。

双臂 Franka Stage 2
~~~~~~~~~~~~~~~~~~~

两台机械臂和 GPU 在同一台主机上，并且冻结的 Stage 1 checkpoint 是 ``pi05_dualfranka_tcp_rot6d`` 时，使用 ``examples/embodiment/config/realworld_dual_franka_rlt_stage2_ac_mlp.yaml``。上面那份单臂配置是插销任务：7 维动作、一路相机。这份配置是 20 维 tcp_rot6d。机械臂执行 10 步，Stage 1 的参考 horizon 是 20。

替换机械臂 IP、相机 serial、夹爪路径、``TASK_DESCRIPTION`` 和目标位姿。把 ``rollout.rlt_feature_model.model_path`` 指到双臂 Stage 1 checkpoint，并把 ``openpi_data.repo_id`` 设成该 checkpoint 的数据集 id。``keyboard_device`` 留空并导出 ``RLINF_KEYBOARD_DEVICE``，或者在 ``env.train`` 和 ``env.eval`` 里都写上 evdev 路径。保持 ``hold_current_when_inactive: False``。松开 grip 后，该臂先保持最后一条遥操作指令，当前 action chunk 结束后才交回 policy。USB 键盘 ``b`` 把 rollout 从 Stage 1 policy 切到 Stage 2 actor。Pico 的 A/B/X/Y 仍是夹爪按键。

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dual_franka_rlt_stage2_ac_mlp

运行 ManiSkill Joint 示例
-------------------------

ManiSkill joint 示例使用 ``PegInsertionSideWideClearance-v1``、RGB 双视角观测、Panda 前 9 维 qpos 和 8 维 ``pd_joint_delta_pos`` action。每个 Stage 2 action 是 10-step chunk，语言指令固定为 ``insert the peg in the hole``。

数据：准备 joint-control LeRobot 数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 和 Stage 2 的 OpenPI feature model 使用 ``pi05_rlt_maniskill_joint`` dataconfig。LeRobot 数据集至少需要包含：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义
   * - ``image``
     - 主视角 RGB 图像。
   * - ``wrist_image``
     - wrist RGB 图像。
   * - ``state``
     - Panda joint proprio，当前示例使用 9 维状态。
   * - ``actions``
     - 8 维 ``pd_joint_delta_pos`` action。
   * - ``task``
     - 语言指令；也可以通过 ``default_prompt`` 固定为 ``insert the peg in the hole``。

可以直接下载参考数据集 `RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ <https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ>`__，或用 ManiSkill 的 Panda motion-planning solver 重新生成，并计算与训练配置一致的归一化统计：

.. tab-set::

   .. tab-item:: 下载参考数据集

      .. code-block:: bash

         export HF_LEROBOT_HOME=./lerobot_root
         hf download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
             --repo-type dataset \
             --local-dir ./lerobot_root/maniskill_peginsertionside_joint
         python toolkits/lerobot/calculate_norm_stats.py \
             --config-name pi05_rlt_maniskill_joint \
             --repo-id maniskill_peginsertionside_joint

   .. tab-item:: 用规划器重新生成

      .. code-block:: bash

         export HF_LEROBOT_HOME=./lerobot_root
         python toolkits/lerobot/collect_maniskill_peg_lerobot_joint.py \
             --repo-id maniskill_peginsertionside_joint \
             --num-episodes 400 \
             --seed 0 \
             --max-attempts 4000 \
             --overwrite
         python toolkits/lerobot/calculate_norm_stats.py \
             --config-name pi05_rlt_maniskill_joint \
             --repo-id maniskill_peginsertionside_joint

.. note::

   ``toolkits/lerobot`` 下的采集脚本依赖 ManiSkill 的 PegInsertionSide Panda motion-planning solver；若报 solver import 错误，先确认当前 ManiSkill 安装包含 motion-planning 示例。

.. warning::

   Stage 1 checkpoint、Stage 2 ``rollout.rlt_feature_model`` 和 OpenPI assets 必须使用同一套 ``norm_stats.json``。如果 SFT 基座和 Stage 2 加载了不同 ``repo_id`` 下的 norm stats，VLA reference action 的尺度会发生偏移。可通过 ``openpi_data.norm_stats_path`` 显式指定上述 ``norm_stats.json`` 路径。

Stage 1：联合训练 ManiSkill OpenPI + RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml`` 中填写数据、基础模型和 ``norm_stats.json`` 路径：

.. code-block:: yaml

   data:
     train_data_paths:
       - dataset_path: ./lerobot_root/maniskill_peginsertionside_joint
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: ./lerobot_root/maniskill_peginsertionside_joint/norm_stats.json

启动训练：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_sft_openpi_pi05

保存出的 FSDP 检查点通常形如：

.. code-block:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Stage 2 中将这个 ``actor`` 目录直接填到 ``rollout.rlt_feature_model.model_path``。

Stage 2：运行 ManiSkill RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`` 中填写 Stage 1 检查点和 ``norm_stats.json`` 路径：

.. code-block:: yaml

   rollout:
     rlt_feature_model:
       model_path: /path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_step/actor
       openpi_data:
         norm_stats_path: ./lerobot_root/maniskill_peginsertionside_joint/norm_stats.json

启动训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_ac_mlp

如需运行 TD3-MLP 变体，在 ``maniskill_rlt_stage2_td3_mlp.yaml`` 中配置相同的 Stage 1 checkpoint，然后执行：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_td3_mlp

该变体面向 ManiSkill 仿真配置，保留冻结的 Stage 1 特征和 transition replay 路径，将 Stage 2 策略和更新目标替换为直接 TD3 actor 与 twin-Q critic。

这个配置会启动 actor、rollout 和 ManiSkill env。rollout 侧冻结 ``rollout.rlt_feature_model``，只同步和执行 Stage 2 MLP actor。ManiSkill route 在 ``ready_for_online`` 之前执行 VLA ``ref_chunk``；达到 ``algorithm.rlt_schedule.warmup_post_collect_updates`` 后，允许 actor 在自动 critical phase 中接管。

调节 ManiSkill Critical Phase 和 Intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManiSkill 的切换逻辑读取 peg insertion 任务信息进入 critical phase（从 VLA 参考策略切到 Stage 2 actor）；开启 expert takeover 后，还会在 actor 连续无进展时请求 expert 接管。仓库自带配置已设置好默认值。

.. dropdown:: 进阶配置（可选）

   ``maniskill_rlt_stage2_ac_mlp.yaml`` 中的 critical phase 调度、自动门控与 expert takeover 参数（注释仅用于说明）：

   .. code-block:: yaml

      algorithm:
        rlt_schedule:
          enable: True                    # 使用 ManiSkill rollout / update budget。
          warmup_post_collect_updates: 30000  # actor 上线前需要完成的 update 数。
          train_every_transitions: 5       # 每 N 条已记录 transition 增加训练预算。
        actor_weight_schedule:
          enable: True                    # 使用 BC/Q 权重 ramp，而不是固定权重。
          warmup_updates: 20000           # warmup BC/Q 权重保持的 update 数。
          ramp_updates: 50000             # 逐步过渡到 online BC/Q 权重。

      env:
        train:
          wrap_obs_mode: rlt_openpi_joint
          rlt_policy_switch:
            enable: True
            task_mode: full_task          # critical phase 前用 VLA，里面用 actor。
            trigger_mode: auto            # 根据任务信息自动计算 critical phase。
            latch_until_done: True        # 一旦进入 critical phase 就保持 actor 控制。
            auto_gate:
              require_grasp: True         # 抓住 peg 后才允许进入。
              require_not_success: True   # 已成功时不再切到 actor。
              near_hole_x_min: -0.16      # 值越大，x 方向进入门控越窄。
              near_hole_yz_margin: 1.5    # 值越大，y/z 进入窗口越宽。
            expert_takeover:
              enable: True                # 只在 train rollout 中启用仿真 expert 干预。
              trigger_mode: stalled_progress  # 连续无进展 chunk 后才接管。
              gate:
                near_hole_x_min: -0.10    # 足够靠近洞口后才检查是否卡住。
                near_hole_yz_margin: 2.0  # y/z 足够接近后才检查是否卡住。
                stuck_chunks_before_takeover: 3  # 接管前需要的无进展 chunk 数。
                min_x_progress: 0.003     # 能重置 stall 计数的前向进展。
                min_yz_progress: 0.0015   # 能重置 stall 计数的对齐进展。
                min_score_progress: 0.002 # 能重置 stall 的综合分数进展。
                progress_yz_weight: 1.0   # 综合进展分数里的 y/z 惩罚权重。

        eval:
          wrap_obs_mode: rlt_openpi_joint
          rlt_policy_switch:
            enable: True
            task_mode: full_task
            trigger_mode: auto
            expert_takeover:
              enable: False               # eval 只测 base policy + actor，不用 expert。

可选：启用 ManiSkill Expert Takeover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManiSkill expert takeover 默认关闭。需要在 critical phase 中用更强的 SFT expert 替换 actor action，并把 expert action 作为 intervention target 写入 replay 时再启用。expert 的 OpenPI dataconfig 和 norm stats 须与 Stage 2 数据保持一致；expert 只用于 train rollout，eval rollout 不执行 expert takeover。

.. dropdown:: 进阶配置（可选）

   .. code-block:: yaml

      rollout:
        expert_model:
          model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_step/actor
          precision: null
          openpi:
            use_rlt: False

      env:
        train:
          rlt_policy_switch:
            expert_takeover:
              enable: True
              trigger_mode: stalled_progress
              gate:
                stuck_chunks_before_takeover: 3
                min_x_progress: 0.003
                min_yz_progress: 0.0015
                min_score_progress: 0.002
                progress_yz_weight: 1.0

Replay Buffer 与监控指标
------------------------

使用 ``loss_type: rlt_ac`` 时，replay buffer 存储提取出的 RLT 特征 ``{z_rl, proprio, ref_chunk}`` 和实际执行的 action chunk，不存储原始图像。切换前的 VLA reference step 与切换后的 actor step 都会进入同一个 replay buffer。``sample_window_size`` 控制近期采样窗口大小，与 ``max_steps_per_rollout_epoch`` 独立。

使用 TensorBoard（``tensorboard --logdir ./logs``）查看日志，完整指标定义见 :doc:`训练指标 <../../reference/metrics>`。核心 RLT 信号：

- **Stage 1 SFT**：``vla_loss``\ （OpenPI 动作预测损失）与 ``rlt_loss``\ （RLT token 损失）。
- **Stage 2 RL**：``env/success_once``、``eval/success_once``、``critic/critic_loss``、``actor/actor_loss``、``actor/bc_loss`` 与 ``train/replay_buffer/size``。
- **ManiSkill 诊断**：``train/replay/actor_switch_rate``、``train/replay/intervention_rate`` 与 ``train/rlt/ready_for_online``。

实验结果
--------

在 RLinf 中的 peg insertion 任务上使用 RL Token 训练的结果如下图。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT_real.png
   :align: center
   :width: 55%

   RLinf peg insertion 任务中的 RL Token 训练结果。

实践建议
--------

- Stage 1 和 Stage 2 的 ``repo_id``、``config_name``、``action_dim``、``proprio_dim``、``ref_num_action_chunks``、``z_dim`` 与 ``norm_stats.json`` 必须一致，推荐显式设置 ``openpi_data.norm_stats_path``。
- Stage 2 中将 ``rollout.rlt_feature_model.model_path`` 指向 Stage 1 FSDP checkpoint 下的 ``actor`` 目录（例如 ``.../checkpoints/global_step_<step>/actor``），不要转成裸 ``model.safetensors``，因为 RLT token 模块保存在完整 wrapper checkpoint 中。
- Stage 2 中不要将 ``actor.model.model_path`` 指向 Stage 1 checkpoint：从头训练 Stage 2 MLP 时保持 ``actor.model.model_path: null``，恢复或加载 Stage 2 MLP 权重使用 ``runner.resume_dir`` 或 ``runner.ckpt_path``。
- ``keyboard_reward_wrapper: rlt_policy_switch`` 仅用于真机按键切换；ManiSkill 仿真使用 ``env.*.rlt_policy_switch``。
