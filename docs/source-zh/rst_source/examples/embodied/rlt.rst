RL Token：借助视觉-语言-动作模型启动在线强化学习
================================================

**RL Token: Bootstrapping Online RL with Vision-Language-Action Models** 在冻结的 VLA 特征模型之上训练一个轻量级强化学习策略。在 RLinf 的配置和代码中，这套流程简称为 **RLT**。整个流程分为两个阶段：

1. 在示范数据上联合训练 VLA 检查点和 RLT token transformer。
2. 冻结第一阶段得到的特征模型，用提取出的 RLT 状态训练一个轻量级 off-policy actor-critic。

当前仓库中的示例配置面向 Franka peg insertion，但 pipeline 本身并不绑定这个任务。后续如果添加仿真环境，只需要替换环境配置、动作维度、状态选择和数据路径，仍然可以复用相同的两阶段结构。

官方项目页：`Precise Manipulation with Efficient Online RL <https://www.pi.website/research/rlt>`_。

概览
----

RLT 将表示学习和在线 RL 控制拆开。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Stage 1
      :text-align: center

      VLA SFT + RLT token transformer

   .. grid-item-card:: Stage 2
      :text-align: center

      轻量级 off-policy actor-critic

   .. grid-item-card:: 状态
      :text-align: center

      ``z_rl`` + proprio + reference chunk

   .. grid-item-card:: 部署
      :text-align: center

      当前是真机示例，结构可扩展到仿真

| **你将完成：** 准备示范数据 -> 训练 Stage 1 -> 在 Stage 2 中加载 Stage 1 检查点 -> 启动 actor-critic 训练 -> 观察 replay buffer 与任务成功率指标。
| **前置条件：** 安装 OpenPI π₀.₅ checkpoint，并准备 :doc:`Franka 真机环境 <../embodied/franka>`。

提供的配置文件
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - 阶段
     - 配置
     - 作用
   * - Stage 1
     - ``examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml``
     - 联合 SFT pi0.5 和 RLT token transformer。
   * - Stage 2
     - ``examples/embodiment/config/rlt_stage2_ac_mlp.yaml``
     - 使用冻结的 Stage 1 特征模型训练 RLT Stage 2 actor-critic。

安装
----

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**方式一：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 为提高国内下载速度，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后，切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**方式二：自建环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 到下面的 install.sh 命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

RLT 如何工作
------------

Stage 1：学习 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 从一个 VLA 检查点开始，在同一批示范数据上优化两个目标：

1. 普通 VLA 动作预测目标，日志中记为 ``vla_loss``。
2. RLT token 目标，日志中记为 ``rlt_loss``。

Stage 1 的总损失为：

.. code:: text

   total_loss = rlt_loss + rlt_alpha * vla_loss

OpenPI 模型会暴露 VLA prefix hidden states。RLT token transformer 读取这些 prefix states，并输出紧凑向量 ``z_rl``。Stage 2 使用 ``z_rl`` 作为学习到的 RL 表示，而不是直接在图像观测上训练 actor-critic。

Stage 1 中比较关键的字段：

.. code:: yaml

   # examples/sft/config/rlt_stage1_sft_openpi_pi05.yaml
   data:
     train_data_paths: "/path/to/data"

   actor:
     openpi_data:
       repo_id: "realworld_peg_insertion_rlt_stage1"
     model:
       model_type: "openpi"
       is_lora: False
       model_path: "/path/to/model"
       num_action_chunks: 20
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         use_rlt: True
         rlt_alpha: 1.0
         rlt_prefix_seq_len: 1024
         rlt_image_only: False
         rlt_use_mask: True

``repo_id`` 和 ``config_name`` 需要与归一化统计和 Stage 2 特征模型配置保持一致。

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

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的动作
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

当前真机配置中，``keyboard_reward_wrapper: rlt_policy_switch`` 会额外提供 ``rlt_switch_flags`` 标记。在操作员按下 ``b`` 之前，实际执行 VLA 的 ``ref_chunk``；按下 ``b`` 之后，实际执行 Stage 2 actor 的动作。仿真配置可以不使用这个 wrapper，也可以替换成自动切换逻辑。

critic 使用 chunked rewards 上的 TD target。一个 action chunk 内的奖励会先按折扣累计，然后用下一状态 Q 值 bootstrap：

.. code:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

如果 episode 已终止，并且 ``bootstrap_type`` 为 ``standard``，则不会加入 bootstrap 项。

Stage 2 中比较关键的字段：

.. code:: yaml

   # examples/embodiment/config/rlt_stage2_ac_mlp.yaml
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
     model:
       model_path: null
       precision: ${actor.model.precision}
       action_dim: ${actor.model.action_dim}
       num_action_chunks: ${actor.model.num_action_chunks}
       ref_num_action_chunks: ${actor.model.ref_num_action_chunks}
     rlt_feature_model:
       model_type: "openpi"
       model_path: "/path/to/stage1/checkpoint"
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         action_chunk: ${actor.model.ref_num_action_chunks}
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

.. code:: text

   actor_loss = -q_weight * Q(obs, pi(obs)) + bc_weight * BC(pi(obs), target_action)

普通 policy step 中，BC target 是 VLA reference action。如果某一步存了 human intervention action，那么这一步的 BC target 会切换成人类动作。

运行当前 Franka 示例
--------------------

数据：采集 Franka 示范并计算归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 需要 LeRobot 格式的 Franka 示范数据，数据目录应直接包含
``data/`` 和 ``meta/``。在控制节点上按照 :doc:`Franka 真机文档
<../embodied/franka>` 的数据采集流程准备机器人和目标位姿，并在采集配置中导出
LeRobot 格式数据：

.. code:: yaml

   env:
     data_collection:
       enabled: True
       export_format: "lerobot"

然后启动采集：

.. code:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

采集完成后，将 LeRobot 数据集放到训练节点，并为当前 RLT OpenPI dataconfig
计算归一化统计。``repo_id`` 需要与 Stage 1 / Stage 2 配置中的
``actor.openpi_data.repo_id`` 和 ``rollout.rlt_feature_model.openpi_data.repo_id`` 保持一致：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state \
       --repo-id realworld_peg_insertion_rlt_stage1

后续将 Stage 1 配置中的 ``data.train_data_paths`` 指向该 LeRobot 数据集目录。

Stage 1：训练 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

启动前先修改 Stage 1 配置中的路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/lerobot_dataset

   actor:
     openpi_data:
       repo_id: "realworld_peg_insertion_rlt_stage1"
     model:
       model_path: /path/to/model
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         rlt_prefix_seq_len: 1024

启动 SFT：

.. code:: bash

   bash examples/sft/run_vla_sft.sh rlt_stage1_sft_openpi_pi05

保存出的检查点目录通常形如：

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>

Stage 2 中需要将这个目录填到 ``rollout.rlt_feature_model.model_path``。
不要把 Stage 1 checkpoint 填到 ``rollout.model.model_path`` 或
``actor.model.model_path``；这两个位置不负责加载 Stage 1 特征模型。

Stage 2：运行 RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 Stage 2 配置：

.. code:: yaml

   rollout:
     model:
       model_path: null
     rlt_feature_model:
       model_path: /path/to/stage1/checkpoint
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         state_indices: []
         rlt_prefix_seq_len: 1024

   cluster:
     node_groups:
       - label: <gpu_node_group>
         node_ranks: <gpu_node_rank>
       - label: <env_node_group>
         node_ranks: <env_node_rank>
         hardware:
           type: <robot_type>
           configs:
             - robot_ip: <robot_ip>

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch
       override_cfg:
         target_ee_pose: [<target_x>, <target_y>, <target_z>, <target_roll>, <target_pitch>, <target_yaw>]

在 master node 上启动异步训练：

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh rlt_stage2_ac_mlp

当前默认键盘模块实现了 RLT 算法中的关键阶段切换：按 ``b`` 进入 Stage 2 actor
控制阶段。其他功能可根据具体任务需求进行定制
（``rlinf/envs/realworld/common/wrappers/keyboard_rlt_policy_switch_wrapper.py``）。

Replay Buffer 逻辑
------------------

当 ``loss_type: rlt_ac`` 时，replay buffer 不会把原始图像观测当作 RL 状态存储。env worker 会等待 rollout worker 返回 RLT 特征，并把这些特征组装成 transition。

这意味着：

- 真机切换前的 step 仍然会用于训练。这些 step 的执行动作是 VLA reference action，transition 也会进入同一个 replay buffer。
- 切换后的 step 执行 actor action，并以相同的 RLT observation 格式存储。
- ``sample_window_size`` 控制从 replay buffer 近期窗口中采样的范围，不需要和 ``max_steps_per_rollout_epoch`` 一致。
- ``max_steps_per_rollout_epoch`` 控制一次 rollout flush 到训练侧之前收集多少环境 step。

监控指标
--------

指标定义见 :doc:`训练指标 <../../reference/metrics>`。RLT 中比较有用的信号包括：

- Stage 1 SFT：

  - ``vla_loss``：OpenPI 动作预测损失。
  - ``rlt_loss``：RLT token 重建 / 压缩损失。

- Stage 2 actor-critic：

  - ``train/sac/critic_loss``：Q 函数 TD loss。
  - ``train/sac/actor_loss``：组合后的 ``-Q + BC`` actor 目标。
  - ``q_pi`` 和 ``q_value_*``：actor 和 critic heads 上的 Q 值。
  - ``bc_loss``、``bc_ref_loss``、``bc_human_loss``：BC 正则相关损失。
  - ``train/replay_buffer/size``：当前 replay transitions 数量。
  - ``env/success_once`` 和 ``env/episode_len``：任务结果指标。

实验结果
--------

在 RLinf 中的 peg_insertion 任务中利用 RL Token 训练结果如下图。

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
      <div style="flex: 0.5; text-align: center;">
        <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT_real.png" style="width: 100%;"/>
       <p><em>RLinf peg_insertion 任务中的 RL Token 训练结果</em></p>
     </div>
   </div>

实践建议
--------

- Stage 1 和 Stage 2 的数据配置必须保持一致：``repo_id``、``config_name``、``action_dim``、``proprio_dim``、``ref_num_action_chunks`` 和 ``z_dim`` 都要对齐；如果保留完整 raw state，可使用 ``state_indices: []``。
- ``rollout.rlt_feature_model`` 指向 Stage 1 检查点；``actor.model`` 是 actor-critic worker 会更新的 Stage 2 MLP 策略。
- ``rollout.model`` 是 Stage 2 MLP 在 rollout worker 上的同步副本。Stage 2 从头训练时保持 ``rollout.model.model_path: null``；恢复 Stage 2 训练使用 ``runner.resume_dir``，加载单个 Stage 2 权重文件使用 ``runner.ckpt_path``。
- 不要配置 ``actor.model.model_path`` 来加载 Stage 1；``actor.model`` 只描述 Stage 2 MLP 的输入输出维度和 Q-head 设置。
- ``keyboard_reward_wrapper: rlt_policy_switch`` 只在需要人工控制关键阶段切换时使用。
- 添加仿真示例时，可以新建仿真环境配置，保留 ``loss_type: rlt_ac`` 和 ``rollout.rlt_feature_model``，再把真机阶段切换逻辑替换成适合仿真的逻辑。
