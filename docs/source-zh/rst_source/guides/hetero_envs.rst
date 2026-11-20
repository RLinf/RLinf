异构环境
========

本页介绍如何在一次运行中，让同一个 policy 在多种不同的环境配置上训练或评估，例如 ManiSkill 与 LIBERO、两个 LIBERO 任务集，或仿真环境与真机。RLinf 将这种用法称为异构环境。

它不同于 ``total_num_envs`` （并行运行多份相同配置），也不同于 :doc:`hetero` （决定各组件运行在哪些机器上）。

运行方式
--------

每个 env worker 仍然只运行一份环境配置，且每个 worker 的并行环境数相同，但不同 worker 可以运行不同的配置。每份环境配置保留自己的循环设置：每个 epoch 走多少步、每个训练 step 跑几个 epoch，以及 episode 何时结束。

rollout 阶段，每个 rollout worker 服务路由到自己的 env worker；某份环境配置跑完自己的 epoch 后，对应的 worker 就退出当前步的 batch。只有落在同一个 rollout worker、且观测格式和 ``unnorm_key`` 等 policy 参数都一致的请求，才会合并成一次前向计算；否则 rollout worker 按分组分别执行前向计算。

所有环境配置完成当前步后，actor 先按每份配置各自的算法设置计算 advantage，再用全部轨迹一起更新共享 policy。

配置异构环境
------------

把各份环境配置列在 ``env.train.envs`` 下，每份配置起一个自定义名称。下面的配置同时在 LIBERO-Goal 和一个 ManiSkill 场景上训练，各有 64 个并行环境：

.. code-block:: yaml

   defaults:
     - env/libero_goal@env.train.envs.libero_goal
     - env/maniskill_put_on_plate_in_scene_25_main@env.train.envs.maniskill
     # model、FSDP 和 weight syncer 预设与平常相同

   env:
     train:
       envs:
         libero_goal:
           total_num_envs: 64
           rollout_epoch: 1
           max_episode_steps: 320
           max_steps_per_rollout_epoch: 320
           group_size: 8
         maniskill:
           total_num_envs: 64
           rollout_epoch: 2
           max_episode_steps: 80
           max_steps_per_rollout_epoch: 80
           ignore_terminations: True
           group_size: 8

每一行 ``defaults`` 把一个环境预设加载到对应名称下（这里是 ``libero_goal`` 和 ``maniskill`` ）。每份环境配置都独立支持 ``env.train`` 的全部字段： ``total_num_envs`` 、 ``rollout_epoch`` 、 ``max_steps_per_rollout_epoch`` 、 ``max_episode_steps`` 、 ``auto_reset`` 、 ``ignore_terminations`` 和 ``group_size`` 只作用于它自己。 ``env.eval`` 的写法相同且与 ``env.train`` 独立，可以在两份环境配置上训练、只在一份上评估。

视频、奖励等环境相关设置同样写在各自的环境配置里。 ``env.train.total_num_envs`` 这类汇总字段由 RLinf 自动推导；配置冲突时会在启动阶段直接报错。

分配 env worker
---------------

RLinf 按列出顺序，根据各份环境配置 ``total_num_envs`` 的比例划分 env worker。假设 ``env`` 有 4 个 worker，配置 48 个 LIBERO-Goal 环境和 16 个 ManiSkill 环境，每个 worker 运行 16 个环境：

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - 环境配置
     - ``total_num_envs``
     - env worker
   * - ``libero_goal``
     - 48
     - 0、1、2
   * - ``maniskill``
     - 16
     - 3

每份环境配置都必须占满整数个 env worker，且每份配置的 GRPO group 数必须能被 actor worker 数整除，RLinf 会在启动阶段校验。随机种子、重置状态和 ``override_cfgs`` 只在各自对应的 worker 之间划分。

worker 运行在哪些硬件上由 ``env`` 的 placement 决定。多个 env worker 可以共用一块 GPU，例如 ``env: 0:0-1`` 。驱动真机时，把该环境配置的 worker 放到机器人所在节点，具体做法见 :doc:`hetero` 。

.. warning::

   worker 按列出顺序分配。调整 ``envs`` 顺序或修改环境数都会改变每个 worker 运行的环境，修改后请重新核对 placement，接入真机时尤其如此。

平衡各环境的训练
----------------

每个训练 step 为每份环境配置采集 ``total_num_envs * rollout_epoch`` 条轨迹，整个 step 的耗时取决于最慢的那份环境配置。如需查看各环境耗时，设置 ``runner.per_worker_log: True`` ，然后查看对应 env worker 的 ``time/env/interact`` 。

默认情况下，无论 episode 长短，每条轨迹在 loss 中的权重相同，因此上面的 64 对 128 示例里 ManiSkill 占三分之二的梯度。可以通过以下配置调整配比：

.. code-block:: yaml

   algorithm:
     env_weighting: trajectory  # 或 sample
     adv_norm_scope: env        # 或 all
     envs:
       maniskill:
         loss_weight: 0.5
   actor:
     mix_envs: True

- ``env_weighting: sample`` 按每份环境配置的名义 horizon（带 done mask 时为 ``max_episode_steps`` ，不带 mask 时为 ``max_steps_per_rollout_epoch`` ）缩放轨迹权重，使 horizon 更长的环境配置占更大比重，同时同一环境内提前结束的轨迹仍保持按轨迹归一化。
- ``algorithm.envs.<name>`` 为单份环境配置覆盖 ``loss_weight`` 、 ``reward_scale`` 、 ``gamma`` 、 ``gae_lambda`` 、 ``rewards_lower_bound`` 或 ``rewards_upper_bound`` ，其余字段沿用 ``algorithm`` 。 ``loss_weight`` 必须为正数，用来缩放该环境在 loss 中的比重。
- ``adv_norm_scope: all`` 在所有环境配置上联合归一化 advantage，而不是分别归一化。它仅适用于整 batch 归一化的 advantage 类型——即 ``gae`` 以及设置 ``normalize_advantages: True`` 的 ``raw`` ；GRPO 在 group 内归一化，不能使用 ``all`` 。
- ``actor.mix_envs: False`` 让每个 optimizer step 只训练一份环境配置的数据，顺序随机且所有 actor worker 一致。此时每份环境配置在单个 actor 上的样本数必须能填满整数个 global batch。

异步训练和 off-policy 训练保持同样配比。异步 PPO 中，每份环境配置按每步发给单个 actor 的轨迹数比例分得 ``algorithm.rollout_store_size_per_rank`` 的配额，等每份配额都填满后才开始更新。off-policy 训练（SAC、DAgger、RLT）为每份环境配置维护独立的 replay buffer，等每个 buffer 都存满 ``min_buffer_size`` 条轨迹后启动训练，并按每步采集的 transition 数乘以 ``loss_weight`` 的比例采样（设置 ``actor.mix_envs: False`` 时，每个 batch 只来自一份环境配置）。

奖励保持各环境原有尺度。PPO 和 SAC 在所有环境配置间共享一个 critic，奖励量级相差较大时请设置 ``algorithm.envs.<name>.reward_scale`` ，避免某一环境主导 critic。该系数在 ``huber_delta`` 和 ``value_clip`` 生效前乘到奖励上。

处理不同的观测
--------------

不同仿真器输出的观测往往不同：LIBERO 输出 256x256 的主相机和腕部相机图像，而 ManiSkill 只输出单路 480x640 图像。rollout 按观测格式（key 与形状）和 policy 参数分组前向计算，actor 训练时每个 micro-batch 也只包含一种观测格式，因此 policy 每次前向计算只看到单一形状。

设置 ``policy_obs`` 可以统一不同仿真器的观测格式：

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` 指定发给 policy 的观测项， ``image_size`` 把所有相机图像缩放到统一高宽。视频录制和数据采集仍然使用原始观测。图像尺寸本来一致时不要设置 ``image_size`` ，以免每步额外缩放。 ``policy_obs`` 不会缩放机器人状态向量。

统一观测格式只消除了一项拆分 rollout 前向计算的原因。请求要合并为同一次 rollout 前向计算，还必须落在同一个 rollout worker 且拥有相同的 policy 参数（例如 ``unnorm_key`` ）。例如仓库自带的 ``maniskill_libero_grpo_openvlaoft`` 配置把每个 env worker 一对一路由到各自的 rollout worker，且为 ManiSkill 单独指定了 ``unnorm_key`` ，因此即使配置了 ``policy_obs`` ，两个环境在 rollout 阶段仍分开前向计算；但 ``policy_obs`` 仍能让它们在 actor 训练阶段共用同一个训练池。

处理不同的动作统计量
--------------------

每个 env worker 各自把动作转换成对应仿真器的格式，因此 policy 只需输出一种动作格式。OpenVLA-OFT 还会用 ``actor.model.unnorm_key`` 指定的数据集统计量把归一化输出还原为动作。若某份环境配置需要不同的统计量，在训练和评估配置下分别指定 ``unnorm_key`` ：

.. code-block:: yaml

   env:
     train:
       envs:
         maniskill:
           unnorm_key: bridge_orig
     eval:
       envs:
         maniskill:
           unnorm_key: bridge_orig

这样 ManiSkill 使用 Bridge 数据集的统计量，其余环境配置继续使用 ``actor.model.unnorm_key`` 。rollout 会把 ``unnorm_key`` 不同的环境配置分到不同的前向计算中；actor 训练不受影响，因为 OpenVLA-OFT 在归一化动作 token 上计算 log probability。只有 OpenVLA-OFT 读取每份环境配置的 ``unnorm_key`` ，换用其他 policy 时启动校验会直接报错。

查看各环境的结果
----------------

env、eval 和 rollout 指标同时记录汇总值（ ``env/success_once`` 、 ``eval/success_once`` ）和分环境值（ ``env/<name>/success_once`` 、 ``eval/<name>/success_once`` ）。训练与耗时指标（ ``train/actor/*`` 、 ``time/*`` ）仅记录汇总值。group 成功率和数据陈旧度依赖各自环境的 group 与 rollout store，因此仅按环境配置记录。

视频保存在 ``<video_base_dir>/seed_<n>/<name>/`` 下，避免不同环境配置互相覆盖文件。

运行示例
--------

``maniskill_libero_grpo_openvlaoft`` 示例用 GRPO 同时在 LIBERO-10 和 ManiSkill 上训练同一个 OpenVLA-OFT policy。

前置条件：

- **硬件**：4 张 GPU（2 个 LIBERO-10 env worker 与 2 个 ManiSkill env worker，与 actor、rollout 共同部署在 GPU 0–3 上）。
- **ManiSkill 资源文件**：按 :doc:`../examples/embodied/maniskill` 的说明下载。
- **模型权重**： ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` （LIBERO-10 基础权重）和 ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora`` （在其上训练的 ManiSkill LoRA adapter）。配置通过 ``merged_loras`` （ ``scale: 0.5`` ）将一半强度的 ManiSkill LoRA 合并进基础权重，使初始 policy 在两个环境里都偶尔能成功，从而为 GRPO 提供有效学习信号，再在其上训练新的 LoRA。

在一个虚拟环境中同时安装两个仿真器：

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

下载基础权重和 ManiSkill LoRA adapter：

.. code-block:: bash

   # 在中国大陆可设置 HF_ENDPOINT=https://hf-mirror.com。
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Main
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Lora

在 ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml`` 中将以下三个路径改为本地目录：

- ``actor.model.model_path`` ：指向 ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
- ``rollout.model.model_path`` ：指向 ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
- ``actor.model.merged_loras[0].path`` ：指向 ``RLinf-OpenVLAOFT-ManiSkill-Base-Lora``

启动训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_libero_grpo_openvlaoft

训练启动后，在终端指标表和 TensorBoard 中确认同时出现 ``env/libero_10/success_once`` （初始约 0.2）、 ``env/maniskill/success_once`` （初始约 0.09）以及汇总的 ``env/success_once`` ，即表示两个环境均已正常运行。

限制
----

- policy 只有一种动作格式，且不接收环境标识。只有 OpenVLA-OFT 支持按环境配置设置 ``unnorm_key`` 。
- NFT 训练要求各环境使用同一种观测格式（设置 ``policy_obs`` ）、设置 ``algorithm.env_weighting: sample`` ，且不能设置分环境 ``loss_weight`` 。
- ``actor.mix_envs: True`` 下，off-policy 训练会把多份环境配置混入同一个 batch，因此所有环境必须存储同一种观测格式（设置 ``policy_obs`` ）。
- ``runner.use_training_pipeline`` 、DAgger 在线 LeRobot 采集、DiffusionART 训练、 ``reward.use_reward_model`` 、 ``smooth_intervene`` 、 ``runner.rtc`` 以及 ``huggingface`` 以外的 rollout 后端（PhyAI、SGLang、ApxInf）都要求单份环境配置。RLT 仿真 replay 要求每份训练环境配置都是 ``maniskill_rlt`` 。
- PPO 和 SAC 在各环境配置间共享一个 critic，奖励尺度不同时请设置 ``algorithm.envs.<name>.reward_scale`` 。SAC 为每份环境配置独立调节 entropy temperature，与 demo buffer 混合采样的样本使用平均 temperature。
- env-decoupled 模式下，rollout worker 从共享队列处理当前 step 属于自己的份额，并在每个 batch 内按观测格式和 policy 参数分组前向计算。
