异构环境
========

本页介绍如何在一次运行中，让同一个 policy 在多种不同的环境配置上训练或评估，例如 ManiSkill 与 LIBERO、两个 LIBERO 任务集，或仿真环境与真机。RLinf 把这种用法称为异构环境。

它与 ``total_num_envs`` 不同， ``total_num_envs`` 是把同一份环境配置并行运行多份；它也不同于 :doc:`hetero` ，后者决定各个 worker 运行在哪些机器上。下面先说明异构环境如何运行，再依次介绍如何配置、如何决定哪些 env worker 运行哪个环境、如何在各环境之间平衡训练、如何处理不同的观测和动作统计量，以及如何查看每个环境的结果。

运行方式
--------

通常所有 env worker 都运行同一份环境配置。使用异构环境时，每个 env worker 仍然只运行一份环境配置，但不同的 worker 可以运行不同的配置。

每份环境配置还保留自己的循环设置：一个 rollout epoch 走多少步、一个训练 step 跑几个 epoch、episode 何时结束。rollout 把所有 env worker 的观测收集成一个 batch，运行一次 policy，再把每个 worker 对应的动作发回去。每个 worker 把动作转换成自己的仿真器或机器人能接受的格式，然后推进环境。某份环境配置提前跑完自己的 epoch 后，它的 worker 就退出 batch；所有环境配置都跑完时，这个训练 step 结束。actor 先按每份环境配置自己的设置计算它的 advantage，再用全部轨迹一起训练。

关于这个 batch，有两点需要先了解。每个 env worker 运行的并行环境数相同，这一点由 RLinf 自动安排。观测则可以因环境配置而不同：rollout 会按观测格式分别运行 policy，因此每次前向计算看到的输入都是同一种形状。如果几份环境配置用不同的统计量把 policy 的输出还原成动作，它们也会分开做前向计算。

配置异构环境
------------

把各份环境配置列在 ``env.train.envs`` 下，每份配置用一个你自己起的名字。下面的配置在 LIBERO-Goal 和一个 ManiSkill 场景上训练，各有 64 个并行环境：

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

每一行 ``defaults`` 把一个环境预设加载到一个名字下，这里是 ``libero_goal`` 和 ``maniskill`` 。每份环境配置可以使用普通 ``env.train`` 的全部设置，而且各份配置可以取不同的值： ``total_num_envs`` 是它自己的并行环境数， ``rollout_epoch`` 、 ``max_steps_per_rollout_epoch`` 、 ``max_episode_steps`` 、 ``auto_reset`` 、 ``ignore_terminations`` 和 ``group_size`` 也只作用于它自己的环境。在这个例子里，每个训练 step 中 LIBERO-Goal 跑一个 320 步的 epoch，ManiSkill 跑两个 80 步的 epoch。 ``env.eval`` 的写法相同，并且与 ``env.train`` 互相独立，例如可以在两份环境配置上训练、只在一份上评估。

其他设置（例如视频或奖励）都写在各自所属的环境配置里。 ``env.train.total_num_envs`` 这类汇总值由 RLinf 自动填写；设置互相冲突时，RLinf 会在启动阶段停止，并给出明确的报错信息。

分配 env worker
---------------

RLinf 按你列出的顺序，按各份环境配置 ``total_num_envs`` 的比例划分 env worker。假设 ``env`` 有 4 个 worker，配置了 48 个 LIBERO-Goal 环境和 16 个 ManiSkill 环境，那么每个 worker 运行 16 个环境：

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

每份环境配置都必须占满整数个 worker。如果只有 2 个 worker，LIBERO-Goal 需要一个半 worker，RLinf 会拒绝这份配置；这时请调整环境数或 worker 数。

在分给自己的 worker 上，一份环境配置的行为与它单独运行时相同：随机种子和重置状态只在它自己的 worker 之间划分， ``override_cfgs`` 列表也是每个 worker 对应一项。

actor worker 会把每份环境配置的轨迹平均分给各自，因此每个 actor worker 训练的环境组成都相同。一个 GRPO group 必须留在同一个 actor worker 上，所以每份环境配置的 group 数必须能被 actor worker 数整除；RLinf 会在启动阶段检查这一点。

这些 worker 运行在哪里，由 ``env`` 的 placement 决定。多个 env worker 可以共用一块 GPU，例如 ``env: 0:0-1`` 。如果要驱动真机，请把这份环境配置的 worker 放到机器人所在的节点上，具体做法见 :doc:`hetero` 。

.. warning::

   worker 按列出的顺序分配。调整 ``envs`` 的顺序或修改环境数，都会改变每个 worker 运行的环境，修改后请重新检查 placement，使用真机时尤其如此。

平衡各环境的训练
----------------

每个训练 step 包含每份环境配置的 ``total_num_envs`` 乘以 ``rollout_epoch`` 条轨迹，一条轨迹就是一个环境的一个 rollout epoch。在上面的例子里，这是 64 条最长 320 步的 LIBERO-Goal 轨迹和 128 条 80 步的 ManiSkill 轨迹。这些数量决定了每份环境配置的数据量；一个 step 的耗时取决于最慢的那份环境配置， ``time/env/<name>/interact`` 记录了每份配置用了多长时间。

默认情况下，无论环境的 episode 有多长，每条轨迹在 loss 中的分量都相同，所以在这个例子里 ManiSkill 占梯度的三分之二。下面这些设置可以调整这一点：

.. code-block:: yaml

   algorithm:
     env_weighting: trajectory  # 或 sample
     adv_norm_scope: env        # 或 all
     envs:
       maniskill:
         loss_weight: 0.5
   actor:
     mix_envs: True

- ``env_weighting: sample`` 改为让每一步的分量相同，于是 episode 更长的环境配置占比更大。
- ``algorithm.envs.<name>`` 为一份环境配置单独设置 ``loss_weight`` 、 ``gamma`` 、 ``gae_lambda`` 、 ``rewards_lower_bound`` 或 ``rewards_upper_bound`` ，其余设置沿用 ``algorithm`` 。 ``loss_weight`` 会乘到这份环境配置每一步的权重上。
- ``adv_norm_scope: all`` 在所有环境配置上一起归一化 advantage，而不是分别归一化。它只影响在整个 batch 上做归一化的 advantage 类型，例如 GAE；GRPO 本来就在每个 group 内归一化。
- ``actor.mix_envs: False`` 让每个 optimizer step 只使用一份环境配置的数据，顺序随机且所有 actor worker 一致。此时每份环境配置的样本数必须能填满整数个 global batch。

异步训练和 off-policy 训练也保持同样的配比。在异步 PPO 中，每份环境配置按它每个 step 发给每个 actor 的轨迹数，分到 ``algorithm.rollout_store_size_per_rank`` 中相应的份额；训练 step 要等每份份额都填满才开始，因此跑得快的环境不会挤占慢的环境。off-policy 训练（SAC、DAgger、RLT）为每份环境配置保留一个 replay buffer，并按每份配置每个 step 采集的 transition 数乘以 ``loss_weight`` 的比例从中采样；设置 ``actor.mix_envs: False`` 时，每个 batch 只来自一份环境配置。

奖励保持各环境自己的尺度。PPO 和 SAC 的 critic 同时服务所有环境配置，如果各环境的奖励大小相差很多，较大的那份会主导 critic，另一个环境得到的 baseline 就会变差。可以在环境配置上设置 ``reward_scale`` ，让各环境的 return 处在相近的量级；它会在训练读取奖励之前乘到这个环境的奖励上，此后 ``huber_delta`` 和 ``value_clip`` 作用于缩放后的值。

处理不同的观测
--------------

来自不同仿真器的环境通常发送不同的观测：LIBERO 发送 256x256 的主相机和腕部相机图像，而这个 ManiSkill 场景只发送一个 480x640 的相机图像。RLinf 会把每一轮 rollout 的请求按观测格式分组，观测格式指观测有哪些 key 以及每个 key 的形状，然后每种格式运行一次前向计算。actor 训练时，每个 micro-batch 也只包含一种格式。因此 policy 每次只会看到一种形状，但它必须能处理收到的每一种格式。

分开处理不同格式会降低速度：每次前向计算的 batch 更小，一轮也需要按格式数运行多次。设置 ``policy_obs`` 可以让各环境的格式一致，所有环境就能重新共用一次前向计算：

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` 列出 policy 读取的观测，其余观测不会发送； ``image_size`` 把所有相机图像缩放到同一个高度和宽度。这样设置后，两个环境都只发送 256x256 的主相机图像和任务描述，这正是 OpenVLA-OFT 需要的输入。

被改动的只是发给 policy 的那一份观测，奖励模型、视频录制和数据采集看到的仍是环境的原始观测。缩放在每一步都会消耗时间，所以图像尺寸本来一致时，不要设置 ``image_size`` 。 ``policy_obs`` 不会缩放机器人状态，状态维度不同的环境配置会保持各自的格式。

处理不同的动作统计量
--------------------

动作的方向正好相反：每个 env worker 把属于自己的那部分动作转换成自己环境的格式，所以 policy 只需要一种动作格式。OpenVLA-OFT 还要用某个训练数据集的统计量把归一化的输出还原成机器人动作，这个数据集由 ``actor.model.unnorm_key`` 指定。如果某份环境配置的机器人来自统计量不同的数据集，就在这份环境配置上设置 ``unnorm_key`` ，训练和评估都要设置：

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

这样 ManiSkill 的动作会使用 Bridge 数据集的统计量，没有设置 ``unnorm_key`` 的环境配置仍使用 ``actor.model.unnorm_key`` 。与观测格式不同时一样，rollout 会把 ``unnorm_key`` 不同的环境配置放在不同的前向计算中。训练不受影响，因为 OpenVLA-OFT 的 log probability 是在归一化后的动作 token 上计算的。

只有 OpenVLA-OFT 会读取每份环境配置的 ``unnorm_key`` ；如果换成会忽略它的 policy，RLinf 会在启动阶段停止。

查看各环境的结果
----------------

每个指标都会记录两份：一份汇总所有环境，一份按环境配置区分。 ``env/success_once`` 统计全部 episode， ``env/libero_goal/success_once`` 只统计 LIBERO-Goal。评估指标以同样的方式记录在 ``eval/`` 下。

视频保存在 ``<video_base_dir>/seed_<n>/<name>/`` 下，因此共用同一个视频目录的环境配置不会互相覆盖文件。

运行示例
--------

端到端测试 ``maniskill_libero_grpo_openvlaoft`` 用 GRPO 在 LIBERO-Goal 和 ManiSkill 上训练 OpenVLA-OFT，并在两者上评估。运行它需要一个同时安装了两个仿真器的虚拟环境、配置中指定的 checkpoint 和 LoRA adapter，以及 2 块 GPU。

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

这组命令依次完成：

1. 安装 OpenVLA-OFT 以及 ManiSkill 和 LIBERO。
2. 在 env worker 0 上运行 LIBERO-Goal，在 env worker 1 上运行 ManiSkill。
3. 运行两个 GRPO step，并把 ``env/<name>/*`` 和 ``eval/<name>/*`` 记录到 ``runner.logger.log_path`` 下的 TensorBoard 中。

这个测试用 8 步的 episode 检查各部分能否正确连通。用于实际训练时，请加长 episode 并增加环境数。ManiSkill 场景还需要准备资源文件，具体做法见 :doc:`../examples/embodied/maniskill` 。

限制
----

- policy 只有一种动作格式，也无法得知某个样本来自哪个环境。只有 OpenVLA-OFT 能按环境配置分别还原动作。
- NFT 训练要求各份环境配置发送同一种观测格式，请用 ``policy_obs`` 让它们一致。
- training pipeline（ ``runner.use_training_pipeline`` ）和 DAgger 的在线 LeRobot 采集需要单份环境配置。
- PPO 和 SAC 在各环境配置之间共享一个 critic；RLinf 不会按环境自动归一化奖励，奖励尺度不同时请设置 ``reward_scale`` 。SAC 为每份环境配置分别调节 entropy temperature，但与 demo buffer 混合采样的样本使用各 temperature 的平均值。
- 在 env-decoupled 模式下，每个 rollout worker 从共享队列中处理一个 step 里相同份额的请求，并对收到的请求按观测格式分别运行前向计算。
- 平滑干预（ ``smooth_intervene`` ）和实时控制评估（ ``runner.rtc`` ）只支持单份环境配置；RLT 仿真 replay 要求训练中的每份环境配置都是 ``maniskill_rlt`` 。配置不满足这些要求时，RLinf 会在启动阶段停止。
