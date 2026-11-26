多 Env 混合训练
===============

在一次运行中，让同一个 policy 在多个不同的环境配置上联合训练或评估，例如 ManiSkill 与 LIBERO 联合训练、多个 LIBERO 任务集组合，或仿真与真机混合训练。

配置多环境
----------

在 ``env.train.envs`` 下为每份环境配置起一个自定义名称并分别配置。下面示例同时在 LIBERO-Goal 和 ManiSkill 场景上训练，各开 64 个并行环境：

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
           group_size: 8

``envs`` 下的每份配置各自独立设置 ``total_num_envs`` 、 ``rollout_epoch`` 、 ``max_episode_steps`` 、 ``max_steps_per_rollout_epoch`` 、 ``auto_reset`` 、 ``ignore_terminations`` 、 ``group_size`` 、奖励和视频选项。 ``env.eval.envs`` 的写法相同且相互独立。

env worker 按列出顺序根据 ``total_num_envs`` 的比例分配：

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - 环境配置
     - ``total_num_envs``
     - env worker（共 4 个）
   * - ``libero_goal``
     - 48
     - 0、1、2
   * - ``maniskill``
     - 16
     - 3

每份配置需占满整数个 env worker，其 GRPO group 数需能被 actor worker 数整除。可通过 ``cluster.component_placement.env`` 将 worker 放到指定 GPU 或真机节点（见 :doc:`hetero` ）。调整 ``envs`` 顺序或环境数会改变 worker 分配，修改后请重新核对 placement。

对齐观测与动作统计量
--------------------

不同环境的相机字段或图像分辨率不一致时，用 ``policy_obs`` 统一传给 policy 的观测格式：

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` 指定发给 policy 的观测项， ``image_size`` 将相机图像缩放到统一高宽，视频录制仍保留原始分辨率。统一观测格式可让多个环境共用 actor 训练池；若请求路由到同一 rollout worker 且 ``unnorm_key`` 等 policy 参数也一致，还能合并 rollout 前向计算。

使用 OpenVLA-OFT 且某个环境需要不同于 ``actor.model.unnorm_key`` 的动作反归一化统计量时，在 ``env.train.envs`` 和 ``env.eval.envs`` 下为该环境指定 ``unnorm_key`` ：

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

调整各环境训练权重
------------------

每个训练 step 为每份环境采集 ``total_num_envs * rollout_epoch`` 条轨迹。默认每条轨迹在 loss 中权重相等：开头配置里 LIBERO-Goal 为 ``64 * 1 = 64`` 条、ManiSkill 为 ``64 * 2 = 128`` 条，ManiSkill 占等权轨迹数的三分之二（实际梯度占比取决于各自梯度）。可通过以下选项调整配比：

.. code-block:: yaml

   algorithm:
     env_weighting: trajectory  # 或 sample
     adv_norm_scope: env        # 或 all
     envs:
       maniskill:
         loss_weight: 0.5
         reward_scale: 1.0
   actor:
     mix_envs: True

- ``algorithm.env_weighting`` ： ``trajectory`` （默认）按轨迹等权； ``sample`` 按各环境名义 horizon（带 done mask 时为 ``max_episode_steps`` ，不带 mask 时为 ``max_steps_per_rollout_epoch`` ）缩放轨迹权重。
- ``algorithm.envs.<name>`` ：按环境覆盖 ``loss_weight`` （须为正数）、 ``reward_scale`` 、 ``gamma`` 、 ``gae_lambda`` 、 ``rewards_lower_bound`` 或 ``rewards_upper_bound`` 。PPO 和 SAC 下各环境奖励量级相差较大时，建议设置 ``reward_scale`` 。
- ``algorithm.adv_norm_scope`` ： ``env`` （默认）按环境分别归一化； ``all`` 跨环境联合归一化（适用于 ``gae`` 及开启 ``normalize_advantages: True`` 的 ``raw`` ）。
- ``actor.mix_envs`` ： ``True`` （默认）混合各环境训练； ``False`` 让每个 optimizer step 只训练单个环境（单环境在每个 actor 上的样本数需填满整数个 global batch）。

异步 PPO 下， ``algorithm.rollout_store_size_per_rank`` 以单个 actor rank 收到的轨迹分片（trajectory shard）数计，需能按各环境分片比例整除（例如两个等份额环境应设为 ``2`` 而非默认的 ``1`` ）。off-policy 回放采样（SAC、DAgger、RLT）按 ``total_num_envs * rollout_epoch * chunk_steps * loss_weight`` 分配比例（不受 ``env_weighting`` 影响），在 ``actor.mix_envs: True`` 时需配置 ``policy_obs`` 统一观测格式。

运行示例
--------

``maniskill_libero_grpo_openvlaoft`` 配置用 GRPO 同时在 LIBERO-10 和 ManiSkill 上训练同一个 OpenVLA-OFT policy。

前置条件：

- **硬件**：4 张 GPU（ ``actor,env,rollout: 0-3`` ）。
- **ManiSkill 资源**：按 :doc:`../examples/embodied/maniskill` 下载。
- **模型权重**： ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main`` 与 ``RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora`` 。配置通过 ``merged_loras`` （ ``scale: 0.5`` ）合并一半强度的 ManiSkill LoRA 到基础权重，再在其上训练新的 LoRA。

1. 在同一个虚拟环境中安装两个仿真器：

   .. code-block:: bash

      bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

2. 下载权重（中国大陆用户可先设置 ``HF_ENDPOINT=https://hf-mirror.com``）：

   .. code-block:: bash

      hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Main
      hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir RLinf-OpenVLAOFT-ManiSkill-Base-Lora

3. 在 ``examples/embodiment/config/maniskill_libero_grpo_openvlaoft.yaml`` 中将以下三个路径改为本地目录：

   - ``actor.model.model_path`` ： ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
   - ``rollout.model.model_path`` ： ``RLinf-OpenVLAOFT-ManiSkill-Base-Main``
   - ``actor.model.merged_loras[0].path`` ： ``RLinf-OpenVLAOFT-ManiSkill-Base-Lora``

4. 启动训练：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh maniskill_libero_grpo_openvlaoft

5. 在终端指标表或 TensorBoard（目录位于 ``logs/<timestamp>-maniskill_libero_grpo_openvlaoft/`` ）中确认同时出现 ``env/libero_10/success_once`` （初始约 0.2）、 ``env/maniskill/success_once`` （初始约 0.09）和汇总的 ``env/success_once`` 。eval 和 rollout 指标采用同样的 ``<group>/<name>/<metric>`` 命名，视频保存在 ``<video_base_dir>/seed_<n>/<name>/`` 。如需查看各环境耗时，可设置 ``runner.per_worker_log: True`` 并查看 ``time/env/interact`` 。

支持范围
--------

- Rollout 后端： ``rollout.rollout_backend: huggingface`` 。
- 分环境 ``unnorm_key`` ：仅支持 OpenVLA-OFT。
- NFT：需配置 ``policy_obs`` 与 ``algorithm.env_weighting: sample`` ，不支持分环境 ``loss_weight`` 。
- 目前仅支持单环境的特性： ``runner.use_training_pipeline`` 、DAgger 在线 LeRobot 采集、DiffusionART、 ``reward.use_reward_model`` 、 ``smooth_intervene`` 、 ``runner.rtc`` ；RLT 仿真 replay 要求全部训练环境为 ``maniskill_rlt`` 。
