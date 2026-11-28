在 Isaac Lab-Arena 中评测 DROID 任务
===============================================

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 安装 RLinf
      :link: ../../start/installation
      :link-type: doc

   .. grid-item-card:: 准备 Arena 环境和 DROID 权重
      :link: ../../examples/embodied/isaaclab_arena
      :link-type: doc

安装与运行
----------

沿用 Arena 示例的终端。如果另开终端，请重新激活 RLinf 环境，并导出 ``MODEL_DIR`` 和 Isaac Sim 许可变量。使用外部 Arena 环境时，还需导出 ``ISAACLAB_ARENA_PYTHON``，指向安装时使用的解释器绝对路径。

将 ``env.eval.init_params.environment`` 设为 Arena 已注册的环境名或场景图 YAML。场景图描述场景、机器人和任务的组合。内置场景图的路径相对于 Arena 仓库根目录，自定义场景图使用绝对路径。机器人需选用 ``droid_abs_joint_pos``。

.. tab-set::

   .. tab-item:: 已注册任务

      下面把默认的香蕉任务切换到已注册的枫木桌抓放任务，并指定要抓取的物体：

      .. code-block:: bash

         bash evaluations/run_eval.sh isaaclab_arena_pi05_eval \
             rollout.model.model_path="$MODEL_DIR" \
             env.eval.init_params.environment=pick_and_place_maple_table \
             +env.eval.init_params.environment_args.pick_up_object=banana_ycb_robolab

      默认使用任务自带的指令，也可以通过 ``env.eval.init_params.language_instruction`` 指定提示词。

   .. tab-item:: 短回合

      用一个环境验证默认的香蕉任务。policy 每次预测 15 个动作，每轮预留 30 步。回合未提前结束时会在第 16 步超时。

      .. code-block:: bash

         bash evaluations/run_eval.sh isaaclab_arena_pi05_eval \
             rollout.model.model_path="$MODEL_DIR" \
             env.eval.total_num_envs=1 env.eval.max_episode_steps=16 \
             env.eval.max_steps_per_rollout_epoch=30

每轮评测中，每个环境只计入一个已结束的回合。物体掉落和超时都按未成功统计，``eval/episode_len`` 记录从重置到首次结束的实际步数。设置 ``env.eval.video_cfg.save_video=true`` 可保存相机录像。

香蕉任务的上限是 40 秒，按 15 Hz 计算为 600 步。若要缩短试跑时间，可按上例同时调整两个步数设置。每轮评测的步数必须覆盖回合上限，且能被动作块长度整除。下一轮评测使用新的放置种子重建场景，上一轮提前结束的时间不会改变下一轮的初始布局。

.. _isaaclab-arena-single-action-ppo:

用单动作 PPO 训练
-----------------

将 ``examples/embodiment/config/polaris_tapeintocontainer_ppo_openpi_pi05.yaml`` 复制为同目录下的 ``arena_droid_ppo.yaml``。

在副本的 ``defaults`` 中，将两处 ``env/polaris_droid_tapeintocontainer`` 替换为 ``env/isaaclab_arena``，保留 ``@env.train`` 和 ``@env.eval`` 后缀。将下面的字段合并到对应配置段。

.. dropdown:: 单动作 PPO 配置
   :open:

   .. code-block:: yaml

      env:
        train:
          auto_reset: true
          ignore_terminations: false
          max_episode_steps: 600
          max_steps_per_rollout_epoch: 600
        eval:
          auto_reset: false
          ignore_terminations: false
          max_episode_steps: 600
          max_steps_per_rollout_epoch: 600
      actor:
        model:
          model_path: ${oc.env:MODEL_DIR}
          action_dim: 8
          num_action_chunks: 1
          openpi:
            task: rl
            pytorch_numerics: true
            config_name: pi05_droid_robolab

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh arena_droid_ppo

训练时每次观测只执行一个动作，超时截断会返回重置前的观测，用于估计状态价值。独立评测支持 15 步动作块。每个环境独立采样初始布局，请保留 ``env.train.group_size=1``，并在训练和评测环境中都关闭 ``use_fixed_reset_state_ids`` 和 ``use_ordered_reset_state_ids``。分组 GRPO 所需的共享布局目前尚未实现。

要复现初始布局的采样顺序，请在启动训练前设置 ``env.train.seed``。训练环境每次调用 ``reset()`` 都会取下一个布局。只有评测时才能在 ``reset(seed=...)`` 中指定种子。Arena 训练和评测都使用 ``data_collection.enabled=false``。
