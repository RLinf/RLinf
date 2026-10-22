基于 BWM 世界模型的强化学习
=============================

本示例在 Boundless World Model（BWM，基于 Wan2.2-TI2V-5B 的动作条件视频世界模型）中，用 PPO 训练单视角 pi0.5 policy 完成 RoboTwin ``adjust_bottle`` 任务。训练过程不调用仿真器，训练后的 checkpoint 在 RoboTwin 中评测。

概述
----

- **世界模型**：在 RoboTwin ``adjust_bottle`` 成功与失败轨迹上微调过的 BWM，由 ``WorldModelEnv`` 的 ``bwm`` 后端在进程内推理。
- **Policy**：单视角 pi0.5 SFT checkpoint（``pi05_aloha_robotwin``，RoboTwin 关节约定）。
- **观测**：世界模型生成的一帧 640x480 头部相机图像；``states`` 为 policy 上一条关节指令（``state_from_last_action``）。
- **动作**：每次 50 条 Aloha 关节指令（14 维）。后端用 URDF 正运动学把它们转换为 BWM 使用的归一化末端位姿；每个 chunk 对应一次 61 帧的 BWM 窗口（9 帧条件、52 帧新生成）。
- **奖励**：冻结的 T5 成功分类器从第 51 帧起逐帧打分；若峰值达到阈值，在第 200 步给 1（``reward_mode: episode_window_max``），到时限时 episode 以终止结束（``terminate_at_max_episode_steps``）。
- **算法**：PPO + GAE（gamma 1.0），每步 48 个环境 x 2 个 rollout epoch，共 14 步：前 4 步只训练 critic，之后 10 步联合更新。critic 读取 VLM 特征（``value_after_vlm``）。

安装
----

1. 按 :doc:`RoboTwin <robotwin>` 安装 openpi 模型与 RoboTwin 环境。仿真器只用于评测和提供动作桥所需的 URDF。

2. BWM 仓库无法用 pip 安装，把源码加入 ``PYTHONPATH``：

   .. code-block:: bash

      git clone https://github.com/boundless-large-model/boundless-world-model
      git -C boundless-world-model checkout cc0bc74
      export PYTHONPATH=$(pwd)/boundless-world-model:$PYTHONPATH

3. BWM 依赖 ``diffsynth==2.0.11``，其 API 与 ``wan`` 后端安装的 DiffSynth 分支不同。把它安装到单独目录，只在运行 BWM 时放在 ``PYTHONPATH`` 最前面：

   .. code-block:: bash

      pip install --no-deps --target /path/to/bwm_pydeps diffsynth==2.0.11 ftfy==6.3.1
      export PYTHONPATH=/path/to/bwm_pydeps:$PYTHONPATH

   另需与已安装 ``torch`` 版本匹配的 ``torchaudio``。

资产
----

.. list-table::
   :header-rows: 1

   * - 资产
     - 来源
     - 配置项
   * - Wan2.2-TI2V-5B DiT 与 VAE
     - ``Wan-AI/Wan2.2-TI2V-5B``
     - ``env.train.model_path``、``env.train.VAE_path``
   * - 微调后的 BWM 权重
     - ``lidavidsh/bwm-robotwin-adjust-bottle-failure-ft``（``step-750.safetensors``）
     - ``env.train.checkpoint_path``
   * - BWM 动作统计
     - BWM 仓库中的 ``demo/stat.json``
     - ``env.train.action_bridge.stat_path``
   * - Aloha URDF / SRDF
     - RoboTwin 资产 ``embodiments/aloha-agilex``
     - ``env.train.action_bridge.urdf_path``、``srdf_path``
   * - 奖励分类器
     - ``lidavidsh/robotwin-adjust-bottle-t5-success-classifier``（``reward_model.pt``）与 ``google-t5/t5-base``
     - ``env.train.reward_model.from_pretrained``、``t5_model_name``
   * - pi0.5 SFT policy
     - `WorldArena/WorldArena <https://huggingface.co/WorldArena/WorldArena>`__ （model 仓库，``fulldata``，归一化统计在 ``assets/robotwin_clean``）
     - ``actor.model.model_path``、``actor.model.openpi_data.norm_stats_path``
   * - 初始帧
     - 48 张 RoboTwin 复位观测（640x480），每张一个 ``episode_XXXXX.npy``；暂未提供导出脚本
     - ``env.train.initial_image_path``

BWM 微调所用的轨迹数据在 ``lidavidsh/robotwin-adjust-bottle-pi05-rollouts``。

运行
----

在 ``examples/embodiment/config/env/bwm_robotwin_adjust_bottle.yaml`` 与 ``bwm_robotwin_adjust_bottle_ppo_openpi_pi05.yaml`` 中填好上述路径，然后：

.. code-block:: bash

   export ROBOT_PLATFORM=ALOHA ROBOTWIN_PATH=/path/to/RoboTwin
   bash examples/embodiment/run_embodiment.sh bwm_robotwin_adjust_bottle_ppo_openpi_pi05

本示例使用 FSDP ``no_shard``（openpi 的要求）。在 Triton 缺少 ``triton_key`` 的 ROCm 环境中，加上 ``actor.model.openpi.torch_compile=False``。

在 RoboTwin 中评测
------------------

用 RoboTwin pi0.5 评测配置评测 checkpoint，并使用本示例的 policy 设置与单头部视角：

.. code-block:: bash

   export ROBOT_PLATFORM=ALOHA ROBOTWIN_PATH=/path/to/RoboTwin
   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_pi05_eval \
     rollout.model.model_path=/path/to/sft runner.ckpt_path=/path/to/full_weights.pt \
     rollout.model.num_steps=4 rollout.model.openpi.task=rl rollout.model.openpi.noise_level=0.5 \
     rollout.model.openpi.num_images_in_input=1 ++rollout.model.use_proprio=true \
     ++rollout.model.openpi.value_after_vlm=True \
     ++rollout.model.openpi.value_head_hidden_sizes=[512,256,128] \
     ++rollout.model.openpi_data.adapt_to_pi=false \
     ++rollout.model.openpi_data.norm_stats_path=/path/to/sft/assets/robotwin_clean \
     env.eval.task_config.camera.head_camera_type=Large_D435 \
     env.eval.task_config.camera.collect_wrist_camera=false

结果
----

使用本后端移植前的原始进程内实现（较早的 RLinf 版本），在 8x MI300X 上训练 14 步 PPO，在 144 个未参与训练的初始状态 x 3 次评测上，相对 SFT policy（0.773）的成功率：

.. list-table::
   :header-rows: 1

   * - Seed
     - 成功率
     - 相对 SFT（90% 置信区间）
   * - 1
     - 0.889
     - +11.6 pp [+8.1, +15.1]
   * - 2
     - 0.884
     - +11.1 pp [+6.7, +15.7]
   * - 3
     - 0.847
     - +7.4 pp [+3.7, +11.3]
   * - 4
     - 0.917
     - +14.4 pp [+10.0, +19.0]
   * - 5
     - 0.813
     - +3.9 pp [+1.6, +6.3]

不同 seed 之间差异较大，应以多个 seed 判断结果。使用移植后的后端训练尚未复现这一提升，详见 `#1713 <https://github.com/RLinf/RLinf/issues/1713>`_。
