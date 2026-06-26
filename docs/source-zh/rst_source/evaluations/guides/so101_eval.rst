SO101 真机评测
==============

在 SO101 六自由度机械臂上评测经过 SFT 的 π₀ 检查点。rollout Worker 对导出的检查点执行推理，
env Worker 驱动真实机械臂，\ :class:`~rlinf.runners.embodied_eval_runner.EmbodiedEvalRunner`
收集每 episode 的成功指标（回报、成功率、episode 长度）。

相关文档：\ :doc:`../../examples/embodied/so101`\ （数据采集）、\ :doc:`../../examples/embodied/so101_sft_openpi`\ （SFT 配方）、\ :doc:`../../examples/embodied/sft_openpi`\ （通用 OpenPI SFT）。

环境准备
--------

**硬件**

- SO101 六自由度机械臂（已组装并校准），通过 USB 连接（\ ``/dev/ttyACM0`` / ``/dev/ttyACM1``\ ）。
- 摄像头（可选——参见 :doc:`../../examples/embodied/so101`\ ；未配置时自动生成空白 ``camera_0``）。
- 带有 SFT 后检查点的 GPU 主机。可以是与机械臂同一台机器（单节点），也可以单独一台 GPU 节点。

**依赖安装**

单节点（GPU + 机械臂在同一台机器上）：

.. code-block:: bash

   bash requirements/install.sh embodied --env so101
   source .venv/bin/activate

双节点（GPU 在一个节点，机械臂在另一个节点——GPU 节点只需要 OpenPI）：

.. code-block:: bash

   # GPU / rollout 节点
   bash requirements/install.sh embodied --model openpi --env so101
   source .venv/bin/activate

   # 机器人控制节点
   bash requirements/install.sh embodied --env so101
   source .venv/bin/activate

**节点拓扑**

- **单节点**\ ：\ ``num_nodes: 1``\ ——\ ``env`` 和 ``rollout`` 都在连接了 SO101 机械臂的主机上运行。
  ``realworld_so101_eval.yaml`` 默认采用这种模式。
- **双节点**\ ：\ ``num_nodes: 2``\ ——\ ``rollout`` 在 rank 0（GPU）、\ ``env`` 在 rank 1（机器人）。
  请相应调整 ``component_placement`` 和 ``node_groups``\ （参考 ``realworld_pnp_eval.yaml``\ ）。

示例配置
--------

评测配置文件位于 ``evaluations/realworld/realworld_so101_eval.yaml``\ 。
它使用 ``pi0_so101`` 数据配置（\ ``action_dim=6``\ 、\ ``num_action_chunks=10``\ 、
两个图像槽位），并在每次重置时启用随机扰动以提升鲁棒性。

首次运行的关键字段：

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 字段
     - 位置
     - 说明
   * - ``port``
     - ``cluster.node_groups[].hardware.configs``
     - Follower 臂 USB 端口（如 ``/dev/ttyACM0``\ ）
   * - ``target_ee_pose``
     - ``env.eval.override_cfg``
     - 任务末端目标 ``(x, y, z)``\ ，单位为米（通过 Hydra override 设置）
   * - ``ckpt_path``
     - ``runner``
     - SFT 导出的 ``full_weights.pt`` 路径
   * - ``model_path``
     - ``rollout.model``
     - π₀ 基座模型目录（需包含 ``so101_data/norm_stats.json``\ ）
   * - ``config_name``
     - ``rollout.model.openpi``
     - 必须为 ``"pi0_so101"``

运行前检查
----------

1. **机械臂连接**\ （在机器人主机上）：

   .. code-block:: bash

      ls /dev/ttyACM*   # 应显示两个设备（leader + follower）
      # 确认 follower 端口与 YAML 配置一致。

2. **校准**\ ：评测前必须已用 lerobot 校准工具完成校准。
   在 ``override_cfg`` 中设置 ``auto_calibrate: False``\ 。

3. **末端目标位姿**\ ：将机械臂移动到期望的任务目标位置，然后运行
   :ref:`compute_ee_pose`\ ：

   .. code-block:: bash

      python toolkits/lerobot/compute_ee_pose.py --port /dev/ttyACM0

   将输出的 ``(x, y, z)`` 填入 ``target_ee_pose`` override。

4. **Ray 集群**\ ：

   .. code-block:: bash

      ray status  # 应至少显示一个节点（双节点模式显示两个）

5. **Dummy 模式（可选）**\ ：在 ``env.eval.override_cfg`` 中设置
   ``is_dummy: True``\ ，在不驱动机械臂的情况下验证集群连接和配置解析。

.. warning::

   启动评测前请确认机械臂工作区域畅通无阻、线缆未缠绕。首次运行建议使用较小的
   ``env.eval.rollout_epoch``\ （如 2），并将急停按钮置于可触及范围内。

启动 Ray 集群
-------------

单节点：

.. code-block:: bash

   ray start --head --port=6379 --node-ip-address=127.0.0.1

双节点（参照 ``realworld_pnp_eval.yaml``\ ）：

.. code-block:: bash

   # GPU 节点（rank 0，head）
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

   # SO101 节点（rank 1）
   export RLINF_NODE_RANK=1
   ray start --address=<head_ip>:6379

端到端流程
----------

**第 1 步：导出检查点**

SFT 训练完成后（参见 :doc:`../../examples/embodied/so101_sft_openpi`\ ），
找到导出的 ``full_weights.pt``\ ：

.. code-block:: text

   so101_sft_openpi/
   └── checkpoints/
       └── global_step_N/
           └── actor/
               └── model_state_dict/
                   └── full_weights.pt

**第 2 步：确定末端目标位姿**

在机器人主机上运行 ``compute_ee_pose.py`` 记录目标位姿。

**第 3 步：启动评测**

.. code-block:: bash

   # 使用启动器
   bash evaluations/run_eval.sh realworld realworld_so101_eval \
       runner.ckpt_path=/path/to/full_weights.pt \
       rollout.model.model_path=/path/to/pi0_base_so101 \
       cluster.node_groups.0.hardware.configs.0.port=/dev/ttyACM0 \
       env.eval.override_cfg.target_ee_pose=[0.35,0.0,0.15]

   # 快速冒烟测试（2 个 episode）
   bash evaluations/run_eval.sh realworld realworld_so101_eval \
       env.eval.rollout_epoch=2 \
       ...

**第 4 步：查看结果**

每 episode 的指标会输出到终端（\ ``eval/success_once``\ 、\ ``eval/return``\ 、
``eval/episode_len``\ ）。日志位于
``logs/<timestamp>-realworld_so101_eval/eval_embodiment.log``\ 。

每个 Episode 结束后运行的重置
------------------------------

当 RL runner 检测到机械臂已终止（奖励 ≥ 1.0 或超时），会在下一个 episode 开始前触发自动重置：

#. env Worker 调用 ``RealWorldEnv._handle_auto_reset()``\ ，进而对完成的 env 调用
   ``SO101PickEnv.reset()``\ 。
#. ``reset()`` 为每个机械臂关节添加小幅均匀扰动（eval 配置启用了
   ``enable_random_reset: True`` 和 ``random_joint_noise_deg: 5.0``\ ，受限于关节限位）。
#. ``go_to_rest()`` 通过 ``robot.send_action()`` 发送加噪后的重置位姿；
   机械臂移动到该位置并保持。
#. ``_update_state()`` 读取最新的关节位置和夹爪状态。

这种随机扰动确保每个评测 episode 的起始位置 **略有不同**\ ，从而测试策略的鲁棒性，
而非让策略从单一起点位置过拟合。重置发生在关节空间（而非 EE 空间），
机械臂返回到 ``reset_joint_qpos`` 附近——与数据采集时使用的相同复位位姿。

评测配置参考
------------

必需的 ``env.eval`` 字段：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - ``rollout_epoch``
     - 评测 episode 数量；首次运行建议 2–5
   * - ``max_episode_steps``
     - 每个 episode 的最大步数；默认 200
   * - ``max_steps_per_rollout_epoch``
     - 每个 rollout 轮次的步数；**必须能被** ``num_action_chunks``\ （10）整除
   * - ``override_cfg.is_dummy``
     - 真机时必须为 ``False``
   * - ``override_cfg.auto_calibrate``
     - 必须为 ``False``\ （机械臂已预先校准）
   * - ``override_cfg.target_ee_pose``
     - EE 目标 ``[x, y, z]``\ ，单位为米
   * - ``override_cfg.reward_threshold_m``
     - 成功半径；默认 0.03（3 cm）
   * - ``override_cfg.enable_random_reset``
     - ``True`` → 带噪声的起始位置（推荐开启）

关键 ``rollout`` 字段：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - ``model.model_path``
     - π₀ 基座模型目录；需包含 ``so101_data/norm_stats.json``
   * - ``model.action_dim``
     - SO101 必须为 6
   * - ``model.num_action_chunks``
     - 必须为 10（与 SFT 保持一致）
   * - ``model.openpi.config_name``
     - 必须为 ``"pi0_so101"``
   * - ``model.openpi.num_images_in_input``
     - 设为 2（前置 + 额外视角摄像头）

常见问题
--------

- **"No cameras configured" 警告**\ ：SO101 评测在 ``camera_cfgs: {}`` 时会生成空白占位帧，
  不影响关节空间评测；要进行视觉条件评测，请在 YAML 中添加摄像头配置。
- **机械臂第一步就不动**\ ：检查 ``override_cfg`` 中 ``is_dummy: False``\ 以及端口配置。
- **奖励始终为 0.0**\ ：检查是否设置了 ``target_ee_pose``\ ，并且 ``reward_threshold_m``
  足够大（可以先宽松一些，例如 0.05 m）。
- **步数错误**\ ：确保 ``max_steps_per_rollout_epoch`` 是
  ``rollout.model.num_action_chunks``\ （10）的整数倍。
- **检查点无法加载**\ ：SFT 导出的 ``full_weights.pt`` 必须来自使用
  ``model.openpi.config_name: pi0_so101`` 的训练运行。
- **Ray 节点数不正确**\ ：检查防火墙和多网卡主机上的 ``RLINF_COMM_NET_DEVICES`` 设置。
