在 Franka 上使用 HG-DAgger
================================================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/hg-dagger.jpg
   :align: center
   :width: 80%

   用于采集干预数据并在线训练 Franka 策略的 Human-Gated DAgger 流程。

使用 Human-Gated DAgger 训练 Franka 真机策略。你将采集干预数据，计算 OpenPI 归一化统计，运行 SFT，然后启动在线 HG-DAgger。在线阶段使用 LeRobot 归档完整的成功 episode：未接管帧保留策略实际执行动作，接管帧保存人工动作和 ``intervene_flag``。启用 ``only_save_expert: True`` 后，训练只采样非 padding 帧全部由人工接管的 action chunk。

概览
----------------------------------------

用人工门控干预在线提升 Franka 真机策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI π₀ / π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      SFT · HG-DAgger

   .. grid-item-card:: 任务
      :text-align: center

      Real-world PnP

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · PICO

| **你将完成:** 采集干预数据 → 计算 norm stats → 运行 SFT → 启动 HG-DAgger → 监控干预.
| **前置条件:** :doc:`franka` · :doc:`franka_vr` · :doc:`sft_openpi` · Ray cluster · trained or base OpenPI checkpoint.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Collection
     - ``realworld_collect_data_pico``
     - 使用 PICO 采集真机示教。
   * - SFT
     - ``realworld_bin_relocation_sft_openpi``
     - 训练 student 初始化。
   * - HG-DAgger
     - ``realworld_pnp_dagger_openpi``
     - 使用 online LeRobot 归档完整成功轨迹，并仅用全接管 action chunk 训练。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - Franka 相机帧与可选机器人状态。
   * - Action
     - OpenPI action 解码为 Franka 真机控制。
   * - Reward
     - 人工门控干预信号与任务结果。
   * - Prompt
     - OpenPI 数据集/配置 metadata 中的任务文本。

安装
----------------------------------------

真实世界流程在机器人节点和 GPU 训练 / rollout 节点上分别使用不同环境（或从 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 拉取预构建镜像）。中国大陆用户可在 ``requirements/install.sh`` 末尾追加 ``--use-mirror``，并在下载 Hugging Face 权重前设置 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: 机器人 / 环境节点

      按 :doc:`franka` 安装基于 Franky 的 ``franka`` 环境（``LIBFRANKA_VERSION`` 须与固件版本匹配，或使用 Franka Docker 镜像中的 ``franky`` venv）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export LIBFRANKA_VERSION=0.19.0
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

   .. tab-item:: 训练与 Rollout 节点

      在 GPU 节点安装与模拟器 Pi0 DAgger（见 :doc:`dagger`）相同的 OpenPI 环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

集群设置
----------------------------------------

在启动采集或训练任务之前，先完成 :doc:`franka` 中介绍的 Ray 集群配置。通常训练 / rollout 节点作为 Ray head（``RLINF_NODE_RANK=0``），Franka 控制节点作为 worker（``RLINF_NODE_RANK=1``）。将 ``HEAD_NODE_IP`` 替换为 head 节点的实际 IP：

.. code-block:: bash

   # 在训练 / rollout 节点
   export RLINF_NODE_RANK=0
   export HEAD_NODE_IP=192.168.1.10
   ray start --head --port=6379 --node-ip-address="$HEAD_NODE_IP"

   # 在机器人 / env 节点
   export RLINF_NODE_RANK=1
   export HEAD_NODE_IP=192.168.1.10
   ray start --address="${HEAD_NODE_IP}:6379"

Ray 会在启动时记录当前 Python 解释器与环境变量，因此务必在 ``ray start`` 之前激活对应环境。

运行
----------------------------------------

1. 采集带人工引导的真实数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

复制 ``examples/embodiment/config/realworld_collect_data_pico.yaml``\ （例如命名为 ``my_realworld_pnp_collect.yaml``），将环境从 peg insertion 切换为 bin relocation，并填写机器人 IP、相机序列号和目标位姿：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 0

   env:
     eval:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

启动采集前，按 :doc:`franka_vr` 启动并检查 PICO 数据流。默认的 ``ipc:///tmp/vr_data.ipc`` 地址要求 publisher 与 env worker 运行在同一台机器；若二者位于不同机器，将 ``env.eval.pico.zmq_addr`` 改为 ``tcp://<publisher_ip>:<port>``。

使用复制后的配置启动采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

遥操作过程中，同一次运行会写出：

- replay-buffer 轨迹到 ``logs/{timestamp}/demo_buffer/``
- LeRobot 数据到 ``logs/{timestamp}/lerobot_dataset/``

关于采集格式，参见 :doc:`../../guides/data_collection`。

.. dropdown:: 进阶配置（可选）

   仓库自带采集配置已设置好 PICO 按键映射、标定和 LeRobot 导出默认项：

   .. code-block:: yaml

      env:
        eval:
          teleop: pico
          pico:
            zmq_addr: "ipc:///tmp/vr_data.ipc"
            hand: "right"
            control_trigger: "grip"
            control_threshold: 0.85
            gripper_close_button: "A"
            gripper_open_button: "B"
            position_scale: 1.0
            rotation_scale: 1.0
            max_stale_s: 0.2
            calibration:
              enabled: True
              required: True
              auto_calibrate_on_start: True
              button: "trigger"
          override_cfg:
            success_hold_steps: 1
          data_collection:
            enabled: True
            save_dir: ${runner.logger.log_path}
            export_format: [replay_buffer, lerobot_dataset]
            only_success: True
            robot_type: "panda"
            fps: 10

2. 计算归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在进行 SFT 或 HG-DAgger 之前，先为采集得到的 LeRobot 数据集计算 OpenPI
归一化统计：

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

这里使用的数据集根目录和数据集 id，需要与后续 SFT 保持一致。更多 OpenPI
数据集说明可参考 :doc:`sft_openpi`。

3. 运行 OpenPI SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

启动前，先修改 ``examples/sft/config/realworld_bin_relocation_sft_openpi.yaml``：

.. code-block:: yaml

   data:
     train_data_paths: "/path/to/realworld-franka-bin-relocation-dataset"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_realworld"

然后执行：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

SFT 导出的 checkpoint 会作为在线阶段的学生模型初始化。更多 OpenPI SFT 细节
可参考 :doc:`sft_openpi`。

4. 在真机上运行异步 HG-DAgger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml`` 中填写 SFT checkpoint、基础模型路径（rollout 会自动沿用 ``actor.model.model_path``）和归一化统计对应的 ``repo_id``：

.. code-block:: yaml

   runner:
     ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi_data:
         repo_id: "realworld_franka_bin_relocation"

同步更新 ``cluster.node_groups``\ （``robot_ip``）和 ``env.{train,eval}.override_cfg``\ （``target_ee_pose``、``camera_serials``），然后在 Ray head 节点启动 HG-DAgger：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

.. dropdown:: 进阶配置（可选）

   仓库自带配置已设置好集群节点分组、在线 LeRobot 归档与 PICO 接管默认项：

   .. code-block:: yaml

      cluster:
        num_nodes: 2
        node_groups:
          - label: "train"
            node_ranks: 0
          - label: franka
            node_ranks: 1
            hardware:
              type: Franka
              configs:
                - robot_ip: ROBOT_IP
                  node_rank: 1

      algorithm:
        dagger:
          only_save_expert: True
          online_lerobot:
            enabled: True
            only_success: True
            robot_type: "panda"
            fps: 10
            finalize_interval: 1
            data_path: ${runner.logger.log_path}/online_lerobot
            rolling_lerobot_window_size: 50000
            min_frames: 1
            lerobot_num_workers: 0

      env:
        train:
          smooth_intervene: True
          teleop: pico
          pico:
            zmq_addr: "ipc:///tmp/vr_data.ipc"
            hand: "right"
            control_trigger: "grip"
            control_threshold: 0.85
            gripper_close_button: "A"
            gripper_open_button: "B"
            position_scale: 1.0
            rotation_scale: 1.0
            max_stale_s: 0.2
            calibration:
              enabled: True
              required: True
              auto_calibrate_on_start: True
              button: "trigger"
          override_cfg:
            target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
            camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
        eval:
          teleop: none
          override_cfg:
            target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
            camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

      actor:
        model:
          openpi:
            config_name: "pi0_realworld"

   ``online_lerobot.enabled: True`` 表示启用在线 LeRobot 数据链路。env worker 按 episode 收集 rollout，并将满足过滤条件的 episode 发送给 actor；actor 将其加入滚动 LeRobot 数据集训练，因此在线训练不再使用 trajectory replay buffer。

   ``smooth_intervene: True`` 会在 PICO 接管持续到 action chunk 最后一帧时绕过下一次策略推理。env worker 使用 dummy chunk 持续驱动遥操 wrapper，并在松开 ``grip`` 或 episode 结束后恢复正常推理。该模式仅支持 PICO：``env.train.teleop`` 必须是 ``pico``，且每个 env worker pipeline stage 只运行一个环境。``env.eval.teleop: none`` 表示评测阶段只运行策略，不启用人工接管。

   ``only_success: True`` 会丢弃失败 episode。``only_save_expert: True`` 仍将完整的成功 episode 保存在 LeRobot 归档中，但在线 sampler 只暴露 action chunk 内所有非 padding 帧均满足 ``intervene_flag=True`` 的起点。``num_action_chunks: 1`` 时，该规则退化为逐帧过滤。每个成功 episode 内：

   * 未接管帧仍保留在物理归档中，但不会作为 expert-only 训练样本暴露；
   * 人工接管帧保存接管设备实际执行的动作，并带有 ``intervene_flag=True``；
   * ``finalize_interval: 1`` 表示每完成一个成功 episode 就立即写出一个 LeRobot shard；
   * ``rolling_lerobot_window_size: 50000`` 表示在线训练只从最近 50,000 个符合专家条件的逻辑 chunk 起点采样，较早的 shard 仍保留在磁盘中。

   真机 DAgger 配置不包含 beta 相关字段，因为没有配置 ``rollout.expert_model``。Beta 只用于模型 expert 和 student 之间的动作混合；真机人工接管由 ``env.train`` 中启用的 PICO intervention wrapper 决定。

可视化与监控
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. 在线 LeRobot 数据写入**

每个成功 episode 会写到本次运行日志目录：

.. code-block:: text

   logs/<timestamp>-realworld_pnp_dagger_openpi/online_lerobot/rank_0/id_0/
   logs/<timestamp>-realworld_pnp_dagger_openpi/online_lerobot/rank_0/id_1/
   ...

失败 episode 不会进入在线 dataset。

**3. 推荐关注的监控指标**

- ``train/dagger/actor_loss``：基于 expert-only action chunk 计算的监督损失。
- ``train/lerobot_dataset/total_episodes``：actor 当前已接收的成功 episode 数量。
- ``train/lerobot_dataset/physical_frames``：已接收的 LeRobot 物理帧数量。
- ``train/lerobot_dataset/logical_samples``：rolling window 内符合专家条件、可训练的 chunk 起点数。
- ``train/lerobot_dataset/num_sub_datasets``：当前加载的 LeRobot shard 数量。
- ``train/actor/lr``：学习率。
- ``train/actor/grad_norm``：梯度范数。
