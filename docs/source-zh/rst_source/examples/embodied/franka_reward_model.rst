Franka 奖励模型
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_reward_model.jpg
   :align: center
   :width: 80%

   Franka 奖励模型闭环：采集带标注的真机数据，训练奖励模型，并在真机 RL 中由其提供奖励信号。

概览
----

在 Franka 真机异步 RLPD 训练中，用学习得到的奖励模型替代手写成功规则。本页支持两种模型：用于视觉上明确成功状态的 ResNet 逐帧分类器，以及提供 5 帧滑动窗口稠密进度引导的 Qwen3-VL-4B（LoRA）动作趋势判别模型；两者的数据标注格式不同，采集前先选定其一。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 机器人
      :text-align: center

      Franka Emika Panda（Franky）

   .. grid-item-card:: 传感器与遥操作
      :text-align: center

      单 / 双 RealSense · SpaceMouse · 键盘

   .. grid-item-card:: 流程
      :text-align: center

      数据采集与标注 → 训练奖励模型 → 在线 RLPD

   .. grid-item-card:: 硬件
      :text-align: center

      控制节点 + NVIDIA GPU 节点

奖励模型对比
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 32 18 32

   * - 类型
     - 奖励信号
     - 模型
     - 适用场景
   * - :ref:`ResNet <franka-resnet-reward-model>`
     - 逐帧成功/失败概率，同时直接驱动环境复位。
     - ResNet 图像分类器
     - 任务成功状态在视觉上清晰明确，且便于标注成功/失败帧。
   * - :ref:`Qwen VLM <franka-vlm-reward-model>`
     - 5 帧滑动窗口的运动趋势得分，外加到达目标时的成功奖励。
     - Qwen3-VL-4B（LoRA）
     - 仅靠稀疏成功信号不足，需要朝目标的稠密进度引导。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - policy 与奖励模型共用相机帧（ResNet 用腕部视角；Qwen VLM 用腕部 + 全局双视角）。
   * - 动作
     - 与基础 Franka 真机环境一致的 6 维笛卡尔动作。
   * - 奖励
     - 奖励模型输出（ResNet 模式下独立触发复位；Qwen VLM 模式下为趋势得分加 ``gt_success_bonus``）。
   * - 提示词
     - 按配置使用任务描述文本（VLM 趋势输入）或固定目标位姿。

硬件与安全准备
--------------

先按 :doc:`franka` 完成 Franka 主机准备（固件检查、实时内核、FCI 激活与 Ray 集群），并阅读 :doc:`../../extending/reward_model` 了解奖励模型通用流程。ResNet 方案需要单台 RealSense 相机与 SpaceMouse（方式一还需实体键盘）；Qwen VLM 方案需要双 RealSense 相机（``wrist_1`` 腕部近景与 ``global`` 全局视角）以及可运行 Qwen3-VL-4B LoRA 微调与 bf16 推理的 NVIDIA GPU（如 RTX 4090）。

.. warning::

   每次数据采集与真机 RL 均须由操作员全程监控并保持急停按钮触手可及。连接前清空工作区域并核对机械臂 IP、相机序列号与目标位姿。ResNet 模式开启 ``standalone_realworld: True`` 后由奖励模型直接触发环境复位，启动自主 RL 前请先离线评估精度，或按 :doc:`../../extending/reward_model` 用真机遥操作做在线推理验证。

.. _franka-resnet-reward-model:
.. _franka-vlm-reward-model:

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: ResNet 奖励模型

      **1. 安装依赖**

      在控制机与 GPU 节点安装 Franka 环境（主机准备见 :doc:`franka`）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. 采集专家演示与奖励标注数据**

      先按 :doc:`franka` 采集 RLPD demo buffer 所需的专家轨迹，确认 ``examples/embodiment/config/realworld_collect_data.yaml`` 中 ``env.eval.data_collection`` 已开启且 ``export_format: [replay_buffer, pickle_episode]``、``only_success: True``。

      接着任选以下一种方式采集奖励模型数据集（完整参数见 :doc:`../../extending/reward_model`）：

      - **方式一：键盘标注（通用任务）**：遥操作过程中按 ``c``\ （成功）和 ``a``\ （失败）逐帧标注；达到目标帧数后自动停止并直接导出 ``train.pt`` 与 ``val.pt``：

        .. code-block:: bash

           bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

      - **方式二：固定位姿标注（目标驱动）**：数据采集时将 ``env.eval.override_cfg.success_hold_steps`` 设为 ``20``\ （缓慢移动获取多样失败帧，到达目标位姿后小幅微调获取多样成功帧），再将 ``.pkl`` 轨迹预处理为 ``train.pt`` 与 ``val.pt``：

        .. code-block:: bash

           python -m examples.reward.preprocess_reward_dataset \
               --raw-data-path logs/xxx/pickle_episode \
               --output-dir logs/xxx/processed_reward_data \
               --fail-success-ratio 3

      **3. 训练 ResNet 奖励模型**

      按 :doc:`../../extending/reward_model` 使用 ``examples/reward/run_reward_training.sh`` 训练分类器（真机数据建议将 ``runner.early_stop.min_delta`` 调低至 ``1e-6``）。训练完成后也可按该文档先在 SpaceMouse 遥操作下做在线推理验证。

      **4. 启动真机 RL**

      在 ``examples/embodiment/config/realworld_charger_sac_cnn_async_standalone_reward.yaml`` 中按 :doc:`franka` 配好集群与机械臂参数，将 ``reward.model.model_path`` 指向训练好的 ResNet 权重，在 Ray head 节点启动训练：

      .. code-block:: yaml

         reward:
           use_reward_model: True
           standalone_realworld: True
           reward_threshold: 0.8
           model:
             model_path: /path/to/reward_model_checkpoint
             model_type: "resnet"

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_charger_sac_cnn_async_standalone_reward

      .. dropdown:: 进阶配置（可选）

         **专家数据导出与早停设置**

         .. code-block:: yaml

            # 在 realworld_collect_data.yaml 中
            env:
              eval:
                data_collection:
                  enabled: True
                  save_dir: ${runner.logger.log_path}
                  export_format: [replay_buffer, pickle_episode]
                  only_success: True

            # 在奖励模型训练配置中（真机推荐）
            runner:
              early_stop:
                min_delta: 1e-6

         **键盘标注关键配置（``realworld_collect_dataset``）**

         - ``runner.num_success_frames`` / ``runner.num_fail_frames``：成功与失败目标帧数，同时达标后停止采集。
         - ``runner.val_split``：验证集划分比例；``runner.fail_success_ratio``：训练集失败帧下采样比例。
         - ``env.eval.keyboard_reward_wrapper: single_stage``：开启 ``c`` / ``a`` 标注；``env.eval.teleop`` 指定遥操作设备；``env.eval.override_cfg.target_ee_pose`` 为任务目标位姿。

         **独立真机模式（``standalone_realworld``）行为**

         开启 ``standalone_realworld: True`` 后，奖励模型经 ``reward_threshold`` 过滤后的成功/失败判定会直接替换环境原生奖励并触发复位，低于阈值的概率置为 ``0``，加权求和相关参数（``reward_mode``、``reward_weight``、``env_reward_weight``）不生效。

   .. tab-item:: Qwen VLM 奖励模型

      Qwen VLM 奖励模型用 Qwen3-VL-4B（LoRA）观察 5 帧双视角滑动窗口，将动作趋势判别映射为标量奖励：``positive`` → ``1.0``、``negative`` → ``-0.2``、``unclear`` → ``0.0``、``invalid`` → ``0.0``。机械臂到达目标并保持后记录 ``success_once``，额外追加 ``gt_success_bonus``\ （``+20.0``）。

      **1. 安装依赖**

      在控制机安装 Franka 环境（GPU 节点的 Qwen3-VL 微调依赖见 :doc:`sft_vlm`）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. 采集双视角轨迹**

      在 ``examples/embodiment/config/realworld_collect_data.yaml`` 中开启 ``pickle_episode`` 导出并设 ``only_success: False``，关闭相对坐标系（``use_relative_frame: False``，让 TCP 坐标保持绝对基坐标系），并映射全局与腕部相机序列号：

      .. code-block:: yaml

         env:
           eval:
             use_relative_frame: False
             override_cfg:
               camera_names:
                 "SERIAL1": "global"
                 "SERIAL2": "wrist_1"
             data_collection:
               enabled: True
               save_dir: /path/to/run
               export_format: pickle_episode
               only_success: False

      缓慢移动机械臂，采集 50 条以上覆盖成功与失败结局的轨迹（确保 ``wrist_1`` 与 ``global`` 视角均能清晰看到末端与目标）：

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh realworld_collect_data

      **3. 预处理并微调 Qwen3-VL-4B（LoRA）**

      将原始 ``.pkl`` 切分为 5 帧双视角窗口（生成 ``dataset_info.json`` 以及含 ``segments.jsonl`` 和 ``pkl/`` 的 ``train/``、``eval/`` 目录）。若原始数据不含 GAE 信号，使用第二条命令传入 ``--target-ee-pose``\ （单位为米，仅需位置时可只传 ``X,Y,Z``）：

      .. code-block:: bash

         # 基于 GAE 或 reward 自动标注
         python -m examples.reward.vlm_trend.preprocess_reward_dataset \
             --raw-data-path /path/to/run/pickle_episode \
             --output-dir /path/to/processed_vlm_trend_reward_data \
             --window-size 5 \
             --task-description "Pick up the peg and insert it into the hole."

         # 或基于到目标位姿的 TCP 距离自动标注（--delta-threshold 单位为米）
         python -m examples.reward.vlm_trend.preprocess_reward_dataset \
             --raw-data-path /path/to/run/pickle_episode \
             --output-dir /path/to/processed_vlm_trend_reward_data \
             --window-size 5 \
             --target-ee-pose "X,Y,Z,RX,RY,RZ" \
             --delta-threshold 0.005

      在 ``examples/sft/config/qwen3vl_sft_vlm_trend_reward.yaml`` 中将 ``actor.model.model_path`` 指向本地 ``Qwen3-VL-4B-Instruct`` 目录，导出 ``VLM_TREND_REWARD_DATA_ROOT`` 并启动 LoRA 微调：

      .. code-block:: bash

         export VLM_TREND_REWARD_DATA_ROOT=/path/to/processed_vlm_trend_reward_data
         bash examples/sft/run_vlm_sft.sh qwen3vl_sft_vlm_trend_reward

      记录训练产出的 LoRA checkpoint 路径（例如 ``checkpoints/global_step_3000``）。

      **4. 启动真机 RL**

      在 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_vlm_reward.yaml`` 中填写 ``reward.model.model_path``\ （基础 Qwen3-VL-4B）、``reward.model.lora_path``\ （LoRA 权重）以及集群配置，然后在 Ray head 节点启动：

      .. code-block:: yaml

         reward:
           model:
             model_path: "/path/to/Qwen3-VL-4B-Instruct"
             lora_path: "/path/to/qwen3-vl-lora-checkpoint"

      .. code-block:: bash

         bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async_vlm_reward

      .. dropdown:: 进阶配置（可选）

         **SFT 配置（``examples/sft/config/qwen3vl_sft_vlm_trend_reward.yaml``）**

         仓库配置默认关闭 ``gradient_checkpointing: false`` 并采用 ``sharding_strategy: no_shard``\ （适配单张 RTX 4090）；如需节省显存，可将 ``gradient_checkpointing`` 改为 ``true``：

         .. code-block:: yaml

            data:
              type: vlm
              dataset_name: "vlm_trend_reward_sft"
              train_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/train/segments.jsonl"
              val_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/eval/segments.jsonl"
              video_root: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}"

            actor:
              model:
                model_type: qwen3_vl
                model_path: /path/to/Qwen3-VL-4B-Instruct
                is_lora: true
                lora_rank: 16
                attn_implementation: flash_attention_2
              fsdp_config:
                gradient_checkpointing: false
                sharding_strategy: no_shard

         **RL 奖励配置（``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_vlm_reward.yaml``）**

         .. code-block:: yaml

            reward:
              use_reward_model: true
              worker_type: model
              group_name: "RewardGroup"
              standalone_realworld: False
              reward_mode: history_buffer
              history_reward_assign: true
              reward_weight: 1.0
              env_reward_weight: 0.0
              reward_threshold: 0.5
              model:
                model_path: "/path/to/Qwen3-VL-4B-Instruct"
                model_type: "buffered_vlm"
                lora_path: "/path/to/qwen3-vl-lora-checkpoint"
                gt_success_bonus: 20.0
                precision: "bf16"
                input_builder_name: vlm_trend_reward_input_builder
                input_builder_params:
                  default_task_description: "Pick up the peg and insert it into the hole."
                reward_parser_name: vlm_trend_reward_parser
                reward_parser_params:
                  positive_reward: 1.0
                  negative_reward: -0.2
                  unclear_reward: 0.0
                  invalid_reward: 0.0
                history_buffers:
                  history_window:
                    history_size: 5
                    min_history_size: 5
                    input_interval: 1
                    history_keys:
                      - main_images
                      - extra_view_images
                    input_on_done: false
                interval_reward: 0.0
                max_new_tokens: 16
                do_sample: false
                temperature: 0.0

            cluster:
              num_nodes: 2
              component_placement:
                actor:
                  node_group: "4090"
                env:
                  node_group: franka
                reward:
                  node_group: "4090"

         **核心字段与奖励计算说明**

         - ``reward_mode: history_buffer`` 按环境缓存最近 5 帧双视角图像（``main_images`` 与 ``extra_view_images``），凑满 ``min_history_size`` 帧后在本地 reward worker 中推理（``worker_type: model``），历史不足时返回 ``0.0``。
         - ``history_reward_assign: true`` 将 VLM 解析出的趋势奖励反向赋给窗口内每一步。
         - ``env_reward_weight: 0.0`` 使训练主要依赖 VLM 趋势引导，到达目标位姿触发 ``success_once`` 后叠加 ``gt_success_bonus: 20.0``。

结果与常见问题
--------------

在 TensorBoard（``tensorboard --logdir ./logs``）中查看奖励分布、VLM 趋势输出和 ``env/success_once``；日志配置见 :doc:`训练指标 <../../reference/metrics>`。

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 现象
     - 解决方法
   * - ResNet 模式下 ``reward_weight`` 或 ``reward_mode`` 不生效
     - 开启 ``standalone_realworld: True`` 后由奖励模型直接替换原生环境奖励并触发复位，属于预期行为。
   * - ResNet 奖励模型在真机数据上过早触发早停
     - 在奖励模型训练配置中将 ``runner.early_stop.min_delta`` 降至 ``1e-6``。
   * - 无 GAE 信号时 VLM 趋势预处理报错或标签全为中性
     - 传入 ``--target-ee-pose "X,Y,Z,RX,RY,RZ"`` 并调整 ``--delta-threshold``\ （单位米，可从 ``0.005`` 起步），且确认数据采集时设置了 ``use_relative_frame: False``。
   * - Qwen3-VL-4B LoRA 微调显存不足
     - 在 ``qwen3vl_sft_vlm_trend_reward.yaml`` 中开启 ``actor.fsdp_config.gradient_checkpointing: true``。
