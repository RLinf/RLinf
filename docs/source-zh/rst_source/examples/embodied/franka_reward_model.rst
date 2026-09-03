  在 Franka 上使用 Reward Model
================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_reward_model.jpg
   :align: center
   :width: 80%

   Franka reward-model 流程：采集带标注的真机数据，训练 reward model，并在真机 RL 中由它产生奖励。

用学习得到的 reward model 替换 Franka 真机上手写的成功信号。
本页记录 **两种** reward model 类型：它们观察的对象、产生的信号、以及信号进入 RL 循环的方式都不同，
数据标注格式也不一样，因此请在开始采集数据之前先确定用哪一种。

.. list-table::
   :header-rows: 1
   :widths: 16 30 18 36

   * - 类型
     - 奖励信号
     - Reward Model
     - 何时选它
   * - :ref:`ResNet <franka-resnet-reward-model>`
     - 逐帧的成功/失败概率，同时驱动环境重置。
     - ResNet 图像分类器
     - 任务的成功状态在视觉上明确，且你能为其标注帧。
   * - :ref:`Qwen VLM <franka-vlm-reward-model>`
     - 对 5 帧滑动窗口的运动趋势判断，外加成功奖励。
     - Qwen3-VL-4B + LoRA
     - 仅靠成功信号过于稀疏，需要朝目标的稠密引导。

概览
--------------------------------------------------------

将训练后的 reward model 用作 Franka 真机任务的成功信号。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy · ResNet reward model · Qwen3-VL-4B (LoRA)

   .. grid-item-card:: 算法
      :text-align: center

      SAC/RLPD · reward-model inference · VLM trend judgment

   .. grid-item-card:: 任务
      :text-align: center

      Charger · Peg Insertion

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · cameras · keyboard labels · dual RealSense

| **你将完成:** 采集专家示教 → 采集 reward 标注 → 预处理数据 → 训练 reward model → 启动真机 RL.
| **前置条件:** :doc:`franka` 到数据采集步骤 · :doc:`Reward model 教程 <../../extending/reward_model>`.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Keyboard labels
     - ``realworld_collect_dataset``
     - 在实时遥操作中标注 success/failure 帧。
   * - Fixed pose labels
     - ``realworld_charger_sac_cnn_async_standalone_reward``
     - 用目标位姿到达情况生成 reward-model 数据。
   * - RL with reward model
     - ``realworld_charger_sac_cnn_async_standalone_reward``
     - 在 Franka env 中使用 reward-model 成功预测。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - policy 与 reward model 共用相机帧。
   * - Action
     - 与基础 Franka 真机环境相同的笛卡尔动作。
   * - Reward
     - reward-model 成功/失败预测替代手写成功信号。
   * - Prompt
     - 由配置决定，使用任务文本或固定目标位姿。

.. _franka-resnet-reward-model:

ResNet Reward Model（逐帧成功判别）
--------------------------------------------------------

用标注帧训练 ResNet 分类器，并将其预测结果用作 Franka 真机任务的成功信号。

安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

请根据 :doc:`franka` 中 ``运行实验`` 的 ``数据采集`` 之前的章节，完成数据采集之前的全部工作。

数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

需要采集两类数据：（1）用于 demo buffer 的专家轨迹数据；（2）用于 reward model 训练和评估的数据。

采集 reward model 训练和评估数据支持两种方式，详细说明请参考
:doc:`../../extending/reward_model` 中的 **数据采集** 部分。
两种方式的核心区别在于标注方式：方式一为手动键盘标注，适用于任意操作任务；
方式二为基于位姿的自动标注，专为固定目标位姿的任务设计。

专家轨迹数据采集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先需要采集专家轨迹数据，该数据会在训练中事先存储在样本缓冲区（demo buffer）中。
具体步骤同 :doc:`franka` 中 ``运行实验`` 的 ``数据采集`` 小节。
注意确认，配置文件 ``examples/embodiment/config/realworld_collect_data.yaml`` 中
``env`` 部分的 ``data_collection`` 已开启：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: True

方式一：键盘标注（通用）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此方式通过键盘在实时 episode 中手动标注每一帧，适用于任何操作任务。
此方式将数据采集、标注和数据集生成整合为一次端到端运行，无需繁琐的离线预处理步骤。

**关键配置：**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — 目标采集帧数，两个阈值均达到时停止采集。
- ``runner.val_split`` — 所有标注帧中用于验证集的比例。
- ``runner.fail_success_ratio`` — 训练集后处理阶段失败帧下采样比例。
- ``env.eval.keyboard_reward_wrapper`` — 设为 ``single_stage`` 以启用键盘标注界面。
- ``env.eval.use_spacemouse`` — 是否使用 SpaceMouse 进行遥操作。
- ``env.eval.override_cfg.target_ee_pose`` — 任务的目标末端执行器位姿。

**启动命令：**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

**按键说明：**

- ``c`` — 将当前帧标注为成功。
- ``a`` — 将当前帧标注为失败。

达到目标帧数后，脚本自动停止、划分数据并保存 ``train.pt`` / ``val.pt`` 文件。
详细配置说明及完整示例请参见 :doc:`../../extending/reward_model` 中的方式一。

方式二：固定位姿（目标驱动）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

此方式专为固定目标位姿的任务设计，无需手动键盘标注，episode 会根据机器人是否到达
配置的 ``target_ee_pose`` 自动驱动成功/失败判定。
可以设置 ``success_hold_steps``，要求机器人在目标位姿保持一定步数后才判定为成功，
有助于采集更多样的成功样本。
此方式采用简化的两步式流程。

**步骤 1：固定位姿 Reward 数据采集**

在上述专家轨迹采集的基础上，将配置中的 ``success_hold_steps`` 字段增大：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

采集技巧：

- 请尽量缓慢移动机械臂，以便获得更多样的失败样本。
- 在到达目标位姿时，在保持目标位姿的前提下进行小范围移动，以便获得更多样的成功样本。

**步骤 2：预处理为 Reward Dataset**

采集好的 ``.pkl`` episode 通过 ``preprocess_reward_dataset.py`` 转换为 ``train.pt`` / ``val.pt``，
建议将 ``fail-success-ratio`` 调高至 ``3``：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

生成的 ``.pt`` 文件符合 ``RewardDatasetPayload`` 约定的标准格式，包含 ``images``、
``labels`` （1 = 成功，0 = 失败）和 ``metadata``。
详细说明及完整示例请参见 :doc:`../../extending/reward_model` 中的方式二。

Reward Model 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本步骤同 :doc:`../../extending/reward_model` 中的 ``2. Reward Model 训练`` 部分。

特别的，在真实世界场景中，建议降低 ``early_stop`` 的 ``min_delta``，例如：

.. code-block:: yaml

  runner:
    early_stop:
      min_delta: 1e-6

如需在真机遥操作中进行在线 reward model 推理（SpaceMouse + GPU 节点，无需 RL 训练循环），
请参考 :doc:`../../extending/reward_model` 中的 **真机遥操作 + 在线 Reward Model 推理** 部分。

集群设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本步骤同 :doc:`franka` 中的 ``运行实验`` 下的 ``集群配置`` 部分。

配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本步骤同 :doc:`franka` 中的 ``配置文件`` 小节，对 ``examples/embodiment/config/realworld_charger_sac_cnn_async_standalone_reward.yaml`` 进行配置。
特别的，还需要启用位于 ``reward`` 段的 reward model 相关参数：

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.8

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"

其中：

- ``reward_mode`` 控制 reward model 在每一步推理，还是仅在终止帧推理。
- ``standalone_realworld`` 利用 reward model 直接判断任务是否成功，进而触发重置。
- ``reward_threshold`` 用于对 reward model 输出的成功概率做阈值过滤；低于阈值的项会被置为 ``0``。
- ``model_path`` 指向用于在线推理的 reward model 权重。

运行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

启动训练后，reward model 会直接基于图像观测判定任务成功/失败，并驱动环境重置。
其余步骤请继续参照 :doc:`franka` 中 ``运行实验`` 章节执行。

Rollout 阶段的 worker 交互
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

与 :doc:`../../extending/reward_model` 中的 ``3.2 Rollout 阶段的 worker 交互`` 和 ``3.3 最终 reward 的计算`` 部分不同的是：
在真机系统中，由于启动了 ``standalone_realworld``，reward model 将不再 `将 env reward 与 reward model output 组合`。

换句话说，reward model 在 RL 中 `不会` 作为 env worker 中的附加 reward 来源参与最终 reward 的构造，
因为系统会直接绕过 ``env_reward`` 和 ``reward_model_output`` 加权求和的过程。
因此，reward_mode、reward_weight、env_reward_weight 均不生效，最终 reward 由 FrankaEnv 内部直接基于 reward model 判定成功/失败后生成。

从系统的角度看，真机系统中的实际行为可以看做：
直接替换 env worker 中的 env_reward，通过沿用原本 env_reward 的功能来实现奖励赋值和控制系统重置等目的，从根本上进行了 reward model 接入。

.. _franka-vlm-reward-model:

Qwen VLM Reward Model（动作趋势判断）
--------------------------------------------------------

与上述 ResNet reward model 直接判断单帧图像"成功/失败"不同，Qwen VLM reward model
通过\ **动作趋势判断**\ 来引导机械臂学习。每 5 帧构成一个滑动历史窗口，Qwen3-VL 模型
判断窗口内机械臂的运动趋势，并将趋势标签转换为标量 reward 参与 RL 训练。

.. code-block:: text

   VLM 输出          Reward 值        含义
   ─────────────────────────────────────────
   positive          1.0              动作趋势正确，正向奖励
   negative          -0.2             动作趋势错误，轻微惩罚
   unclear           0.0              趋势不明确，不给信号
   invalid           0.0              模型输出无法解析，不给信号

当机械臂到达目标位姿并保持足够步数后（ ``terminated=True`` ），
``RealWorldEnv`` 会在传给 reward model 的 episode 指标中记录 ``success_once``。
``gt_success_bonus`` （默认 +20.0）据此追加奖励，帮助 Agent 识别成功状态。

工作流程
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

完整流程包含三个阶段：

1. **数据采集** — 在真机上采集包含双视角（腕部 + 全局）图像序列的 episode 数据。
2. **监督微调（SFT）** — 将采集数据预处理为 VLM trend 格式，对 Qwen3-VL-4B 进行 LoRA 微调。
3. **真机强化学习** — 在 RLPD 训练中接入微调后的 VLM reward model，通过 ``history_buffer`` 模式在线推理并引导策略学习。

阶段一：数据采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 :doc:`franka` 中的真机训练流程采集 episode 数据。建议开启 ``data_collection``，
将每个 episode 保存为 ``.pkl`` 文件：

.. code-block:: yaml

   env:
     eval:
       data_collection:
         enabled: True
         save_dir: /path/to/collected_data
         export_format: "pickle"
         only_success: False

采集技巧
^^^^^^^^^^^^^^^^^^^^^^^^

- **缓慢移动机械臂**，使采集数据包含丰富的中间状态，便于 VLM 学习趋势判断。
- **确保双相机视角清晰**：``main_images`` （腕部相机）和 ``extra_view_images`` （全局相机）
  都能清晰看到机械臂末端和目标孔位。
- **采集足够 episode** （建议 50+），覆盖成功和失败两种结局。
- **正确配置** ``camera_names``，确保 ``SERIAL1`` / ``SERIAL2`` 与实际相机序列号一致：

.. code-block:: yaml

   env:
     train:
       override_cfg:
         camera_names:
           "SERIAL1": "global"    # 替换为全局相机序列号
           "SERIAL2": "wrist_1"   # 替换为腕部相机序列号

两个相机的作用：

- **腕部相机** （ ``main_images`` ）：近距离观察夹爪和插孔的精细交互。
- **全局相机** （ ``extra_view_images`` ）：提供工作台整体布局的上下文信息。

双视角输入让 VLM 能同时关注局部操作细节和全局空间关系，提高趋势判断的准确性。

在更新 ``examples/embodiment/config/realworld_collect_data.yaml`` （相机序列号、
``target_ee_pose`` 和 ``data_collection.save_dir``）后，使用标准真机采集入口：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

阶段二：监督微调（SFT）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

预处理为 VLM Trend 数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

采集到的 ``.pkl`` episode 需要通过 ``preprocess_vlm_trend_reward_dataset.py``
转换为 VLM trend SFT 格式。运行前需要激活虚拟环境并设置 ``PYTHONPATH``：

.. code-block:: bash

   source .venv/bin/activate
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

该脚本将 episode 按滑动窗口切分为 5 帧片段，
提取双视角图像，并根据 GAE、reward 或 TCP 距离变化自动标注：

.. code-block:: bash

   python examples/reward/preprocess_vlm_trend_reward_dataset.py \
       --raw-data-path /path/to/collected_data \
       --output-dir /path/to/processed_vlm_trend_reward_data \
       --window-size 5 \
       --task-description "Pick up the peg and insert it into the hole."

当采集数据没有 GAE 信号时，通过 ``--target-ee-pose`` 使用 TCP 距离代替 reward 作为趋势信号：

.. code-block:: bash

   python examples/reward/preprocess_vlm_trend_reward_dataset.py \
       --raw-data-path /path/to/collected_data \
       --output-dir /path/to/processed_vlm_trend_reward_data \
       --window-size 5 \
       --target-ee-pose "X,Y,Z,RX,RY,RZ"

``X,Y,Z,RX,RY,RZ`` 替换为你的任务目标位姿。获取方式：

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_controller

将机械臂移动到目标位置后，终端会打印当前 TCP 位姿。只需位置时可填入 ``X,Y,Z`` （3 个值），方向被忽略。

输出目录结构：

.. code-block:: text

   processed_vlm_trend_reward_data/
   ├── dataset_info.json
   ├── train/
   │   ├── segments.jsonl
   │   └── pkl/              # 每个窗口一个 pkl，含 main_frames + extra_view_frames
   └── eval/
       ├── segments.jsonl
       └── pkl/

微调 Qwen3-VL-4B
^^^^^^^^^^^^^^^^^^^^^^^

修改 SFT 配置文件 ``examples/sft/config/qwen3vl_sft_vlm_trend_reward.yaml`` 中的路径：

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
     gradient_checkpointing: true     # 使用 gradient checkpointing 节省显存
     sharding_strategy: no_shard      # RTX 4090 使用 no_shard

设置环境变量并启动训练：

.. code-block:: bash

   export VLM_TREND_REWARD_DATA_ROOT=/path/to/processed_vlm_trend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_vlm_trend_reward

训练完成后，LoRA checkpoint 路径（如 ``checkpoints/global_step_3000`` ）将通过
``reward.model.lora_path`` 在 RL 训练中引用。

阶段三：真机强化学习
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

配置文件
^^^^^^^^^^^^^^

使用 ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_vlm_reward.yaml``
作为 RL 训练配置。该配置基于 RLPD CNN 异步训练模板，核心 reward 配置如下：

.. code-block:: yaml

   reward:
     use_reward_model: true            # 启用 reward model
     worker_type: model                # 本地 HuggingFace 推理
     group_name: "RewardGroup"
     standalone_realworld: False  
     reward_mode: history_buffer       # 历史窗口趋势判断
     history_reward_assign: true       # 将 VLM 奖励反向分配给历史步
     reward_weight: 1.0                # VLM 奖励权重
     env_reward_weight: 0.0            # 原生环境奖励权重（纯 VLM）
     reward_threshold: 0.5

     model:
       model_path: "/path/to/Qwen3-VL-4B-Instruct"
       model_type: "buffered_vlm"
       lora_path: "/path/to/sft_output/checkpoints"
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

关键配置字段说明
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - ``reward_mode: history_buffer``
     - Env worker 维护滑动窗口，积累到 ``min_history_size`` 帧后才发送给 VLM 推理。
   * - ``history_reward_assign: true``
     - 将 VLM 返回的趋势奖励反向分配到历史窗口内的每一步。
   * - ``env_reward_weight: 0.0``
     - Franka 原生 reward 为稀疏信号（到达=1.0），训练主要依赖 VLM 趋势判断。
       到达目标时由 ``gt_success_bonus`` 提供奖励。
   * - ``gt_success_bonus: 20.0``
     - 当 ``env_infos["episode"]["success_once"]`` 表示成功时追加 +20.0。
   * - ``history_buffers.history_window.history_keys``
     - ``[main_images, extra_view_images]`` — 历史窗口缓存双视角帧，供 VLM 推理使用。
   * - ``history_buffers.history_window``
     - 缓存最近 5 帧的 ``main_images`` 和 ``extra_view_images``，最少 5 帧后触发推理。
   * - ``worker_type: model``
     - 在 reward worker 进程中直接加载模型进行本地推理。

奖励计算流程
^^^^^^^^^^^^^^^^^

每一步 RL 训练中，最终奖励由以下流程合成：

.. code-block:: text

   FrankaEnv.step()
     -> 原生 reward 与终止信号
          |
          v
   RealWorldEnv.step()
     -> env_infos["episode"]["success_once"]
          |
          v
   EnvWorker.get_reward_model_output()
     -> HistoryManager 构建 history_input
     -> reward worker 接收 reward_input
          |
          v
   EmbodiedRewardWorker.compute_image_rewards()
     -> BufferedVLMRewardModel.compute_reward()
        -> 历史不足：返回 0.0
        -> 历史满足：Qwen3-VL 推理并解析为 1.0 / -0.2 / 0.0
     -> apply_gt_success_bonus()：success_once 为 True 时追加 20.0
          |
          v
   final_reward = env_reward_weight * env_reward
                + reward_weight * vlm_reward_with_bonus

成功信号
^^^^^^^^^^^^

``FrankaEnv`` 在到达目标时返回稀疏 reward 1.0。``RealWorldEnv`` 将该信号
转换为 episode 内持续有效的 ``success_once`` 指标。Env worker 把 episode 指标
传给 reward model，``apply_gt_success_bonus`` 再读取 ``success_once``。

启动训练
^^^^^^^^^^^^^^

确认硬件部署和配置无误后，在 Ray head 节点执行：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh \
       realworld_peginsertion_rlpd_cnn_async_vlm_reward

训练启动后，日志中可以看到 VLM 推理输出、reward 分布和成功信号。
