Reward Model 使用指南
======================

在 RLinf 中使用 reward model——包括 ``ResNetRewardModel`` 这类图像分类 reward，
以及基于 ``VLMRewardModel`` 的 VLM reward。``BufferedVLMRewardModel`` 继承
``VLMRewardModel``，用于处理 env worker 维护的历史窗口。

仿真场景 Reward Model
---------------------

完整流程包括四个阶段：

1. 数据收集：在 RL 运行过程中采集原始 episode 数据。
2. 数据转换：将原始 episode 转成图像分类数据或 VLM SFT 数据。
3. Reward model 训练：训练 ResNet reward model，或微调 VLM reward model。
4. Reward model 在 RL 中推理：将训练好的模型接入在线 rollout，参与最终 reward 计算。

1. 数据收集
^^^^^^^^^^^

reward model 的训练数据通常来自 episode 级数据采集。RLinf 提供了统一的数据采集封装，
相关用法可参考 :doc:`数据采集教程 <../guides/data_collection>`。

对于 reward model 场景，建议先以 ``pickle`` 格式保存原始 episode 数据，再通过预处理脚本转换为训练集。

1.1 启用数据采集
""""""""""""""""

在 YAML 配置文件的 ``env`` 部分开启 ``data_collection``：

.. code-block:: yaml

   env:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "pickle"
       only_success: False

启动训练或评估后，环境会自动将 episode 保存到 ``save_dir``。当 ``export_format="pickle"`` 时，
每个 episode 会被写入一个独立的 ``.pkl`` 文件，便于后续离线预处理。

对于 VLM Trend reward，RLinf 也提供了可直接运行的数据采集配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_vlm_trend_reward_collect

该配置保持 ``reward.use_reward_model: false``，并在 eval 环境上开启数据采集。
保存下来的 episode 会包含 VLM 流程后续需要的双视角图像观测，例如
``main_images`` 和 ``extra_view_images``。

1.2 预处理为 ResNet reward dataset
""""""""""""""""""""""""""""""""""

原始 ``pickle`` 文件不能直接用于 reward model 训练，需要使用
``examples/reward/preprocess_reward_dataset.py`` 进行转换。该脚本会读取采集到的 ``.pkl`` episode，
从观测中提取 ``main_images``，并基于逐步 ``info["success"]`` 生成二分类标签，最终保存为
``RewardBinaryDataset`` 可直接加载的 ``.pt`` 数据文件。

预处理命令示例：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data

默认会生成：

.. code-block:: text

   logs/xxx/processed_reward_data/
   ├── train.pt
   └── val.pt

生成后的 ``.pt`` 文件满足 ``RewardDatasetPayload`` 约定的标准格式：

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

其中：

- ``images`` 存放训练样本图像。
- ``labels`` 存放二分类标签。
- ``metadata`` 记录原始数据路径、采样参数、划分比例等信息。

训练阶段，``RewardBinaryDataset`` 会直接加载上述 ``RewardDatasetPayload`` 格式的 ``train.pt`` / ``val.pt``。

1.3 转换为 VLM Trend reward dataset
""""""""""""""""""""""""""""""""""""""""""""""""

VLM Trend reward 使用短时间双视角历史窗口，而不是单张图像。使用
``examples/reward/vlm_trend/preprocess_reward_dataset.py`` 可以将采集到的
episode 切成 5 帧窗口，提取 ``main_images`` 和 ``extra_view_images``，并给每个
窗口标注 ``positive``、``negative`` 或 ``unclear``。

命令示例：

.. code-block:: bash

   python examples/reward/vlm_trend/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_vlm_trend_reward_data \
       --window-size 5 \
       --stride 1 \
       --delta-threshold 0.05

默认会生成 JSONL manifest 和逐样本 pickle 文件：

.. code-block:: text

   logs/xxx/processed_vlm_trend_reward_data/
   ├── dataset_info.json
   ├── train/
   │   ├── segments.jsonl
   │   └── pkl/
   └── eval/
       ├── segments.jsonl
       └── pkl/

train/eval 按 episode 划分，因此同一个 episode 中切出的窗口不会混到不同 split 中。

2. Reward Model 训练
^^^^^^^^^^^^^^^^^^^^

RLinf 支持两条 reward 训练路径。``examples/reward/run_reward_training.sh``
用于训练 ResNet 图像 reward model，``examples/sft/run_vlm_sft.sh``
用于微调 VLM Trend reward model。

2.1 在线 Reward Model 类型
""""""""""""""""""""""""""

在线 embodied reward worker 通过 ``reward.model.model_type`` 选择实现类：

.. code-block:: python

   reward_model_registry = {
       "resnet": ResNetRewardModel,
       "vlm": VLMRewardModel,
       "buffered_vlm": BufferedVLMRewardModel,
       "vlm_trend_success_potential": VLMTrendSuccessPotentialRewardModel,
   }

其中：

- ``resnet``：单帧图像二分类，输出 sigmoid 概率。
- ``vlm``：对当前观测运行 VLM（单步/终止帧等，由 ``reward_mode`` 决定）。
- ``buffered_vlm``：对 env worker 维护的历史窗口运行 VLM；具体 prompt、视频格式与
  标量映射由 ``input_builder_name`` / ``reward_parser_name`` 决定。
  标准 VLM Trend reward 即 ``buffered_vlm`` + ``vlm_trend_reward_*`` 插件组合。
- ``vlm_trend_success_potential``：加载独立的 Potential 和 Success LoRA adapter，
  执行 scalar potential head，并维护 episode 内的 shaping 状态。

2.2 微调 ResNet Reward Model
""""""""""""""""""""""""""""

2.2.1 配置 ResNet 数据路径
..........................

训练前需要先修改 ``examples/reward/config/reward_training.yaml`` 中的数据路径，
指向上一步预处理得到的文件：

.. code-block:: yaml

   data:
     train_data_paths: "logs/processed_reward_data/train.pt"
     val_data_paths: "logs/processed_reward_data/val.pt"

.. note::

   当前 ``run_reward_training.sh`` 主要负责组织启动命令与日志目录；
   训练数据路径以 ``reward_training.yaml`` 中的 ``data.train_data_paths`` 和
   ``data.val_data_paths`` 配置为准。

2.2.2 配置 ResNet 模型
......................

对于 ResNet 路径，需要将 ``actor.model.model_type`` 设置为 ``"resnet"``：

.. code-block:: yaml

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 224, 224]

如果需要从已有权重继续训练，可以通过 ``model_path`` 指定 checkpoint；
如果希望从头训练，则保持 ``model_path: null``。

2.2.3 启动 ResNet 训练
......................

完成数据与模型配置后，执行：

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

训练日志会保存到新建的 ``logs/<timestamp>-reward_training`` 目录下。

2.3 微调 VLM Trend Reward Model
""""""""""""""""""""""""""""""""""""""""""""""""

使用 ``examples/reward/vlm_trend/preprocess_reward_dataset.py`` 转换数据后，将
``VLM_TREND_REWARD_DATA_ROOT`` 指向处理后的数据根目录，然后启动 VLM SFT：

.. code-block:: bash

   export VLM_TREND_REWARD_DATA_ROOT=/path/to/processed_vlm_trend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_vlm_trend_reward

对应配置会读取 JSONL manifest 和逐样本 pickle 文件：

.. code-block:: yaml

   data:
     type: vlm
     dataset_name: "vlm_trend_reward_sft"
     train_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/train/segments.jsonl"
     val_data_paths: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}/eval/segments.jsonl"
     video_root: "${oc.env:VLM_TREND_REWARD_DATA_ROOT}"
     video_nframes: 5

   actor:
     model:
       model_type: qwen3_vl
       model_path: /path/to/Qwen3-VL-4B-Instruct
       attn_implementation: flash_attention_2
       is_lora: true
       lora_rank: 16

训练得到的 LoRA checkpoint 后续可通过 ``reward.model.lora_path`` 传给在线 reward 配置。

2.3.1 训练 Success + Potential 分支
........................................

当你需要同一个 VLM 同时提供稀疏终止奖励和稠密进度 shaping 时，按以下
四个阶段训练：

#. 从 rollout state 训练 state-success teacher。
#. 分别训练 Success LoRA 和 Potential LoRA。
#. 冻结 Potential LoRA，提取 prompt feature，再训练 scalar head。
#. 在线组合 one-shot Success bonus 与 potential-difference shaping。

运行各阶段前，先在 YAML 中配置路径：

- 在 ``examples/reward/vlm_trend/config/pipeline.yaml`` 中设置
  ``paths.raw_data_root``、``paths.pipeline_root``、``paths.model_path``，并在
  Potential SFT 完成后设置 ``paths.potential_checkpoint``。teacher、feature
  和 scalar-head 路径都通过 YAML 引用复用这些值。
- 在 ``examples/sft/config/vlm_trend_success_sft.yaml`` 和
  ``vlm_trend_potential_sft.yaml`` 中设置 ``paths.pipeline_root`` 与
  ``paths.model_path``；数据和输出路径由 YAML 自动派生。

使用 ``pipeline.yaml`` 中 ``auxiliary.teacher.raw_data_paths`` 列出的
checkpoint 目录训练 teacher：

.. code-block:: bash

   python examples/reward/vlm_trend/train_auxiliary.py auxiliary.stage=teacher

生成两套数据。每个采集 checkpoint 目录都重复传入一次
``--raw-data-path``；下面的简化示例展示参数形式：

.. code-block:: bash

   python examples/reward/vlm_trend/preprocess_terminal_success.py \
     --raw-data-path /path/to/vlm_trend_uniform_collection/step0 \
     --raw-data-path /path/to/vlm_trend_uniform_collection/step20 \
     --output-dir /path/to/vlm_trend_success_potential/success_data

   python examples/reward/vlm_trend/preprocess_potential.py \
     --raw-data-path /path/to/vlm_trend_uniform_collection/step0 \
     --raw-data-path /path/to/vlm_trend_uniform_collection/step20 \
     --value-checkpoint /path/to/vlm_trend_success_potential/teacher/best.pt \
     --output-dir /path/to/vlm_trend_success_potential/potential_data

这些命令会保留与在线采样一致的 ``0``/``1`` terminal-success 窗口，不执行
balanced sampling。两条分支都在首次成功 transition 截断，与在线环境的终止边界一致。
Potential 预处理使用 teacher 生成绝对 potential 数字，以及
``up``/``same``/``down`` progress pair。

两个薄配置继承现有 Trend 配方，路径已经在 YAML 中配置，无需再导出路径环境变量：

.. code-block:: bash

   bash examples/sft/run_vlm_sft.sh vlm_trend_success_sft
   bash examples/sft/run_vlm_sft.sh vlm_trend_potential_sft

将 ``pipeline.yaml`` 中的 ``paths.potential_checkpoint`` 指向选中的
Potential VLM SFT 权重文件（通常是
``.../global_step_*/actor/model_state_dict/full_weights.pt``）。如果自行导出过
PEFT adapter，也可以填写包含 ``adapter_config.json`` 的目录。

Feature 提取不要求必须使用四张 GPU。默认 RLinf placement 使用所有可用 accelerator；如果只使用一张 GPU，
在 YAML 中改为：

.. code-block:: yaml

   cluster:
     component_placement:
       feature_extractor: 0  # 使用全部可用 accelerator 时设为 all

RLinf 会从 placement 自动得到每个 shard 的 rank 和 world size。Python 入口会
处理两个 split 和两种 sample type，替代原来的嵌套 shell 循环：

.. code-block:: bash

   python examples/reward/vlm_trend/extract_potential_features.py
   python examples/reward/vlm_trend/train_auxiliary.py auxiliary.stage=scalar_head

使用现有 ``eval/eval_accuracy`` 选择 Success adapter，并使用
``scalar_head/metrics.jsonl`` 中的 ``model_potential`` 选择 scalar head。

3. Reward Model 在 RL 中推理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RLinf 提供了多个 reward model 接入 RL 的示例配置：

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``
- ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_vlm_trend_reward.yaml`` （VLM Trend reward，本地 Hugging Face）
- ``examples/embodiment/config/maniskill_ppo_mlp_vlm_trend_reward_sglang.yaml`` （VLM Trend reward，SGLang API）

这些配置展示了如何在 RL 训练中启用 reward worker，同时让策略网络继续使用状态观测，
而 reward model 使用图像观测或 VLM 观测。

3.1 基本配置项
""""""""""""""

在 RL 配置中，reward model 相关参数位于 ``reward`` 段：

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     reward_mode: "terminal"   # 或 "per_step" / "history_buffer"
     reward_threshold: 0.5
     reward_weight: 1.0
     env_reward_weight: 0.0

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"    # 或 "vlm" / "buffered_vlm"

其中：

- ``reward_mode`` 控制 reward model 在每一步、终止帧，还是历史窗口上推理。
- ``reward_weight`` 和 ``env_reward_weight`` 控制 learned reward 与环境 reward 的加权组合。
- ``reward_threshold`` 仅对 ``model_type: resnet`` 生效：低于阈值的 sigmoid 概率会被置为 ``0``。
  对 ``buffered_vlm`` / VLM Trend reward，标量 reward 由 ``reward_parser_params`` 定义；
  配置里的 ``reward_threshold`` 当前不会被 VLM 路径读取。
- ``model_path`` 指向用于在线推理的 reward model 权重。

3.2 Rollout 阶段的 worker 交互
""""""""""""""""""""""""""""""

在线 RL 阶段，``env``、``rollout``、``reward`` 三类 worker 会协同工作。整体流程如下：

.. code-block:: text

   Env worker
      | 1. 与环境交互，获得 obs / env reward / done
      | 2. 将 obs 发送给 Rollout worker 生成动作
      | 3. 当启用 reward model 时，将 reward input dict 发送给 Reward worker
      v
   Reward worker
      | 4. 执行 compute_reward(...)，返回 reward model output
      v
   Env worker
      | 5. 接收 Rollout worker 的 bootstrap values
      | 6. 将 env reward 与 reward model output 组合
      v
   Final reward -> 写入 rollout 结果并参与后续 RL 更新

在实现上，``EnvWorker`` 会在 rollout 过程中向 reward worker 请求 reward model 输出，
再统一计算最终 reward。

3.3 最终 reward 的计算
""""""""""""""""""""""

当 reward channel 已启用时，``EnvWorker`` 会先获取 ``reward_model_output``，
随后在 ``compute_bootstrap_rewards`` 中与环境原始 reward 合并：

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

之后，若当前算法配置启用了 bootstrap，RLinf 还会按配置将 bootstrap value 加到最后一步 reward 中。

因此，从系统视角看，reward model 在 RL 中并不会替代原有的 bootstrap reward，
而是作为 env worker 中的附加 reward 来源参与最终 reward 的构造。

3.4 部署 VLM Trend reward 进行 MLP RL
""""""""""""""""""""""""""""""""""""""""""""""""

进行 VLM reward 推理前，需要安装带 VLM reward 支持的 embodied 依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --env maniskill_libero --model qwen3_vl \
     --torch 2.8.0 --sglang 0.5.4 --transformers 4.57.1

标准 VLM Trend reward 使用 ``model_type: buffered_vlm``，并通过
``vlm_trend_reward_input_builder`` 和 ``vlm_trend_reward_parser`` 构造输入、解析 reward。
Success + Potential 配方的本地推理使用专用
``model_type: vlm_trend_success_potential``。API 推理仍只支持 ``buffered_vlm``。

两条本地 VLM Trend 路径都使用历史窗口输入。标准路径使用：

- ``input_builder_name: vlm_trend_reward_input_builder``
- ``reward_parser_name: vlm_trend_reward_parser``
- ``reward_mode: history_buffer`` 与 ``history_buffers`` （双视角 5 帧窗口）
- ``interval_reward`` ：历史窗口尚未填满时使用的默认标量（通常为 ``0.0`` ）

.. note::

   env worker 在 ``reward_mode: history_buffer`` 下 **每步都会** 向 reward worker 发送
   ``history_input``；窗口未满足 ``min_history_size`` 时，reward worker 会返回
   ``interval_reward``，而不是跳过 RPC。

3.4.1 本地 Hugging Face 推理
............................

不设置 ``reward.worker_type``\ （默认 ``model``，使用 ``EmbodiedRewardWorker``）。
参考 ``maniskill_ppo_mlp_vlm_trend_reward.yaml``：

.. code-block:: yaml

   reward:
     use_reward_model: true
     group_name: "RewardGroup"
     reward_mode: history_buffer
     history_reward_assign: true
     reward_weight: 1.0
     env_reward_weight: 0.0
     model:
       model_path: "/path/to/Qwen3-VL-4B-Instruct"
       model_type: "buffered_vlm"
       lora_path: "/path/to/qwen3-vl-lora-checkpoint"
       gt_success_bonus: 20.0
       precision: "bf16"
       input_builder_name: vlm_trend_reward_input_builder
       input_builder_params:
         default_task_description: "Pick up the red cube and place it on the green spot on the table."
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
       infer_micro_batch_size: 64
       max_new_tokens: 16
       do_sample: false
       temperature: 0.0

启动：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_vlm_trend_reward

先使用 ManiSkill 原有的稠密环境奖励预热 MLP policy。这里直接复用未修改的
``maniskill_ppo_mlp``，预热阶段不要启用 reward overlay。使用同一组 1,024 个
固定 reset state 评估保存的 checkpoint，并选择 ``success_once`` 非零且接近
5%（约 3--7%）的一个；正式 PPO 不要从零成功率 checkpoint 开始：

.. code-block:: bash

   export EMBODIED_PATH="$(pwd)/examples/embodiment"
   python examples/embodiment/train_embodied_agent.py \
     --config-path config \
     --config-name maniskill_ppo_mlp \
     runner.max_steps=60 runner.save_interval=5 runner.val_check_interval=5 \
     env.eval.total_num_envs=1024

然后使用选中的低成功率 checkpoint 和上述三个 reward 产物启动双输出分支。
该配方基于现有 VLM Trend 配方，使用 1,024 个环境，每 5 个 PPO step 评估一次，
关闭环境奖励，默认训练 160 个 step：

在
``examples/embodiment/config/maniskill_ppo_mlp_vlm_trend_success_potential.yaml``
的 ``vlm_trend_paths`` 中填写五个产物路径。运行字段会直接引用这段 YAML：

.. code-block:: yaml

   vlm_trend_paths:
     policy_checkpoint: /path/to/evaluated_approximately_5pct/actor/model_state_dict/full_weights.pt
     model_path: /path/to/Qwen3-VL-4B-Instruct
     success_checkpoint: /path/to/success/selected_global_step/actor/model_state_dict/full_weights.pt
     potential_checkpoint: /path/to/potential/selected_global_step/actor/model_state_dict/full_weights.pt
     scalar_head: /path/to/scalar_head/best.pt

然后启动已配置的配方：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     maniskill_ppo_mlp_vlm_trend_success_potential

专用 ``VLMTrendSuccessPotentialRewardModel`` 计算
``scale * (gamma * potential_t - potential_{t-1})``，并在满足连续确认窗口后，
每个 episode 仅添加一次 ``success_bonus``。两组状态都会在 ``done`` 时重置。

.. warning::

   ``reward.model.lora_path`` 和 ``reward.model.success_lora_path`` 必须是
   ``full_weights.pt`` 的完整文件路径（VLM SFT 产物通常为
   ``.../actor/model_state_dict/full_weights.pt``）。如果你自行导出过 PEFT
   adapter，也可以传包含 ``adapter_config.json`` 的目录。不要只传
   checkpoint 根目录。

3.4.2 SGLang API 推理
.....................

设置 ``reward.worker_type: api``\ （``EmbodiedAPIRewardWorker``）。可指向外部
OpenAI-compatible endpoint，或留空 ``reward.api.api_base`` 并由 RLinf 按
:doc:`../guides/sglang_server` 拉起 Ray 托管的 SGLang server/router。
参考 ``maniskill_ppo_mlp_vlm_trend_reward_sglang.yaml``：

.. code-block:: yaml

   reward:
     use_reward_model: true
     worker_type: api
     group_name: "RewardGroup"
     reward_mode: history_buffer
     history_reward_assign: true
     reward_weight: 1.0
     env_reward_weight: 0.0
     api:
       api_base: null
       model: Qwen3-VL-4B-Instruct
       sampling_params:
         max_tokens: 16
         temperature: 0.0
     model:
       model_path: "/path/to/Qwen3-VL-4B-Instruct"
       model_type: "buffered_vlm"
       gt_success_bonus: 20.0
       precision: "bf16"
       input_builder_name: vlm_trend_reward_input_builder
       input_builder_params:
         default_task_description: "Pick up the red cube and place it on the green spot on the table."
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

SGLang 路径额外说明：

- ``router_server_args`` 使用标准 SGLang server/router 配置。
- ``cluster.component_placement.reward_server`` 决定 SGLang server worker 的放置位置。
- 当 ``reward.api.api_base`` 为空且配置了 ``router_server_args`` 时，
  ``train_embodied_agent.py`` 会在创建 reward worker 前解析 endpoint 并写入
  ``reward.api.api_base``。

启动：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_vlm_trend_reward_sglang

4. 总结
^^^^^^^^^^^^

完整工作流如下：

1. 在环境配置中开启 ``data_collection``，并将数据保存为 ``pickle`` 格式。
2. 对于 ResNet reward，使用 ``preprocess_reward_dataset.py`` 构建 ``train.pt`` / ``val.pt``，再用 ``run_reward_training.sh`` 训练。
3. 对于 VLM Trend reward，使用 ``examples/reward/vlm_trend/preprocess_reward_dataset.py`` 构建双视角历史窗口数据，再用 ``run_vlm_sft.sh`` 微调。
4. 在 RL YAML 中开启 ``reward.use_reward_model=True``，并通过示例配置接入 reward worker 完成在线推理。


真机场景 Reward Model
---------------------

在真实世界的 Franka 机械臂上直接采集并预处理 reward model 训练数据集。
支持两种数据采集方式：**通用键盘标注方式** 和 **固定位姿方式** （通过预定的目标位姿驱动 episode 成功/失败）。

在开始前，强烈建议先阅读以下文档：

1. :doc:`../examples/embodied/franka` 以熟悉 Franka 机械臂真机训练全流程。
2. :doc:`reward_model` 以了解 RLinf 中标准的 reward model 工作流（通过 ``pickle`` 采集数据、离线预处理、训练、RL 推理）。
3. :doc:`../examples/embodied/franka_reward_model` 以了解在训练好 reward model 后如何接入真机 RL 流程。

工作流概览
^^^^^^^^^^

方式一将数据采集、标注和数据集生成整合为一次端到端运行；方式二采用简化的两步式流程。

.. code-block:: text

   真机数据集采集（本指南）
   ├── 方式一：键盘标注（通用）
   │   1. 使用 SpaceMouse / 键盘遥操作启动单个 RealWorld episode。
   │   2. 按 'c'（成功）或 'a'（失败）标注每一帧。
   │   3. 达到阈值或 max_steps 时停止。
   │   4. 对 fail:success 比例进行采样，并划分训练/验证集。
   │   5. 直接保存 train.pt / val.pt（无中间 .pkl 文件）。
   │
   └── 方式二：固定位姿（目标驱动）
       1. 配置目标末端执行器位姿（无需键盘标注）。
       2. 机器人到达目标位姿时 episode 自动终止。
       3. 保存 episode 轨迹为 .pkl 文件。
       4. 从 episode 轨迹中自动提取成功/失败帧。
       5. 通过 preprocess_reward_dataset.py 预处理并生成 train.pt / val.pt。

预备工作
^^^^^^^^

请根据 :doc:`../examples/embodied/franka` 中的 **Prerequisites** 和 **Hardware Setup** 章节，
完成机器人连接和环境验证步骤。

数据采集
^^^^^^^^

方式一：键盘标注（通用）
""""""""""""""""""""""""

此方式通过键盘在实时 episode 中手动标注每一帧，适用于任何操作任务。

**配置文件** — ``examples/reward/config/realworld_collect_dataset.yaml``，
环境参数从 ``env/realworld_bin_relocation.yaml`` 继承：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       env:
         node_group: franka
         placement: 0
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 0

   runner:
     task_type: embodied
     logger:
       log_path: null
       project_name: rlinf
       experiment_name: "collect-dataset"
       logger_backends: ["tensorboard"]
     num_success_frames: 50    # 目标采集的成功帧数
     num_fail_frames: 150      # 目标采集的失败帧数
     val_split: 0.2            # 用于验证集的帧比例
     fail_success_ratio: 2.0   # 训练集后处理时将失败帧下采样至 success * ratio
     random_seed: 42

   env:
     group_name: "EnvGroup"
     eval:
       no_gripper: False
       use_spacemouse: True
       max_episode_steps: 10000
       keyboard_reward_wrapper: single_stage
       override_cfg:
         target_ee_pose: TARGET_EE_POSE

**关键配置字段说明：**

- ``runner.num_success_frames`` / ``runner.num_fail_frames`` — 目标采集帧数。两个阈值均达到时停止采集。
- ``runner.val_split`` — 所有标注帧中用于验证集的比例。
- ``runner.fail_success_ratio`` — 训练集后处理阶段，失败帧会被下采样，使 ``num_fail = num_success * fail_success_ratio``。设为 ``0`` 可禁用下采样。
- ``env.eval.keyboard_reward_wrapper`` — 设为 ``single_stage``（或任务对应的 ``stage``）以启用键盘标注界面。
- ``env.eval.use_spacemouse`` — 是否使用 SpaceMouse 进行遥操作（step info 中的 ``intervene_action`` 会覆盖默认零动作）。
- ``env.eval.override_cfg.target_ee_pose`` — 任务的目标末端执行器位姿。

**启动命令：**

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh

或者显式指定配置名称：

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset

终端会实时打印进度条：

.. code-block:: text

   success: 12/50 [############----------------]  fail: 28/150 [#####################-----------]

在 episode 过程中使用以下按键：

- ``c`` — 将当前帧标注为成功。
- ``a`` — 将当前帧标注为失败。
- ``keyboard_reward_wrapper`` 中的键盘操作也会控制 episode 是否继续或重置。

当 ``num_success_frames`` 和 ``num_fail_frames`` 两个阈值均达到后，
脚本自动停止、划分数据并保存 ``.pt`` 文件。


方式二：固定位姿（目标驱动）
""""""""""""""""""""""""""""

此方式专为固定目标位姿的任务设计（例如到达预定箱体位置）。
无需手动键盘标注，episode 会根据机器人是否到达配置的 ``target_ee_pose`` 自动驱动成功/失败判定。
可以设置 ``success_hold_steps``，要求机器人在目标位姿保持一定步数后才判定为成功，
有助于采集更多样的成功样本。

此方式的数据采集流程同 :doc:`../examples/embodied/franka_reward_model`，
但预处理步骤与方式一相同，使用同一脚本。


步骤 1：固定位姿 Reward 数据采集
.......................................

为了得到高质量的 reward model，需要采集更多的数据用来训练和评估。
在上述专家轨迹采集的基础上，进一步对采集脚本做以下修改：

将配置中的 ``success_hold_steps`` 字段增大，以便在有限的采集轮次内得到更多的成功数据。
机械臂末端在到达目标位姿后不会立刻判定为成功并重置，
而是需要到达目标位姿并保持一定步数（``success_hold_steps``）后才会判定为成功。
如果中途退出成功状态，会重新开始计数。

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         success_hold_steps: 20

采集技巧：

- 请尽量缓慢移动机械臂，以便获得更多样的失败样本。
- 在到达目标位姿时，在保持目标位姿的前提下进行小范围移动，以便获得更多样的成功样本。

步骤 2：预处理为 Reward Dataset
.......................................

采集好的 ``.pkl`` episode 通过 ``preprocess_reward_dataset.py`` 转换为 ``train.pt`` / ``val.pt``。
建议调高 ``fail-success-ratio`` 至 ``3``：

.. code-block:: bash

   python examples/reward/preprocess_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_reward_data \
       --fail-success-ratio 3

生成文件如下：

.. code-block:: text

   logs/xxx/processed_reward_data/
   ├── train.pt
   └── val.pt

生成的 ``.pt`` 文件符合 ``RewardDatasetPayload`` 约定的标准格式：

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],
       "metadata": dict[str, Any],
   }

其中：

- ``images`` — 训练样本图像。
- ``labels`` — 二分类标签（1 = 成功，0 = 失败）。
- ``metadata`` — 原始数据路径、采样参数、划分比例等信息。


输出
""""

采集完成后（两种方式均适用），两个 ``.pt`` 文件会保存到 ``runner.logger.log_path``
（默认为 Hydra run dir）：

.. code-block:: text

   logs/<timestamp>-collect-dataset/
   ├── train.pt
   └── val.pt
   └── run_collect_process.log   # （仅方式一）

每个 ``.pt`` 文件遵循 ``RewardDatasetPayload`` 约定的标准格式：

.. code-block:: python

   {
       "images": list[torch.Tensor],
       "labels": list[int],             # 1 = 成功，0 = 失败
       "metadata": dict,                # 采集统计信息和配置参数
   }

``metadata`` 字典包含以下字段：

- ``num_success_frames`` / ``num_fail_frames`` — 比例采样前的原始帧数。
- ``fail_success_ratio`` / ``val_split`` / ``random_seed`` — 采样参数。
- ``num_train_samples`` / ``num_val_samples`` — 最终数据集大小。

生成的 ``.pt`` 文件可直接用于 ``RewardBinaryDataset`` 进行训练，
具体用法与上方仿真场景 Reward Model 第 2 节描述一致。

数据采集方式对比
""""""""""""""""

.. list-table::
   :header-rows: 1

   * -
     - 键盘标注
     - 固定位姿（目标驱动）
   * - **标注方式**
     - 手动逐帧（``c`` / ``a``）
     - 自动（episode 成功/失败信号）
   * - **Episode 终止**
     - 由键盘封装器驱动
     - 由到达 ``target_ee_pose`` 驱动
   * - **成功保持**
     - 不适用
     - ``success_hold_steps`` 捕获多样成功样本
   * - **输出流程**
     - 直接生成 .pt（一个脚本）
     - ``.pkl`` episode → ``preprocess_reward_dataset.py`` → .pt
   * - **适用场景**
     - 任意操作任务
     - 具有固定目标位姿的任务

Reward Model 训练
^^^^^^^^^^^^^^^^^

完成以上步骤后，继续参考上方仿真场景 Reward Model 第 2 节（**Reward Model 训练**）
使用生成的 ``train.pt`` / ``val.pt`` 文件进行 reward model 训练。

训练好 reward model 后，有两种方式在真机上使用：

- **真机遥操作 + 在线推理** （见下文）——使用 SpaceMouse 遥操作机械臂，
  同时让 reward model 在 GPU 节点上运行，实时向终端输出成功概率。
  无需启动完整 RL 训练循环。
- **真机 RL 训练** （参见 :doc:`../examples/embodied/franka_reward_model`）——
  将 reward model 接入物理 Franka 上的完整 RL 训练循环。

真机遥操作 + 在线 Reward Model 推理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

获得 reward model checkpoint 后，``examples/reward/eval_realworld_teleop.py`` 提供了一种
遥操作模式：SpaceMouse 控制机器人运动，reward model 在 GPU 节点上运行，
实时在终端打印每步成功概率。

此功能的适用场景：

- 对 reward model 在真实机器人观测上的准确性进行冒烟测试（sanity check）。
- 采集符合人类判断的成功/失败数据，用于进一步扩充数据集。
- 定性评估 reward model 对当前场景的泛化能力。

集群配置
^^^^^^^^

遥操作脚本需要两个节点：一个用于 Franka 机器人，一个用于运行 reward model 推理的 GPU：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
       reward:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

Reward worker 被部署在 GPU 节点（``"4090"``）上，与机器人节点（``franka``）上的遥操作 worker 分离。
这是一种解聚式部署（disaggregated placement）。

配置文件
^^^^^^^^

默认配置为 ``examples/reward/config/realworld_teleop.yaml``，
环境参数从 ``env/realworld_bin_relocation.yaml`` 继承：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
       reward:
         node_group: "4090"
         placement: 0
     node_groups:
       - label: "4090"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   env:
     group_name: "EnvGroup"
     eval:
       no_gripper: True
       use_spacemouse: True
       max_episode_steps: 10000
       override_cfg:
         target_ee_pose: TARGET_EE_POSE
         camera_serials: ["0123456789"]

   reward:
     use_reward_model: True
     use_reward_prob: True    # 打印每步原始 sigmoid 概率到终端
     standalone_realworld: True
     reward_mode: "per_step"
     reward_threshold: 0.2
     model:
       model_path: path/to/reward_model_checkpoint
       model_type: "resnet"
       arch: "resnet18"
       image_size: [3, 128, 128]

关键配置字段说明：

- ``reward.use_reward_model: True`` — 启用 reward model 推理。
- ``reward.use_reward_prob: True`` — 每步将原始 sigmoid 概率打印到终端。
- ``reward.standalone_realworld: True`` — 利用 reward model 直接判断成功/失败并触发重置。
- ``reward.reward_threshold`` — 概率阈值，低于该值的成功判定将被抑制。根据模型校准情况调整。
- ``reward.model.model_path`` — 指向训练好的 reward model checkpoint。

启动
^^^^

设置环境变量并运行：

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh

或显式指定配置名称：

.. code-block:: bash

   bash examples/reward/run_realworld_teleop.sh realworld_teleop

终端每步输出如下：

.. code-block:: text

   [TeleopWorker] Starting teleoperation loop.
   [TeleopWorker] EmbodiedRewardWorker ready: type=EmbodiedRewardWorker | reward_threshold=0.200
   Step 0      | rm_reward: 0 | success: False
   Step 1      | rm_reward: 0 | success: False
   Step 10     | rm_reward: 0 | success: False
   Step 123    | rm_reward: 1 | success: True
   Step 124    | rm_reward: 1 | success: True

SpaceMouse 控制说明：

- **移动** — 遥操作机械臂。
- **左键** — 合拢夹爪。
- **右键** — 张开夹爪。
- **Ctrl+C** — 停止。

工作原理
^^^^^^^^

``TeleopWorker`` 内部流程：

1. ``RealWorldEnv`` 以 ``use_spacemouse=True`` 初始化，包装了 ``SpacemouseIntervention``。
   当 SpaceMouse 输入非零（或按下按钮）时，用 SpaceMouse 动作覆盖零 dummy 动作，持续 0.5 秒。
2. ``EmbodiedRewardWorker`` 通过 ``EmbodiedRewardWorker.launch_for_realworld(...)``
   在 GPU 节点上启动，在启动时一次性完成初始化。
3. 每步遥操作中，从观测中提取腕部相机图像（``obs["main_images"]``）并发送给 reward worker 进行推理。
4. 原始 sigmoid 概率被打印到终端。当 ``standalone_realworld=True`` 时，
   reward model 还直接驱动成功/失败判定和环境重置。

与 :doc:`../examples/embodied/franka_reward_model` 中的完整 RL 流程相比，
遥操作脚本不运行策略、actor 或 rollout worker——它纯粹是人在回路的 reward model 评估。
