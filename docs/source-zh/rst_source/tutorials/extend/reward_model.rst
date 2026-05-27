Reward Model 使用指南
========================

本文档介绍如何在 RLinf 中使用 reward model，覆盖 ``ResNetRewardModel``
这类图像分类 reward，以及 QwenTrend / ``HistoryVLMRewardModel`` 这类 VLM reward。
这里的 QwenTrend 指使用 Qwen3-VL 模型判断一段历史视频中的动作趋势，并据此转换为标量 reward。

完整流程包括四个阶段：

1. 数据收集：在 RL 运行过程中采集原始 episode 数据。
2. 数据转换：将原始 episode 转成图像分类数据或 VLM SFT 数据。
3. Reward model 训练：训练 ResNet reward model，或微调 VLM reward model。
4. Reward model 在 RL 中推理：将训练好的模型接入在线 rollout，参与最终 reward 计算。

1. 数据收集
----------------------------

reward model 的训练数据通常来自 episode 级数据采集。RLinf 提供了统一的数据采集封装，
相关用法可参考 :doc:`数据采集教程 <../components/data_collection>`。

对于 reward model 场景，建议先以 ``pickle`` 格式保存原始 episode 数据，再通过预处理脚本转换为训练集。

1.1 启用数据采集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

对于 QwenTrend VLM reward，RLinf 也提供了可直接运行的数据采集配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_collect

该配置保持 ``reward.use_reward_model: false``，并在 eval 环境上开启数据采集。
保存下来的 episode 会包含 VLM 流程后续需要的双视角图像观测，例如
``main_images`` 和 ``extra_view_images``。

1.2 预处理为 ResNet reward dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

1.3 转换为 QwenTrend VLM dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QwenTrend 使用短时间双视角历史窗口，而不是单张图像。使用
``examples/reward/preprocess_qwentrend_reward_dataset.py`` 可以将采集到的
episode 切成 5 帧窗口，提取 ``main_images`` 和 ``extra_view_images``，并给每个
窗口标注 ``positive``、``negative`` 或 ``unclear``。

命令示例：

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
       --raw-data-path logs/xxx/collected_data \
       --output-dir logs/xxx/processed_qwentrend_reward_data \
       --window-size 5 \
       --stride 1 \
       --delta-threshold 0.05

默认会生成 JSONL manifest 和逐样本 pickle 文件：

.. code-block:: text

   logs/xxx/processed_qwentrend_reward_data/
   ├── dataset_info.json
   ├── train/
   │   ├── segments.jsonl
   │   └── pkl/
   └── eval/
       ├── segments.jsonl
       └── pkl/

train/eval 按 episode 划分，因此同一个 episode 中切出的窗口不会混到不同 split 中。

2. Reward Model 训练
----------------------------

RLinf 支持两条 reward 训练路径。``examples/reward/run_reward_training.sh``
用于训练 ResNet 图像 reward model，``examples/sft/run_vlm_sft.sh``
用于微调 QwenTrend 这类 VLM reward model。

2.1 微调 ResNet Reward Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.1.1 配置 ResNet 数据路径
"""""""""""""""""""""""""""

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

2.1.2 配置 ResNet 模型
""""""""""""""""""""""

对于 ResNet 路径，需要将 ``actor.model.model_type`` 设置为 ``"resnet"``：

.. code-block:: yaml

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 128, 128]

如果需要从已有权重继续训练，可以通过 ``model_path`` 指定 checkpoint；
如果希望从头训练，则保持 ``model_path: null``。

在线 reward worker 的模型注册表目前包含以下类型：

.. code-block:: python

   reward_model_registry = {
       "resnet": ResNetRewardModel,
       "vlm": VLMRewardModel,
       "history_vlm": HistoryVLMRewardModel,
   }

``resnet`` 是图像分类 reward 路径；``vlm`` 会基于当前观测运行 VLM；
``history_vlm`` 会基于 env worker 维护的历史窗口运行 VLM。

2.1.3 启动 ResNet 训练
""""""""""""""""""""""

完成数据与模型配置后，执行：

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

训练日志会保存到新建的 ``logs/<timestamp>-reward_training`` 目录下。

2.2 微调 QwenTrend VLM Reward Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``preprocess_qwentrend_reward_dataset.py`` 转换数据后，将
``DUALVIEW_SFT_DATA_ROOT`` 指向处理后的数据根目录，然后启动 VLM SFT：

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_qwentrend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

对应配置会读取 JSONL manifest 和逐样本 pickle 文件：

.. code-block:: yaml

   data:
     type: vlm
     dataset_name: "qwentrend_progress_sft"
     train_data_paths: "${oc.env:DUALVIEW_SFT_DATA_ROOT}/train/segments.jsonl"
     val_data_paths: "${oc.env:DUALVIEW_SFT_DATA_ROOT}/eval/segments.jsonl"
     video_root: "${oc.env:DUALVIEW_SFT_DATA_ROOT}"
     video_nframes: 5

   actor:
     model:
       model_type: qwen3_vl
       model_path: /path/to/Qwen3-VL-4B-Instruct
       attn_implementation: flash_attention_2
       is_lora: true
       lora_rank: 16

训练得到的 LoRA checkpoint 后续可通过 ``reward.model.lora_path`` 传给在线 reward 配置。

3. Reward Model 在 RL 中推理
----------------------------

RLinf 提供了多个 reward model 接入 RL 的示例配置：

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``
- ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``（设置 ``inference_backend: hf``）

这些配置展示了如何在 RL 训练中启用 reward worker，同时让策略网络继续使用状态观测，
而 reward model 使用图像观测或 VLM 观测。

3.1 基本配置项
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
       model_type: "resnet"    # 或 "vlm" / "history_vlm"

其中：

- ``reward_mode`` 控制 reward model 在每一步、终止帧，还是历史窗口上推理。
- ``reward_weight`` 和 ``env_reward_weight`` 控制 learned reward 与环境 reward 的加权组合。
- ``reward_threshold`` 用于对 reward model 输出的成功概率做阈值过滤；低于阈值的项会被置为 ``0``。
- ``model_path`` 指向用于在线推理的 reward model 权重。

3.2 Rollout 阶段的 worker 交互
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在线 RL 阶段，``env``、``rollout``、``reward`` 三类 worker 会协同工作。整体流程如下：

.. code-block:: text

   Env worker
      | 1. 与环境交互，获得 obs / env reward / done
      | 2. 将 obs 发送给 Rollout worker 生成动作
      | 3. 当启用 reward model 时，将 reward input dict 发送给 Reward worker
      v
   Reward worker
      | 4. 执行 ``compute_reward(...)``，返回 reward model output
      v
   Env worker
      | 5. 接收 Rollout worker 的 bootstrap values
      | 6. 将 env reward 与 reward model output 组合
      v
   Final reward -> 写入 rollout 结果并参与后续 RL 更新

在实现上，``EnvWorker`` 会在 rollout 过程中向 reward worker 请求 reward model 输出，
再统一计算最终 reward。

3.3 最终 reward 的计算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当 reward channel 已启用时，``EnvWorker`` 会先获取 ``reward_model_output``，
随后在 ``compute_bootstrap_rewards`` 中与环境原始 reward 合并：

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

之后，若当前算法配置启用了 bootstrap，RLinf 还会按配置将 bootstrap value 加到最后一步 reward 中。

因此，从系统视角看，reward model 在 RL 中并不会替代原有的 bootstrap reward，
而是作为 env worker 中的附加 reward 来源参与最终 reward 的构造。

3.4 部署 QwenTrend 进行 MLP RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

进行 VLM reward 推理前，需要安装带 VLM reward 支持的 embodied 依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --env maniskill_libero --model qwen3_vl

这会安装 MLP policy 示例所需的 Qwen3-VL reward runtime，并包含 SGLang
reward 后端所需的依赖。

随后在 reward 配置中使用 ``model_type: history_vlm``。设置
``inference_backend: hf`` 会使用 Hugging Face/Transformers 后端；设置
``inference_backend: sglang`` 会启用 SGLang 后端。QwenTrend 示例使用
``reward_mode: history_buffer``，因此 env worker 会按 env 维护历史窗口，只在窗口
有效时将历史输入发送给 reward worker：

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
       model_type: "history_vlm"
       inference_backend: hf
       gt_success_bonus: 20.0
       precision: "bf16"
       input_builder_name: qwentrend_input_builder
       input_builder_params:
         default_task_description: "Pick up the red cube and place it on the green spot on the table."
       reward_parser_name: qwentrend_reward_parser
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
       use_chat_template: true

关键字段说明：

- ``model_type`` 选择 reward model 类型。历史窗口 VLM reward 使用 ``history_vlm``。
- ``inference_backend`` 选择推理后端。设置为 ``hf`` 时使用
  Hugging Face/Transformers，设置为 ``sglang`` 时使用 SGLang。
- ``history_buffers`` 定义需要缓存的 observation key、窗口长度和最小有效历史长度。
- ``input_builder_name`` 将历史窗口转换为双视角 VLM 输入。
- ``reward_parser_name`` 将模型生成的标签映射为标量 reward，标量由 ``positive_reward``、``negative_reward``、``unclear_reward`` 和 ``invalid_reward`` 控制。
- ``gt_success_bonus`` 可以从环境 info 中读取成功信号并额外加分。

使用 Hugging Face/Transformers 后端时，保留 ``inference_backend: hf``，
并按需配置 ``lora_path``：

.. code-block:: yaml

   reward:
     model:
       model_type: "history_vlm"
       inference_backend: hf
       lora_path: "/path/to/qwen3-vl-lora-checkpoint"

SGLang 后端复用同一套 QwenTrend input builder、reward parser 和 history
buffer 配置。当设置 ``inference_backend: sglang`` 时，embodied runner 会启动一个
外部 ``python -m sglang.launch_server`` 进程，reward worker 通过其
OpenAI-compatible ``/v1/chat/completions`` API 同步请求推理。历史帧会作为同一条
chat message 中的多个 ``image_url`` data URL 发送，因此该后端不再使用 in-process
``sglang.Engine``，也不再生成临时 MP4/video 输入。

通过覆盖后端字段选择 SGLang：

.. code-block:: yaml

   reward:
     model:
       model_type: "history_vlm"
       inference_backend: sglang

runner 默认启动本地 server。可以覆盖 server 参数，或指定用户自行管理的 server：

.. code-block:: yaml

   reward:
     model:
       model_type: "history_vlm"
       inference_backend: sglang
       server_port: 30000
       server_startup_timeout: 600
       served_model_name: qwentrend-reward
       # api_base: "http://127.0.0.1:30000/v1"  # 设置后跳过 runner 启动
       # sglang_server_args:
       #   tp_size: 2

启动 MLP RL：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

总结
----------------------------

完整工作流如下：

1. 在环境配置中开启 ``data_collection``，并将数据保存为 ``pickle`` 格式。
2. 对于 ResNet reward，使用 ``preprocess_reward_dataset.py`` 构建 ``train.pt`` / ``val.pt``，再用 ``run_reward_training.sh`` 训练。
3. 对于 QwenTrend VLM reward，使用 ``preprocess_qwentrend_reward_dataset.py`` 构建双视角历史窗口数据，再用 ``run_vlm_sft.sh`` 微调。
4. 在 RL YAML 中开启 ``reward.use_reward_model=True``，并通过示例配置接入 reward worker 完成在线推理。
