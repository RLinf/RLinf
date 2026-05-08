Reward Model 使用指南
========================

本文档介绍如何在 RLinf 中使用 reward model，内容包括三个部分：

1. 数据收集：在 RL 运行过程中采集原始 episode 数据。
2. Reward model 训练：将原始数据预处理后训练图像或 VLM 奖励模型。
3. Reward model 在 RL 中推理：将训练好的模型接入在线 rollout，参与最终 reward 计算。

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

1.2 预处理为 reward dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

2. Reward Model 训练
----------------------------

RLinf 提供了训练脚本 ``examples/reward/run_reward_training.sh``，
用于启动 reward model 训练流程。

2.1 配置数据路径
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

2.2 配置模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于图像分类式 reward 路径，使用 ``ResNetRewardModel``，并将
``actor.model.model_type`` 设置为 ``"resnet"``：

.. code-block:: yaml

   actor:
     model:
       model_type: "resnet"
       arch: "resnet18"
       pretrained: False
       image_size: [3, 128, 128]

如果需要从已有权重继续训练，可以通过 ``model_path`` 指定 checkpoint；
如果希望从头训练，则保持 ``model_path: null``。

对于 VLM reward，在线 reward worker 还可以使用 ``vlm``、``history_vlm`` 或
``history_vlm_sglang`` 等冻结推理模型。QwenTrend ManiSkill 流程使用 VLM SFT runner
训练 VLM LoRA，而不是通过 ``run_reward_training.sh`` 训练；数据转换与 SFT 步骤见
:doc:`../../examples/embodied/maniskill_vlm_reward`。

2.3 启动训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

完成数据与模型配置后，执行：

.. code-block:: bash

   bash examples/reward/run_reward_training.sh

训练日志会保存到新建的 ``logs/<timestamp>-reward_training`` 目录下。

3. Reward Model 在 RL 中推理
----------------------------

RLinf 提供了两个 reward model 接入 RL 的示例配置：

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``
- ``examples/embodiment/config/maniskill_sac_mlp_resnet_reward_async.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``

这两个配置展示了如何在 RL 训练中启用 reward worker，同时让策略网络继续使用状态观测，
而 reward model 使用图像观测。

3.1 基本配置项
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 RL 配置中，reward model 相关参数位于 ``reward`` 段：

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     reward_mode: "terminal"   # 或 "per_step"
     reward_threshold: 0.5
     reward_weight: 1.0
     env_reward_weight: 0.0

     model:
       model_path: /path/to/reward_model_checkpoint
       model_type: "resnet"

其中：

- ``reward_mode`` 控制 reward model 在每一步推理，还是仅在终止帧推理。
- ``reward_weight`` 和 ``env_reward_weight`` 控制 learned reward 与环境 reward 的加权组合。
- ``reward_threshold`` 用于对 reward model 输出的成功概率做阈值过滤；低于阈值的项会被置为 ``0``。
- ``model_path`` 指向用于在线推理的 reward model 权重。

对于基于历史片段的 VLM reward model，使用 ``reward_mode: history_buffer``，并在
``reward.model.history_buffers`` 下配置历史窗口：

.. code-block:: yaml

   reward:
     use_reward_model: True
     group_name: "RewardGroup"
     reward_mode: history_buffer
     history_reward_assign: True
     reward_weight: 1.0
     env_reward_weight: 0.0

     model:
       model_type: history_vlm        # 或 history_vlm_sglang
       model_path: /path/to/Qwen3-VL-4B-Instruct
       lora_path: /path/to/qwen3-vl-reward-lora
       input_builder_name: qwentrend_input_builder
       reward_parser_name: qwentrend_reward_parser
       history_buffers:
         history_window:
           history_size: 5
           min_history_size: 5
           input_interval: 1
           history_keys: [main_images, extra_view_images]
           input_on_done: false

``history_vlm`` 会在 reward worker 进程内通过 Hugging Face 加载 VLM。
``history_vlm_sglang`` 使用相同的 history/input/parser 约定，但通过 SGLang 版 reward
后端完成生成；server host、port 和 model name 等字段按该后端配置填写。

3.2 Rollout 阶段的 worker 交互
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在线 RL 阶段，``env``、``rollout``、``reward`` 三类 worker 会协同工作。整体流程如下：

.. code-block:: text

   Env worker
      | 1. 与环境交互，获得 obs / env reward / done
      | 2. 将 obs 发送给 Rollout worker 生成动作
      | 3. 当启用 reward model 时，将 ``main_images`` 发送给 Reward worker
      v
   Reward worker
      | 4. 对图像做前向推理，返回 reward model output
      v
   Env worker
      | 5. 接收 Rollout worker 的 bootstrap values
      | 6. 将 env reward 与 reward model output 组合
      v
   Final reward -> 写入 rollout 结果并参与后续 RL 更新

在实现上，``EnvWorker`` 会在 rollout 过程中向 reward worker 请求 reward model 输出，
再统一计算最终 reward。

在 async 具身训练中，runner 会启动一次 ``EmbodiedRewardWorker.compute_rewards_async``。
之后 reward worker 会常驻，并消费 env worker 通过 ``train_reward_input`` channel key
发送的排队 reward 输入，再通过 ``reward_output`` key 返回切分后的结果。当 async runner
配置了 ``reward.use_reward_model=True`` 时，ResNet 与 VLM reward worker 都走这一 queued 路径。

3.3 最终 reward 的计算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当 reward channel 已启用时，``EnvWorker`` 会先获取 ``reward_model_output``，
随后在 ``compute_bootstrap_rewards`` 中与环境原始 reward 合并：

.. code-block:: python

   reward = env_reward_weight * env_reward + reward_weight * reward_model_output

之后，若当前算法配置启用了 bootstrap，RLinf 还会按配置将 bootstrap value 加到最后一步 reward 中。

因此，从系统视角看，reward model 在 RL 中并不会替代原有的 bootstrap reward，
而是作为 env worker 中的附加 reward 来源参与最终 reward 的构造。

总结
----------------------------

完整工作流如下：

1. 在环境配置中开启 ``data_collection``，并将数据保存为 ``pickle`` 格式。
2. 对 ResNet reward，使用 ``examples/reward/preprocess_reward_dataset.py`` 将原始 ``.pkl`` 转成 ``RewardDatasetPayload`` 格式的 ``train.pt`` / ``val.pt``；对历史 VLM reward，使用 VLM 专用预处理流程。
3. 在 ``reward_training.yaml`` 中完成数据与模型配置后，使用 ``examples/reward/run_reward_training.sh`` 启动 ``ResNetRewardModel`` 训练；VLM reward 则通过 VLM SFT runner 训练 LoRA。
4. 在 RL YAML 中开启 ``reward.use_reward_model=True``，并接入 reward worker 完成在线推理。进程内 Hugging Face serving 使用 ``history_vlm``，SGLang 版历史 VLM serving 使用 ``history_vlm_sglang``。
