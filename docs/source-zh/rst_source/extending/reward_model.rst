Reward Model 使用指南
======================

在 RLinf 中使用 reward model——包括 ``ResNetRewardModel`` 这类图像分类 reward，
以及基于 ``VLMRewardModel`` 的 VLM reward。``BufferedVLMRewardModel`` 继承
``VLMRewardModel``，用于处理 env worker 维护的历史窗口。

仿真场景 Reward Model
---------------------

推荐的 VLM Trend Success 流程
-----------------------------

在 ``BufferedVLMRewardModel`` 上用两路信号搭 PPO reward：稀疏的终局成功 LoRA，
加上稠密的 potential head（``inference_mode=scalar_head``）。标签只来自环境
success 和轨迹时序，不写死任务规则。该双线路由与本指南后文的单线 VLM Trend
reward（``inference_mode=generate`` + ``vlm_trend_reward_parser``）不同。

所有命令都在仓库根目录执行。稀疏支路（步骤 2–3）和稠密支路（步骤 4–6）都依赖
步骤 1；采完数据后两条支路可以并行。

下面的示例默认使用 4 张 GPU（placement ``0-3``）和 ``NUM_ENVS=1024``。
训练与 PPO 使用仓库统一入口加 YAML config-name
（``run_vlm_sft.sh``、``run_embodiment.sh``）。数据集 / teacher / scalar-head
步骤使用 ``examples/reward/`` 下统一 CLI
（``preprocess_vlm_trend_dataset.py``、``run_vlm_trend.py``、
``run_vlm_trend_scalar_head.py``）。

本指南用到的 ``preprocess_vlm_trend_dataset.py`` mode：

- ``terminal_success`` — 稀疏 0/1 窗口（步骤 2）
- ``potential`` — 由 state-value teacher 生成稠密 potential **与** progress
  窗口（步骤 4）；双线 Success 流程请用这个
- ``trend_reward`` — 经典 GAE-delta trend reward 窗口（后文单独章节）

``--mode progress`` 是兼容用的 **legacy** 路径（旧的 state-value delta
progress 标注），不属于推荐的 Success 流程；新实验请用 ``--mode potential``。

步骤 1 — 采集 rollout
^^^^^^^^^^^^^^^^^^^^^

从覆盖策略分布的 checkpoint 各跑一轮固定 50 步轨迹。

.. code-block:: bash

   export CHECKPOINT_TEMPLATE_EARLY='/path/to/clean_gt_0_120/checkpoints/global_step_%d/actor/model_state_dict/full_weights.pt'
   export CHECKPOINT_TEMPLATE_LATE='/path/to/clean_gt_0_200/checkpoints/global_step_%d/actor/model_state_dict/full_weights.pt'
   export OUTPUT_ROOT=/path/to/vlm_trend_uniform_collection
   export CUDA_DEVICES=0,1,2,3
   export PLACEMENT=0-3
   export NUM_ENVS=1024
   export SEED=0
   PYTHON_BIN=${PYTHON_BIN:-python}
   for step in 0 20 40 60 80 100 120 140 160 180 200; do
     if ((step == 0)); then checkpoint=null
     elif ((step <= 120)); then checkpoint=$(printf "${CHECKPOINT_TEMPLATE_EARLY}" "${step}")
     else checkpoint=$(printf "${CHECKPOINT_TEMPLATE_LATE}" "${step}"); fi
     run_dir="${OUTPUT_ROOT}/runs/step${step}_seed${SEED}_env${NUM_ENVS}"
     data_dir="${OUTPUT_ROOT}/step${step}"
     mkdir -p "${run_dir}" "${data_dir}"
     CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
       EMBODIED_PATH="${PWD}/examples/embodiment" \
       "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
       --config-path ../examples/embodiment/config \
       --config-name maniskill_ppo_mlp_vlm_trend_reward_collect \
       runner.only_eval=true \
       runner.ckpt_path="${checkpoint}" \
       runner.logger.log_path="${run_dir}" \
       cluster.component_placement.env="${PLACEMENT}" \
       cluster.component_placement.rollout="${PLACEMENT}" \
       'rollout.model=${actor.model}' \
       rollout.enable_torch_compile=false \
       rollout.enable_cuda_graph=false \
       env.eval.total_num_envs="${NUM_ENVS}" \
       env.eval.seed="${SEED}" \
       env.eval.wrap_obs_mode=simple \
       env.eval.ignore_terminations=true \
       env.eval.max_episode_steps=50 \
       env.eval.max_steps_per_rollout_epoch=50 \
       env.eval.data_collection.enabled=true \
       env.eval.data_collection.save_dir="${data_dir}" \
       env.eval.data_collection.only_success=false
   done

这一步会：

- 把 episode pickle 写到 ``${OUTPUT_ROOT}/step0``、``step20``、…、``step200``。
- step 0 是随机策略；20–120 用 early 模板；140–200 用 late 模板。每个
  checkpoint 固定 seed 0、四卡 Ray 任务。
- 使用 ``simple`` observation，忽略提前终止，始终跑满 50 步，避免“很快成功”
  被存成短失败样本。

只重采失败的几步时：

.. code-block:: bash

   # Re-run only failed steps by shrinking the ``for step in ...`` list, e.g.
   # ``for step in 80 120 160; do ...; done``

同名 episode 文件会被覆盖。

步骤 2 — 构建稀疏 success SFT 数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

把采集结果打成双视角、5 帧窗口，标签为 ``0`` / ``1``。

.. code-block:: bash

   export UNIFORM_DATA_ROOT=/path/to/vlm_trend_uniform_collection
   export DUALVIEW_SFT_DATA_ROOT=/path/to/vlm_trend_success_sft
   python examples/reward/preprocess_vlm_trend_dataset.py \
       --mode terminal_success \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step0" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step20" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step40" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step60" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step80" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step100" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step120" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step140" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step160" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step180" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step200" \
       --output-dir "${DUALVIEW_SFT_DATA_ROOT}" \
       --window-size 5 \
       --online-interval 5 \
       --workers 32

这一步会：

- 先按 source episode 划分 train/eval（有泄漏会直接退出）。
- 在 observation 索引 ``5, 10, …, 50`` 各生成一个窗口；标签来自
  ``infos[end_step]["success"]``，保留自然类别比例。
- 在 ``${DUALVIEW_SFT_DATA_ROOT}/{train,eval}/`` 写 manifest，引用原始 pickle
  （不复制图像）。只加载可信 pickle。

步骤 3 — 训练稀疏 success LoRA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

微调 Qwen3-VL-4B，让它输出终局成功的 ``0`` / ``1``
（配置 ``vlm_trend_sft_success``）。

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/vlm_trend_success_sft
   export VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   # 可选: export OUTPUT_ROOT=/path/to/vlm_trend_sparse_success_sft
   bash examples/sft/run_vlm_sft.sh vlm_trend_sft_success

这一步会：

- 通过共享的 ``run_vlm_sft.sh`` + YAML 默认超参做 LoRA SFT（micro 4 /
  global 256、400 步、类别加权 success loss、warmup 20）。
- 按 **balanced accuracy** 选 checkpoint（同时看 positive recall 和 negative
  accuracy），不要只看整体 accuracy。
- 选出的目录稍后设为 ``VLM_TREND_SUCCESS_CHECKPOINT``。
  SFT 会保留 ``actor/model_state_dict/full_weights.pt`` 作为完整模型权重，
  并通过 ``PeftModel.save_pretrained`` 把 Peft adapter 写到
  ``actor/lora_adapter/``。共享加载逻辑在 ``rlinf/utils/lora_adapter.py``，
  会读取该 adapter 产物（并兼容旧版 ``full_weights.pt`` fallback）。

步骤 4 — 构建稠密 potential SFT 数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

先训一个小的 state-value teacher，再用同一批采集数据生成 potential / progress
窗口。

.. code-block:: bash

   export UNIFORM_DATA_ROOT=/path/to/vlm_trend_uniform_collection
   export STATE_VALUE_ROOT=/path/to/vlm_trend_state_success_value
   export POTENTIAL_SFT_DATA_ROOT=/path/to/vlm_trend_potential_sft
   export FLAT_ROOT=/path/to/vlm_trend_uniform_collection_flat
   mkdir -p "${FLAT_ROOT}"
   for step in 0 20 40 60 80 100 120 140 160 180 200; do
     for f in "${UNIFORM_DATA_ROOT}/step${step}"/*.pkl; do
       ln -s "$(realpath "$f")" "${FLAT_ROOT}/step${step}_$(basename "$f")"
     done
   done
   python examples/reward/run_vlm_trend.py \
       --stage train_teacher \
       --raw-data-path "${FLAT_ROOT}" \
       --output-dir "${STATE_VALUE_ROOT}"
   python examples/reward/preprocess_vlm_trend_dataset.py \
       --mode potential \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step0" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step20" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step40" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step60" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step80" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step100" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step120" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step140" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step160" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step180" \
       --raw-data-path "${UNIFORM_DATA_ROOT}/step200" \
       --value-checkpoint "${STATE_VALUE_ROOT}/best.pt" \
       --output-dir "${POTENTIAL_SFT_DATA_ROOT}"

这一步会：

- 产出 ``${STATE_VALUE_ROOT}/best.pt``，并把 potential SFT manifest 写到
  ``${POTENTIAL_SFT_DATA_ROOT}``。

步骤 5 — 训练稠密 potential LoRA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 potential / progress 标签上微调 Qwen3-VL
（配置 ``vlm_trend_sft_potential``）。

.. code-block:: bash

   export VLM_TREND_REWARD_DATA_ROOT=/path/to/vlm_trend_potential_sft
   export VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   # 可选: export OUTPUT_ROOT=/path/to/vlm_trend_dense_potential_sft
   bash examples/sft/run_vlm_sft.sh vlm_trend_sft_potential

这一步会：

- 超参与步骤 3 同类（YAML 默认：micro 4 / global 256、400 步），走
  potential 配置（读取 ``VLM_TREND_REWARD_DATA_ROOT``）。
- 在 SFT 日志目录下选出 checkpoint，供步骤 6–7 使用。该目录保留完整的
  ``full_weights.pt``，并单独写出 ``actor/lora_adapter/`` 供 reward /
  特征提取加载。

步骤 6 — 训练 scalar potential head
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

冻结 potential LoRA，用 ``VLMRewardModel.extract_prompt_features`` 抽特征，
再训一个小的标量头。

.. code-block:: bash

   export VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   export POTENTIAL_SFT_DATA_ROOT=/path/to/vlm_trend_potential_sft
   export VLM_TREND_POTENTIAL_CHECKPOINT=/path/to/potential_lora_ckpt
   export FEAT_ROOT=/path/to/vlm_trend_potential_features
   export SCALAR_OUTPUT_ROOT=/path/to/vlm_trend_scalar_head
   python examples/reward/run_vlm_trend_scalar_head.py \
       --stage all \
       --model-path "${VLM_MODEL_PATH}" \
       --checkpoint "${VLM_TREND_POTENTIAL_CHECKPOINT}" \
       --data-root "${POTENTIAL_SFT_DATA_ROOT}" \
       --feat-root "${FEAT_ROOT}" \
       --output-dir "${SCALAR_OUTPUT_ROOT}" \
       --cuda-devices 0,1,2,3

这一步会：

- 多卡分片抽特征，训练得到 ``${SCALAR_OUTPUT_ROOT}/best.pt``。

步骤 7 — 跑 PPO
^^^^^^^^^^^^^^^

把两路 reward 接到 embodied PPO，从策略 checkpoint 起步。
在线推理使用 ``vlm_trend_reward_input_builder``（各路径通过
``prompt_template`` 区分），稠密项为 ``inference_mode=scalar_head``，
稀疏 success 项使用 ``vlm_trend_binary_digit_reward_parser``。

.. code-block:: bash

   export VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   export VLM_TREND_POTENTIAL_CHECKPOINT=/path/to/potential_lora_ckpt
   export VLM_TREND_SCALAR_HEAD=/path/to/vlm_trend_scalar_head/best.pt
   export VLM_TREND_SUCCESS_CHECKPOINT=/path/to/vlm_trend_sparse_success_sft/.../global_step_300
   export POLICY_CHECKPOINT=/path/to/policy/full_weights.pt
   export PPO_OUTPUT_ROOT=/path/to/ppo-output
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_vlm_trend_success

这一步会：

- 稠密项：potential LoRA + scalar head → 受限的 potential difference。
- 稀疏项：success LoRA 生成 ``1`` 时发一次 ``+1``（``0`` / 非法输出为 0）。
  环境 reward 关闭；VLM 每 5 步推理一次；episode 固定 50 步，与预处理对齐。
- 每 5 步评估、每 20 步保存；默认把 actor / rollout / env / reward 放在 GPU
  ``0-3``。可用 Hydra ``cluster.component_placement.*=...`` 覆盖。
- 续训：传 ``runner.resume_dir=/path/to/checkpoints/global_step_N``，并提高
  ``runner.max_steps`` 到目标总步数。


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
``examples/reward/preprocess_vlm_trend_dataset.py --mode trend_reward`` 可以将采集到的
episode 切成 5 帧窗口，提取 ``main_images`` 和 ``extra_view_images``，并给每个
窗口标注 ``positive``、``negative`` 或 ``unclear``。

命令示例：

.. code-block:: bash

   python examples/reward/preprocess_vlm_trend_dataset.py \
       --mode trend_reward \
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
   }

其中：

- ``resnet``：单帧图像二分类，输出 sigmoid 概率。
- ``vlm``：对当前观测运行 VLM（单步/终止帧等，由 ``reward_mode`` 决定）。
- ``buffered_vlm``：对 env worker 维护的历史窗口运行 VLM；具体 prompt、视频格式与
  标量映射由 ``input_builder_name`` / ``reward_parser_name`` 决定。
  VLM Trend reward 即 ``buffered_vlm`` + ``vlm_trend_reward_*`` 插件组合。

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

使用 ``preprocess_vlm_trend_dataset.py --mode trend_reward`` 转换数据后，将
``VLM_TREND_REWARD_DATA_ROOT`` 指向处理后的数据根目录，然后启动 VLM SFT。

``vlm_trend_sft_reward.yaml`` 是 **单线** Trend reward
（``inference_mode=generate`` + ``vlm_trend_reward_parser``）的默认配置
（micro 4 / global 256、``max_steps=3000``）。如需调整训练预算，可继续加
override，例如：

.. code-block:: bash

   export VLM_TREND_REWARD_DATA_ROOT=/path/to/processed_vlm_trend_reward_data
   export VLM_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
   # 可选: export OUTPUT_ROOT=/path/to/vlm_trend_reward_sft
   bash examples/sft/run_vlm_sft.sh vlm_trend_sft_reward \
       runner.max_steps=3000 \
       runner.max_epochs=3000 \
       actor.optim.total_training_steps=3000

双线 Success 稠密支路（步骤 5）请改用专用配置
``vlm_trend_sft_potential.yaml``（400 步），与本节的单线配置互不影响。

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

.. note::

   SFT 会保留框架的 ``actor/model_state_dict/full_weights.pt``，并通过
   ``PeftModel.save_pretrained`` 在 ``actor/lora_adapter/`` 下单独导出 Peft
   adapter（``adapter_config.json`` + ``adapter_model.safetensors``）。在线 reward
   模型通过共享的 ``rlinf.utils.lora_adapter`` API 把 ``lora_path`` 解析到该
   adapter 产物（并保留对旧版 ``full_weights.pt`` 的回退）。

   与 ``rlinf.models.apply_lora`` 的分工：

   * ``apply_lora`` — 在根据 ``cfg.actor.model`` 构造训练 actor 时挂载或创建
     LoRA（``is_lora``、``lora_path`` 等）。
   * ``rlinf.utils.lora_adapter`` — 解析 SFT checkpoint 目录布局
     （``global_step_*/actor/lora_adapter``），在保留 ``full_weights.pt`` 的同时
     导出 adapter，并将这些产物（或 legacy ``full_weights.pt``）加载到冻结的
     reward 模型与离线特征脚本中。

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

VLM Trend reward 使用 buffered VLM 接口（ ``model_type: buffered_vlm`` ），并通过
``vlm_trend_reward_input_builder`` 和 ``vlm_trend_reward_parser`` 构造输入、解析 reward。
本地推理会实例化 ``BufferedVLMRewardModel`` ；API 推理则使用
``EmbodiedAPIRewardWorker`` ，并遵循相同的 input builder 和 reward parser 约定。

VLM Trend reward 在线推理共用以下核心字段（ ``model_type`` 始终为 ``buffered_vlm`` ）：

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

不设置 ``reward.worker_type``（默认 ``model``，使用 ``EmbodiedRewardWorker``）。
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

3.4.2 SGLang API 推理
.....................

设置 ``reward.worker_type: api``（``EmbodiedAPIRewardWorker``）。可指向外部
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
3. 对于 VLM Trend reward，使用 ``preprocess_vlm_trend_dataset.py --mode trend_reward`` 构建双视角历史窗口数据，再用 ``run_vlm_sft.sh`` 微调。
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
