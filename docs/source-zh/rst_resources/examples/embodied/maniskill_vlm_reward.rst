ManiSkill PPO（基于 VLM Reward Model）
========================================

本文档给出在 RLinf 框架内使用 **MLP policy + Qwen3-VL reward model** 运行 ManiSkill PPO 训练的完整说明。
主要参考配置为 ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``。

主要目标是让训练流程具备以下能力：

1. **状态策略学习**：actor 仍然基于 ``states`` 使用轻量 ``mlp_policy`` 生成动作。
2. **视觉奖励判断**：reward worker 使用 Qwen3-VL 对图像历史片段和任务描述进行判断。
3. **历史片段打分**：通过 ``history_buffer`` 模式为一小段轨迹分配 learned reward。
4. **后端选择**：reward model 可使用进程内 Hugging Face 后端（``history_vlm``），
   也可在配置对应后端时使用进程内 ``sglang.Engine`` 后端（``history_vlm_sglang``）。
5. **强化学习优化**：结合 PPO 与 reward worker 输出更新策略。

环境
-----------------------

**ManiSkill3 环境**

- **Environment**：ManiSkill3 仿真平台
- **Task**：以 ``PickCube3View-v1`` 为代表的机械臂操作任务
- **Policy Observation**：``states``
- **Reward Observation**：``main_images``、``extra_view_images``，以及任务描述文本
- **Action Space**：8 维连续动作

**Reward 输入结构**

- **States**：供 ``mlp_policy`` 使用的状态向量
- **Main Images**：供 Qwen3-VL reward worker 使用的主视角历史图像
- **Extra View Images**：与主视角同步的第三人称历史图像
- **Task Descriptions**：任务文本描述
- **History Buffer**：按 ``history_size`` 和 ``input_interval`` 组织的短视频片段

算法
-----------------------------------------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE 进行优势估计
   - 使用裁剪目标更新 actor
   - 使用 value loss 优化 critic

2. **MLP 策略网络**

   - actor 仅消费 ``states``
   - 推理与训练开销较低

3. **Qwen3-VL Reward Model**

   - reward worker 使用 ``HistoryVLMRewardModel``
   - 输入为任务文本与短视频历史
   - 输出通过 ``reward_parser`` 转为标量 reward
   - async 运行中，reward worker 会常驻并通过 RLinf channel 消费排队的历史片段请求

依赖安装
---------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

对于 OpenPI + Qwen3-VL Hugging Face reward 实验，请使用 OpenPI 具身环境，并固定
Qwen3-VL 兼容的 ``transformers`` 版本。RLinf 仓库内的 OpenPI 本地 patch 已兼容该统一环境，
因此不会因为安装了 OpenPI 就要求 Qwen3-VL reward worker 使用单独的 venv：

.. code:: bash

   source switch_env openpi
   uv pip install --upgrade "transformers==4.57.1" "tokenizers>=0.22,<0.23"

SGLang 是可选依赖，仅当 ``reward.model.model_type=history_vlm_sglang`` 时需要。
RLinf 使用进程内 ``sglang.Engine`` reward 后端，不走外部 SGLang server 路径。目标
依赖栈是 ``sglang==0.5.4``、``transformers==4.57.1``，以及匹配的 SGLang torch、
xgrammar 和 flashinfer runtime。只有选择该后端时才把这些依赖安装到同一个 OpenPI
venv；SGLang runtime 安装完成后，RLinf 会重新应用 OpenPI 的
``transformers==4.57.1`` patch。

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
   bash requirements/install.sh embodied --model openpi --env maniskill_libero --vlm-reward
   source .venv/bin/activate

``--vlm-reward`` 会固定 ``transformers==4.57.1``，并安装 Qwen3-VL 所需的
``tokenizers`` 版本范围。该环境通过 RLinf 仓库内的 OpenPI 本地 patch 与 OpenPI
共用，不需要为 Qwen reward 单独创建 venv；它不会下载 reward checkpoint，也不会自动启动
或安装 SGLang。

如果需要同一 venv 内的 OpenPI + Qwen3-VL + 进程内 SGLang reward 环境，请改用
``--vlm-reward-sglang``：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero --vlm-reward-sglang
   source .venv/bin/activate

该 SGLang 路径面向 ``sglang==0.5.4`` 和 ``transformers==4.57.1``，并安装
SGLang 专用的 torch、xgrammar 与 flashinfer runtime 到同一个 OpenPI venv。仅当
reward 配置设置 ``reward.model.model_type: history_vlm_sglang`` 时使用。SGLang
runtime 固化后，安装脚本会重新应用 RLinf 的 OpenPI ``transformers==4.57.1`` patch。

如果只需要 reward worker 环境，不需要 OpenPI 或具身 env，请继续使用 reward-only model
入口：

.. code:: bash

   bash requirements/install.sh embodied --model qwen_vlm_reward --vlm-reward
   bash requirements/install.sh embodied --model qwen_vlm_reward --vlm-reward-sglang

资源下载
----------------

下载 ManiSkill 资源文件：

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

模型下载
--------------

在开始训练之前，你需要准备 reward worker 所需的基础模型和 LoRA 权重：

.. code:: bash

   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com

   # 下载基础模型
   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /path/to/Qwen3-VL-4B-Instruct

   # Reward LoRA 权重目录请替换成你自己的路径
   ls /path/to/Qwen3-VL-4B-Instruct_lora

下载完成后，请确保在配置 yaml 中正确指定：

- ``reward.model.model_path``
- ``reward.model.lora_path``

如果还需要准备或微调 reward worker 使用的 Qwen3-VL checkpoint / LoRA，可以参考下面的
QwenTrend 数据与 SFT 流程；通用 VLM SFT runner 说明也可参考
:doc:`/rst_source/examples/embodied/sft_vlm`。

QwenTrend Reward Model 数据与 SFT 流程
------------------------------------------------------------

如果你已经有训练好的 QwenTrend reward checkpoint，可以跳过本节。否则，reward model
的准备流程分为三步：先收集 ManiSkill episode pkl，再改造成 5 帧双视角 progress
标签数据，最后运行 VLM SFT。

**1. 收集原始 episode pickle 数据**

.. code-block:: bash

   cd <path_to_RLinf>
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_collect

这条命令会用 ``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_collect.yaml``
启动 ``examples/embodiment/train_embodied_agent.py``。它是数据收集任务，不是在线
VLM-reward PPO 任务：

- ``reward.use_reward_model`` 为 ``false``，因此不会调用 VLM reward worker。
- 只有 ``env.eval.data_collection.enabled`` 为 ``true``，train 环境的数据收集是关闭的。
- 每到配置的 eval interval，eval 环境会被 ``CollectEpisode`` 包装并保存完整 episode。
- 默认输出目录是 ``logs/<timestamp>-maniskill_ppo_mlp_qwentrend_collect/collected_data``。
- 每个 episode pkl 会保存 observations、actions、rewards、done flags、info dicts 和 success 信息。由于 eval 环境使用 ``obs_mode: rgb`` 且 ``use_3rd_view_as_extra: true``，observation 中会包含 ``main_images`` 和 ``extra_view_images``。

**2. 将 episode 转成 QwenTrend SFT 数据**

.. code-block:: bash

   python examples/reward/preprocess_qwentrend_reward_dataset.py \
      --raw-data-path logs/<timestamp>-maniskill_ppo_mlp_qwentrend_collect/collected_data \
      --output-dir logs/processed_qwentrend_reward_data \
      --max-samples-per-label 5000 \
      --load-workers 32 \
      --write-workers 32

预处理脚本会扫描 ``*.pkl`` episode，把每个 episode 切成双视角窗口，并写出：

- ``<output-dir>/train/segments.jsonl``
- ``<output-dir>/train/pkl/*.pkl``
- ``<output-dir>/eval/segments.jsonl``
- ``<output-dir>/eval/pkl/*.pkl``

几个重要默认值如下：

- ``--window-size 5`` 表示每个视角导出 5 帧。
- ``--stride 1`` 表示相邻窗口重叠滑动。
- ``--delta-threshold 0.05`` 会把很小的 progress delta 标成 ``unclear``。
- ``--tail-unclear-ratio 0.15`` 会强制把每个 episode 尾部窗口标成 ``unclear``。
- ``--val-split 0.1`` 会按 episode 划分 train/eval。
- ``--balance-labels`` 和 ``--reverse-positive-as-negative`` 默认开启。
- ``--fps`` 只是为了兼容旧命令保留；当前 pkl 导出路径不会用 FPS 做重采样。

如果原始 episode 里没有任务文本，建议额外传入 ``--task-description``，避免使用脚本的通用 fallback prompt。

**3. 训练 Qwen3-VL QwenTrend reward model**

先设置处理后数据的根目录，并检查 ``examples/sft/config/qwen3vl_sft_qwentrend.yaml``
里的 ``actor.model.model_path`` 和 ``runner.output_dir``，然后启动 VLM SFT：

.. code-block:: bash

   export DUALVIEW_SFT_DATA_ROOT=/path/to/processed_qwentrend_reward_data
   bash examples/sft/run_vlm_sft.sh qwen3vl_sft_qwentrend

这个启动脚本实际会调用 ``examples/sft/train_vlm_sft.py``。``qwen3vl_sft_qwentrend``
配置使用 ``dataset_name: qwentrend_progress_sft``，并读取：

- ``${DUALVIEW_SFT_DATA_ROOT}/train/segments.jsonl``
- ``${DUALVIEW_SFT_DATA_ROOT}/eval/segments.jsonl``

数据集会根据每条样本的 ``pkl_path`` 读取两路内存中的 5 帧视频数组，并直接交给
Qwen3-VL processor。配置里对应的是 ``video_nframes: 5`` 和 ``video_fps: null``。

SFT checkpoint 会保存到：

.. code-block:: text

   logs/<sft-timestamp>/<experiment_name>/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt

后续在线 PPO 使用这个 reward model 时，``reward.model.model_path`` 指向 Qwen3-VL
基础模型目录，``reward.model.lora_path`` 指向 checkpoint step 目录，例如
``logs/<sft-timestamp>/<experiment_name>/checkpoints/global_step_<N>``。

运行脚本
-------------------

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         actor: 0-0
         env: 0-0
         rollout: 0-0
         reward: 0-0

   env:
      train:
         wrap_obs_mode: simple_prompt
         use_full_state: True
         init_params:
            id: PickCube3View-v1
            obs_mode: rgb
         use_3rd_view_as_extra: True

   reward:
      use_reward_model: True
      reward_mode: history_buffer
      history_reward_assign: True
      reward_weight: 1.0
      env_reward_weight: 0.0
      model:
         model_type: history_vlm
         model_path: /path/to/Qwen3-VL-4B-Instruct
         lora_path: /path/to/Qwen3-VL-4B-Instruct_lora
         gt_success_bonus: 20.0
         input_builder_name: qwentrend_input_builder
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
         infer_micro_batch_size: 64

这些参数的作用如下：

- ``component_placement.reward`` 用于放置在线 reward worker。
- ``wrap_obs_mode: simple_prompt`` 会同时暴露 ``states``、``main_images``、``extra_view_images`` 和 ``task_descriptions``。
- ``use_full_state: True`` 保持 actor 继续基于 ``states`` 运行 ``mlp_policy``。
- ``use_3rd_view_as_extra: True`` 会把 ManiSkill 的第三视角相机作为 ``extra_view_images`` 暴露出来。
- ``reward_mode: history_buffer`` 表示 reward worker 消费一段历史片段，而不是只看当前帧。
- ``history_reward_assign: True`` 表示当前窗口的 reward 会回填到窗口覆盖的更早几个 step。
- ``infer_micro_batch_size`` 用于限制 Hugging Face reward model 的推理微批大小，可按 reward worker 显存调整。

**Reward 后端变体**

两种历史 VLM reward 后端共享 history buffer、input builder 和 reward parser 配置：

- ``reward.model.model_type: history_vlm`` 会在 reward worker 进程内通过 Hugging Face
  ``AutoModelForVision2Seq`` 加载 Qwen3-VL。
- ``reward.model.model_type: history_vlm_sglang`` 会在 reward worker 内通过进程内
  ``sglang.Engine`` 加载 Qwen3-VL，不是外部 server endpoint 路径。这种方式仍保留
  相同的 ``history_buffers``、``input_builder_name`` 和 ``reward_parser_name``；
  请使用 ``--vlm-reward-sglang`` 环境，因为该后端目标是 ``sglang==0.5.4``，并带有
  匹配的 torch、xgrammar 和 flashinfer runtime。

**2. 配置文件**

可以直接参考以下配置文件：

- 主 QwenTrend 示例：``examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_qwentrend_reward

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- ``env/success_once``：建议优先关注的成功率指标。
- ``env/reward``：环境原始 step-level reward。
- ``rollout/rewards``：混合后的 rollout reward。
- ``train/actor/policy_loss``：策略优化情况。

在线调用链
----------

VLM reward 的主路径如下：

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker(history_buffer)
      -> HistoryManager
      -> EmbodiedRewardWorker
      -> HistoryVLMRewardModel
      -> InputBuilder + Qwen3-VL generate() + RewardParser

具体流程为：

1. ``train_embodied_agent.py`` 会在 ``reward.use_reward_model=True`` 时创建 ``EmbodiedRewardWorker``。
2. ``EmbodiedRunner.run`` 会在 ``global_step >= reward.use_output_step`` 后激活 reward channel。
3. ``EnvWorker.get_reward_model_output`` 会在 ``reward_mode="history_buffer"`` 时把当前观测追加进 ``HistoryManager``。
4. ``HistoryManager.build_history_input`` 负责提取配置中的历史窗口。
5. ``EmbodiedRewardWorker`` 会根据 ``reward.model.model_type`` 实例化对应的历史 VLM
   reward model。``history_vlm`` 对应进程内 Hugging Face 路径，``history_vlm_sglang``
   对应进程内 ``sglang.Engine`` 路径。
6. ``HistoryVLMRewardModel.compute_reward`` 使用 ``input_builder_name`` 构造多模态输入，调用 ``AutoModelForVision2Seq.generate()``，并用 ``reward_parser_name`` 解析生成文本。
7. ``EnvWorker.compute_bootstrap_rewards`` 会把 reward model 输出写到当前 step；如果 ``history_reward_assign=True``，``EnvWorker.assign_history_reward`` 还会把同一个 reward 回填到当前历史窗口覆盖的更早几个 step。

在 async 具身训练中，runner 会启动一次 ``EmbodiedRewardWorker.compute_rewards_async``，
让 reward worker 常驻服务 reward 请求。Env worker 会用稳定的 ``train_reward_input``
channel key 发送切分后的 reward 输入，reward worker 再用 ``reward_output`` key 返回切分后的输出。
这就是 queued reward-worker 路径；它避免每个 rollout step 重启 VLM，但最终 reward 仍由
``EnvWorker`` 统一组装。

当前实现说明
------------

- 这些 YAML 在顶层 ``reward`` 段配置了 ``reward_threshold``，但当前 ``history_vlm`` 实现并不会在 reward 推理阶段应用这个阈值。
- ``qwentrend_input_builder`` 会从 ``history_input`` 中同时读取 ``main_images`` 和 ``extra_view_images``，因此历史缓冲里需要同时记录这两个键，才能组成同步双视角输入。
- ``qwentrend_reward_parser`` 会按照 ``positive_reward``、``negative_reward``、``unclear_reward`` 和 ``invalid_reward`` 直接映射成带符号标量 reward，不会把输出截断到 ``[0, 1]``。
- ``gt_success_bonus`` 配置在 ``reward.model`` 下，并在 reward model 内部生效，而不是在 reward worker 前端额外注入。
- ``history_vlm_sglang`` 应与 ``history_vlm`` 使用相同的 prompt/input 与 parser 约定。两者差异主要在进程内 SGLang runtime 与 batching 行为；比较 reward 曲线前建议先检查生成标签是否一致。
