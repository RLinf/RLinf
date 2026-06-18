基础配置
=========

下面给出 RLinf 核心配置参数的完整参考说明，适用于所有工作负载。
我们对 YAML 中每个重要键做了说明，便于你针对自己的集群、模型或研究需求进行调整。
各参数按顶层键分组组织。

本节涵盖具身智能与智能体训练共用的基础 GPU 和集群配置。
任务专用配置请参见 :doc:`embodiment_config` 和 :doc:`agentic_config`。

.. contents::
   :depth: 1
   :local:

hydra
~~~~~~

.. code:: yaml

  hydra:
    run:
      dir: .
    output_subdir: null

``hydra.run.dir``：Hydra 运行的工作目录。

``hydra.output_subdir``：输出子目录（设为 null 则不创建子目录）。


cluster
~~~~~~~~~~~~~~~

.. code:: yaml

  cluster:
    num_nodes: 1
    component_placement:
      actor,inference,rollout: all

``cluster.num_nodes``：用于训练的物理节点数量。

``cluster.component_placement``：
各组件（进程）的 *放置策略*。

在上面运行于GPU节点的简单示例中：

- 键 (key) 是组件的名称，例如 ``rollout``，或 ``rollout,inference,actor``
- 值 (value) 是分配给这些组件的全局 GPU Rank，可以是：
   - "all"：使用集群中的所有 GPU
   - 单个整数，例如 "3"：使用 GPU 3
   - 逗号分隔的整数列表，例如 "0,2,3"：使用 GPU 0、2 和 3
   - 连字符分隔的整数范围，例如 "0-3"：使用 GPU 0、1、2 和 3
   - 上述两种方式的组合，例如 "0-3,5,7"：使用 GPU 0、1、2、3、5 和 7

而对于更高级的组件放置用法（例如，异构集群中使用不同型号的 GPU、机器人硬件或仅 CPU 节点）以及代码中的自定义，请参见 :doc:`../concepts/placement`。

runner
~~~~~~~~~~~~~~~

.. code:: yaml

  runner:
    task_type: math
    logger:
      log_path: ${runner.output_dir}/${runner.experiment_name}
      project_name: rlinf
      experiment_name: ${runner.experiment_name}
      logger_backends: ["tensorboard"] # wandb, swanlab

    max_epochs: 5
    max_steps: -1

    val_check_interval: 1
    save_interval: 50

    seq_length: 2048

    resume_dir: null
    experiment_name: grpo-1.5b
    output_dir: ../results

``runner.task_type``：任务类型标识（math 或 embodied）。

**logger：**

``runner.logger.log_path``：日志输出的根目录。

``runner.logger.project_name``：实验跟踪的项目名。

``runner.logger.experiment_name``：实验名称。

``runner.logger.logger_backends``：日志后端（tensorboard、wandb、swanlab）。

关于日志后端详见 :doc:`logger`。

``runner.max_epochs``：最大训练 epoch 数。

``runner.max_steps``：最大全局步数；为 -1 时，依据 ``runner.max_epochs`` 自动确定。

``runner.val_check_interval``：验证 rollout 的触发频率（-1 关闭）。

``runner.save_interval``：保存 checkpoint 的步数间隔。

``runner.seq_length``：输入到模型的总序列长度（提示 + 生成）。

algorithm
~~~~~~~~~~~~~~~

.. code:: yaml

  algorithm:
    group_size: 2

    logprob_forward_micro_batch_size: 1

    val_rollout_batch_size_per_gpu: 4

    loss_type: ppo
    loss_agg_func: "token-mean"
    kl_beta: 0.0
    kl_penalty_type: low_var_kl
    ratio_clip_eps: 0.2
    entropy_bonus: 0.0
    calculate_entropy: False
    clip_ratio_c: null

    adv_type: grpo
    normalize_advantages: True
    early_stop_imp_ratio: 5.0
    use_valid_token_scale: False

    sampling_params:
      do_sample: True
      temperature: 1.0
      top_k: 1000000
      top_p: 1.0
      repetition_penalty: 1.0

``algorithm.group_size``：每个提示采样的响应个数（>1 时启用组基线）。

``algorithm.logprob_forward_micro_batch_size``：log-prob 前向的微批大小。

``algorithm.val_rollout_batch_size_per_gpu``：验证阶段每 GPU 的 rollout 微批大小。

``algorithm.loss_type``：策略损失类型（如 ppo）。

``algorithm.loss_agg_func``：token 损失的聚合方式（如 token-mean）。

``algorithm.kl_beta``：加入到奖励中的 KL 权重。

``algorithm.kl_penalty_type``：KL 形态（如 low_var_kl）。

``algorithm.ratio_clip_eps``：PPO 比率裁剪阈值。

``algorithm.entropy_bonus``：熵奖励系数。

``algorithm.calculate_entropy``：是否计算/记录熵项。

``algorithm.adv_type``：优势函数估计类型（如 grpo）。

``algorithm.normalize_advantages``：是否对优势进行归一化。

``algorithm.early_stop_imp_ratio``：当重要性比超出阈值时提前终止本次更新。

``algorithm.use_valid_token_scale``：是否按有效 token 掩码缩放损失/优势。

**sampling_params：**

``algorithm.sampling_params.do_sample``：False 时使用贪心解码。

``algorithm.sampling_params.temperature``：采样温度。

``algorithm.sampling_params.top_k``：top-k 截断（设很大值等于禁用）。

``algorithm.sampling_params.top_p``：nucleus 采样阈值。

``algorithm.sampling_params.repetition_penalty``：重复惩罚系数。

rollout
~~~~~~~~~~~~~~~

.. code:: yaml

  rollout:
    group_name: "RolloutGroup"

    gpu_memory_utilization: 0.55

    model:
      model_path: ../../model/DeepSeek-R1-Distill-Qwen-1.5B/
      model_type: qwen2.5

    recompute_logprobs: True

``rollout.gpu_memory_utilization``：目标 GPU 显存占用比例。

``rollout.group_name``：rollout / inference worker 的逻辑分组名。

``rollout.model.model_path``：生成后端所用 HF 模型路径。

``rollout.model.model_type``：后端内部使用的模型架构标记（如 qwen2.5）。

``rollout.recompute_logprobs``：是否为采样序列重新计算对数概率。

actor
~~~~~~~~~~~~~~~

.. code:: yaml

  actor:
    group_name: "ActorGroup"

    model:
      megatron_checkpoint: null

    seed: 1234

**顶层：**

``actor.group_name``：训练（actor）worker 的逻辑分组名。

``actor.model.megatron_checkpoint``：训练前加载的模型 Megatron checkpoint 路径。

``actor.seed``：全局随机种子，便于复现。

reward
~~~~~~~~~~~~~~~

.. code:: yaml

  reward:
    use_reward_model: false

``reward.use_reward_model``：是否使用奖励模型。

critic
~~~~~~~~~~~~~~~

.. code:: yaml

  critic:
    use_critic_model: false

``critic.use_critic_model``：是否使用价值网络（critic）。
