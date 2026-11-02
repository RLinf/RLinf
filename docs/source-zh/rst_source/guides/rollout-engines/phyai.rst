PhyAI 具身 Rollout 后端
=======================

本页介绍如何在 LIBERO 上用 PPO 训练 PI0.5 policy，并由 `PhyAI <https://github.com/mingti-org/phyai>`_ 生成 rollout。PhyAI 是面向 VLA 等物理 AI 模型的推理框架。每个 RLinf rollout worker 在自己的 GPU 上运行一个 PhyAI engine，用 CUDA graph 和 FlashInfer kernel 采样动作，替代 Hugging Face 模型代码。其余部分仍由 RLinf 负责，包括 LIBERO 环境、观测与动作变换、FSDP actor 和权重同步。

下面依次完成安装、配置 checkpoint 和启动训练。训练前有一节说明为什么 PhyAI 训练需要开启重要性采样修正：actor 训练时用到的动作不是它自己采样的，这个设置决定了 actor 如何处理这些动作。

准备工作
--------

需要一台配备 NVIDIA GPU 的 Linux 机器和 Python 3.12。目前该后端支持在 LIBERO 上用同步 PPO 训练 PI0.5，每个 rollout worker 使用一张 GPU。训练流程已使用 RLinf PI0.5 LIBERO SFT checkpoint，在两张 NVIDIA H800 80 GB GPU 上完成 smoke test。

安装
----

在 RLinf 仓库根目录下，将 RLinf、OpenPI、LIBERO 和 PhyAI 安装到同一个虚拟环境，然后激活该环境：

.. code-block:: bash

   bash requirements/install.sh \
     --python 3.12 \
     --torch 2.11.0 \
     --install-rlinf \
     embodied \
     --model openpi \
     --env libero \
     --engine phyai
   source .venv/bin/activate

这条命令做了两件事：

1. 按 Hugging Face rollout 的方式安装 RLinf、OpenPI 和 LIBERO。
2. 按已安装 Torch 的 CUDA 版本，安装 RLinf 测试过的 PhyAI 版本。PhyAI 自带的依赖会替换 OpenPI 所需的 Torch 和 Transformers 版本，因此安装器保留原有版本；如果 PhyAI 的 CUDA 版本与 Torch 不一致，安装会直接停止。

准备 Checkpoint
---------------

先下载 PI0.5 LIBERO SFT checkpoint：

.. code-block:: bash

   # For mainland China users: export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

然后在 ``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` 中把 ``actor.model.model_path`` 设为下载目录。rollout 读取同一路径，因此 PhyAI 与 actor 加载相同的权重，并用同一份统计量归一化观测。其他 PI0.5 checkpoint 及下载镜像见 :doc:`π₀ 与 π₀.₅ 模型强化学习训练 <../../examples/embodied/pi0>`。

如果要从 RL 训练得到的 checkpoint 继续，还需将 ``runner.ckpt_path`` 设为对应的 ``.pt`` 文件。每个 PhyAI engine 会在下载的 checkpoint 之上加载这份 state dict，因此第一次 rollout 使用的权重与 actor 一致，冻结参数也不例外。

配置后端
--------

示例配置已经选择了 PhyAI，关键的几行如下：

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   algorithm:
     recompute_logprobs: True
     importance_sampling_fix: True
     importance_sampling_clip: 1.0

   rollout:
     rollout_backend: phyai
     phyai:
       max_batch_size: null

``rollout.rollout_backend: phyai`` 让训练脚本启动 PhyAI rollout worker，而不是 Hugging Face rollout worker。``hybrid_engines/phyai`` 提供 kernel 策略，固定 PhyAI 在 PI0.5 上使用的 attention 和激活函数 kernel，使其数值在两种实现允许的范围内尽量接近 actor。``max_batch_size`` 限制单个 engine 一次处理的观测数量；设为 ``null`` 时，按分配给该 worker 的环境数量确定。

``algorithm`` 中的三个配置项用于处理 PhyAI 与 actor 之间剩余的数值差异，下一节说明原因。

修正 Rollout 与 Actor 的差异
----------------------------

PhyAI 和 FSDP actor 使用同一份权重，但计算所用的 kernel 不同，例如 paged attention、融合的激活函数和不同的归约顺序。因此，actor 为 PhyAI 采样出的动作计算 log-probability 时，结果会与 PhyAI 报告的值略有出入。

PPO 以产生数据的 policy 为基准衡量每次更新。如果这个基准来自 PhyAI，kernel 带来的差异就会进入每一个 PPO ratio，甚至单独触发 clipping。开启 ``recompute_logprobs`` 后，actor 会在训练前为每个采样动作重新计算 log-probability，并以自己的结果作为基准。``importance_sampling_fix`` 再按 actor 与 PhyAI 对同一动作的概率之比为每个样本加权，权重上限为 ``importance_sampling_clip``，从而让 actor 能正确使用并非由自己采样的数据。这三个配置项需要一起开启：只开启 ``importance_sampling_fix`` 而未开启 ``recompute_logprobs`` 时，启动阶段会报错。

修正方法及其指标的详细说明见 :doc:`PPO <../../reference/algorithms/ppo>`。

启动训练
--------

设置好 checkpoint 路径后，在仓库根目录启动示例：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

该命令通过具身训练入口启动 FSDP actor、PhyAI rollout worker 和 LIBERO 环境 worker。每个训练 step 先由 PhyAI 收集 rollout，再由 actor 重新计算 log-probability 并更新参数，最后在下一轮 rollout 前把新权重同步到所有 PhyAI engine。

检查训练结果
------------

启动脚本将日志写入 ``logs/<timestamp>-<config-name>/``。``env/success_once`` 反映任务成功率。``actor/importance_sampling_weight`` 接近 1、``actor/importance_sampling_clip_fraction`` 较小，说明 PhyAI 与 actor 的结果仍然一致；如果权重逐渐偏离 1，说明二者的差异在扩大。其余指标的定义见 :doc:`训练指标 <../../reference/metrics>`。
