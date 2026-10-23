PhyAI 具身 Rollout 后端
=======================

使用 PhyAI 为 LIBERO 生成 PI0.5 rollout，同时由 RLinf 的 FSDP actor 训练 policy。本页依次介绍环境安装、checkpoint 配置和同步 PPO 示例的启动方法。RLinf 管理环境交互、观测变换、动作执行和权重同步；每个 rollout worker 在一张 GPU 上运行进程内的 PhyAI engine。

准备工作
--------

准备配备 NVIDIA GPU 的 Linux 机器和 PI0.5 LIBERO checkpoint。安装器使用 Python 3.12 和 Torch 2.11.0。当前接入已使用 RLinf PI0.5 LIBERO SFT checkpoint，在两张 NVIDIA A800 80 GB GPU 上完成训练流程的 smoke test。

安装
----

在 RLinf 仓库根目录，将 RLinf、OpenPI、LIBERO 和 PhyAI 安装到同一个虚拟环境，然后激活该环境：

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

安装器保留 OpenPI 已安装的 Torch 和 Transformers 版本，从 ``requirements/agentic/phyai_0.1.1_cu13.txt`` 安装额外运行时依赖，再使用 ``--no-deps`` 从官方 PhyAI wheel index 安装三个 PhyAI 包。包版本限定在 ``0.1.1`` 发布线内，包括预发布版本。

配置 Checkpoint 与后端
----------------------

在 ``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` 中，将 ``actor.model.model_path`` 设置为 PI0.5 LIBERO checkpoint 目录。目录中应包含 ``model.safetensors`` 和 ``physical-intelligence/libero/norm_stats.json``。rollout model 继承 actor 的权重路径与归一化统计，确保两侧使用一致的观测和动作变换。

示例通过 ``rollout.rollout_backend`` 选择 PhyAI，并使用 Hydra defaults 引入内核配置：

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   rollout:
     rollout_backend: phyai

``examples/embodiment/config/hybrid_engines/phyai.yaml`` 以 YAML mapping 的形式提供 ``rollout.phyai.kernel_config``，选择 PI0.5 adapter 使用的 attention 和 activation 内核。通过 ``rollout.phyai.max_batch_size`` 设置每个 engine 的 batch 容量；值为 ``null`` 时，根据分配给 worker 的环境数量推导容量。

启动训练
--------

设置 checkpoint 路径后，在仓库根目录启动示例：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

该命令通过标准具身训练入口启动 FSDP actor、PhyAI rollout worker 和 LIBERO 环境 worker。RLinf 采样 action chunk、更新 actor，再将更新后的权重同步给 PhyAI engine。

示例使用 bucket 权重同步，并启用 actor log-probability 重算和重要性采样修正。请同时启用 ``algorithm.recompute_logprobs`` 与 ``algorithm.importance_sampling_fix``。修正方法及其诊断指标见 :doc:`PPO <../../reference/algorithms/ppo>`。

当前训练接入要求使用 PI0.5，在 VLM 后添加 value head，并设置 ``value_vlm_mode: mean_token`` 和 ``joint_logprob: False``。训练采用同步、非流水线模式，配置校验会在 worker 启动前检查这些要求。

观察训练结果
------------

启动脚本将日志写入 ``logs/<timestamp>-<config-name>/``。通过 ``env/success_once`` 观察任务成功率，通过 PPO 的重要性采样指标检查 rollout 与 actor 的 log-probability 差异。各项指标的定义见 :doc:`训练指标 <../../reference/metrics>`。
