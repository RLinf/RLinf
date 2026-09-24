LingBot-VLA 2.0 的 PPO 训练
===========================

使用 PPO 和 GAE 在 RoboTwin ``click_bell`` 上微调 LingBot-VLA 2.0，再评估保存的 policy。RLinf 通过 FSDP2 训练 action expert 和 value head，冻结 Qwen3-VL。默认训练配置使用 4 张 48 GB GPU。模型包独立安装，接入边界见 :doc:`../../extending/model_adaptation`。

安装依赖
--------

准备包含 RLinf ``robotwin.envs.vector_env`` 接口的 RoboTwin 源码。安装器会固定官方 V2 源码版本，并应用 ``requirements/embodied/models/lingbotvlav2/`` 下的兼容补丁，补齐 checkpoint 元数据、数据模块延迟导入，以及概率重算和激活检查点所需的 MoE 接口。请与 :doc:`LingBot-VLA 1.0 <lingbotvla>` 使用独立 venv 和源码目录：两个包都叫 ``lingbotvla``。

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin \
     --torch 2.9.0 --transformers 4.57.6
   source .venv/bin/activate
   export LINGBOT_VLA_V2_PATH="${LINGBOT_VLA_V2_PATH:-$VIRTUAL_ENV/lingbot-vla-v2}"

复用已有 checkout 时，在安装前设置 ``LINGBOT_VLA_V2_PATH``。安装器根据 ``source.json`` 检查 revision 和补丁前后哈希，遇到冲突改动会停止。本次验证的环境为 PyTorch 2.9.0+cu126、Transformers 4.57.6 和 FlashAttention 2.8.3.post1。

Docker target 使用同一安装器：

.. code-block:: bash

   docker buildx build -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-robotwin-lingbotvlav2 \
     -t rlinf:robotwin-lingbotvlav2 .

模型与资源
----------

下载官方 `V2 RoboTwin 发布模型 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`_ 和 `Qwen3-VL-4B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`_。保留完整的 V2 发布目录：checkpoint config 只有模型类别标记，严格加载权重还需要上层的 ``lingbotvla_cli.yaml`` 和 Qwen metadata。

.. code-block:: bash

   export LINGBOT_VLA_V2_CKPT=/path/to/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
   export LINGBOT_VLA_V2_TRAIN_CONFIG=/path/to/lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
   export QWEN3_VL_PATH=/path/to/Qwen3-VL-4B-Instruct
   # RoboTwin 要求填写包含 assets/ 的根目录。
   export ROBOTWIN_ASSETS_PATH="$ROBOTWIN_PATH"
   export REPO_PATH=$(pwd)
   export PYTHONPATH="$REPO_PATH:$LINGBOT_VLA_V2_PATH:$ROBOTWIN_PATH"

源码和两个模型卡均声明 Apache-2.0；兼容补丁保留原始声明并附带上游许可证。2026-09-22 检查时，这两个公开模型仓库均不需要 gated-access 审批。

训练
----

先运行两轮更新的 smoke：2 个环境，每个 episode 为 100 个仿真步，每次更新前检查 rollout 与 actor 的概率重算一致性。默认 actor 使用 GPU 0–1、rollout 使用 GPU 2、环境使用 GPU 3；通过 ``cluster.component_placement`` 调整资源。

.. code-block:: bash

   bash tests/e2e_tests/embodied/run.sh robotwin_ppo_lingbotvlav2 \
     runner.logger.log_path=/path/to/results/vla2_smoke

模型输入为头部和双腕 HWC RGB 图像、14D 状态和语言指令，预处理与机器人布局由上游 ``FeatureTransform`` 负责。动作 shape 为 ``[B, 50, 14]``；模型 latent 为 55 维，夹爪位于索引 28、29。V2 保留 prefix 中的 depth/video query token。PPO log-prob 为 ``[B, 2750]``，value 为 ``[B, 1]``。

PPO 对完整 flow-SDE latent 轨迹计算概率，保持 ``algorithm.logprob_type=chunk_level``、``algorithm.reward_type=chunk_level`` 和 ``logprob_dim=55``。BF16 权重下，记录的 latent 和概率计算仍使用 FP32，因此必须设置 ``actor.fsdp_config.mixed_precision.cast_forward_inputs=false``。``forward_micro_batch_size=1`` 保持 MoE batch shape 一致；完整轨迹概率误差超过 ``logprob_replay_atol`` 时停止更新。

Smoke 在两轮更新后保存 checkpoint。使用 ``runner.resume_dir`` 续训，并保留原始 ``actor.model.model_path`` 以构造架构。检查通过后启动完整配置：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2 ALOHA

在 TensorBoard 检查 ``train/actor/total_loss``、``train/actor/grad_norm``、``train/actor/rollout_train_joint_logprob_gap_max`` 和环境成功率。仓库变量 ``LINGBOT_VLA_V2_CI=true`` 可启用 CI 作业，需要在 ``/workspace/dataset`` 预置权重和 RoboTwin 资产，并提供 4 张 GPU。全新安装和 Docker 构建需独立验证，不能用既有环境的运行结果替代。

评估
----

独立评估使用完整 rollout 配置，采用 flow 积分。将 ``runner.ckpt_path`` 指向 RLinf actor 的 ``model_state_dict/full_weights.pt``；省略该项则评估原始 checkpoint。仍需保留原始模型路径以构造架构。

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
     runner.ckpt_path=/path/to/actor/model_state_dict/full_weights.pt

上游 benchmark 使用 FP32，并提示 BF16 可能改变任务成功率。短程 BF16 PPO 检查只验证执行流程，不能证明收敛或复现 benchmark。本示例尚未验证导出为上游部署格式。
