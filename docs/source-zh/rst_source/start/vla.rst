快速开始
========

选择 **具身智能** 或 **Agent 与 LLM** 两条单卡路径之一，从安装环境到跑通第一次 RLinf 训练并查看指标输出。

安装与运行
----------

.. tab-set::

   .. tab-item:: 具身智能（ManiSkill3）
      :sync: embodied

      在单卡上基于 `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ 内置 ``PickCube-v1`` 任务运行 PPO 训练（约需 4 GB 显存，无需额外下载权重或资产）。

      1. 克隆仓库并安装 ManiSkill3 虚拟环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env maniskill_libero
         source .venv/bin/activate

      2. 启动训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   .. tab-item:: Agent 与 LLM（GRPO）
      :sync: agentic

      在单卡上使用 SGLang + Megatron 对 ``DeepSeek-R1-Distill-Qwen-1.5B`` 运行数学推理 GRPO 训练（配置 ``examples/reasoning/config/math/qwen2.5-1.5b-single-gpu.yaml``）。

      1. 克隆仓库并安装 Agent 与 LLM 虚拟环境：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang
         source .venv/bin/activate

      2. 下载模型与数据集，然后启动训练：

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data
         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-single-gpu \
           rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-1.5B \
           actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-1.5B \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

如需使用预构建 Docker 镜像、国内镜像加速（``--use-mirror`` 与 ``HF_ENDPOINT``）、其他模型/仿真器组合或查看多卡硬件要求，见 :doc:`安装与硬件参考 <installation>`。命令行中的配置名对应 ``examples/embodiment/config/`` 或 ``examples/reasoning/config/math/`` 下的同名 YAML 文件，完整配置目录见 :doc:`../reference/configuration`。

.. dropdown:: 常用操作与参数覆盖（可选）

   .. code-block:: bash

      # 从已保存的 checkpoint 目录断点续训
      bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
        runner.resume_dir=./logs/<run>/checkpoints/global_step_50

      # 限定到 0 号 GPU 运行
      bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
        cluster.component_placement.actor,rollout,env=0

      # 清理残留的本地 Ray 会话（共享机器上先用 ray status 确认无他人任务）
      ray stop

结果与常见问题
--------------

日志、TensorBoard 事件与 checkpoint 保存在配置指定的输出目录下（具身默认 ``logs/<时间戳>-<配置名>/``，推理默认 ``../results/grpo-1.5b/``）。

启动 ``tensorboard --host 0.0.0.0 --logdir ./logs`` 查看曲线：具身任务重点关注 ``rollout/env_info/return`` 与 ``rollout/env_info/success_once``，Agent / LLM 任务重点关注 ``rollout/rewards`` 与响应长度。

具身训练每步会在终端输出指标表，完整运行数百轮后 ``success_once`` 会升至 ``0.90`` 以上：

.. code-block:: text

   ╭───────────────────────────────── Metric Table ─────────────────────────────────╮
   │ Global Step: 10/1000     │ Progress: ████░░░░░░░░░░░░ │ 1.0%                   │
   │ Elapsed: 00:51           │ ETA: 01:24:09              │ Step Time: 5.122s      │
   ├───────────────────────────────── Environment ──────────────────────────────────┤
   │ episode_len=50.0         │ num_trajectories=1024      │ return=7.187           │
   ├──────────────────────────────── Training/Actor ────────────────────────────────┤
   │ actor/policy_loss=-0.013 │ actor/clip_fraction=0.179  │ critic/value_loss=0.04 │
   ╰────────────────────────────────────────────────────────────────────────────────╯

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 解决方法
   * - 训练时显存不足（OOM）
     - 调小 ``actor.micro_batch_size`` 与 ``env.train.total_num_envs``，详见 :ref:`显存不足（OOM） <faq-oom>`。
   * - EGL 或 Vulkan 渲染报错
     - 按 :ref:`渲染问题 <faq-rendering>` 检查驱动与图形渲染配置。
   * - 重复运行时报 Ray 或 NCCL 错误
     - 先执行 ``ray stop`` 清理残留进程，详见 :ref:`CUDA 与显存问题 <faq-cuda>`。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 框架导览
      :link: walkthrough
      :link-type: doc

      通过图示了解训练闭环、多卡扩展方式以及配置和工具的目录位置。

   .. grid-item-card:: 具身智能
      :link: ../examples/index
      :link-type: doc

      查看 OpenVLA、OpenPI、LIBERO、RoboTwin、SFT 与世界模型示例。

   .. grid-item-card:: Agent 与 LLM
      :link: ../examples/agentic/index
      :link-type: doc

      查看 Search-R1、WideSeek-R1、rStar2、AgentLightning 与数学推理示例。
