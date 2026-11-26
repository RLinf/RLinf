在 LIBERO 中训练 RPent 规划器
=============================

在 RLinf 的多轮视觉 Agent 强化学习流程中接入 `RPent <https://github.com/RLinf/RPent>`__，在 LIBERO-Pro 与 LIBERO 模拟器中训练 Qwen3.5-4B 规划器。RPent 在独立模拟器进程中执行相机渲染、SAM3 分割感知与机械臂末端工具调用，RLinf 按模拟器原生成功判定给出 ``0`` / ``1`` 终局奖励；从 SFT checkpoint 出发经 96 步同步 GRPO 更新后，准入轨迹成功率从 **26.88%** 提升至 **75.78%**。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen3.5-4B（Megatron-Bridge · SGLang）

   .. grid-item-card:: 算法
      :text-align: center

      同步 / 异步 GRPO

   .. grid-item-card:: 工具
      :text-align: center

      RPent Episode · SAM3 · LIBERO-Pro

   .. grid-item-card:: 硬件
      :text-align: center

      单机 4×A100 80 GB

| **你将完成：** 安装训练与 RPent 模拟器环境 → 下载 Qwen3.5-4B、SAM3 权重及 LIBERO-Pro 资产并生成任务清单 → 启动 SAM3 与 ``run_train.sh``。
| **前置条件：** :doc:`安装与硬件参考 </rst_source/start/installation>` · 已执行 ``hf auth login`` 并获准访问 `facebook/sam3 <https://huggingface.co/facebook/sam3>`__ · `RPent 模拟器文档 <https://github.com/RLinf/RPent/blob/762a57c2b78278aff8d5c2c8f752ee42501bd5dc/docs/source-zh/rst_source/simulators/rl.rst>`__。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 模式
     - 任务集
     - 配置（``examples/agent/rpent/config/``）
     - 说明
   * - 同步 GRPO
     - LIBERO-Pro（``goal``、``object``、``spatial``）
     - ``train_qwen3.5_4b_grpo.yaml``
     - 每个初态采样 8 条轨迹，每次发起 5 组并取最先完成的 4 组，Actor 重算 logprob（TIS 上限 2.0）
   * - 异步 GRPO
     - LIBERO-Pro（``goal``、``object``、``spatial``）
     - ``train_qwen3.5_4b_async.yaml``
     - rollout 采样与模型更新重叠执行，支持缓冲区断点恢复
   * - 独立评估
     - LIBERO-Pro 保留初态
     - ``eval_qwen3.5_4b.yaml``
     - 评估导出权重并将各任务集成功率写入 ``eval_results.json``

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - 观测
     - ``1024 × 1024`` 自适应相机视角图像与 SAM3 分割掩码
   * - 动作
     - 感知工具调用、每次移动调用最多依次执行三个末端位姿或夹爪动作，以及结束回合的 ``finish`` 调用
   * - 奖励
     - 模拟器原生终局判定：环境判定任务完成（``terminated``）时为 ``1``；若规划器调用 ``finish`` 时未完成目标或达到轮数/步数上限则为 ``0``
   * - 提示词
     - 任务自然语言指令及 JSON 格式的任务集、任务编号和初态种子

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: Qwen3.5-4B（Megatron · 4×A100）

      **1. 安装环境**

      在仓库根目录安装 RLinf agentic 训练环境，再为已验证的 RPent 版本（``762a57c2``）创建独立的模拟器环境（会自动安装配套的 ``rpent-libero`` 与 ``rpent-liberopro``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UNINSTALL_FA4=1 bash requirements/install.sh agentic --sglang 0.5.12.post1 && source .venv/bin/activate

      .. code-block:: bash

         export RLINF_ROOT="$PWD"
         git clone --no-checkout https://github.com/RLinf/RPent.git ../RPent-episode
         git -C ../RPent-episode fetch --depth 1 origin 762a57c2b78278aff8d5c2c8f752ee42501bd5dc
         git -C ../RPent-episode checkout --detach FETCH_HEAD
         (cd ../RPent-episode && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --prerelease=allow --torch-backend cu128 -r requirements/episode-ppo-env.txt)

      **2. 下载模型并生成任务**

      如尚未登录，先执行一次 ``hf auth login``，随后下载 ``Qwen/Qwen3.5-4B``\ （或将 ``RPENT_MODEL`` 指向经过 SFT 初始化的 checkpoint）、``sam3.pt`` 与 LIBERO-Pro 资产：

      .. code-block:: bash

         hf download Qwen/Qwen3.5-4B --local-dir ./models/Qwen3.5-4B
         hf download facebook/sam3 sam3.pt --local-dir ./models
         env -u LIBERO_PRO_ASSET_PATH ../RPent-episode/.venv/bin/liberopro-download-assets --assets-dir "$PWD/libero_pro_assets" --skip-existing --no-cache

      配置环境变量，将资产目录写入 ``libero_config/config.yaml``，并生成训练与评估 JSONL 清单：

      .. code-block:: bash

         export RPENT_PYTHON="$PWD/../RPent-episode/.venv/bin/python"
         export RPENT_MODEL="$PWD/models/Qwen3.5-4B"
         export LIBERO_CONFIG_PATH="$PWD/libero_config"
         export RPENT_RUN_DIR="$PWD/logs"
         export RPENT_TRAIN_DATA="$PWD/data/rpent/train.jsonl"
         export RPENT_EVAL_DATA="$PWD/data/rpent/eval.jsonl"
         "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --assets-dir ./libero_pro_assets --seeds 0-31 --output-dir data/rpent
         "$RPENT_PYTHON" examples/agent/rpent/prepare_tasks.py --libero-type pro --seeds 32-39 --split eval --output-dir data/rpent

      另开一个终端，在 RLinf 根目录于 3 号卡启动 SAM3 感知服务：

      .. code-block:: bash

         SAM3_CHECKPOINT_PATH="$PWD/models/sam3.pt" "$PWD/../RPent-episode/.venv/bin/python" -m rpent.robots.components.sam3_server --cuda-device 3 --port 8101

      **3. 启动训练与评估**

      在 4 张 GPU 上启动同步 GRPO 训练（0、1 号卡运行 TP2 Actor，2 号卡运行 SGLang 服务，3 号卡运行 SAM3 与 EGL 模拟器；默认 ``runner.max_steps=150`` 且每 10 步保存一次，可在命令行追加 ``runner.max_steps=2 runner.save_interval=2`` 快速验证）：

      .. code-block:: bash

         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_grpo

      若需让 rollout 采样与模型更新异步重叠执行，运行：

      .. code-block:: bash

         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_async

      评估导出的 Hugging Face checkpoint 时，将 ``RPENT_MODEL`` 指向对应目录后执行：

      .. code-block:: bash

         export RPENT_MODEL="$RPENT_RUN_DIR/rpent-grpo/checkpoints/global_step_150/actor/hf_checkpoint"
         bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b

      .. dropdown:: 进阶配置（可选）

         - 如果模拟器环境中的 ``transformers`` 版本较旧、无法读取导出 checkpoint 里的新 tokenizer 类，可在命令末尾追加 ``tools.rpent.tokenizer_path=./models/Qwen3.5-4B`` 以指定兼容的 tokenizer 与图像处理器目录，模型权重仍从 ``RPENT_MODEL`` 加载。
         - ``actor.optim.optimizer_cpu_offload`` 默认开启以适配 4×80 GB 显存；如需将优化器状态留在显存中，可追加 ``actor.optim.optimizer_cpu_offload=false``，长训前请先核验当前 Megatron-Core 版本的断点恢复兼容性。
         - 如需在标准 LIBERO 上训练或评估，生成清单时将 ``prepare_tasks.py`` 的参数改为 ``--libero-type standard``，并在启动时覆盖 ``tools.rpent.libero_type=standard``。

可视化与结果
------------

使用 ``tensorboard --logdir "$RPENT_RUN_DIR"`` 观察训练奖励、重要性采样比例与 rollout staleness 曲线；完整指标说明见 :doc:`训练指标 </rst_source/reference/metrics>`。

每条轨迹会在 ``$RPENT_RUN_DIR/<experiment_name>/episodes/`` 下保存请求参数、交互记录与环境判定结果，评估脚本会在 ``$RPENT_RUN_DIR/rpent-eval/eval_results.json`` 汇总总体及分任务集成功率。训练样本通过 ``response_mask`` 屏蔽中间工具观测，使损失与重要性采样仅作用于模型采样的 policy token。

.. list-table:: **LIBERO-Pro 规划器训练结果**
   :header-rows: 1
   :widths: 22 20 28 30

   * - 基准任务
     - 配置
     - 工作负载
     - 实测结果
   * - LIBERO-Pro（30 个任务）
     - 32×H100 · 从 SFT checkpoint 出发的同步 GRPO
     - ``goal``、``object``、``spatial``\ （每任务 50 个初态，40 组取 32 组 × 8 条轨迹，96 步更新）
     - 准入轨迹成功率从第 1–10 步的 ``688/2,560``\ （**26.88%**）提升至第 87–96 步的 ``1,940/2,560``\ （**75.78%**）
