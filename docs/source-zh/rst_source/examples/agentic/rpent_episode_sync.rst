混合仿真器上的 RPent 训练
=========================

用同步 GRPO 在 LIBERO 和 RoboTwin 上训练 Qwen3.5-4B 规划器。RLinf 在模型 GPU 上启动推理和训练 worker。各仿真器使用独立 Python 环境。RPent toolkit 根据请求启动环境子进程，奖励来自仿真器判定的任务结果。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      从完整 SFT checkpoint 初始化 Qwen3.5-4B 规划器。

   .. grid-item-card:: 算法

      同步 GRPO，为每种仿真器单独设置组数配额。

   .. grid-item-card:: 任务

      LIBERO-Pro 和 RoboTwin 操作任务。

   .. grid-item-card:: 硬件

      单机示例用四张 NVIDIA GPU。384 环境示例用六台八卡节点。

**前置条件：** 准备 Linux NVIDIA 主机、仿真资产、自带预处理器的规划器 checkpoint，以及 SAM3 权重。各训练节点的源码和模型路径须一致。

安装与运行
----------

.. tab-set::

   .. tab-item:: NVIDIA CUDA

      1. 安装

      将 RLinf 和 RPent 克隆到各模型节点可访问的目录。试用 Draft 版本时，按 PR 正文检出配套的两个版本。

      .. code-block:: bash

         export PROJECT_DIR="$PWD"
         git clone https://github.com/RLinf/RLinf.git
         git clone https://github.com/RLinf/RPent.git

      已有 ``reason`` 镜像时，激活 ``/opt/venv/reason``。全新 Linux 主机可从 RLinf 根目录安装 NVIDIA agentic 环境，详见 :doc:`安装指南 <../../start/installation>`。

      .. code-block:: bash

         cd "$PROJECT_DIR/RLinf"
         bash requirements/install.sh agentic --engine sglang --sglang 0.5.12.post1
         source .venv/bin/activate

      模型节点使用 :doc:`../../guides/rpent_multimodal` 中的单机版本。384 环境使用该指南指定的 SGLang 原生构建。

      .. code-block:: bash

         export RLINF_ROOT="$PROJECT_DIR/RLinf"
         export RPENT_ROOT="$PROJECT_DIR/RPent"
         export RPENT_PYTHON="$(command -v python)"
         export RPENT_MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-4B"
         export RPENT_LIBERO_TASKS="$PROJECT_DIR/data/tasks/libero/train.jsonl"
         export RPENT_ROBOTWIN_TASKS="$PROJECT_DIR/data/tasks/robotwin/train.jsonl"
         export RPENT_RUN_DIR="$PROJECT_DIR/runs/rpent"
         uv pip install --python "$VIRTUAL_ENV/bin/python" \
           -r "$RPENT_ROOT/requirements/episode-planner.txt"
         uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"

      2. 准备资产和任务池

      将 Hugging Face 格式的 Qwen3.5-4B SFT checkpoint 放在 ``RPENT_MODEL_PATH``。目录中应有权重、tokenizer、聊天模板和图像预处理器，各模型节点使用同一个绝对路径。

      在仿真和 SAM3 主机上，按配套 RPent 仓库的 ``docs/source-zh/rst_source/guides/robot_episodes.rst`` 安装并下载资产。把解释器和资产路径填入 RPent 部署 YAML。单机示例需要 LIBERO-PRO 网格与原生初态、RoboTwin 资产快照及 ``sam3.pt``。

      任务池逐行指定仿真器怎样初始化。轨迹与奖励由训练在线生成。用公开的 LIBERO-PRO 初态和 RPent 的 RoboTwin seed 表生成两份任务池。

      .. code-block:: bash

         "$PROJECT_DIR/envs/libero/bin/python" "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
           --libero-config-path "$PROJECT_DIR/data/libero-config" \
           --suites libero_goal libero_spatial libero_object libero_10 \
           --output-dir "$PROJECT_DIR/data/tasks/libero"
         python "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
           --robotwin-seeds "$RPENT_ROOT/robots/robotwin/eval/demo_randomized.json" \
           --output-dir "$PROJECT_DIR/data/tasks/robotwin"

      两条命令各生成一份 ``train.jsonl`` 和一份记录行数与 SHA-256 的清单。单机每域至少需要 2 行，384 环境每域至少需要 24 行。LIBERO 每行指定 suite、任务和初态，RoboTwin 每行指定任务、随机化配置和 seed。自建 RoboTwin 任务池时，可在 ``answer`` 中加入 ``fallback_seeds`` 列表，初始化失败后整组一起切换场景。

      示例使用公开的 RoboTwin 评测场景作为任务池。使用此任务池训练后，请选择独立场景评测。若要复现某次实验，需取得该次训练的任务 JSONL、资产版本和起始 checkpoint。

      3. 启动最小双域示例

      在四卡单机上，从项目目录另开终端。复制部署文件，将其中的 ``/work`` 换成项目目录。GPU 0 运行 SAM3，GPU 1 运行仿真，GPU 2–3 运行 SGLang 和 learner。

      .. code-block:: bash

         export PROJECT_DIR="$PWD"
         cd "$PROJECT_DIR/RPent"
         cp tools/rl/deployment-local.yaml "$PROJECT_DIR/deployment-local.yaml"
         "${EDITOR:-vi}" "$PROJECT_DIR/deployment-local.yaml"
         "$PROJECT_DIR/envs/sam3/bin/python" -m robots.rl.services \
           --config "$PROJECT_DIR/deployment-local.yaml" --node local

      让服务留在单独的终端中。两个 toolkit 都就绪后，回到第 1 步激活训练环境并导出路径的终端。先解析 YAML，再启动训练。

      .. code-block:: bash

         cd "$RLINF_ROOT"
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1 --cfg job --resolve
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1

扩展到 384 个并行环境
---------------------

384 示例使用六台八卡节点。按 RPent 指南，在 ``sim`` 和 ``model0`` 到 ``model3`` 安装 LIBERO 与 RoboTwin，在 ``twin`` 安装 RoboTwin，在 ``sim`` 安装 SAM3。将 RPent 的 ``tools/rl/deployment-384.yaml`` 复制到 ``$PROJECT_DIR/deployment-384.yaml``，填写路径与可达地址后分发到六台节点。各节点将 ``RPENT_NODE`` 设为自己的角色名，再启动服务。

.. code-block:: bash

   cd "$PROJECT_DIR/RPent"
   "$PROJECT_DIR/envs/robotwin/bin/python" -m robots.rl.services \
     --config "$PROJECT_DIR/deployment-384.yaml" --node "$RPENT_NODE"

在 ``model0`` 运行 ``ray start --head --port=29710``，在另三个模型节点运行 ``ray start --address=<head-ip>:29710``。启动前各自设定不同的 ``RLINF_NODE_RANK``，并激活相同训练环境、设置 ``MEGATRON_PATH``。替换示例 IP，核对各节点的通信接口并确保模型、轨迹目录在四台模型节点上分别可读、可写。用路由的实际可达地址填写 ``mixed_384.yaml``。跨网段用部署侧的受限转发。

在 head 启动训练。

.. code-block:: bash

   cd "$RLINF_ROOT"
   export RAY_ADDRESS="<head-ip>:29710"
   export RPENT_LIBERO_ENDPOINT="http://<libero-router>:29840"
   export RPENT_ROBOTWIN_ENDPOINT="http://<robotwin-router>:29841"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
     +deployment@_global_=mixed_384

``data.rollout_batch_size`` 与 ``data.sampling_batch_size`` 按组填写。384 示例为 32 个有效组、48 个候选组，每组 8 条。``env.rpent`` 中每域填轨迹个数，默认按整组平分。可为各域指定不同配额，合计须匹配全局设置。每种仿真器各采集 192 条候选轨迹，其中 128 条用于训练。每条轨迹最多 40 轮，每 5 步保存。

使用 ``group_router: true`` 接入组路由。设置 ``full_cohort_ready_timeout_s`` 后，runner 等本批所有仿真轨迹就绪，再开始生成。检查日志中的两域就绪计数，以及 ``sampling/`` 中的就绪与释放记录。组选择、仿真释放和生成队列清理完成后才开始更新。

可视化与结果
------------

运行 ``tensorboard --logdir "$RPENT_RUN_DIR"``，查看 :doc:`训练指标 <../../reference/metrics>`。单机示例从每域的两个候选组中各选一组，每组两条轨迹，执行一次更新。轨迹存到配置的输出目录。``sampling/`` 保存每批选组与清理记录。

恢复训练
--------

保留模型、任务池、域配额及原有轨迹目录。重启所需的节点服务，然后在同一训练命令后添加 ``runner.resume_dir=$RPENT_RUN_DIR/grpo/checkpoints/global_step_5``，选择已完整保存的 checkpoint 目录。恢复后从保存的采样位置继续。单域可继续使用基础配方，在 ``data.train_data_paths`` 和 ``tools.rpent.remote_toolkit_endpoint`` 分别设置任务与 toolkit。
