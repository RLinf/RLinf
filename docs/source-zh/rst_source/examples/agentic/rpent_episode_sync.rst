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

      将 RLinf 和 RPent 克隆到各模型节点可访问的共享目录。

      .. code-block:: bash

         export PROJECT_DIR="$PWD"
         git clone https://github.com/RLinf/RLinf.git
         git clone https://github.com/RLinf/RPent.git

      模型节点使用 ``reason`` 镜像，或在 RLinf 根目录用 ``bash requirements/install.sh agentic --engine sglang`` 安装 NVIDIA agentic 环境。详见 :doc:`安装指南 <../../start/installation>`。各模型节点使用相同版本。384 环境配方使用 SGLang 原生 TokenSpace，运行环境见 :doc:`../../guides/rpent_multimodal`。

      .. code-block:: bash

         source /opt/venv/reason/bin/activate
         export RLINF_ROOT="$PROJECT_DIR/RLinf"
         export RPENT_ROOT="$PROJECT_DIR/RPent"
         export RPENT_PYTHON="$(command -v python)"
         export RPENT_MODEL_PATH="$PROJECT_DIR/models/Qwen3.5-4B"
         export RPENT_LIBERO_TASKS="$PROJECT_DIR/data/tasks/libero.jsonl"
         export RPENT_ROBOTWIN_TASKS="$PROJECT_DIR/data/tasks/robotwin.jsonl"
         export RPENT_RUN_DIR="$PROJECT_DIR/runs/rpent"
         uv pip install --python "$VIRTUAL_ENV/bin/python" \
           -r "$RPENT_ROOT/requirements/episode-planner.txt"
         uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$RPENT_ROOT"

      将完整 SFT 模型及其 tokenizer、图像预处理器放在 ``$RPENT_MODEL_PATH``，确保模型节点看到同一路径。仿真和 SAM3 节点按 `RPent 安装与部署指南 <https://github.com/RLinf/RPent/blob/main/docs/source-zh/rst_source/guides/robot_episodes.rst>`_ 安装独立环境、配置资产和 SAM3 权重。

      2. 准备模型和任务池

      每域一份 JSONL，记录中包含 ``prompt`` 和 JSON 编码的 ``answer``。以下各行分别保存到 ``$RPENT_LIBERO_TASKS`` 和 ``$RPENT_ROBOTWIN_TASKS``。每个文件至少提供两条任务记录才能运行下文最小示例。将示例中的 RoboTwin seed 换成当前资产上验证过的初态。

      .. code-block:: json

         {"prompt":"Solve the LIBERO task","answer":"{\"environment\":\"libero\",\"suite\":\"libero_spatial\",\"task\":0,\"seed\":0}"}
         {"prompt":"Solve the RoboTwin task","answer":"{\"environment\":\"robotwin\",\"task_name\":\"adjust_bottle\",\"task_config\":\"demo_randomized\",\"seed\":100000}"}

      使用 LIBERO 环境为大规模配置导出任务池。

      .. code-block:: bash

         "$LIBERO_PYTHON" "$RLINF_ROOT/examples/agent/rpent/prepare_tasks.py" \
           --libero-config-path "$LIBERO_CONFIG_PATH" --output-dir "$PROJECT_DIR/data/tasks/libero-native"
         cp "$PROJECT_DIR/data/tasks/libero-native/train.jsonl" "$RPENT_LIBERO_TASKS"

      正式训练使用事先准备好的完整任务池，384 示例每域至少需要 24 条任务记录。在 ``answer`` 中加入 ``fallback_seeds``，可在 RoboTwin 初始化失败时换用备用初态。整组在首次请求模型前一起切换。

      3. 启动最小双域示例

      在四卡单机上，从 RPent 根目录启动本机服务。先填写 YAML 内的解释器、模型和资产路径。GPU 0 运行 SAM3，GPU 1 运行仿真，GPU 2–3 运行 SGLang 和 learner。

      .. code-block:: bash

         cd "$RPENT_ROOT"
         python -m robots.rl.services --config tools/rl/deployment-local.yaml --node local

      另开终端，进入模型环境并启动训练。先解析 YAML，确认路径和配额。训练会创建 SGLang 模型副本，独立采集用的模型服务可以在这里省略。

      .. code-block:: bash

         cd "$RLINF_ROOT"
         source /opt/venv/reason/bin/activate
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1 --cfg job --resolve
         bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
           +deployment@_global_=mixed_local cluster.num_nodes=1

扩展到 384 个并行环境
---------------------

384 示例使用六台八卡节点。按 RPent 指南，在 ``sim`` 和 ``model0`` 到 ``model3`` 安装 LIBERO 与 RoboTwin，在 ``twin`` 安装 RoboTwin，在 ``sim`` 安装 SAM3。六台机器各执行以下命令，填自己的角色名。

.. code-block:: bash

   cd "$RPENT_ROOT"
   python -m robots.rl.services --config tools/rl/deployment-384.yaml --node <node>

在 ``model0`` 启动 Ray head，在另三个模型节点运行 ``ray start --address=<head-ip>:29710``。启动前各自设定不同的 ``RLINF_NODE_RANK``，并激活相同训练环境、设置 ``MEGATRON_PATH``。替换示例 IP，核对各节点的通信接口并确保模型、轨迹目录在四台模型节点上分别可读、可写。用路由的实际可达地址填写 ``mixed_384.yaml``。跨网段用部署侧的受限转发。

在 head 启动训练。

.. code-block:: bash

   export RAY_ADDRESS="<head-ip>:29710"
   export RPENT_LIBERO_ENDPOINT="http://<libero-router>:29840"
   export RPENT_ROBOTWIN_ENDPOINT="http://<robotwin-router>:29841"
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_episode_sync_grpo \
     +deployment@_global_=mixed_384

``data.rollout_batch_size`` 与 ``data.sampling_batch_size`` 按组填写。384 示例为 32 个有效组、48 个候选组，每组 8 条。``env.rpent`` 中每域填轨迹个数，默认按整组平分。可为各域指定不同配额，合计须匹配全局设置。384 示例每域候选 192 条、入训 128 条，最多 40 轮，每 5 步保存。

使用 ``group_router: true`` 接入组路由。设置 ``full_cohort_ready_timeout_s`` 后，runner 等本批所有仿真轨迹就绪，再开始生成。检查日志中的两域就绪计数，以及 ``sampling/`` 中的就绪与释放记录。组选择、仿真释放和生成队列清理完成后才开始更新。

可视化与结果
------------

运行 ``tensorboard --logdir "$RPENT_RUN_DIR"``，查看 :doc:`训练指标 <../../reference/metrics>`。单机示例从每域的两个候选组中各选一组，每组两条轨迹，执行一次更新。轨迹存到配置的输出目录。``sampling/`` 保存每批选组与清理记录。

恢复训练
--------

保留模型、任务池、域配额及原有轨迹目录。重启所需的节点服务，然后在同一训练命令后添加 ``runner.resume_dir=$RPENT_RUN_DIR/grpo/checkpoints/global_step_5``，选择已完整保存的 checkpoint 目录。恢复后从保存的采样位置继续。单域可继续使用基础配方，在 ``data.train_data_paths`` 和 ``tools.rpent.remote_toolkit_endpoint`` 分别设置任务与 toolkit。
