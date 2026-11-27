Cosmos3 SGLang 评测
===================

使用 SGLang Diffusion server 或原生 Hugging Face 后端在 LIBERO 仿真器上评测 Cosmos3-Nano 与 Cosmos3-Edge。

工作原理
--------

每个 rollout worker 会在分配给自己的 GPU 上独立拉起并管理一个 SGLang Diffusion server，通过 ``POST /v1/actions/generations`` 提供动作推理。每一步仿真中，worker 按 prompt 将并行环境分组，再按 ``rollout.sglang.server.batching_max_size`` 切分成批量请求发出，最后将返回动作按环境顺序拼回。

响应中每个环境对应一段归一化后的 10 维 rot6d 动作块。rollout worker 按 ``input_index`` 排序，用 LIBERO quantile 统计文件去归一化，将 rot6d 转为 3 维 axis-angle，再把 7 维动作送入 LIBERO 执行。

.. code-block:: text

   LIBERO env worker --观测（图像 + 指令）--> Rollout worker
     按 prompt 分组，每组最多 batching_max_size 个环境
        --POST /v1/actions/generations-->
   Rollout worker 管理的 SGLang Diffusion server
        --返回归一化 [horizon, 10] rot6d 动作块-->
   Rollout worker（按 input_index 排序 → 去归一化 → rot6d 转 axis-angle）
        --[N, 16, 7] 环境动作-->
   LIBERO env worker 推进仿真

安装
----

按 Cosmos3 文档中的 :ref:`安装 SGLang 推理环境 <cosmos3-sglang-install>` 配置 ``.venv-sglang`` 并安装固定版本的 Cosmos3 SGLang 源码。

准备 Checkpoint
---------------

SGLang 评测读取 Diffusers 组件目录 ``model_diffusers``。请按 :doc:`Cosmos3 监督微调与 NFT 训练 <../../examples/embodied/sft_cosmos3>` 中的“转换 Checkpoint”一节导出 RLinf SFT 权重。

.. note::

   导出完成后评测可完全离线运行，tokenizer 和 processor 文件已包含在 ``model_diffusers`` 目录中。

评测 Cosmos3-Edge（LIBERO-Spatial）
-----------------------------------

编辑 ``evaluations/libero/libero_spatial_cosmos3_edge_eval_sglang.yaml``，填写导出的 ``model_diffusers`` 路径和统计文件：

.. code-block:: yaml

   rollout:
     model:
       model_path: ./assets/cosmos3-edge-policy-liberoall-10k/model_diffusers
       action_stats_path: ./cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json

在仓库根目录启动评测：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_edge_eval_sglang

单卡快速验证（4 个并行环境、32 步、1 个 epoch）：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_edge_eval_sglang \
     cluster.component_placement.env=0 cluster.component_placement.rollout=0 \
     env.eval.total_num_envs=4 env.eval.specific_reset_id=0 \
     env.eval.max_episode_steps=32 env.eval.max_steps_per_rollout_epoch=32

评测 Cosmos3-Nano（LIBERO-Spatial）
-----------------------------------

使用 SGLang 评测 Cosmos3-Nano 时，编辑 ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml`` 中的 ``rollout.model.model_path`` 和 ``rollout.model.action_stats_path``，然后运行：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

关键配置
--------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 字段
     - 说明
   * - ``rollout.model.model_path``
     - 导出的 ``model_diffusers`` 目录。
   * - ``rollout.model.action_stats_path``
     - 与 SFT 同源的 quantile 统计文件（``*_rot6d.json``）。
   * - ``rollout.model.action_normalization``
     - ``quantile_rot``，须与训练一致。
   * - ``rollout.model.raw_action_dim`` / ``action_dim``
     - 模型侧 10 维（rot6d）与环境侧 7 维（axis-angle）。
   * - ``rollout.model.num_action_chunks``
     - 单次请求返回的动作步数（16）；``env.eval.max_steps_per_rollout_epoch`` 须能被其整除。
   * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
     - 扩散步数与输入视频尺寸，须与训练一致。
   * - ``rollout.sglang.server.tp_size`` / ``sp_degree`` / ``dp_size``
     - 单实例并行度。每个 server 占用的 GPU 数由 ``dp_size * tp_size * sp_degree * cfg_parallel_degree`` 自动推导。
   * - ``rollout.sglang.server.batching_max_size``
     - 单次 HTTP 请求最多批处理的环境数（示例默认为 8）。
   * - ``rollout.sglang.http_timeout_s``
     - 单次 HTTP 请求超时秒数，建议设为 ``600``。
   * - ``env.eval.total_num_envs``
     - 全部 worker 上的并行 LIBERO 环境总数。

验证
----

查看 ``logs/<时间戳>-<config>/eval_embodiment.log``，确认依次出现以下节点：

1. 启动 server：``Launching sglang server (server_type=embodied) ...``
2. server 就绪：``sglang server ready at http://...``
3. 逐 episode 输出：``[libero eval] task_id=.., trial_id=.., success=..``
4. 汇总表格：``success_once``、``success_at_end`` 与 ``num_trajectories``。

LIBERO 轨迹计数规则见 :ref:`libero-eval-config`。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 现象
     - 处理
   * - SGLang 找不到模型组件
     - 确认 ``rollout.model.model_path`` 指向包含 ``model_index.json`` 的 ``model_diffusers`` 目录。
   * - 动作异常或成功率为 0
     - 核对 ``action_normalization``、``action_stats_path``、``num_inference_steps``、``num_frames`` 与 ``size`` 是否与 SFT 一致。
   * - 首批请求 HTTP 超时
     - 调大 ``rollout.sglang.http_timeout_s`` 与 ``rollout.sglang.http_max_retries``。
   * - 本地请求被代理拦截
     - 启动前导出 ``NO_PROXY=127.0.0.1,localhost``。
   * - LIBERO 渲染报错
     - ``run_eval.sh`` 默认使用 EGL；无 EGL 的机器请导出 ``MUJOCO_GL=osmesa`` 与 ``PYOPENGL_PLATFORM=osmesa``。
   * - 异常退出后 GPU 未释放
     - 运行 ``ray stop --force``，确认 ``nvidia-smi`` 中无残留 rollout worker 或 server 进程。
