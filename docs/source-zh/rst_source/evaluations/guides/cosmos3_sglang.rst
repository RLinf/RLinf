Cosmos3 SGLang 评测
===================

概览
----

用 SGLang embodied backend 在 LIBERO 仿真器上评测 Cosmos3。每张 GPU 运行一个独立的 SGLang server（``server_type: embodied``），通过 ``POST /v1/actions/generations`` 提供动作推理服务；rollout worker 作为轻量 HTTP 客户端按 prompt 分组发送观测、接收动作块并推进仿真。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      LIBERO-Spatial（``libero``）

   .. grid-item-card:: 支持模型
      :text-align: center

      Cosmos3（SGLang diffusers 后端）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>`

一键上手
--------

``sglang`` 目标会安装 LIBERO 环境与带 ``diffusion`` extra 的 SGLang 0.5.19，固定 PyTorch 2.13.0 并按驱动支持的版本选择 CUDA 12 或 CUDA 13 的 wheel。启动前，先按 :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>` 的“转换 Checkpoint”一节将 SFT checkpoint 转换为 diffusers 目录 ``model_diffusers``。

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: Cosmos3 SGLang + LIBERO

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model sglang --env libero && source .venv/bin/activate

         # 2. 启动 LIBERO-Spatial 评测（替换为本地 model_diffusers 与统计文件路径）
         bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang \
           rollout.model.model_path=/path/to/model_diffusers \
           rollout.model.action_stats_path=/path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json \
           env.eval.total_num_envs=128

默认配置为 ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``\ （每张 GPU 启动一个 SGLang server），按 GPU 数量和显存调整 ``env.eval.total_num_envs``\ （例如 8 卡用 ``128``）。转换 checkpoint 时 tokenizer 已复制到 ``model_diffusers/text_tokenizer/``，评测阶段离线运行，无需访问 Hugging Face 或 Qwen3-VL 缓存。

.. dropdown:: SGLang 服务端架构

   每张 GPU 启动一个运行 ``Cosmos3OmniDiffusersPipeline`` 的 SGLang server，eval driver 将 server URL 下发给 rollout worker。server 只批处理 token 长度相同的 prompt，因此 adapter 先按补全后的 prompt 给环境分组，再按 ``rollout.sglang.server.batching_max_size`` 切分每组（示例为 ``8``，未设置时默认 ``1``）。响应中每条 ``data[i].action.values`` 为 ``[horizon, raw_action_dim]`` 归一化 rot6d 动作块；adapter 按 ``input_index`` 排序，完成 quantile 去归一化并转为 7 维 axis-angle 后，按原始环境顺序拼回送入 LIBERO。

   .. code:: text

      EnvWorker(libero) --观测(图像+指令)--> Cosmos3SGLangAdapter.request_groups
        按 prompt 给 N 个环境分组，每组最多 batching_max_size 个环境
           --每组发一次 POST /v1/actions/generations-->
      SGLang server（Cosmos3OmniDiffusersPipeline，扩散 num_inference_steps 步）
           --响应中每个环境一条 data[i].action.values = [horizon, 10]（归一化 rot6d）-->
      Cosmos3SGLangAdapter 按 input_index 顺序逐条解析：
        裁前 10 通道 → quantile 去归一化 → rot6d(6) 转 axis-angle(3)
      worker 把各组结果按环境顺序拼成 [N, 16, 7]
           --[N, 16, 7]-->
      EnvWorker.chunk_step 推进仿真

.. dropdown:: 关键配置

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - 字段
        - 说明
      * - ``rollout.model.model_path``
        - 包含 ``model_index.json`` 的 diffusers checkpoint 目录。
      * - ``rollout.model.action_stats_path``
        - 动作去归一化的 quantile 统计文件，须与 SFT 同源（``*_rot6d.json``）。
      * - ``rollout.model.action_normalization``
        - ``quantile_rot``，须与训练一致。
      * - ``rollout.model.raw_action_dim`` / ``action_dim``
        - 模型侧 ``10``\ （rot6d）/ 环境侧 ``7``\ （axis-angle），两者均必填。
      * - ``rollout.model.num_action_chunks``
        - 每次请求返回的动作步数（示例 ``16``）；``env.eval.max_steps_per_rollout_epoch`` 须能被其整除。
      * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
        - 扩散步数与输入视频规格，须与训练一致。
      * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
        - 每个 server 占用 GPU 数与 TP；单卡部署均为 ``1``\ （每卡一个 server）。
      * - ``rollout.sglang.server.batching_max_size``
        - 单次请求最多批处理的环境数（示例为 ``8``，未设置时采用 SGLang 默认值 ``1``）。
      * - ``rollout.sglang.http_timeout_s``
        - HTTP 超时时间；扩散推理耗时较长，建议设为 ``600``。
      * - ``env.eval.total_num_envs``
        - 并行环境数，按 GPU 数量与显存调整。

结果与常见问题
--------------

日志写入 ``logs/<timestamp>-<config>/eval_embodiment.log``\ （指标与视频说明见 :doc:`../reference/results`，LIBERO 轨迹计数规则见 :ref:`LIBERO 评测配置 <libero-eval-config>`）。确认 ``eval_embodiment.log`` 依次出现以下节点：

1. ``Launching sglang server (server_type=embodied) ...``
2. ``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <秒数>``\ （缺少该行说明权重加载卡住）
3. ``sglang server assigned: rank=i -> http://...``
4. ``[libero eval] task_id=.., trial_id=.., success=..``
5. 汇总行：``success_once``、``success_at_end`` 与 ``num_trajectories``

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 现象
     - 处理
   * - SGLang 找不到 model components
     - 确认两处 ``model_path`` 都指向含 ``model_index.json`` 的 ``model_diffusers`` 目录。
   * - 动作异常或成功率为零
     - 核对 ``action_normalization``、``action_stats_path``、``num_inference_steps``、``num_frames`` 与 ``size`` 与 SFT 一致。
   * - 首批请求 HTTP 超时
     - 调大 ``rollout.sglang.http_timeout_s`` 与 ``http_max_retries``。
   * - 本地请求被 proxy 拦截
     - 启动前设置 ``NO_PROXY=127.0.0.1,localhost``。
   * - LIBERO 渲染报错
     - ``run_eval.sh`` 默认使用 EGL 渲染；机器不支持 EGL 时设 ``MUJOCO_GL=osmesa``、``PYOPENGL_PLATFORM=osmesa``。
   * - 重跑前 GPU 未释放
     - 确认上次 ``ray stop`` 已完成，``nvidia-smi`` 显示空闲，且无残留 ``ray::SGLangServerGroup`` 进程。
