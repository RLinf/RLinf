Cosmos3 SGLang 评测
===================

概览
----

使用 RLinf 的 SGLang 具身后端在 LIBERO 上评测 Cosmos3。每张 GPU 运行一个 SGLang server（``server_type: embodied``），通过 ``POST /v1/actions/generations`` 提供动作推理服务；rollout worker 按 prompt 组批发送 HTTP 请求，并用返回的动作块推进仿真。

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

安装与运行
----------

``sglang`` 目标会安装 LIBERO 与带 ``diffusion`` extra 的 SGLang 0.5.19，固定 PyTorch 2.13.0 并按驱动选择 CUDA 12 或 CUDA 13 的 wheel。启动前，先按 :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>` 将 SFT checkpoint 转换为 ``./model_diffusers``，并准备好 NVIDIA ``cosmos-framework`` 中的 ``libero_native_frame_wise_relative_rot6d.json`` 统计文件。

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: LIBERO

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model sglang --env libero && source .venv/bin/activate

      **2. 启动 LIBERO-Spatial 评测**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang \
           rollout.model.model_path=./model_diffusers \
           rollout.model.action_stats_path=./libero_native_frame_wise_relative_rot6d.json \
           env.eval.total_num_envs=128

默认配置为 ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``\ （每张 GPU 启动一个 SGLang server），按 GPU 数量与显存调整 ``env.eval.total_num_envs``\ （例如 8 卡设 ``128``）。转换 checkpoint 时已将 tokenizer 复制到 ``model_diffusers/text_tokenizer/``，评测阶段离线运行，无需访问 Hugging Face。

.. dropdown:: SGLang 服务工作流程

   每张 GPU 启动一个 SGLang server 并绑定给对应的 rollout worker，单步推理按以下四步流转：

   1. **按 prompt 分组：** rollout worker 接收 LIBERO 图像与任务指令，按补全后的 prompt 给并行环境分组（server 仅批处理 token 长度相同的 prompt），再按 ``rollout.sglang.server.batching_max_size`` 切分批次（示例 ``8``，默认 ``1``）。

   2. **扩散推理：** 每个批次通过 ``POST /v1/actions/generations`` 发往 SGLang server，执行 ``num_inference_steps`` 步扩散推理，为每个环境返回 ``[horizon, 10]`` 的归一化 rot6d 动作块。

   3. **动作后处理：** rollout worker 按 ``input_index`` 排序响应，对 10 维输出做 quantile 去归一化，并将 6 维 rot6d 旋转转为 3 维 axis-angle，得到 7 维 LIBERO 动作。

   4. **推进仿真：** rollout worker 按原始环境顺序拼回 ``[N, 16, 7]`` 动作块，送入 LIBERO env worker 推进仿真。

.. dropdown:: 关键配置

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - 字段
        - 说明
      * - ``rollout.model.model_path``
        - 包含 ``model_index.json`` 的 diffusers checkpoint 目录。
      * - ``rollout.model.action_stats_path``
        - 动作去归一化的 quantile 统计 JSON，须与 SFT 同源（``*_rot6d.json``）。
      * - ``rollout.model.action_normalization``
        - ``quantile_rot``，须与训练一致。
      * - ``rollout.model.raw_action_dim`` / ``action_dim``
        - 模型侧 ``10``\ （rot6d）与环境侧 ``7``\ （axis-angle），两者均必填。
      * - ``rollout.model.num_action_chunks``
        - 每次请求返回的动作步数（示例 ``16``）；``env.eval.max_steps_per_rollout_epoch`` 须能被该值整除。
      * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
        - 扩散步数与输入视频规格，须与训练一致。
      * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
        - 每个 server 占用 GPU 数与张量并行度；单卡部署均为 ``1``\ （每卡一个 server）。
      * - ``rollout.sglang.server.batching_max_size``
        - 单次请求最多批处理的环境数（示例 ``8``，未设置时默认 ``1``）。
      * - ``rollout.sglang.http_timeout_s``
        - HTTP 超时秒数；扩散推理耗时较长，建议设为 ``600``。
      * - ``env.eval.total_num_envs``
        - 并行环境数，按 GPU 数量与显存调整。

结果与常见问题
--------------

日志写入 ``logs/<timestamp>-<config>/eval_embodiment.log``\ （指标与视频说明见 :doc:`../reference/results`，LIBERO 轨迹计数规则见 :ref:`LIBERO 评测配置 <libero-eval-config>`）。确认日志依次出现以下节点：

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
     - 确认 ``rollout.model.model_path`` 指向含 ``model_index.json`` 的 ``model_diffusers`` 目录。
   * - 动作异常或成功率为零
     - 核对 ``action_normalization``、``action_stats_path``、``num_inference_steps``、``num_frames`` 与 ``size`` 与 SFT 一致。
   * - 首批请求 HTTP 超时
     - 调大 ``rollout.sglang.http_timeout_s`` 与 ``rollout.sglang.http_max_retries``。
   * - 本地请求被代理拦截
     - 启动前导出 ``NO_PROXY=127.0.0.1,localhost``。
   * - LIBERO 渲染报错
     - ``run_eval.sh`` 默认使用 EGL；无 EGL 时设置 ``MUJOCO_GL=osmesa`` 与 ``PYOPENGL_PLATFORM=osmesa``。
   * - 重跑前 GPU 未释放
     - 用 ``nvidia-smi`` 检查是否仍有残留 SGLang server 进程；独占机器上可执行 ``ray stop`` 清理 Ray 会话（共享机器上先用 ``ray status`` 确认无他人任务，因为 ``ray stop`` 会影响整机）。
