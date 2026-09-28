Cosmos3 SGLang 评测
=====================

用 SGLang 后端在 LIBERO 仿真器上评测 Cosmos3：模型放进独立的 SGLang server 进程，rollout worker 作为客户端发观测、收动作、送入仿真。适用于只做推理评测的场景。

工作原理
----------------------------------------

每张 GPU 起一个 SGLang server（``server_type: embodied``），运行 ``Cosmos3OmniDiffusersPipeline``，把动作策略暴露成 HTTP 端点 ``POST /v1/actions/generations``。eval driver 把 server URL 下发给 rollout worker，worker 把 N 个环境的一步观测拆成一个或多个请求发出。

server 只会把 token 长度相同的 prompt 放在同一批，因此 ``Cosmos3SGLangAdapter.request_groups`` 先按补全后的 prompt 给环境分组，再把每组按 ``rollout.sglang.server.batching_max_size`` 切成若干份，每份最多这么多个环境。示例配置中该值为 ``8``；不设置时 adapter 采用 SGLang 自身的默认值 ``1``，即每个环境单独发一次请求。worker 逐组发送，再把结果按原来的环境顺序拼回去。

响应中每个环境各占一条记录：``data[i].action.values`` 是该环境的 ``[horizon, raw_action_dim]`` 归一化 rot6d 动作块。adapter 先按 ``input_index`` 排序，再去归一化并转成 7 维 axis-angle 送入 LIBERO。

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

安装
----------------------------------------

安装 LIBERO 环境和带 ``diffusion`` extra 的 SGLang 0.5.19。``sglang`` model 会固定 torch 2.13.0，并按驱动支持的版本选择 CUDA 12 或 CUDA 13 的 wheel：

.. code-block:: bash

   bash requirements/install.sh embodied --model sglang --env libero
   source .venv/bin/activate

准备 Checkpoint
----------------------------------------

评测输入是 diffusers 组件目录 ``model_diffusers``，由 SFT checkpoint 经 cosmos-framework 转换得到，完整四步见 :doc:`Cosmos3 SFT <../../examples/embodied/sft_cosmos3>` 的「转换 Checkpoint」一节。

.. note::

   评测**不需要**联网拉 HuggingFace 或 Qwen3-VL 缓存：转换时 tokenizer 已拷进 ``model_diffusers/text_tokenizer/``，server 直接从该目录读。

运行 LIBERO-Spatial
----------------------------------------

默认配置 ``evaluations/libero/libero_spatial_cosmos3_eval_sglang.yaml``。开跑前改 YAML 指向本地 ``model_diffusers``：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model_diffusers          # 评测输入的 diffusers 目录
       action_stats_path: /path/to/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json  # cosmos-framework 的 rot6d 统计文件

   env:
     eval:
       total_num_envs: 128   # 按 GPU 数 / 显存调整（示例 8 卡用 128）

改好后运行：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_cosmos3_eval_sglang

每张 GPU 起一个 Cosmos3 SGLang server，逐 episode 打印成功与否，最后汇总成功率；日志写到 ``logs/<时间戳>-<config>/eval_embodiment.log``。

关键配置
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 说明
   * - ``rollout.model.model_path``
     - diffusers checkpoint 目录（评测输入）。
   * - ``rollout.model.action_stats_path``
     - 动作去归一化的 quantile 统计文件，须与 SFT 同源（``*_rot6d.json``）。
   * - ``rollout.model.action_normalization``
     - ``quantile_rot``，须与训练一致。
   * - ``rollout.model.raw_action_dim`` / ``action_dim``
     - 模型侧 10（rot6d）/ 环境侧 7（axis-angle）；**两者都必填** （见常见问题）。
   * - ``rollout.model.num_action_chunks``
     - 每次请求返回的动作步数（示例 16）；``env.eval.max_steps_per_rollout_epoch`` 须能被其整除。
   * - ``rollout.model.num_inference_steps`` / ``num_frames`` / ``size``
     - 扩散步数与输入视频规格，须与训练一致。
   * - ``rollout.sglang.server.num_gpus`` / ``tp_size``
     - 每 server 占用 GPU 与 TP；单卡部署均为 1（每卡一个 server）。
   * - ``rollout.sglang.server.batching_max_size``
     - 单次请求中 server 最多批处理的环境数（示例为 8）。adapter 按该值切分每个 prompt 分组；不设置时采用 SGLang 默认值 1，即每个环境单独一次请求。
   * - ``rollout.sglang.http_timeout_s``
     - HTTP 超时；扩散推理慢，建议 ``600``。
   * - ``env.eval.total_num_envs``
     - 并行环境数，按 GPU / 显存调整。

验证
----------------------------------------

看 ``eval_embodiment.log``，确认这几个节点依次出现：

1. 起 server：``Launching sglang server (server_type=embodied) ...``
2. 权重加载完成：``[RunAI Streamer] Overall time to stream 28.3 GiB ... to cpu: <秒数>`` （本地盘通常数十秒内；**缺这行** 说明加载卡住，见常见问题）
3. server 就绪：``sglang server assigned: rank=i -> http://...``
4. 逐 episode 结果：``[libero eval] task_id=.., trial_id=.., success=..``
5. 汇总：``success_once`` / ``success_at_end`` / ``num_trajectories``

LIBERO 轨迹计数规则见 :ref:`libero-eval-config`。

常见问题
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 现象
     - 处理
   * - SGLang 找不到 model components
     - 确认两处 ``model_path`` 都指向含 ``model_index.json`` 的 ``model_diffusers`` 目录。
   * - 动作乱 / 完全不成功
     - 核对 ``action_normalization`` / ``action_stats_path`` / ``num_inference_steps`` / ``num_frames`` / ``size`` 与 SFT 一致。
   * - 首批请求 HTTP 超时
     - 调大 ``rollout.sglang.http_timeout_s`` 与 ``http_max_retries``。
   * - 本地请求被 proxy 拦截
     - 启动前设 ``NO_PROXY=127.0.0.1,localhost``。
   * - LIBERO 渲染报错
     - ``run_eval.sh`` 默认使用 EGL 渲染；若机器不支持 EGL，设 ``MUJOCO_GL=osmesa``、``PYOPENGL_PLATFORM=osmesa``。
   * - 重跑前 GPU 未释放
     - 确认上次 ``ray stop`` 已彻底、``nvidia-smi`` 全空、无残留 ``ray::SGLangServerGroup`` 进程。
