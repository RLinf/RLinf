在 SGLang Server 中适配具身模型
====================================

本文介绍如何将一个已适配 SGLang Server 的具身 VLA 模型接入
RLinf 的评测 rollout，并使用 RLinf 支持的各类仿真器进行模型评测。

全文分为两部分：

- 第一部分说明适配任意新模型时需要完成的步骤和接口约定；
- 第二部分以 DreamZero 为例，逐项说明需要修改的代码和 YAML 配置。

.. note::

   本文覆盖 **SGLang 具身 rollout** 路径（同时支持评测与 RL 训练采样）。FSDP actor 或 SFT 模型的训练侧注册请参考 :doc:`使用 FSDP 添加新模型 <new_model_fsdp>` 与 :doc:`添加新的 SFT 模型 <new_model_sft>`。


第一部分：适配新模型
====================

整体架构
--------

SGLang 具身路径将通用调度逻辑与模型逻辑分开：

- ``EmbodiedComponentPlacement`` 按 ``dp_size * tp_size * sp_degree * cfg_parallel_degree`` 自动推导每个 SGLang Diffusion server 实例占用的 GPU 数，并将 rollout 硬件 rank 按实例分组；
- 每个 ``SGLangEmbodiedWorker`` 在 ``init_worker()`` 中于分配到的 GPU 上独立拉起并管理一个 ``SGLangServerBackend``\ （``server_type="embodied"``），建立本地回环控制与推理客户端，加载模型对应的 **sglang adapter**，并通过 channel 与 ``EnvWorker`` 交换观测和动作。配置 ``rollout.sglang.launch_router: true`` 且 adapter 设置 ``supports_router = True`` 后，驱动脚本会调用 ``launch_embodied_sglang_router`` 启动可选 router 转发推理请求，控制与权重同步请求仍留在本地回环；
- **sglang adapter** 继承 ``SGLangEmbodiedAdapter``\ （如 ``DreamZeroSGLangAdapter``、``Cosmos3SGLangAdapter``），负责请求构造（``build_request``）、响应解析（``parse_response``）以及可选的训练采样和权重映射钩子（``supports_training``、``validate_training``、``create_weight_target``、``map_weight_name``、``parse_rollout_response``）。异步 HTTP 通信由基类借助 worker 的 ``InferenceHTTPClient`` 统一完成。

因此，接入新模型通常 **不需要修改** ``rlinf/workers/rollout/sglang/sglang_embodied_worker.py``。server 参数由 YAML 的 ``rollout.sglang.server`` 直接传递给 ``ServerArgs.from_kwargs``，模型由 ``rollout.model.model_type`` 选择，调用关系如下：

.. code-block:: text

   rollout.model.model_type: "<your_model>"
                 │
                 ▼
   EmbodiedComponentPlacement  （按 dp * tp * sp * cfg 为每个 server 分组 GPU）
                 │
                 ▼
   SGLangEmbodiedWorker.init_worker()
                 │
                 ├── get_sglang_adapter_cls("<your_model>") → YourSglangAdapter(cfg, rank)
                 └── SGLangServerBackend(server_type="embodied").initialize()
                       (rollout.sglang.server → ServerArgs.from_kwargs → dispatch_launch)
                 │
                 ▼ （可选：rollout.sglang.launch_router: true 时）
   驱动脚本: launch_embodied_sglang_router() → rollout_group.set_sglang_router_url()

   SGLangEmbodiedWorker.predict(env_obs, mode)   # 由 EmbodiedEvalRunner / EmbodiedRunner 驱动
                 │
                 ├── adapter.async_build_request(env_obs, mode) → (payload, state)
                 └── adapter.async_send_request(client, payload, state, mode) → [N, H, D] 动作


进入该调用链需满足以下配置项：

.. code-block:: yaml

   runner:
     task_type: embodied_eval   # RL 训练时为 embodied
     only_eval: true            # RL 训练时为 false

   rollout:
     rollout_backend: sglang
     sglang:
       serving_mode: embodied   # rollout worker = SGLangEmbodiedWorker
       backend_type: server
       launch_server: true      # 由各 worker 启动并管理自己的 server 子进程
       launch_router: false     # 设为 true 时启用 SGLang router
     model:
       model_type: "<your_model>"

``rollout.sglang.serving_mode`` 选择 ``SGLangEmbodiedWorker``\ （其固定通过 ``sglang.multimodal_gen.runtime.launch_server.dispatch_launch`` 拉起 ``server_type="embodied"``），``rollout.model.model_type`` 决定从 adapter registry 加载的模型适配器。


适配步骤
--------

下面按照推荐顺序说明新模型需要完成的工作。

步骤一：确认 SGLang Server 的动作接口
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 的 sglang adapter 为 SGLang Server 构造请求、解析响应（实际的 HTTP 收发由
worker 完成）。在编写 RLinf 代码前，先确认 SGLang 侧已经具备以下能力：

1. ``sglang serve`` 能够加载目标模型或目标 Pipeline；
2. SGLang Server 已为该模型提供接收批量 observation、返回批量 action 的 VLA 接口；
3. 请求和响应格式固定，并能够表达模型所需的图像、文本、状态和缓存信息。

.. warning::

   RLinf 仓库中的 sglang adapter 只负责构造 action 请求、解析响应。模型 Pipeline、
   动作路由及相关 ``sglang serve`` 参数仍需在所使用的 SGLang 版本中实现。


步骤二：注册 ``model_type``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``rlinf/config.py`` 中注册模型，使配置校验能够识别该名称：

.. code-block:: python

   SupportedModel.YOUR_MODEL = SupportedModel.register("your_model", force=True)

如果该模型需要通过 embodied 配置校验，还需要将它加入 ``EMBODIED_MODEL``：

.. code-block:: python

   EMBODIED_MODEL = {
       # ...
       SupportedModel.YOUR_MODEL,
   }

``"your_model"``、``register_sglang_adapter`` 注册时使用的名称和 YAML 中的
``rollout.model.model_type`` 必须一致。adapter registry 查找时不区分大小写，
但仍建议全部使用小写。


步骤三：实现 sglang adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在模型自己的目录下新建 adapter 文件（放在 model 层，与训练侧模型代码同级）：

.. code-block:: text

   rlinf/models/embodiment/<your_model>/sglang_adapter.py

adapter 继承 ``rlinf/models/embodiment/sglang_adapter.py`` 中的 :class:`SGLangEmbodiedAdapter`，构造签名为 ``(cfg, rank)``，实现 ``build_request`` 和 ``parse_response`` 并声明 ``action_path``\ （若依赖单 server 会话状态，设置 ``supports_router = False``）：

.. code-block:: python

   import torch
   from rlinf.models.embodiment.sglang_adapter import SGLangEmbodiedAdapter


   class YourSglangAdapter(SGLangEmbodiedAdapter):
       action_path = "/v1/actions/generations"

       def __init__(self, cfg, rank):
           self.cfg_rollout = cfg.rollout
           self.model_cfg = cfg.rollout.model
           self.rank = rank
           # 在这里创建轻量的 transform（不要加载大模型权重）。

       def build_request(self, env_obs, mode="eval"):
           # 1. 将 RLinf env_obs 转换为模型输入并归一化；
           # 2. 组装 server 的请求 payload（dict）；
           # 3. 返回 (payload, state)，state 会原样回传给 parse_response。
           payload = {...}
           state = ...            # 例如反归一化需要用到的中间 observation
           return payload, state

       def parse_response(self, resp, state):
           # 1. 从 server 响应里取出归一化动作；
           # 2. 反归一化为环境尺度动作。
           actions = ...
           info = {
               "prev_logprobs": ...,
               "prev_values": ...,
               "forward_inputs": ...,
           }
           return torch.as_tensor(actions, dtype=torch.float32), info

接口设计：

- **HTTP 传输封装在 ``SGLangEmbodiedAdapter`` 基类中**：worker 将 ``InferenceHTTPClient`` 传给 ``adapter.async_request_actions``\ （或分批时调用 ``async_build_request`` 与 ``async_send_request``），由基类在工作线程构造请求、异步发送至 ``action_path``\ （训练时为 ``rollout_path``）并解析响应；
- ``build_request`` 输入的 ``env_obs`` 是按 batch 组织的环境观测字典；通用字段包括 ``main_images``、``task_descriptions``，模型还可使用 ``wrist_images``、``states`` 或其他视角。返回 ``(payload, state)``，其中 ``payload`` 为请求体，``state`` 供 ``parse_response`` 复用；
- ``parse_response`` 返回 ``(actions, info)``：``actions`` 是形状为 ``[N, num_action_chunks, action_dim]`` 的 Tensor，``info`` 携带 ``prev_logprobs``、``prev_values`` 和 ``forward_inputs``；
- 仅支持评测的 adapter 保留默认的 ``supports_training = False``。如需支持 RL 训练采样和在线权重同步（参考 ``Cosmos3SGLangAdapter``），将 ``supports_training = True`` 并实现 ``validate_training``、``map_weight_name``、``create_weight_target`` 与 ``parse_rollout_response``。

.. important::

   不要在 adapter 中加载模型。模型应只存在于 ``sglang serve`` 子进程中。adapter
   中只保留数据变换和少量请求上下文，否则会造成重复加载权重和额外显存占用。


步骤四：注册 adapter
~~~~~~~~~~~~~~~~~~~~

adapter 使用 **lazy builder** 注册在
``rlinf/models/embodiment/sglang_adapter.py`` 的
``_register_builtin_sglang_adapters()`` 中（与 ``rlinf.models`` 的模型注册风格
一致）：把重依赖的 import 放进 builder 内部，只有真正 lookup 到该 ``model_type``
时才会导入你的 adapter 模块：

.. code-block:: python

   # rlinf/models/embodiment/sglang_adapter.py
   def _register_builtin_sglang_adapters():
       def _build_your_model():
           from rlinf.models.embodiment.your_model.sglang_adapter import (
               YourSglangAdapter,
           )

           return YourSglangAdapter

       register_sglang_adapter(
           SupportedModel.YOUR_MODEL.value,
           _build_your_model,
           force=True,
       )

worker 在 ``init_worker`` 时通过
``get_sglang_adapter_cls(model_type)`` 查表。如果漏掉注册，Worker 初始化时会报告
没有为对应 ``model_type`` 注册 sglang adapter。


步骤五：添加模型 YAML 和评测 YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

推荐分别维护模型配置和 SGLang 评测配置：

.. code-block:: text

   examples/embodiment/config/model/<your_model>.yaml
   evaluations/<benchmark>/<your_model>_eval_sglang.yaml

模型配置描述模型本身的固定结构，例如动作维度、动作时域和输入图像大小；
评测 YAML 描述 checkpoint、环境、资源、Server 启动参数和 HTTP 参数。这样，同一个
模型配置可以被多个 YAML 配置文件复用。


步骤六：测试和调试
~~~~~~~~~~~~~~~~~~~~~~~~~

第一次运行时建议将 ``env.eval.total_num_envs`` 降低，并依次确认：

1. 日志中的 rollout Worker 类型为 ``SGLangEmbodiedWorker``，server Worker 类型为
   ``SGLangServerWorker``；
2. 日志打印的 ``multimodal_gen sglang serve: launching in-process ...`` 行包含
   正确的 ``pipeline``、``backend``、``tp_size`` 和 GPU；
3. ``/health`` 能在超时前返回（server 首次编译 + 权重加载耗时较长，需留足时间）；
4. Policy 发出的请求能被 action endpoint 正确解析；
5. Server 输出的 action 维度和 dtype 符合约定；
6. 反归一化后的 action shape 与仿真器要求一致；
7. 小规模成功后再增加环境并行数和模型并行度。


第二部分：以 DreamZero 为例
===========================

DreamZero 的 SGLang 评测路径由以下文件组成：

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 文件
     - 作用
   * - ``rlinf/config.py``
     - 注册 ``dreamzero`` 并加入 ``EMBODIED_MODEL``
   * - ``rlinf/models/embodiment/dreamzero/sglang_adapter.py``
     - ``DreamZeroSGLangAdapter``：观测变换、请求构造、响应解析和动作后处理
   * - ``rlinf/models/embodiment/sglang_adapter.py``
     - adapter registry（``register_sglang_adapter`` / ``get_sglang_adapter_cls``）， DreamZero adapter
   * - ``examples/embodiment/config/model/dreamzero_5b.yaml``
     - DreamZero 5B 模型配置
   * - ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``
     - LIBERO-Spatial 的 SGLang 评测 YAML


代码适配
--------

注册模型
~~~~~~~~

DreamZero 在 ``rlinf/config.py`` 中的注册如下：

.. code-block:: python

   SupportedModel.DREAMZERO = SupportedModel.register("dreamzero", force=True)

同时，``SupportedModel.DREAMZERO`` 被加入 ``EMBODIED_MODEL``。因此评测 YAML
可以使用：

.. code-block:: yaml

   rollout:
     model:
       model_type: dreamzero


Adapter 适配
~~~~~~~~~~~~~~~~~

``sglang_adapter.py`` 用\ **一个类** ``DreamZeroSGLangAdapter`` 承载全部 DreamZero
适配逻辑（观测/动作变换 + 请求组装 + 响应解析）：

1. ``__init__(cfg, rank)`` 复用训练数据变换（``build_dreamzero_composed_transform``）
   完成观测转换与动作反变换的准备；不加载大模型、不建 HTTP client；
2. ``build_request(env_obs, mode)`` 把 env observation 转成 server 请求 payload，
   并返回一份 ``state``\ （转换后的 observation）供 ``parse_response`` 复用；
3. ``parse_response(resp, state)`` 从响应里取出归一化动作，反归一化为环境尺度动作；
4. ``action_path`` 声明动作端点 ``/v1/actions/generations``。

**HTTP 请求由 ``SGLangEmbodiedWorker`` 执行**——adapter 只构造 payload、解析响应，
不发请求；server 参数也完全来自 YAML 的 ``rollout.sglang.server`` 块。

adapter 通过 builder 注册（见
``rlinf/models/embodiment/sglang_adapter.py``）：

.. code-block:: python

   register_sglang_adapter(
       SupportedModel.DREAMZERO.value,
       _build_dreamzero_sglang_adapter,   # imports DreamZeroSGLangAdapter
       force=True,
   )

一次推理的完整数据流为：

.. code-block:: text

   RLinf env_obs
       │
       ├── build_request()
       │     ├── _observation_convert()
       │     │     main_images       → video.image
       │     │     wrist_images      → video.wrist_image
       │     │     states            → state.state
       │     │     task_descriptions → annotation.task
       │     ├── _normalize_obs()   dataset transform + metadata 归一化
       │     │     （文本不在客户端 tokenize，交由 SGLang server 处理）
       │     └── 组装 payload（原始 prompt 文本 + 归一化 observation）
       │
       ├── worker: POST /v1/actions/generations（msgpack）
       │
       └── parse_response()
             ├── _unapply()  [B, H, max_action_dim] → 环境尺度动作
             └── actions [B, H, action_dim]

以 ``embodiment_tag: libero_sim`` 为例，``main_images`` 和
``wrist_images`` 会被转换成外部相机与腕部相机两个视频模态，``states`` 转换为
机器人状态，``task_descriptions`` 转换为语言指令。反变换会按照 metadata
切出 LIBERO 所需的动作维度，并将 gripper 动作二值化为 ``-1`` 或 ``1``。

如果要让 DreamZero 支持新的仿真器，通常还需要在
``rlinf/data/datasets/dreamzero/data_transforms/`` 中添加对应的
``embodiment_tag``、``RolloutObsLayout``、模态定义、训练 prompt 格式和
embodiment id。


HTTP 请求
~~~~~~~~~~~~~

adapter 的 ``build_request`` 组装发往 ``POST /v1/actions/generations`` 的 payload，
由 worker 用 ``http_client.post(..., msgpack=True)`` 发送。payload 的逻辑结构如下
（用 msgpack 传输，Tensor 和 ndarray 不需要先展开成大列表）：

.. code-block:: json

   {
     "model": "/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-repacked",
     "input": {
       "prompt": [
         "<training-format prompt>"
       ],
       "observation": {}
     },
     "parameters": {
       "session_ids": [
         "rlinf-eval-r0-stage0-slot0"
       ],
       "reset_mask": [
         false
       ],
       "negative_prompts": [
         "<negative prompt>"
       ],
       "seed": 1140
     },
     "runtime": {
       "response_format": "envelope",
       "output_format": "numpy"
     }
   }

其中：

- ``input.prompt`` 是每个环境的 **原始指令文本**\ （不在客户端 tokenize，由 SGLang
  server 自行 tokenize）；
- ``input.observation`` 是归一化后的模型输入（``_normalize_obs`` 的输出，去掉
  prompt/token 相关键）；
- ``parameters.session_ids`` 标识每个逻辑环境槽位，用于 Server 复用视频或文本缓存；
- ``parameters.reset_mask`` 用于在下一次请求前清理对应 session 的缓存（评测默认
  全为 ``false``）；
- ``parameters.negative_prompts`` 是负向提示词；
- ``parameters.seed`` 来自 ``rollout.sglang.seed``。

worker 从以下位置读取 Server 返回的归一化动作，并交给 ``parse_response``：

.. code-block:: python

   response["data"][0]["action"]["values"]

该动作仍处于 DreamZero 的归一化、补齐后的动作空间，不能直接发送给环境，必须经过
``DreamZeroSGLangAdapter._unapply``\ （在 ``parse_response`` 内部调用）。


Server 参数和 Pipeline 配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

server 的启动参数完全来自 YAML 的 ``rollout.sglang.server`` 块。该块被原样转发给
``sglang.multimodal_gen`` 的 ``ServerArgs.from_kwargs(**)``：顶层参数
``ServerArgs`` 字段，``pipeline_config`` 子块由 ``from_kwargs`` dispatch 到
``DreamZeroPipelineConfig``\ （按 ``pipeline_class_name`` 匹配），并把
``backend`` / ``disagg_role`` 等字符串转成枚举。结构如下：

.. code-block:: yaml

   rollout:
     sglang:
       server:
         model_path: ${rollout.model.model_path}
         backend: sglang
         pipeline_class_name: DreamZeroPipeline
         tp_size: ${..tensor_parallel_size}
         num_gpus: 1
         attention_backend: TORCH_SDPA
         dit_cpu_offload: false
         cfg_parallel_degree: 1
         sp_degree: 1
         pipeline_config:            # → DreamZeroPipelineConfig
           cfg_scale: 5.0
           default_num_inference_steps: 16
           action_horizon: 16
           num_frames: 33
           synthetic_height: 160
           synthetic_width: 320
           dreamzero_compile_components: true
           dreamzero_max_sessions: 128

关键说明：

- ``host`` / ``port`` / ``master_port`` 由 ``SGLangServerWorker`` 在运行时
  填入（用 PortLock 串行分配的空闲端口），YAML 中不要手设；
- 顶层 ``ServerArgs`` 字段直接对应 ``sglang serve`` 参数（``backend``、
  ``attention_backend``、``tp_size``、``num_gpus``、``dit_cpu_offload``、
  ``cfg_parallel_degree``、``sp_degree`` 等）；
- ``pipeline_config`` 子块对应 ``DreamZeroPipelineConfig`` 字段，命名与 sglang
  侧一致（``cfg_scale``、``default_num_inference_steps``、
  ``dreamzero_compile_components``、``dreamzero_max_sessions`` 等）；
- 模型路径都指向 ``rollout.model.model_path``，由 Server 按 checkpoint 布局
  加载不同组件。

模型 shape 相关字段（``num_frames``、tile 参数、``synthetic_*``、
``action_horizon`` 等）必须与 checkpoint 训练配置保持一致，不能只根据显存情况
随意修改。


模型 YAML
---------

DreamZero 模型配置位于
``examples/embodiment/config/model/dreamzero_5b.yaml``。与 SGLang 评测直接相关的
字段可以概括为：

.. code-block:: yaml

   model_type: "dreamzero"

   model_path: null
   metadata_json_path: null

   action_dim: 32
   state_horizon: 1
   action_horizon: 24
   num_action_per_block: 24
   max_action_dim: 32
   max_state_dim: 64

   target_video_height: 176
   target_video_width: 320

   action_head_cfg:
     config:
       num_frames: 33
       tile_size_height: 34
       tile_size_width: 34
       tile_stride_height: 18
       tile_stride_width: 16
       tiled: false

主要字段含义如下：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 字段
     - 含义
   * - ``model_type``
     - adapter registry 的查找键，必须为 ``dreamzero``
   * - ``model_path``
     - 完整 checkpoint 路径；评测 YAML 中必须覆盖
   * - ``metadata_json_path``
     - 数据集统计量，用于状态和动作归一化及反归一化
   * - ``action_dim`` / ``max_action_dim``
     - 模型动作宽度；``max_action_dim`` 是多 embodiment 使用的 padded 宽度
   * - ``action_horizon``
     - Server 一次预测的动作时域
   * - ``num_action_per_block``
     - DreamZero DiT 每个 action block 使用的动作数量
   * - ``target_video_height`` / ``target_video_width``
     - 数据变换和 Pipeline 使用的视频分辨率
   * - ``action_head_cfg.config``
     - DreamZero 网络结构与视频、tile 参数；通常应与训练 checkpoint 保持一致

模型配置中的值为默认值。评测 YAML 可以覆盖这些值，但
``action_horizon``、``num_action_per_block``、视频尺寸和模型结构相关字段必须与
当前 checkpoint 匹配。例如，本节的 LIBERO SGLang 配置将 horizon 从默认的
``24`` 覆盖为 ``16``，这是该评测 checkpoint 的要求，不是通用推荐值。


评测 YAML 详解
--------------

完整示例位于
``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``。下面按配置块说明。

Hydra defaults
~~~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     - env/libero_spatial@env.eval
     - model/dreamzero_5b@rollout.model
     - override hydra/job_logging: stdout

   hydra:
     searchpath:
       - file://${oc.env:EMBODIED_PATH}/config/

这里将 ``libero_spatial`` 仿真器配置组合到 ``env.eval``，将
``dreamzero_5b`` 模型配置组合到 ``rollout.model``。``run_eval.sh`` 会设置
``EMBODIED_PATH``。


集群与 Runner 配置
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       env,rollout: all

   runner:
     task_type: embodied_eval
     max_epochs: 1
     only_eval: True
     ckpt_path: null

字段说明：

- ``component_placement`` 指定 env 和 rollout 的放置方式；
- ``task_type: embodied_eval`` 选择 ``EmbodiedEvalRunner``\ （RL 训练使用 ``task_type: embodied`` 和 ``only_eval: false``）；
- server 直接从 ``rollout.model.model_path`` 加载权重时，``ckpt_path`` 保持为 ``null``；评测 RLinf checkpoint 时可将其指向目标权重文件。


环境并行与评测步数
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 1
       total_num_envs: 128
       auto_reset: True
       ignore_terminations: True
       max_episode_steps: 480
       max_steps_per_rollout_epoch: 1920
       group_size: 1
       use_fixed_reset_state_ids: True
       use_ordered_reset_state_ids: True
       is_eval: True

``total_num_envs`` 是并行环境总数，必须能被实际的环境 Worker 数、
``pipeline_stage_num`` 和 ``group_size`` 整除。

``max_steps_per_rollout_epoch`` 必须能被
``rollout.model.num_action_chunks`` 整除。Worker 按下面的公式计算每轮需要请求多少
次动作：

.. code-block:: python

   n_eval_chunk_steps = (
       env.eval.max_steps_per_rollout_epoch
       // rollout.model.num_action_chunks
   )

示例中 ``1920 // 16 = 120``。``num_action_chunks`` 必须与一次推理返回、环境实际
执行的动作 chunk 长度保持一致。


SGLang Worker 分发
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     rollout_backend: "sglang"
     pipeline_stage_num: 1
     return_logprobs: false

     sglang:
       serving_mode: "embodied"      # rollout worker = SGLangEmbodiedWorker
       backend_type: server
       launch_server: true           # 每个 worker 自行管理 server 子进程
       launch_router: false          # 设为 true 可由 SGLang router 统一转发请求
       router_group_name: SGLangRouterGroup

- ``rollout_backend: sglang`` 选择 SGLang 后端；
- ``serving_mode: embodied`` 选择 ``SGLangEmbodiedWorker``，由每个 rollout worker 在分配到的 GPU 上拉起具身 SGLang Diffusion server（经由 ``sglang.multimodal_gen`` 的 ``dispatch_launch``）；
- ``launch_router: false`` 会让请求直接发往各自 worker 持有的本地 loopback server。不依赖单实例会话状态的适配器可以在使用支持转发 ``/v1/actions/generations`` 与 ``/rollout/generate`` 的 router 构建时启用 ``launch_router: true``。

``pipeline_stage_num`` 会参与每个 rollout rank 的 eval batch size 计算；``return_logprobs: false`` 表示评测不需要策略概率。


Server 启动与并行配置
~~~~~~~~~~~~~~~~~~~~~

``rollout.sglang`` 顶层放 RLinf 私有参数（HTTP 客户端、启动开关、router 配置），``server`` 子块放转发给 SGLang 的 ``ServerArgs`` 字段：

.. code-block:: yaml

   rollout:
     sglang:
       tensor_parallel_size: 1
       pipeline_parallel_size: 1
       launch_server: true
       launch_router: false
       seed: 1140

       server:                       # → ServerArgs.from_kwargs
         backend: sglang
         pipeline_class_name: DreamZeroPipeline
         tp_size: ${..tensor_parallel_size}
         attention_backend: TORCH_SDPA
         dit_cpu_offload: false
         cfg_parallel_degree: 1
         sp_degree: 1
         pipeline_config:            # → DreamZeroPipelineConfig
           cfg_scale: 5.0
           default_num_inference_steps: 16
           dreamzero_compile_components: true

RLinf 私有参数字段说明：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 字段
     - 含义
   * - ``launch_server`` / ``launch_router``
     - 是否由每个 rollout worker 拉起本地 server 子进程（必须为 ``true``）以及驱动脚本是否启动可选的 router
   * - ``seed``
     - 每次 action 请求使用的随机种子

``server`` 子块的字段对应 ``ServerArgs`` 字段（``backend``、``attention_backend``、``dp_size``、``tp_size``、``sp_degree``、``cfg_parallel_degree``、``dit_cpu_offload`` 等），``pipeline_config`` 子块对应 ``DreamZeroPipelineConfig`` 字段（``cfg_scale``、``default_num_inference_steps``、``dreamzero_*`` 等）。请不要在配置中设置 ``server.num_gpus``：``EmbodiedComponentPlacement`` 会自动按 ``dp_size * tp_size * sp_degree * cfg_parallel_degree`` 推导每个 server 实例占用的 GPU 数，并将 rollout 硬件 rank 打包为对应数量的 worker 与 server。

**HTTP 端口、master_port 与 scheduler_ports 由 worker 的 PortLock 自动分配空闲端口**，无需也不要在 YAML 中手设 ``host`` / ``port`` / ``port_base`` / ``master_port_base``，以避免并发 rank 争抢同一端口。


HTTP Client 配置
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     sglang:
       http_timeout_s: 600
       http_max_retries: 5
       http_retry_backoff_s: 1.0

这些字段由 ``SGLangEmbodiedWorker`` 读取，用于创建向 server 发请求的
:class:`InferenceHTTPClient`。字段说明：

- ``http_timeout_s`` 是单次 action 请求的超时时间；
- ``http_max_retries`` 是遇到连接错误或可重试 5xx 时的重试次数；
- ``http_retry_backoff_s`` 是线性退避的基础等待时间。

.. note::

   具身 action 请求固定使用 **msgpack**\ （图像/大 batch 下 ndarray 不必展开成巨大的
   JSON 列表），worker 调用 ``http_client.post(..., msgpack=True)``，无需也不再有
   ``http_payload_format`` 配置项。


DreamZero 模型与数据配置
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     model:
       model_type: "dreamzero"
       precision: bf16
       model_path: /path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-repacked
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: "libero_sim"

       action_horizon: 16
       num_action_chunks: 16
       num_action_per_block: 16
       target_video_height: 160
       target_video_width: 320

这些字段中：

- ``model_path`` 是 SGLang Server 加载的 checkpoint；
- ``metadata_json_path`` 提供训练数据的归一化统计量。若未显式指定，代码只会尝试
  ``model_path/experiment_cfg/metadata.json``；
- ``embodiment_tag`` 选择观测布局、模态变换、动作后处理和 embodiment id；
- ``action_horizon`` 是模型一次生成的动作长度；
- ``num_action_chunks`` 是 RLinf 每次向环境发送并执行的动作长度；
- ``num_action_per_block`` 是 DreamZero 网络结构参数；
- ``target_video_height`` 和 ``target_video_width`` 必须与 checkpoint 及 Server
  Pipeline 一致。

当前示例中的三个动作长度均为 ``16``。如果新 checkpoint 的生成 horizon 与实际
执行的 chunk 不同，需要同时确认 Server 输出、Policy 切片和环境执行策略，而不能只
修改其中一个字段。


生成 metadata
~~~~~~~~~~~~~

DreamZero 的观测归一化和动作反归一化依赖数据集 metadata。LIBERO 示例可以使用：

.. code-block:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

生成后将路径写入 ``rollout.model.metadata_json_path``。metadata 必须来自与训练
checkpoint 匹配的数据和 embodiment；使用错误统计量可能不会立即报错，但会导致动作
尺度错误。


运行评测
--------

准备好 DreamZero 依赖、支持 ``DreamZeroPipeline`` 的 SGLang 环境、checkpoint 和
metadata 后，在仓库根目录运行：

.. code-block:: bash

   export DREAMZERO_PATH=/path/to/DreamZero

   bash evaluations/run_eval.sh \
     libero \
     libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
     rollout.model.metadata_json_path=/path/to/metadata.json

详细的 DreamZero SGLang evaluation 流程见 :doc:`../evaluations/guides/dreamzero_sglang`。

首次联调可以覆盖环境数量：

.. code-block:: bash

   bash evaluations/run_eval.sh \
     libero \
     libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
     rollout.model.metadata_json_path=/path/to/metadata.json \
     env.eval.total_num_envs=4

``run_eval.sh`` 会设置 ``EMBODIED_PATH``，并将 ``DREAMZERO_PATH`` 加入
``PYTHONPATH``。请确保 ``DREAMZERO_PATH`` 指向包含 ``groot`` 包的 DreamZero
代码目录。

常见问题
--------

Worker 不是 ``SGLangEmbodiedWorker``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

检查是否同时设置：

.. code-block:: yaml

   rollout:
     rollout_backend: sglang
     sglang:
       serving_mode: embodied


提示 sglang adapter 未注册
~~~~~~~~~~~~~~~~~~~~~~~~~~~

确认以下三个名称一致，并确认 adapter 已在
``rlinf/models/embodiment/sglang_adapter.py`` 的
``_register_builtin_sglang_adapters()`` 中注册：

.. code-block:: text

   SupportedModel.register("dreamzero")
   register_sglang_adapter("dreamzero", _build_dreamzero_sglang_adapter, force=True)
   rollout.model.model_type: dreamzero


Server 无法启动
~~~~~~~~~~~~~~~

日志会打印 ``multimodal_gen sglang serve: launching in-process ...`` 行和
server 子进程的输出（直接流到 Ray actor 的 stdout/stderr）。优先检查：

- 当前 SGLang 安装是否包含 ``DreamZeroPipeline`` 和 action endpoint；
- checkpoint 路径和组件布局是否正确；
- ``num_gpus``、``tp_size``、``sp_degree`` 与可用 GPU 是否匹配；
- 端口是否被占用（HTTP/master_port 由 PortLock 自动分配，但仍可能与外部进程
  冲突）；
- server 首次编译 + 权重加载耗时较长时，是否给了足够的启动等待时间。


请求本地 Server 超时
~~~~~~~~~~~~~~~~~~~~

``NO_PROXY`` 必须包含 ``127.0.0.1,localhost``，否则 ``/health`` 和 action 请求可能
被发送到上游代理。Worker 启动本地 Server 时会自动设置；若手动启动或单独测试 Client，
需要自行检查代理环境变量。


动作 shape 或尺度错误
~~~~~~~~~~~~~~~~~~~~~

依次核对：

1. Server 输出是否为 ``[B, H, max_action_dim]``；
2. ``action_horizon``、``num_action_chunks``、``num_action_per_block`` 是否与
   checkpoint 一致；
3. ``embodiment_tag`` 是否选择了正确的数据变换；
4. ``metadata_json_path`` 是否来自匹配的训练数据；
5. ``unapply`` 后的动作维度是否符合环境；
6. 图像分辨率和视角顺序是否与训练时一致。


显存占用异常
~~~~~~~~~~~~

确认 sglang adapter 没有导入并创建本地 DreamZero 大模型。模型只能由
``sglang serve`` 加载。还需要检查 ``max_sessions``、eval batch size、编译选项和
并行配置；DreamZero 默认将 ``max_sessions`` 设为当前 Worker 的 eval batch size，
增加并行环境数也会增加 Server 侧缓存需求。