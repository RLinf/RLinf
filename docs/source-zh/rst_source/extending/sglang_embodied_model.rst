在 SGLang Server 中适配具身模型
====================================

本文介绍如何将一个已适配 SGLang Server 的具身 VLA 模型接入
RLinf 的评测 rollout，并使用 RLinf 支持的各类仿真器进行模型评测。

全文分为两部分：

- 第一部分说明适配任意新模型时需要完成的步骤和接口约定；
- 第二部分以 DreamZero 为例，逐项说明需要修改的代码和 YAML 配置。

.. note::

   本文只介绍 **eval rollout / sglang-serve** 路径。该路径负责在评测时将环境观测
   转换成动作，不包含训练侧的模型注册、FSDP Policy 或 SFT 适配。训练侧适配请参考
   :doc:`使用 FSDP 添加新模型 <new_model_fsdp>` 和
   :doc:`添加新的 SFT 模型 <new_model_sft>`。


第一部分：适配新模型
====================

整体架构
--------

SGLang 具身评测路径将通用逻辑与模型逻辑分开：

- ``SGLangEmbodiedWorker`` 负责启动和关闭 ``sglang serve`` 子进程、轮询
  ``/health``、分配端口，以及通过 channel 与环境 Worker 交换 observation 和 action；
- ``EmbodiedActionPolicy`` 的子类负责模型特有的 observation 预处理、HTTP 请求、
  响应解析和 action 后处理。

因此，接入新模型时通常 **不需要修改**
``rlinf/workers/rollout/sglang/sglang_embodied_worker.py``。模型由
``rollout.model.model_type`` 选择，调用关系如下：

.. code-block:: text

   rollout.model.model_type: "<your_model>"
                 │
                 ▼
   SGLangEmbodiedWorker.init_worker()
                 │
                 ├── get_action_policy_cls("<your_model>")
                 ├── 启动 sglang serve
                 └── 创建 YourActionPolicy
                              │
                              ├── 将 env_obs 转换为模型输入
                              ├── 请求模型的 action endpoint
                              └── 返回 [N, H, D] 动作

要进入这条调用链，配置中必须同时满足以下四项：

.. code-block:: yaml

   runner:
     task_type: embodied_eval
     only_eval: true

   rollout:
     rollout_backend: sglang
     sglang:
       serving_mode: embodied
     model:
       model_type: "<your_model>"

其中 ``serving_mode: embodied`` 不可省略。否则，RLinf 会创建普通的
``SGLangWorker``，而不是 ``SGLangEmbodiedWorker``，导致模型无法正常工作。


适配步骤
--------

下面按照推荐顺序说明新模型需要完成的工作。

步骤一：确认 SGLang Server 的动作接口
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 的 Action Policy 是 SGLang Server 的客户端。在编写 RLinf 代码前，先确认
SGLang 侧已经具备以下能力：

1. ``sglang serve`` 能够加载目标模型或目标 Pipeline；
2. SGLang Server 已为该模型提供接收批量 observation、返回批量 action 的 VLA 接口；
3. 请求和响应格式固定，并能够表达模型所需的图像、文本、状态和缓存信息。

.. warning::

   RLinf 仓库中的 Action Policy 只负责 action 请求与转换。模型 Pipeline、动作路由及
   相关 ``sglang serve`` 参数仍需在所使用的 SGLang 版本中实现。


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

``"your_model"``、Action Policy 装饰器中的名称和 YAML 中的
``rollout.model.model_type`` 必须一致。Policy Registry 查找时不区分大小写，
但仍建议全部使用小写。


步骤三：实现 Action Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

在以下目录中新建模型文件：

.. code-block:: text

   rlinf/workers/rollout/sglang/action_policies/<your_model>.py

Policy 需要继承 ``EmbodiedActionPolicy``，并通过装饰器注册：

.. code-block:: python

   import torch

   from rlinf.workers.rollout.sglang.action_policies.base import (
       EmbodiedActionPolicy,
   )
   from rlinf.workers.rollout.sglang.action_policies.registry import (
       register_action_policy,
   )


   @register_action_policy("your_model")
   class YourActionPolicy(EmbodiedActionPolicy):
       def __init__(self, cfg, server_url, rank):
           super().__init__(cfg, server_url, rank)
           # 在这里创建轻量的 transform 和 HTTP client。

       def infer(self, env_obs, mode="eval"):
           # 1. 将 RLinf env_obs 转换为模型输入。
           # 2. 归一化并向 SGLang Server 发送请求。
           # 3. 解析响应并将动作反归一化。
           actions = ...
           info = {
               "prev_logprobs": ...,
               "prev_values": ...,
               "forward_inputs": ...,
           }
           return torch.as_tensor(actions, dtype=torch.float32), info

``infer`` 是唯一必须实现的方法，其接口输入为：

- 输入 ``env_obs`` 是按 batch 组织的环境观测字典；
- 通用字段包括 ``main_images``、``task_descriptions``，模型还可以使用
  ``wrist_images``、``states`` 或其它视角；
- 输出 ``actions`` 必须是 Tensor，shape 为
  ``[N, num_action_chunks, action_dim]``；
- 输出 ``info`` 是附加信息字典。该接口为未来的训练扩展预留；目前可以像 DreamZero
  一样返回 ``prev_logprobs``、``prev_values`` 和 ``forward_inputs``；
- 当前 SGLang embodied Worker 只用于评测。若 Policy 不支持训练模式，应对
  ``mode != "eval"`` 明确抛出 ``NotImplementedError``。后续计划支持将 SGLang
  作为 rollout worker，用于具身模型训练。

.. important::

   不要在 Action Policy 中加载模型。模型应只存在于 ``sglang serve``
   子进程中。Policy 中只保留数据变换、HTTP Client 和少量请求上下文，否则会造成
   重复加载权重和额外显存占用。


步骤四：导入 Policy，触发注册
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``rlinf/workers/rollout/sglang/action_policies/__init__.py`` 中添加：

.. code-block:: python

   from rlinf.workers.rollout.sglang.action_policies import your_model  # noqa: F401,E401

装饰器只有在模块被导入时才会执行。如果漏掉这一步，Worker 初始化时会报告没有为
对应 ``model_type`` 注册 Action Policy。


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

1. 日志中的 Worker 类型为 ``SGLangEmbodiedWorker``；
2. 日志打印的 ``sglang serve`` 命令包含模型专用参数；
3. ``/health`` 能在 ``spawn_timeout`` 内返回；
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
   * - ``rlinf/workers/rollout/sglang/action_policies/dreamzero.py``
     - 观测变换、HTTP Client、Serve 参数和动作后处理
   * - ``rlinf/workers/rollout/sglang/action_policies/__init__.py``
     - 导入 DreamZero Policy
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


Policy 适配
~~~~~~~~~~~~~~~~~

``dreamzero.py`` 将模型适配拆成四步：

1. ``DreamZeroActionRequest`` 和 ``DreamZeroActionResult`` 定义请求与响应；
2. ``HttpDreamZeroActionClient`` 负责编码、重试、发送和解析 HTTP 请求；
3. ``_DreamZeroActionAdapter`` 复用训练数据变换，完成观测转换与动作转换；
4. ``DreamZeroActionPolicy`` 实现 RLinf 接口，并生成 DreamZero Server 启动参数。

Policy 的注册代码如下：

.. code-block:: python

   @register_action_policy("dreamzero")
   class DreamZeroActionPolicy(EmbodiedActionPolicy):
       ...

一次推理的完整数据流为：

.. code-block:: text

   RLinf env_obs
       │
       ├── observation_convert()
       │     main_images       → video.image
       │     wrist_images      → video.wrist_image
       │     states            → state.state
       │     task_descriptions → annotation.task
       │
       ├── normalize_obs()
       │     dataset transform + metadata 归一化 + prompt tokenize
       │
       ├── POST /v1/actions/generations
       │
       ├── unapply()
       │     [B, H, max_action_dim] → 环境尺度动作
       │
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

DreamZero Client 调用 ``POST /v1/actions/generations``。JSON 形式如下；
使用 msgpack 时逻辑字段保持一致，但 Tensor 和 ndarray 不需要先展开成大列表：

.. code-block:: json

   {
     "model": "/path/to/dreamzero_checkpoint",
     "parameters": {
       "action_input": {},
       "session_ids": [
         "rlinf-eval-r0-stage0-slot0"
       ],
       "reset_mask": [
         false
       ],
       "prompts": [
         "<training-format prompt>"
       ],
       "negative_prompts": [
         "text_negative:missing"
       ],
       "seed": 1140
     },
     "runtime": {
       "response_format": "envelope",
       "output_format": "numpy"
     }
   }

其中：

- ``action_input`` 是 ``_DreamZeroActionAdapter.normalize_obs`` 的输出；
- ``session_ids`` 标识每个逻辑环境槽位，用于 Server 复用视频或文本缓存；
- ``reset_mask`` 用于在下一次请求前清理对应 session 的缓存；
- Python dataclass 中的 ``prompt_cache_keys`` 在 HTTP payload 中发送为
  ``prompts``，``negative_prompt_cache_keys`` 发送为 ``negative_prompts``；
- ``seed`` 来自 ``rollout.sglang.seed``。

Client 从以下位置读取 Server 返回的归一化动作：

.. code-block:: python

   response["data"][0]["action"]["values"]

该动作仍处于 DreamZero 的归一化、补齐后的动作空间，不能直接发送给环境，必须经过
``_DreamZeroActionAdapter.unapply``。


Server 参数和 Pipeline 配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SGLangEmbodiedWorker._model_specific_sglang_serve_args`` 会追加以下类型的参数：

.. code-block:: text

   --backend sglang
   --pipeline DreamZeroPipeline
   --pipeline-config-path <tmpdir>/dreamzero_pipeline_rank<rank>.json
   --sp-degree <sp_size>
   --cfg-parallel-size <cfg_parallel_degree>
   --dreamzero-dit-path <model_path>
   --dreamzero-vae-path <model_path>
   --dreamzero-text-encoder-path <model_path>
   --dreamzero-image-encoder-path <model_path>

这些模型路径都指向 ``rollout.model.model_path``，由 Server 根据 checkpoint 布局
加载模型的不同组件。

Policy 还会生成 ``DreamZeroPipelineConfig`` 使用的 JSON。主要映射关系如下：

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Pipeline JSON 字段
     - 来源
   * - ``dreamzero_compile_components``
     - ``rollout.sglang.compile_components``
   * - ``dreamzero_sequence_parallel_size``
     - ``rollout.sglang.sp_degree``，未设置时读取 ``sp_size``
   * - ``dreamzero_max_sessions``
     - ``rollout.sglang.max_sessions``，默认等于本 Worker 的 eval batch size
   * - ``cfg_scale``
     - ``rollout.sglang.cfg_scale``
   * - ``action_horizon``
     - ``rollout.model.action_horizon``
   * - ``num_inference_steps``
     - ``rollout.sglang.num_inference_steps``
   * - ``num_frames``、tile 参数
     - ``rollout.model.action_head_cfg.config``
   * - ``synthetic_height`` / ``synthetic_width``
     - ``rollout.model.target_video_height`` / ``target_video_width``

这意味着模型 shape 相关字段必须与 checkpoint 训练配置保持一致，不能只根据显存情况
随意修改。


模型 YAML
---------

DreamZero 模型配置位于
``examples/embodiment/config/model/dreamzero_5b.yaml``。与 SGLang 评测直接相关的
字段可以概括为：

.. code-block:: yaml

   model_type: "dreamzero"

   model_path: null
   tokenizer_path: null
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
     - Action Policy Registry 的查找键，必须为 ``dreamzero``
   * - ``model_path``
     - 完整 checkpoint 路径；评测 YAML 中必须覆盖
   * - ``tokenizer_path``
     - 文本 tokenizer 路径；DreamZero 示例使用 ``google/umt5-xxl``
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
- ``task_type: embodied_eval`` 选择 ``EmbodiedEvalRunner``；
- ``only_eval: True`` 是必填项，``SGLangEmbodiedWorker`` 会对此断言；
- ``ckpt_path`` 在本路径中可以为 ``null``，Server 从
  ``rollout.model.model_path`` 加载权重；
- 当前脚本仅支持评测，不支持训练。


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
       serving_mode: "embodied"

- ``rollout_backend: sglang`` 选择 SGLang 后端；
- ``serving_mode: embodied`` 进一步选择 ``SGLangEmbodiedWorker``。

``pipeline_stage_num`` 会参与每个 rollout rank 的 eval batch size 计算；
``return_logprobs: false`` 表示评测不需要策略概率。


Server 启动与并行配置
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     sglang:
       spawn_timeout: 900
       attention_backend: "TORCH_SDPA"
       compile_components: true
       num_gpus: 1
       tp_size: 1
       sp_size: 1
       cfg_parallel_degree: 1
       dit_cpu_offload: false
       cfg_scale: 5.0
       num_inference_steps: 16
       seed: 1140

字段说明：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 字段
     - 含义
   * - ``spawn_timeout``
     - Worker 等待 ``/health`` 成功的最长秒数
   * - ``attention_backend``
     - 传给 ``sglang serve`` 的注意力后端
   * - ``compile_components``
     - 是否编译 DreamZero Pipeline 的相关组件
   * - ``num_gpus``
     - 每个 Server 使用的 GPU 数
   * - ``tp_size``
     - 传给 Server 的 tensor parallel 大小
   * - ``sp_size`` / ``sp_degree``
     - DreamZero sequence parallel 大小；若两者都设置，``sp_degree`` 优先
   * - ``cfg_parallel_degree``
     - classifier-free guidance 并行大小
   * - ``dit_cpu_offload``
     - 是否将 DiT 相关部分 offload 到 CPU
   * - ``cfg_scale``
     - DreamZero 推理使用的 CFG scale
   * - ``num_inference_steps``
     - 流匹配/扩散推理步数
   * - ``seed``
     - 每次 action 请求使用的随机种子

如果有多个 rollout rank，每个 rank 会启动独立的 Server。默认服务端口为
``port_base + rank * port_stride``。需要自定义时可以增加：

.. code-block:: yaml

   rollout:
     sglang:
       host: 127.0.0.1
       port_base: 30010
       port_stride: 100
       master_port_base: 30100

端口范围必须避免与其它任务或其它 rank 冲突。


HTTP Client 配置
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     sglang:
       http_timeout_s: 600
       http_max_retries: 5
       http_retry_backoff_s: 1.0
       http_payload_format: "msgpack"
       debug_sessions: false
       debug_batch: false

字段说明：

- ``http_timeout_s`` 是单次 action 请求的超时时间；
- ``http_max_retries`` 是遇到连接错误或可重试 5xx 时的重试次数；
- ``http_retry_backoff_s`` 是线性退避的基础等待时间；
- ``http_payload_format`` 支持 ``json`` 和 ``msgpack``。图像和大 batch 推荐使用
  ``msgpack``，避免把 ndarray 展开成巨大的 JSON 列表；
- ``debug_sessions`` 和 ``debug_batch`` 可在联调缓存或 shape 问题时打开，不建议在
  大规模评测中长期启用。


DreamZero 模型与数据配置
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   rollout:
     model:
       model_type: "dreamzero"
       precision: bf16
       model_path: /path/to/dreamzero_checkpoint
       tokenizer_path: google/umt5-xxl
       metadata_json_path: /path/to/metadata.json
       embodiment_tag: "libero_sim"

       action_horizon: 16
       num_action_chunks: 16
       num_action_per_block: 16
       target_video_height: 160
       target_video_width: 320

这些字段中：

- ``model_path`` 是 SGLang Server 加载的 checkpoint；
- ``tokenizer_path`` 只用于文本 tokenizer，不要用它代替 ``model_path``；
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
     rollout.model.model_path=/path/to/dreamzero_checkpoint \
     rollout.model.metadata_json_path=/path/to/metadata.json

详细的 DreamZero SGLang evaluation 流程见 :doc:`../evaluations/guides/dreamzero_sglang`。

首次联调可以覆盖环境数量：

.. code-block:: bash

   bash evaluations/run_eval.sh \
     libero \
     libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/dreamzero_checkpoint \
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


提示 Action Policy 未注册
~~~~~~~~~~~~~~~~~~~~~~~~~

确认以下三个名称一致，并确认 Policy 模块已在 ``action_policies/__init__.py`` 中
导入：

.. code-block:: text

   SupportedModel.register("dreamzero")
   @register_action_policy("dreamzero")
   rollout.model.model_type: dreamzero


Server 无法启动
~~~~~~~~~~~~~~~

Worker 会打印完整的 ``sglang serve`` 命令和 Server 日志路径。优先检查：

- 当前 SGLang 安装是否包含 ``DreamZeroPipeline`` 和 action endpoint；
- checkpoint 路径和组件布局是否正确；
- ``num_gpus``、``tp_size``、``sp_size`` 与可用 GPU 是否匹配；
- 端口是否被占用；
- ``spawn_timeout`` 是否足够覆盖首次编译和权重加载时间。


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

确认 Action Policy 没有导入并创建本地 DreamZero 大模型。模型只能由
``sglang serve`` 加载。还需要检查 ``max_sessions``、eval batch size、编译选项和
并行配置；DreamZero 默认将 ``max_sessions`` 设为当前 Worker 的 eval batch size，
增加并行环境数也会增加 Server 侧缓存需求。