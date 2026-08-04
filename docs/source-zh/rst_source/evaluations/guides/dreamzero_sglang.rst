DreamZero SGLang 评测
=====================

本文档说明如何通过 RLinf 的 SGLang embodied backend 运行 DreamZero LIBERO 评测。该路径用于只做推理评测的场景，要求 DreamZero checkpoint 已转换为 SGLang-native 的组件化目录。

与 :doc:`../../examples/embodied/sft_dreamzero` 中的原始 DreamZero eval 路径相比，SGLang backend 把庞大的 DreamZero 网络放在独立拉起的 ``sglang serve`` 进程里，而不是在 rollout worker 中加载。但 worker 仍然需要安装 DreamZero 的 transform 栈：``rlinf.models.embodiment.dreamzero.sglang_adapter`` 会 import ``rlinf.data.datasets.dreamzero.data_transforms``，后者依赖由 ``install_dreamzero_deps`` 安装的 ``groot`` 包。由 eval driver 拉起 SGLang server group 并把每个 server URL 下发给各 rollout worker；worker 只是一个瘦客户端，通过 VLA action API 向本 rank 分配到的 URL 发送 batched observation，并将返回的 action chunk 反归一化后送入 LIBERO 环境。

安装测试环境
------------

安装 embodied、LIBERO 和 DreamZero SGLang 依赖：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env libero --model dreamzero \
     --venv /path/to/dreamzero_test

DreamZero 支持仍在 SGLang PR 中，需使用包含
`sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_
的 SGLang 代码，并启用 ``diffusion`` extra：

.. code-block:: bash

   source /path/to/dreamzero_test/bin/activate
   cd /path/to/sglang_dreamzero
   pip install -e "python[diffusion]"

准备 Checkpoint
---------------

从
`RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_
下载 LIBERO SFT Diffusers checkpoint。将 ``rollout.model.model_path`` 指向下载后的 checkpoint 目录。

.. code-block:: bash

   hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
     --local-dir /path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

checkpoint 中通常应包含 ``experiment_cfg/metadata.json``。如果 checkpoint 中没有 metadata，可从 LIBERO 数据集生成，并显式设置 ``rollout.model.metadata_json_path``：

.. code-block:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root /path/to/libero \
     --output-metadata /path/to/metadata.json

运行 LIBERO-Spatial
-------------------

默认 SGLang 评测配置为 ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``。

.. code-block:: bash

   cd /path/to/RLinf
   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

如果使用自定义 metadata 文件，额外加入：

.. code-block:: bash

   rollout.model.metadata_json_path=/path/to/metadata.json

关键配置
--------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 字段
     - 作用
   * - ``rollout.rollout_backend: sglang``
     - 选择 SGLang rollout backend。
   * - ``rollout.sglang.serving_mode: embodied``
     - 选择 ``SGLangEmbodiedWorker``；SGLang serve 组由驱动脚本
       （``launch_sglang_router_and_server``）拉起，worker 连接 driver 分配的
       server URL 进行推理。
   * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
     - 控制 server 侧 DreamZero denoising steps。
   * - ``rollout.sglang.server.pipeline_config.cfg_scale``
     - action inference 使用的 classifier-free guidance scale。
   * - ``rollout.sglang.server.cfg_parallel_degree``
     - 设置为 ``2`` 时，将 CFG 的 positive / negative 分支切到不同 rank。
   * - ``rollout.sglang.server.tp_size``
     - DreamZero DiT tensor parallel size。
   * - ``rollout.sglang.server.sp_degree``
     - DreamZero DiT attention sequence parallel size。
   * - ``rollout.model.model_path``
     - SGLang 加载的 DreamZero Diffusers checkpoint 目录。
   * - ``rollout.model.metadata_json_path``
     - action inference 前后归一化使用的统计信息。
   * - ``rollout.model.num_action_chunks``
     - 每次模型请求返回的 action 数量；``env.eval.max_steps_per_rollout_epoch`` 必须能被该值整除。

并行覆盖参数
------------

正式支持的 DreamZero SGLang evaluation 入口是 ``libero_spatial_dreamzero_eval_sglang``。本地实验需要调整并行度时，直接覆盖这个配置中的字段，不切换到其他 YAML：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.sglang.server.num_gpus=2 \
     rollout.sglang.server.cfg_parallel_degree=2 \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

验证
----

评测日志写入 ``logs/<timestamp>-<config>/``。检查 ``eval_embodiment.log``，其中包含 SGLang server 启动命令、endpoint readiness、逐 episode 结果和最终 ``eval/success_once``。

LIBERO-Spatial SGLang 配置使用 ``auto_reset: True`` 和 ordered reset states，用较少并行环境覆盖完整 suite。LIBERO 轨迹计数规则见 :ref:`libero-eval-config`。

常见问题
--------

- 如果 SGLang 找不到 model components，确认 ``rollout.model.model_path`` 指向下载后的 Diffusers checkpoint 目录。
- 如果 metadata 加载失败，将 ``rollout.model.metadata_json_path`` 设置为为 ``libero_sim`` 生成的现有 ``metadata.json``。
- 如果本地 HTTP 请求意外经过 proxy，启动前设置 ``NO_PROXY=127.0.0.1,localhost``。
