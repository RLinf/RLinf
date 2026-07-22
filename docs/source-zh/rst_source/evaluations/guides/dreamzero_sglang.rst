DreamZero SGLang 评测
=====================

本文档说明如何通过 RLinf 的 SGLang embodied backend 运行 DreamZero LIBERO 评测。该路径用于只做推理评测的场景，要求 DreamZero checkpoint 已转换为 SGLang-native 的组件化目录。

与 :doc:`../../examples/embodied/sft_dreamzero` 中的原始 DreamZero eval 路径相比，SGLang backend 在运行时不需要 ``DREAMZERO_PATH``，也不依赖外部 DreamZero Python 包。RLinf rollout worker 会启动并管理本地 SGLang action server，通过 VLA action API 发送 batched observation，并将返回的 action chunk 反归一化后送入 LIBERO 环境。

安装测试环境
------------

安装 embodied、LIBERO 和 DreamZero SGLang 依赖：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env libero --model dreamzero-sglang \
     --torch 2.11.0 --python 3.11.14 --venv /path/to/dreamzero_test

安装包含 DreamZero 支持的 SGLang，并启用 ``diffusion`` extra：

.. code-block:: bash

   source /path/to/dreamzero_test/bin/activate
   cd /path/to/sglang_dreamzero
   pip install -e "python[diffusion]"

准备 Repacked Checkpoint
------------------------

从 `RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000 <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000>`_ 下载 LIBERO SFT checkpoint。
DreamZero SGLang backend 需要 repacked checkpoint。从 RLinf 仓库中将原始 checkpoint 转换一次：

.. code-block:: bash

   cd /path/to/RLinf
   python toolkits/difusser-like-weight-convert/dreamzero_repack.py \
     --path /path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000

默认输出目录与原 checkpoint 同级，并带 ``-repacked`` 后缀。后续将 ``rollout.model.model_path`` 指向该 repacked 目录。

repacked checkpoint 中通常应包含 ``experiment_cfg/metadata.json``。如果 checkpoint 中没有 metadata，可从 LIBERO 数据集生成，并显式设置 ``rollout.model.metadata_json_path``：

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
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000-repacked \
     rollout.model.tokenizer_path=/path/to/umt5-xxl

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
     - 启动 ``SGLangEmbodiedWorker``，由 worker 拉起并管理本地 SGLang action server。
   * - ``rollout.sglang.http_payload_format: msgpack``
     - 使用 DreamZero evaluation 路径的二进制 action payload。
   * - ``rollout.sglang.num_inference_steps``
     - 控制 server 侧 DreamZero denoising steps。
   * - ``rollout.sglang.cfg_scale``
     - action inference 使用的 classifier-free guidance scale。
   * - ``rollout.sglang.cfg_parallel_degree``
     - 设置为 ``2`` 时，将 CFG 的 positive / negative 分支切到不同 rank。
   * - ``rollout.sglang.tp_size``
     - DreamZero DiT tensor parallel size。
   * - ``rollout.sglang.sp_size``
     - DreamZero DiT attention sequence parallel size。
   * - ``rollout.model.model_path``
     - SGLang 加载的 repacked DreamZero checkpoint 目录。
   * - ``rollout.model.metadata_json_path``
     - action inference 前后归一化使用的统计信息。
   * - ``rollout.model.num_action_chunks``
     - 每次模型请求返回的 action 数量；``env.eval.max_steps_per_rollout_epoch`` 必须能被该值整除。

并行覆盖参数
------------

正式支持的 DreamZero SGLang evaluation 入口是 ``libero_spatial_dreamzero_eval_sglang``。本地实验需要调整并行度时，直接覆盖这个配置中的字段，不切换到其他 YAML：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.sglang.num_gpus=2 \
     rollout.sglang.cfg_parallel_degree=2 \
     rollout.model.model_path=/path/to/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000-repacked \
     rollout.model.tokenizer_path=/path/to/umt5-xxl

验证
----

评测日志写入 ``logs/<timestamp>-<config>/``。检查 ``eval_embodiment.log``，其中包含 SGLang server 启动命令、endpoint readiness、逐 episode 结果和最终 ``eval/success_once``。

LIBERO-Spatial SGLang 配置使用 ``auto_reset: True`` 和 ordered reset states，用较少并行环境覆盖完整 suite。LIBERO 轨迹计数规则见 :ref:`libero-eval-config`。

常见问题
--------

- 如果 SGLang 找不到 model components，确认 ``rollout.model.model_path`` 指向 repacked checkpoint，而不是原始 DreamZero checkpoint。
- 如果 metadata 加载失败，将 ``rollout.model.metadata_json_path`` 设置为为 ``libero_sim`` 生成的现有 ``metadata.json``。
- 如果本地 HTTP 请求意外经过 proxy，启动前设置 ``NO_PROXY=127.0.0.1,localhost``。
