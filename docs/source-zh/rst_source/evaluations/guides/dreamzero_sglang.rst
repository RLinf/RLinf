DreamZero SGLang 评测
=====================

概览
----

使用 RLinf 的 SGLang 具身后端在 LIBERO 上评测 SGLang-native Diffusers 格式的 DreamZero checkpoint。评测启动脚本拉起独立的 ``sglang serve`` 服务组并为各 rollout worker 分配 server URL；worker 运行 DreamZero 数据变换，通过 HTTP 请求动作块并推进 LIBERO 仿真。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      LIBERO-Spatial（``libero``）

   .. grid-item-card:: 支持模型
      :text-align: center

      DreamZero WAN2.2-5B（SGLang 后端）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`DreamZero SFT <../../examples/embodied/sft_dreamzero>`

安装与运行
----------

需要 CUDA 13.0 及以上版本。在 `sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_ 合入上游前，将包含该 PR 的 SGLang 仓库放在 ``./sglang``，并在同一虚拟环境中安装 ``diffusion`` extra。

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: LIBERO

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env libero --model dreamzero && source .venv/bin/activate
         pip install -e "./sglang/python[diffusion]"

      **2. 下载 LIBERO SFT Diffusers Checkpoint**

      .. code-block:: bash

         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers --local-dir ./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

      **3. 启动 LIBERO-Spatial 评测**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
           rollout.model.model_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

默认配置为 ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml``，使用 `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_。

.. dropdown:: 生成缺失的 Metadata

   checkpoint 通常自带 ``experiment_cfg/metadata.json``。如果缺失，可从本地 LIBERO 数据集目录（``./libero_dataset``）生成，并在启动命令中追加 ``rollout.model.metadata_json_path=./metadata.json``：

   .. code-block:: bash

      python toolkits/lerobot/generate_dreamzero_metadata.py \
        --preset libero_sim \
        --dataset-root ./libero_dataset \
        --output-metadata ./metadata.json

.. dropdown:: 关键配置与并行覆盖

   .. list-table::
      :header-rows: 1
      :widths: 38 62

      * - 字段
        - 作用
      * - ``rollout.rollout_backend: sglang``
        - 选择 SGLang rollout backend。
      * - ``rollout.sglang.serving_mode: embodied``
        - 启用具身 SGLang 服务模式，拉起 SGLang server 组并为各 rollout worker 分配推理 URL。
      * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
        - server 侧 DreamZero 去噪步数。
      * - ``rollout.sglang.server.pipeline_config.cfg_scale``
        - 动作推理的 classifier-free guidance 系数。
      * - ``rollout.sglang.server.cfg_parallel_degree``
        - 设为 ``2`` 时将 CFG 正负分支分配到不同 rank。
      * - ``rollout.sglang.server.tp_size``
        - DreamZero DiT 的 tensor parallel size。
      * - ``rollout.sglang.server.sp_degree``
        - DreamZero DiT 注意力序列的 sequence parallel size。
      * - ``rollout.model.model_path``
        - SGLang 加载的 DreamZero Diffusers checkpoint 目录。
      * - ``rollout.model.metadata_json_path``
        - 动作推理前后归一化所需的统计文件。
      * - ``rollout.model.num_action_chunks``
        - 每次请求返回的动作步数；``env.eval.max_steps_per_rollout_epoch`` 须能被该值整除。

   通过 Hydra 覆盖参数调整本地并行度：

   .. code-block:: bash

      bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
        rollout.sglang.server.num_gpus=2 \
        rollout.sglang.server.cfg_parallel_degree=2 \
        rollout.model.model_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

结果与常见问题
--------------

评测日志保存在 ``logs/<timestamp>-<config>/eval_embodiment.log``，记录 SGLang server 启动、端口就绪、逐回合结果与 ``eval/success_once``\ （详见 :doc:`../reference/results`）。该配置开启 ``auto_reset: True`` 与有序初始状态以较少并行环境覆盖完整套件，轨迹计数规则见 :ref:`LIBERO 评测配置 <libero-eval-config>`。

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 现象
     - 处理
   * - SGLang 找不到 model components
     - 确认 ``rollout.model.model_path`` 指向下载后的 Diffusers checkpoint 目录。
   * - metadata 加载失败
     - 将 ``rollout.model.metadata_json_path`` 设为针对 ``libero_sim`` 生成的 ``metadata.json``。
   * - 本地 HTTP 请求误走代理
     - 启动评测前导出 ``NO_PROXY=127.0.0.1,localhost``。
