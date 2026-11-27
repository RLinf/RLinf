DreamZero SGLang 评测
=====================

通过 RLinf 的 SGLang embodied 后端，使用 SGLang 原生 Diffusers 格式的 DreamZero checkpoint 在 LIBERO 上评测 DreamZero。

与 :doc:`../../examples/embodied/sft_dreamzero` 中的原生评测相比，每个 rollout worker 会在分配的 GPU 上独立启动并管理一个 SGLang Diffusion server 进程，不在 worker 内加载庞大的 DreamZero 网络。rollout worker 保留 DreamZero 数据变换栈，通过 VLA action API 向本地 server 发送批量观测，再将返回的动作块反归一化后送入 LIBERO。

安装测试环境
------------

DreamZero SGLang 评测需要 CUDA 13.0 及以上版本。

克隆 RLinf 并安装 embodied、LIBERO 与 DreamZero 依赖：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git ./RLinf-dreamzero
         cd ./RLinf-dreamzero
         bash requirements/install.sh embodied --env libero --model dreamzero --venv .venv-dreamzero
         source .venv-dreamzero/bin/activate

   .. tab-item:: 国内镜像

      .. code-block:: bash

         git clone https://gh-proxy.com/github.com/RLinf/RLinf.git ./RLinf-dreamzero
         cd ./RLinf-dreamzero
         bash requirements/install.sh embodied --env libero --model dreamzero --venv .venv-dreamzero --use-mirror
         source .venv-dreamzero/bin/activate

若虚拟环境中未包含 ``RLinf/dreamzero``，克隆并导出 ``DREAMZERO_PATH``：

.. code-block:: bash

   git clone https://github.com/RLinf/dreamzero.git ../dreamzero
   export DREAMZERO_PATH="$PWD/../dreamzero"

DreamZero 的 SGLang 服务支持在 `sgl-project/sglang#30679 <https://github.com/sgl-project/sglang/pull/30679>`_ 中维护。拉取该 PR 分支并安装带 ``diffusion`` extra 的版本：

.. code-block:: bash

   git clone https://github.com/sgl-project/sglang.git ../sglang-dreamzero
   git -C ../sglang-dreamzero fetch origin pull/30679/head:pr-30679
   git -C ../sglang-dreamzero checkout pr-30679
   uv pip install -e "../sglang-dreamzero/python[diffusion]"

准备 Checkpoint
---------------

从 `RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers <https://huggingface.co/RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers>`_ 下载 LIBERO SFT Diffusers checkpoint，并将 ``rollout.model.model_path`` 指向该目录：

.. tab-set::

   .. tab-item:: 海外源

      .. code-block:: bash

         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
           --local-dir ./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

   .. tab-item:: 国内镜像

      .. code-block:: bash

         export HF_ENDPOINT=https://hf-mirror.com
         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers \
           --local-dir ./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

checkpoint 中通常包含 ``experiment_cfg/metadata.json``。如缺失，可从 LIBERO 数据集生成并显式设置 ``rollout.model.metadata_json_path``：

.. code-block:: bash

   python toolkits/lerobot/generate_dreamzero_metadata.py \
     --preset libero_sim \
     --dataset-root ./data/libero \
     --output-metadata ./assets/metadata.json

运行 LIBERO-Spatial
-------------------

在 ``evaluations/libero/libero_spatial_dreamzero_eval_sglang.yaml`` 中配置 ``rollout.model.model_path``\ （或通过命令行覆盖）后启动评测：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.model.model_path=./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

如需自定义 metadata 文件，追加 ``rollout.model.metadata_json_path=./assets/metadata.json``。

关键配置
--------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 字段
     - 作用
   * - ``rollout.rollout_backend: sglang``
     - 选择 SGLang rollout 后端。
   * - ``rollout.sglang.serving_mode: embodied``
     - 启用具身 SGLang rollout worker，由每个 worker 在分配的 GPU 上启动并管理本地 server。
   * - ``rollout.sglang.server.pipeline_config.default_num_inference_steps``
     - server 侧 DreamZero 去噪步数。
   * - ``rollout.sglang.server.pipeline_config.cfg_scale``
     - 动作推理的 classifier-free guidance 系数。
   * - ``rollout.sglang.server.cfg_parallel_degree``
     - 设为 ``2`` 时将 CFG 正负分支切分到不同 rank。
   * - ``rollout.sglang.server.tp_size``
     - DreamZero DiT 的 tensor 并行度。
   * - ``rollout.sglang.server.sp_degree``
     - DreamZero DiT 注意力序列并行度。
   * - ``rollout.model.model_path``
     - SGLang 加载的 DreamZero Diffusers checkpoint 目录。
   * - ``rollout.model.metadata_json_path``
     - 动作推理前后归一化所用的统计文件。
   * - ``rollout.model.num_action_chunks``
     - 单次请求返回的动作步数；``env.eval.max_steps_per_rollout_epoch`` 须能被该值整除。

并行覆盖参数
------------

RLinf 会根据 ``dp_size * tp_size * sp_degree * cfg_parallel_degree`` 自动推导每个 server 实例占用的 GPU 数。要在 ``libero_spatial_dreamzero_eval_sglang`` 上开启 2 卡 CFG 并行，直接覆盖 ``cfg_parallel_degree`` 即可：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval_sglang \
     rollout.sglang.server.cfg_parallel_degree=2 \
     rollout.model.model_path=./assets/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Diffusers

验证
----

评测日志写入 ``logs/<timestamp>-<config>/``。检查 ``eval_embodiment.log`` 中的 server 启动日志、就绪信息、逐 episode 结果与最终 ``eval/success_once`` 指标。

LIBERO-Spatial SGLang 配置使用 ``auto_reset: True`` 与有序重置状态，以较少并行环境覆盖完整套件。LIBERO 轨迹计数规则见 :ref:`libero-eval-config`。

常见问题
--------

- 如果 SGLang 找不到模型组件，确认 ``rollout.model.model_path`` 指向下载后的 Diffusers checkpoint 目录。
- 如果 metadata 加载失败，将 ``rollout.model.metadata_json_path`` 设为针对 ``libero_sim`` 生成的 ``metadata.json`` 文件。
- 如果本地 HTTP 请求被代理拦截，启动前导出 ``NO_PROXY=127.0.0.1,localhost``。
