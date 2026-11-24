快速体验
========

以 **LIBERO Spatial + OpenPI π₀.₅** 为例（``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``），演示一次完整的具身评测流程。

安装与运行
----------

如果 ``.venv`` 里已经安装了其他模型或环境，在 ``install.sh`` 末尾加 ``--venv .venv-eval`` 并激活 ``.venv-eval``，避免覆盖原有环境。国内镜像加速与 Docker Hub 预构建镜像见 :doc:`../../start/installation`。

.. tab-set::

   .. tab-item:: LIBERO Spatial（OpenPI π₀.₅）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. 启动评测**\ （首次运行自动缓存下载 `RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__，环境与权重就绪后在 24 GB+ 单卡上约 5 分钟完成）

      在 1、2、4 或 8 张 GPU 上做快速验证，设置 ``env.eval.total_num_envs=8``：

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval env.eval.total_num_envs=8

      如需评测完整 500 条轨迹套件，使用 128 并行环境配合自动重置，避免耗尽主机内存（单个 LIBERO 环境约占 2 GiB 主机内存，默认 500 并行约需 1 TiB，详见 :doc:`../guides/libero`）：

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
           env.eval.total_num_envs=128 \
           env.eval.max_steps_per_rollout_epoch=960

      离线环境可先将 ``RLinf/RLinf-Pi05-LIBERO-SFT`` 下载到本地，并在命令末尾追加 ``rollout.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT``。

结果与下一步
------------

评测结束后会在终端打印指标汇总表，并同步写入 ``metrics.log``。``RLinf/RLinf-Pi05-LIBERO-SFT`` 在 LIBERO Spatial 上的 ``eval/success_once`` 预期约为 **0.95+**\ ：

.. code-block:: text

   ╭────────────────────────────── Metric Table ──────────────────────────────╮
   │ Global Step:    1/1 │ Progress: ████████████████████ │ 100.0%            │
   │ Elapsed: 03:45      │ ETA: 00:00                     │ Step Time: 225.0s │
   ├────────────────────────────── Evaluation ────────────────────────────────┤
   │ episode_len=142.3   │ num_trajectories=8             │ success_once=1.00 │
   ╰──────────────────────────────────────────────────────────────────────────╯

- 日志与指标表保存在 ``logs/<时间戳>-libero_spatial_openpi_pi05_eval/eval_embodiment.log`` 与 ``metrics.log``。
- 该配置默认开启视频录制（``save_video: True``），视频保存在 ``<log_path>/video/eval/``；如需关闭，传入 ``env.eval.video_cfg.save_video=False``\ （详见 :doc:`../reference/results`）。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: YAML 配置参考
      :link: ../reference/configuration
      :link-type: doc

      了解评测配置的字段结构与覆盖方式。

   .. grid-item-card:: Benchmark 评测指南
      :link: ../guides/index
      :link-type: doc

      探索 LIBERO、ManiSkill OOD、RoboTwin、BEHAVIOR-1K 等基准。

   .. grid-item-card:: CLI 选项参考
      :link: ../reference/cli
      :link-type: doc

      查看 ``run_eval.sh`` 全部命令行参数与用法。
