评测快速体验
============

以 **LIBERO Spatial + OpenPI π₀.₅** 为例（``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``），约 5 分钟完成第一次具身评测。

一键上手
--------

如果 ``.venv`` 里已经安装了其他模型或环境，在 ``install.sh`` 末尾加 ``--venv .venv-eval`` 并激活 ``.venv-eval``，避免覆盖原有环境。国内镜像加速与 Docker Hub 预构建镜像见 :doc:`../../start/installation`。

.. tab-set::

   .. tab-item:: LIBERO Spatial（OpenPI π₀.₅）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. 启动评测**\ （首次运行自动缓存下载 `RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__）

      在单张 24 GB GPU 上做快速验证（约 5 分钟），传入 ``env.eval.total_num_envs=10``：

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=RLinf/RLinf-Pi05-LIBERO-SFT \
           env.eval.total_num_envs=10

      去掉 ``env.eval.total_num_envs=10`` 即按 YAML 默认的 500 个环境运行完整套件。离线环境可先将 ``RLinf/RLinf-Pi05-LIBERO-SFT`` 下载到本地，再把 ``rollout.model.model_path`` 改为本地目录路径。

结果与下一步
------------

- 终端会输出 ``eval/success_once``、``eval/return`` 等评测指标。
- 日志保存在 ``logs/<时间戳>-libero_spatial_openpi_pi05_eval/eval_embodiment.log``。
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
