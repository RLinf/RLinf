快速体验
========

以 **LIBERO Spatial + OpenPI π₀.₅** 为例（``evaluations/libero/libero_spatial_openpi_pi05_eval.yaml``），约 5 分钟完成第一次具身评测。

一键上手
--------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``\ （或前往 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 拉取预构建镜像）。

.. tab-set::

   .. tab-item:: LIBERO Spatial（OpenPI π₀.₅）

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. 下载预训练模型**\ （`RLinf/RLinf-Pi05-LIBERO-SFT <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__）

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      **3. 启动评测**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=$PWD/RLinf-Pi05-LIBERO-SFT

结果与下一步
------------

- 终端会输出 ``eval/success_once``、``eval/return`` 等评测指标。
- 日志保存在 ``logs/<时间戳>-libero_spatial_openpi_pi05_eval/eval_embodiment.log``。
- 设置 ``env.eval.video_cfg.save_video=True`` 可在 ``<log_path>/video/eval/`` 保存评测视频，详见 :doc:`../reference/results`。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: YAML 配置参考
      :link: ../reference/configuration
      :link-type: doc

      了解评测配置的字段结构与覆盖方式。

   .. grid-item-card:: Benchmark 评测指南
      :link: ../guides/libero
      :link-type: doc

      探索 LIBERO、ManiSkill OOD、RoboTwin、BEHAVIOR-1K 等基准。

   .. grid-item-card:: CLI 选项参考
      :link: ../reference/cli
      :link-type: doc

      查看 ``run_eval.sh`` 全部命令行参数与用法。
