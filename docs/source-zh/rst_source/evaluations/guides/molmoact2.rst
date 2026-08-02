MolmoAct2 评测
==============

通过 RLinf 的统一 LIBERO 入口评测官方 MolmoAct2-LIBERO checkpoint。当前接入仅支持评测。

安装
----

在仓库根目录安装 MolmoAct2 和 LIBERO：

.. code-block:: bash

   bash requirements/install.sh embodied --model molmoact2 --env libero
   source .venv/bin/activate

本命令会：

1. 使用 Python 3.12 配置 MolmoAct2 环境。
2. 安装 LIBERO 环境和 RLinf embodied 依赖。
3. 检出 MolmoAct2 adapter 使用的固定 LeRobot revision。

下载模型
--------

下载官方 `allenai/MolmoAct2-LIBERO <https://huggingface.co/allenai/MolmoAct2-LIBERO>`__ checkpoint：

.. code-block:: bash

   hf download allenai/MolmoAct2-LIBERO \
     --local-dir /path/to/models/MolmoAct2-LIBERO

运行
----

启动 ``libero_10_molmoact2_eval`` 配置，并覆盖其中的模型占位路径：

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
     rollout.model.model_path=/path/to/models/MolmoAct2-LIBERO

本命令会：

1. 通过 MolmoAct2 model adapter 加载官方 checkpoint。
2. 使用 ``evaluations/libero/libero_10_molmoact2_eval.yaml`` 中的评测设置运行 LIBERO-Long suite。
3. 将终端输出和 ``eval/success_once`` 写入带时间戳的评测日志。

.. warning::

   默认配置会覆盖完整 LIBERO-Long suite，可能需要数小时。如果只需 smoke test，请通过 ``env.eval`` Hydra 覆盖项缩小评测规模。

输入与推理设置
--------------

RLinf 将 ``main_images`` 映射为 MolmoAct2-LIBERO 所需的 agent view，将 ``wrist_images`` 映射为 wrist view。模型 preset 已设置连续动作推理、``norm_tag: libero`` 和 ``num_steps: 10``，无需在命令行中重复设置。

查看结果
--------

终端会输出 ``eval/success_once``。日志写入：

.. code-block:: text

   logs/<timestamp>-libero_10_molmoact2_eval/eval_embodiment.log

评测协议见 :doc:`LIBERO 评测 <libero>`，指标解释见 :doc:`评测结果 <../reference/results>`。
