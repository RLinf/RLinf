RMBench 评测
=============

用 SFT 得到的 π₀.₅ Short Memory checkpoint，在 `RMBench <https://rmbench.github.io/>`__ 的双臂 Aloha 任务上跑评测。开始前准备好 :doc:`训练示例 <../../examples/embodied/rmbench>` 用的 RMBench 代码和 ``norm_stats.json``。

安装
----

在 RLinf 根目录给模拟器单独建一个 Python 环境。CuRobo 需要用 CUDA Toolkit 编译，运行模拟器需要 NVIDIA GPU。装好后，到 RMBench 目录下载仿真资产。

.. code-block:: bash

   bash requirements/install.sh embodied --env rmbench --venv .rmbench-venv
   source .rmbench-venv/bin/activate
   (cd RMBench && bash scripts/_download_assets.sh)
   source .venv/bin/activate

最后切回 OpenPI 环境。OpenPI 环境负责跑 policy，启动脚本会用 ``.rmbench-venv`` 跑模拟器。

启动评测
--------

在 ``evaluations/rmbench/rmbench_openpi_pi05_eval.yaml`` 中填写以下路径。

.. list-table::
   :header-rows: 1

   * - 配置项
     - 取值
   * - ``rmbench_path``
     - 已下载资产的 RMBench 目录。
   * - ``sim_python``
     - ``.rmbench-venv/bin/python``。
   * - ``rollout.model.model_path``
     - SFT 保存的 ``global_step_<N>`` 目录。
   * - ``rollout.model.openpi_data.norm_stats_path``
     - SFT 用的 ``norm_stats.json`` 文件。

默认评测 ``observe_and_pickup``，使用 ``seen`` 指令集。RMBench 先用 expert 筛选 seed，再用通过筛选的 seed 跑 policy，共评测 100 个 episode。第一次可以把 ``episodes`` 设为 1，确认跑通后改回 100。

在 RLinf 根目录启动评测。

.. code-block:: bash

   bash evaluations/run_eval.sh rmbench rmbench_openpi_pi05_eval

模拟器每个控制步都记录观测，包括 action chunk 内的控制步。policy 每次读取 8 帧，从当前帧往前每隔 53 步取一帧。模型按演示数据的标称频率 15 Hz 计算时间偏移。

如果只用当前帧评测，把 ``rollout.model.openpi.short_memory`` 设为 ``false``，并填入对应的 checkpoint。

结果
----

终端会输出 RMBench 的 ``Success rate``。同一份输出也写入 ``logs/<run>/eval_embodiment.log``。如果只跑 1 个 episode，确认日志里有本次结果即可。``save_video: true`` 可以把视频保存到运行目录，需要系统安装 ``ffmpeg``。
