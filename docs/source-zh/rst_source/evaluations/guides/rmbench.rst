RMBench 评测
=============

用 SFT 得到的 π₀.₅ Short Memory checkpoint，在 `RMBench <https://rmbench.github.io/>`__ 的双臂 Aloha 任务上评测。

**前置条件：** 模拟器需要 NVIDIA GPU，编译 CuRobo 需要 CUDA Toolkit。按 :doc:`RLinf 安装指南 <../../start/installation>` 准备基础环境，然后备好 :doc:`训练示例 <../../examples/embodied/rmbench>` 中的 OpenPI 环境、RMBench 源码、SFT checkpoint 和归一化文件。

安装与运行
----------

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. 安装模拟器**

      在 RLinf 根目录为模拟器单独创建环境。

      .. code-block:: bash

         bash requirements/install.sh embodied --env rmbench --venv .rmbench-venv
         source .rmbench-venv/bin/activate

      **2. 下载仿真资产**

      在 RMBench 目录下载资产，然后切回 OpenPI 环境运行推理。模拟器继续使用 ``.rmbench-venv``。

      .. code-block:: bash

         (cd RMBench && bash scripts/_download_assets.sh)
         source .venv/bin/activate

      **3. 启动评测**

      在 ``evaluations/rmbench/rmbench_openpi_pi05_eval.yaml`` 中填写以下路径。

      .. list-table::
         :header-rows: 1

         * - 配置项
           - 取值
         * - ``env.eval.init_params.rmbench_path``
           - 已下载资产的 RMBench 目录。
         * - ``env.eval.init_params.sim_python``
           - ``.rmbench-venv/bin/python``。
         * - ``rollout.model.model_path``
           - SFT 保存的 ``global_step_<N>`` 目录。
         * - ``rollout.model.openpi_data.norm_stats_path``
           - SFT 用的 ``norm_stats.json`` 文件。

      默认评测 ``observe_and_pickup``，使用 ``seen`` 指令集。RMBench 先用 expert 筛选 seed，再用通过筛选的 seed 跑 policy，共评测 100 个 episode。初次运行可把 ``env.eval.rollout_epoch`` 设为 1，先跑一个 episode。

      OpenPI policy 用 ``rollout.model.openpi.eval_seed`` 固定动作采样的随机种子。

      在 RLinf 根目录启动评测。

      .. code-block:: bash

         bash evaluations/run_eval.sh rmbench rmbench_openpi_pi05_eval

启用 Short Memory 后，模拟器会记录每个控制步的观测，包括 action chunk 内的控制步。policy 每次读取 8 帧，从当前帧往前每隔 53 步取一帧。模型按演示数据的标称频率 15 Hz 计算时间偏移。

要换任务，修改 ``env.eval.init_params.task_name``。在 RMBench 的 ``env_cfg/task_config/_eval_step_limit.yml`` 里查该任务的步数上限，填入 ``env.eval.max_episode_steps``。将上限向上取整到 32 的倍数，填入 ``env.eval.max_steps_per_rollout_epoch``，这样最后一个 action chunk 也能执行。

只读取当前帧的 checkpoint 需要关闭 Short Memory。把 ``rollout.model.openpi.short_memory`` 设为 ``false``，并将 ``env.eval.short_memory`` 设为 ``null``。

可视化与结果
------------

在 ``logs/<run>/eval_embodiment.log`` 或 TensorBoard 中查看成功率 ``eval/success_once``。将 ``env.eval.video_cfg.save_video`` 设为 ``true``，即可把视频保存到运行目录。
