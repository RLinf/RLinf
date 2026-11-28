RMBench 评测
============

概览
----

在 `RMBench <https://rmbench.github.io/>`__ 的双臂 Aloha 操作任务上评测微调后的 π₀.₅ Short Memory checkpoint，输出任务级指标 ``eval/success_once``。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      RMBench（双臂 Aloha）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenPI π₀.₅（Short Memory）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/rmbench/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`RMBench 训练 <../../examples/embodied/rmbench>`

**前置条件：** :doc:`环境安装 </rst_source/start/installation>` · 1 张 NVIDIA GPU（≥ 24 GB 显存）及用于编译 CuRobo 的 CUDA Toolkit，并备好 :doc:`RMBench 训练 <../../examples/embodied/rmbench>` 中的 ``.venv`` 环境、RMBench 仓库、SFT checkpoint 与 ``norm_stats.json``。

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在下载资产前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      这里使用两套虚拟环境：``.venv`` 运行 OpenPI SFT、数据准备与推理，``.rmbench-venv`` 隔离通过 ``sim_python`` 启动的 SAPIEN/CuRobo 模拟器进程并用于下载仿真资产。

      **1. 安装模拟器**

      在 RLinf 根目录创建 ``.rmbench-venv`` 并安装 RMBench 模拟器依赖：

      .. code-block:: bash

         bash requirements/install.sh embodied --env rmbench --venv .rmbench-venv
         source .rmbench-venv/bin/activate

      **2. 下载仿真资产**

      在 RMBench 目录下载资产后，切回 ``.venv`` 运行策略推理：

      .. code-block:: bash

         (cd RMBench && bash scripts/_download_assets.sh)
         source .venv/bin/activate

      **3. 启动评测**

      通过 Hydra CLI 参数传入本地路径并启动 ``rmbench_openpi_pi05_eval``\ （默认评测 ``observe_and_pickup`` 的 ``seen`` 指令集，共跑 100 个经 expert 筛选的 episode；初次运行可追加 ``env.eval.rollout_epoch=1`` 先跑 1 个 episode）：

      .. code-block:: bash

         bash evaluations/run_eval.sh rmbench rmbench_openpi_pi05_eval \
           env.eval.init_params.rmbench_path="$PWD/RMBench" \
           env.eval.init_params.sim_python="$PWD/.rmbench-venv/bin/python" \
           rollout.model.model_path="$PWD/logs/<run>/rmbench_pi05_short_memory_sft/checkpoints/global_step_<N>" \
           rollout.model.openpi_data.norm_stats_path="$PWD/rmbench_lerobot/norm_stats.json"

      OpenPI policy 通过 ``rollout.model.openpi.eval_seed`` 固定动作采样随机种子。开启 Short Memory 后，模拟器会记录每个控制步的观测（含 action chunk 内部步），按 15 Hz 间隔 53 步提取 8 帧。

.. dropdown:: 切换任务、当前帧对照与保存视频（可选）

   - **切换任务：** 追加 ``env.eval.init_params.task_name=<task>``\ ，按 ``RMBench/env_cfg/task_config/_eval_step_limit.yml`` 设置 ``env.eval.max_episode_steps``\ ，并将 ``env.eval.max_steps_per_rollout_epoch`` 向上取整到 32 的倍数，使最后一个 action chunk 完整执行。
   - **当前帧对照：** 对无 Short Memory 的 checkpoint，追加 ``rollout.model.openpi.short_memory=false env.eval.short_memory=null``。
   - **录制视频：** 追加 ``env.eval.video_cfg.save_video=true`` 将 MP4 保存到 ``logs/rmbench/``。

结果与常见问题
--------------

终端会输出 ``eval/success_once``\ ，日志与视频保存在 ``logs/rmbench/``\ （详见 :doc:`../reference/results`\ ，也可用 ``tensorboard --logdir ./logs/rmbench`` 查看）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - ``install.sh`` 编译 CuRobo 报错
     - 确认主机具备 NVIDIA GPU 与版本匹配的 CUDA Toolkit（``nvcc``\ ），再在 ``.rmbench-venv`` 中安装。
   * - 提示资产缺失或 Python 路径错误
     - 运行 ``_download_assets.sh`` 后，传入 ``env.eval.init_params.rmbench_path="$PWD/RMBench"`` 与 ``env.eval.init_params.sim_python="$PWD/.rmbench-venv/bin/python"``。
   * - 回合在最后一个 action chunk 结束前被截断
     - 将 ``env.eval.max_episode_steps`` 对齐 ``_eval_step_limit.yml``\ ，并把 ``env.eval.max_steps_per_rollout_epoch`` 向上取整到 32 的倍数。
   * - 并行环境数或 reset 参数校验失败
     - 保持 ``env.eval.total_num_envs=1``\ 、``env.eval.auto_reset=false`` 与 ``env.eval.ignore_terminations=false``\ ，以保证种子筛选顺序。
