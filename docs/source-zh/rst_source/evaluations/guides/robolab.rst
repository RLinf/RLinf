在 RoboLab 上评测 π₀.₅
======================

使用 RLinf 在 RoboLab 的 DROID joint-position 操作任务上评测 π₀.₅。先安装 RoboLab 和 OpenPI，再下载转换后的 checkpoint，查询任务元数据，最后运行单任务或完整的 120 任务 benchmark。观测和动作格式见 :doc:`RoboLab 概览 <../../examples/embodied/robolab>`。

安装环境
--------

准备 Linux、Python 3.11、``uv`` 和支持 Vulkan 的 NVIDIA GPU。已有评测使用 NVIDIA A800 80 GB 显卡及 570.172.08 驱动。按照 :doc:`安装指南 </rst_source/start/installation>` 准备系统依赖后，在 RLinf 仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robolab
   source .venv/bin/activate

需要镜像源时，在安装命令中添加 ``--use-mirror``。

下载模型
--------

下载 ``RLinf/RLinf-Pi05-RoboLab-droid-jointpos``，它由 RoboLab 使用的 ``pi05_droid_jointpos`` checkpoint 转换为 PyTorch 格式。此配置预测关节位置，应选择 joint-position checkpoint。

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
       --local-dir /path/to/model/RLinf-Pi05-RoboLab-droid-jointpos

下载目录应包含 ``model.safetensors``、``config.json`` 和 ``assets/droid/norm_stats.json``。将 ``evaluations/robolab/pi05_droid_jointpos.yaml`` 中的 ``rollout.model.model_path`` 改为该目录。

查询 Benchmark 任务
-------------------

使用任务导出脚本查询当前安装版本的 benchmark 任务。激活环境后，在 RLinf 仓库根目录执行：

.. code-block:: bash

   python -m toolkits.standalone_eval_scripts.robolab.export_tasks

脚本会在仓库根目录创建 ``robolab_tasks.tsv``，包含任务名称、原生控制步数上限（``max_steps``）、难度、能力属性、指令和源码文件名。可通过 ``--output /path/to/tasks.tsv`` 指定其他输出位置。任务实现位于已安装 RoboLab 源码目录的 ``robolab/tasks/<filename>``。

配置评估
--------

运行前编辑 ``evaluations/robolab/pi05_droid_jointpos.yaml``。默认配置使用 ``BowlStackingLeftOnRightTask``，原生上限为 300 个控制步，共评测 10 个回合。使用默认任务时，只需修改 checkpoint 路径；选择其他任务时，按下表设置：

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - 字段
     - 设置方法
   * - ``rollout.model.model_path``
     - 本地 checkpoint 目录，应包含上文列出的三个文件。
   * - ``env.eval.init_params.task``
     - TSV 中 ``task`` 列的任务名称，默认为 ``BowlStackingLeftOnRightTask``。
   * - ``env.eval.max_episode_steps``
     - 同一行 ``max_steps`` 列的原生步数上限，默认为 300。
   * - ``env.eval.max_steps_per_rollout_epoch``
     - 对于这 120 个 benchmark 任务，设置为同一行的 ``max_steps``，默认为 300。
   * - ``env.eval.total_num_envs``
     - 并行环境数，默认为 10；显存不足时可减小。
   * - ``env.eval.rollout_epoch``
     - 评估轮数，默认为 1。总回合数等于该值乘以 ``total_num_envs``；使用 1 个环境、10 轮可以顺序评测 10 个回合。
   * - ``env.eval.init_params.output_dir``
     - 任务配置与轨迹的输出目录，默认为 ``./logs/robolab/BowlStackingLeftOnRightTask``。不同任务或实验应使用独立目录。

导出脚本已经将任务时长换算为控制步数，用户无需手动换算。使用单任务命令切换任务时，将该行的 ``max_steps`` 同时填入上表的两个步数字段；只修改任务名称不会自动更新这两个字段。例如切换到 ``BananaInBowlTask`` 时，两个字段都应设为 750。

这 120 个任务的原生步数均为当前动作块长度 15 的整数倍。如果自行缩短回合上限，应将 ``max_steps_per_rollout_epoch`` 设为不小于该上限的最小 15 的整数倍，例如回合上限为 17 时，rollout 步数设为 30。原生任务终止仍可能让回合提前结束。

运行单任务
----------

保存 YAML 配置后，在仓库根目录启动评估：

.. code-block:: bash

   bash evaluations/run_eval.sh robolab pi05_droid_jointpos

使用默认任务和回合设置时，该命令评测 ``BowlStackingLeftOnRightTask`` 的 10 个回合。其余 policy 和环境默认设置已写入 YAML。

运行全部 120 个任务
-------------------

导出 TSV 并在 YAML 中设置 checkpoint 路径后，保留 ``total_num_envs: 10`` 和 ``rollout_epoch: 1``，共评测 1200 回合。在仓库根目录启动全量评估：

.. code-block:: bash

   python -m toolkits.standalone_eval_scripts.robolab.run_benchmark \
       --output-dir robolab_benchmark

结果保存在仓库根目录的 ``robolab_benchmark/`` 下。该工具读取 ``robolab_tasks.tsv`` 和 ``evaluations/robolab/pi05_droid_jointpos.yaml``，按 YAML 的 GPU 放置配置依次运行任务。工具自动将 TSV 中的 ``max_steps`` 设置为该任务的 ``env.eval.max_episode_steps``，并将 ``env.eval.max_steps_per_rollout_epoch`` 向上取整到完整动作块；使用这个入口时，无需逐任务手动修改 YAML 中的任务名称和两个步数字段。使用 ``--config`` 或 ``--tasks`` 可以指定其他 YAML 或任务 TSV；如需评测子集，可复制 TSV 并只保留需要的任务行。

终端每 30 秒显示已完成任务数、当前任务及尝试次数、经过时间、成功率和归一化得分，同步以原子写入方式保存到输出目录下的 ``progress.json``。每个任务成功完成后保存 ``result.json``；每次尝试分别保存 ``console.log``、TensorBoard 事件、配置和 ``native/`` 轨迹。

每个任务通过 ``evaluations/run_eval.sh`` 执行。``--output-dir`` 用于创建新实验，目录应为空或尚不存在。工具将配置和任务列表保存到该目录；任务失败时记录错误并以非零状态码退出，修复问题后使用 ``--resume-dir`` 继续：

.. code-block:: bash

   python -m toolkits.standalone_eval_scripts.robolab.run_benchmark \
       --resume-dir robolab_benchmark

恢复时直接使用该目录保存的配置和任务列表，无需再次指定 ``--config`` 或 ``--tasks``。如果指定了这两个参数，内容必须与保存的实验一致。已完成任务自动跳过，未完成任务从第一个回合重新评测，之前的尝试日志会保留。``--output-dir`` 和 ``--resume-dir`` 只能选择一个；需要修改配置或任务列表时，使用新的 ``--output-dir``。续跑时保持 checkpoint 内容不变。

查看结果
--------

终端和 TensorBoard 会记录已完成回合的指标。``evaluations/run_eval.sh`` 将控制台日志以及默认的 TensorBoard 事件保存到 ``logs/<timestamp>-pi05_droid_jointpos/``。``env.eval.init_params.output_dir`` 单独指定 RoboLab 任务配置和 ``run_<index>.hdf5`` 轨迹文件的保存目录。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指标
     - 含义
   * - ``eval/success_once``
     - 已完成回合的成功比例，乘以 100 即成功率百分比。
   * - ``eval/score``
     - 范围为 [0, 1] 的平均归一化子任务得分；成功回合记为 1。通过 ``env.eval.init_params.enable_subtask: true`` 启用。
   * - ``eval/num_trajectories``
     - 本次结果包含的已完成回合数。
   * - ``eval/episode_len``
     - 每回合实际执行控制步数的均值。

跨任务汇总时，累加成功回合数，并按回合数加权计算得分。论文中的 SPARC 和速度需要从记录的末端轨迹计算，RLinf runner 不直接输出这两项指标；这里的速度是机器人运动速度。日志的通用用法见 :doc:`../reference/results`。
