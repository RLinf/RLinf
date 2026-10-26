在 RoboLab 上评测 π₀.₅
======================

通过 RLinf 的环境 worker 和具身评估 runner 运行 DROID joint-position 操作任务。先安装 RoboLab 和 OpenPI，再下载转换后的 checkpoint，查询任务元数据，最后运行单任务或完整的 120 任务 benchmark。观测和动作格式见 :doc:`RoboLab 概览 <../../examples/embodied/robolab>`。

安装环境
--------

准备 Linux、Python 3.11、``uv`` 和支持 Vulkan 的 NVIDIA GPU。已有评测使用 NVIDIA A800 80 GB 显卡及 570.172.08 驱动。按照 :doc:`安装指南 </rst_source/start/installation>` 准备系统依赖后，在 RLinf 仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robolab
   source .venv/bin/activate

需要镜像源时，在安装命令中添加 ``--use-mirror``。安装器创建 ``.venv``，安装 RoboLab 所需的 Isaac Sim 5.0 / Isaac Lab 2.2 及 RLinf OpenPI。RoboLab 通过安装器共享的 Torch 版本配置选择 PyTorch 2.7.1；图形系统库和 Git LFS 由 ``requirements/sys_deps.sh`` 安装。

默认源码目录为 ``.venv/robolab``，固定到 ``b6cfcf6ccd2113c4cd1216d35b33024a7d5691da``。运行时需要保留已安装源码旁的任务元数据和资产。若要复用已有源码，在安装前设置 ``ROBOLAB_PATH=/path/to/RoboLab``，安装器会使用该目录的现有版本；运行时适配器从包安装元数据中定位源码。

.. note::

   已有评测验证了当前运行环境，本次贡献尚未完成全新环境安装和 Docker 构建验证。对现有环境执行 ``uv pip check --python .venv/bin/python`` 会报告依赖声明冲突，包括 Isaac Sim 固定的 Torch 版本和 OpenPI 的 Transformers fork。部署到新环境前，请查看 PR 中的安装检查记录。

下载模型
--------

下载 ``RLinf/RLinf-Pi05-RoboLab-droid-jointpos``，它由 RoboLab 使用的 ``pi05_droid_jointpos`` checkpoint 转换为 PyTorch 格式。此配置预测关节位置，应选择 joint-position checkpoint。

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos \
       --local-dir /path/to/model/RLinf-Pi05-RoboLab-droid-jointpos

下载目录应包含 ``model.safetensors``、``config.json`` 和 ``assets/droid/norm_stats.json``。将 ``evaluations/robolab/pi05_droid_jointpos.yaml`` 中的 ``rollout.model.model_path`` 改为该目录，或像下文一样通过命令行指定。

查询 Benchmark 任务
-------------------

使用已安装 RoboLab 版本中的元数据查询任务及其原生回合时长。下面导出 120 个 benchmark 任务的名称、指令、难度、能力属性和源码文件名：

.. code-block:: bash

   mkdir -p data
   python - <<'PYTHON'
   import csv
   import json
   from decimal import Decimal
   from importlib.util import find_spec
   from pathlib import Path

   package = Path(find_spec("robolab").origin).parent
   metadata = package / "tasks/_metadata/task_metadata.json"
   tasks = [t for t in json.loads(metadata.read_text())
            if t["filename"].startswith("benchmark/")]
   with Path("data/robolab_tasks.tsv").open("w", newline="") as stream:
       writer = csv.writer(stream, delimiter="	")
       writer.writerow(["task", "max_steps", "difficulty", "attributes", "instruction", "filename"])
       for task in sorted(tasks, key=lambda t: t["task_name"]):
           writer.writerow([
               task["task_name"], int(Decimal(task["episode_s"]) * 15),
               task["difficulty_label"], task["attributes"],
               task["instruction"], task["filename"],
           ])
   print(f"{len(tasks)} tasks; metadata: {metadata}")
   print("Saved data/robolab_tasks.tsv")
   PYTHON

DROID 配置的控制频率为 15 Hz，仿真时间步为 1/120 秒，decimation 为 8，因此 ``max_steps = episode_s × 15``。``data/robolab_tasks.tsv`` 可用作本地任务索引；原始 JSON 还包含指令变体和子任务定义。任务实现位于已安装源码目录的 ``robolab/tasks/<filename>``。

运行单任务
----------

使用 10 个并行环境评测 ``BowlStackingLeftOnRightTask``，共运行 10 个回合。该任务的原生上限为 300 个控制步：

.. code-block:: bash

   bash evaluations/run_eval.sh robolab pi05_droid_jointpos \
       rollout.model.model_path=/path/to/model/RLinf-Pi05-RoboLab-droid-jointpos \
       env.eval.init_params.task=BowlStackingLeftOnRightTask \
       env.eval.max_episode_steps=300 \
       env.eval.max_steps_per_rollout_epoch=300 \
       env.eval.total_num_envs=10 \
       env.eval.rollout_epoch=1 \
       env.eval.init_params.output_dir=./logs/robolab/BowlStackingLeftOnRightTask

RLinf 通过 ``EnvWorker`` 创建 ``RoboLabEnv``，由 rollout worker 执行 policy 推理，仿真子进程归环境适配器管理。Hydra 配置默认将这两个 worker 放在 GPU 0；修改放置位置时使用 ``cluster.component_placement``。

配置使用 ``env_type: robolab``、``auto_reset: false`` 和 ``ignore_terminations: false``。每个 rollout epoch 重置整批环境。评测 10 回合时，可设置 ``total_num_envs=10, rollout_epoch=1``，也可设置 ``total_num_envs=1, rollout_epoch=10``。``max_episode_steps`` 是 RLinf 设置的上限，原生任务终止或超时可以提前结束回合。``max_steps_per_rollout_epoch`` 应覆盖该上限，并设置为动作块长度 15 的整数倍。

复现本次对比实验时，使用元数据中的任务原生步数，保留评估配置中的数值对齐选项：

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/model/RLinf-Pi05-RoboLab-droid-jointpos
       openpi:
         pytorch_numerics: true
         eval_batch_size: 1
         eval_seed: 0
         sample_compile: false
         torch_compile: false

10 个仿真环境与 ``eval_batch_size: 1`` 控制的是不同层次：环境并行推进，policy 逐条处理仍在运行的环境观测，并跳过已结束的环境。每个任务的 policy 随机数生成器从 seed 0 开始，这与原生多任务服务持续使用同一随机数流的方式不同。

运行全部 120 个任务
-------------------

导出 TSV 后，可以在一张 GPU 上依次评测全部任务。每个任务使用各自的原生步数和 10 个回合，共 1200 回合。在仓库根目录的 Bash 中执行：

.. code-block:: bash

   while IFS=$'	' read -r task max_steps rest; do
       bash evaluations/run_eval.sh robolab pi05_droid_jointpos \
           rollout.model.model_path=/path/to/model/RLinf-Pi05-RoboLab-droid-jointpos \
           "env.eval.init_params.task=${task}" \
           "env.eval.max_episode_steps=${max_steps}" \
           "env.eval.max_steps_per_rollout_epoch=${max_steps}" \
           env.eval.total_num_envs=10 \
           env.eval.rollout_epoch=1 \
           "env.eval.init_params.output_dir=./logs/robolab/${task}" || break
   done < <(tail -n +2 data/robolab_tasks.tsv)

命令失败时循环停止。该循环没有自动断点续跑功能；可复制 TSV 并删除已完成任务的行，选择剩余任务。新实验应使用新的输出目录，跨多次运行汇总时以任务日志为准。

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
     - 范围为 [0, 1] 的平均归一化子任务得分；成功回合记为 1。通过 ``init_params.enable_subtask: true`` 启用。
   * - ``eval/num_trajectories``
     - 本次结果包含的已完成回合数。
   * - ``eval/episode_len``
     - 每回合实际执行控制步数的均值。

跨任务汇总时，累加成功回合数，并按回合数加权计算得分。论文中的 SPARC 和速度需要从记录的末端轨迹计算，RLinf runner 不直接输出这两项指标；这里的速度是机器人运动速度。日志的通用用法见 :doc:`../reference/results`。
