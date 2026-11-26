RoboLab 评测
============

概览
----

使用 RLinf 在 RoboLab 的 120 个 DROID joint-position 操作任务上评测 π₀.₅。在单个环境中安装 RoboLab 与 OpenPI，下载转换后的 PyTorch checkpoint，即可运行单任务或全部 120 个任务。观测与动作格式见 :doc:`RoboLab 概览 <../../examples/embodied/robolab>`。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      RoboLab（120 个 DROID joint-position 任务）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenPI π₀.₅（``pi05_droid_jointpos``）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/robolab/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`RoboLab 概览 <../../examples/embodied/robolab>`

安装与运行
----------

需要 Linux、Python 3.11、``uv`` 与支持 Vulkan 的 NVIDIA GPU（已在 NVIDIA A800 80 GB + 570.172.08 驱动验证，系统依赖见 :doc:`安装指南 </rst_source/start/installation>`）。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: 单任务（默认）

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robolab && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos --local-dir ./RLinf-Pi05-RoboLab-droid-jointpos

      **3. 启动默认任务（BowlStackingLeftOnRightTask，10 个回合）**

      .. code-block:: bash

         bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos \
           rollout.model.model_path=./RLinf-Pi05-RoboLab-droid-jointpos

   .. tab-item:: 全部 120 任务评测

      保持 ``total_num_envs=10`` 与 ``rollout_epoch=1``\ （共 1200 个回合），在单张 GPU 上依次跑完全部 120 个 benchmark 任务：

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robolab && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboLab-droid-jointpos --local-dir ./RLinf-Pi05-RoboLab-droid-jointpos

      **3. 导出任务表并运行全部 120 个任务**

      .. code-block:: bash

         python -m toolkits.standalone_eval_scripts.robolab.export_tasks
         while IFS=$'\t' read -r task max_steps rest; do
             bash evaluations/run_eval.sh robolab robolab_pi05_droid_jointpos \
                 rollout.model.model_path=./RLinf-Pi05-RoboLab-droid-jointpos \
                 "env.eval.init_params.task=${task}" \
                 "env.eval.max_episode_steps=${max_steps}" \
                 "env.eval.max_steps_per_rollout_epoch=${max_steps}" \
                 "env.eval.init_params.output_dir=./logs/robolab/${task}" || break
         done < <(tail -n +2 robolab_tasks.tsv)

``RLinf/RLinf-Pi05-RoboLab-droid-jointpos`` 是 RoboLab ``pi05_droid_jointpos`` checkpoint 的 PyTorch 转换版本（包含 ``model.safetensors``、``config.json`` 与 ``assets/droid/norm_stats.json``）。120 任务循环遇错会停止且不自动续跑；复制 ``robolab_tasks.tsv`` 删除已完成的行，并为新实验换用独立输出目录即可继续。

.. dropdown:: 查询 Benchmark 任务

   导出已安装 RoboLab 版本的全部 120 个任务到 ``robolab_tasks.tsv``\ （可用 ``--output`` 指定路径）：

   .. code-block:: bash

      python -m toolkits.standalone_eval_scripts.robolab.export_tasks

   TSV 包含任务名称、原生步数上限（``max_steps``）、难度、能力属性、指令与文件名：

   .. list-table::
      :header-rows: 1

      * - ``task``
        - ``max_steps``
        - ``difficulty``
        - ``attributes``
        - ``instruction``
        - ``filename``
      * - ``BowlStackingLeftOnRightTask``
        - ``300``
        - ``simple``
        - ``spatial``
        - ``Stack the left bowl on the right bowl``
        - ``benchmark/bowl_stacking_left_on_right.py``
      * - ``BananaInBowlTask``
        - ``750``
        - ``simple``
        - ``semantics``
        - ``Pick up the banana and place it in the bowl``
        - ``benchmark/banana_in_bowl_task.py``

   切换任务时，将 ``task`` 填入 ``env.eval.init_params.task``，并将同一行的 ``max_steps`` 同时填入 ``env.eval.max_episode_steps`` 与 ``env.eval.max_steps_per_rollout_epoch``。

.. dropdown:: 配置评测参数

   ``evaluations/robolab/robolab_pi05_droid_jointpos.yaml`` 默认评测 ``BowlStackingLeftOnRightTask``\ （300 步，10 个回合）。切换任务或调整规模时修改以下字段：

   .. list-table::
      :header-rows: 1
      :widths: 38 62

      * - 字段
        - 设置方法
      * - ``rollout.model.model_path``
        - 包含 ``model.safetensors``、``config.json`` 与 ``assets/droid/norm_stats.json`` 的本地目录。
      * - ``env.eval.init_params.task``
        - TSV ``task`` 列的任务名（默认 ``BowlStackingLeftOnRightTask``）。
      * - ``env.eval.max_episode_steps``
        - 同一行 ``max_steps`` 列的原生步数上限（默认 ``300``）。
      * - ``env.eval.max_steps_per_rollout_epoch``
        - 对 120 个 benchmark 任务设为同一行的 ``max_steps``\ （默认 ``300``）。
      * - ``env.eval.total_num_envs``
        - 并行环境数（默认 ``10``，显存不足时调小）。
      * - ``env.eval.rollout_epoch``
        - 评测轮数（默认 ``1``，总回合数 = ``rollout_epoch × total_num_envs``）。
      * - ``env.eval.init_params.output_dir``
        - RoboLab 任务配置与 ``run_<index>.hdf5`` 轨迹输出目录（默认 ``./logs/robolab/BowlStackingLeftOnRightTask``）。

   120 个任务的原生步数均为动作块长度 15 的整数倍；自定义上限时，将 ``max_steps_per_rollout_epoch`` 向上取整为 15 的整数倍（例如 17 步回合需设 30）。

结果与常见问题
--------------

控制台日志与 TensorBoard 事件保存在 ``logs/<timestamp>-robolab_pi05_droid_jointpos/``\ （详见 :doc:`../reference/results`），``env.eval.init_params.output_dir`` 单独保存 RoboLab 任务配置与 ``run_<index>.hdf5`` 轨迹文件。核心指标包括 ``eval/success_once``\ （成功率比例）、``eval/score``\ （开启 ``env.eval.init_params.enable_subtask: true`` 时的平均归一化子任务得分，范围 ``[0, 1]``）、``eval/num_trajectories`` 与 ``eval/episode_len``。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 切换任务后回合步数不对
     - 仅修改 ``env.eval.init_params.task`` 不会同步更新步数；将 ``robolab_tasks.tsv`` 同一行的 ``max_steps`` 同时填入 ``max_episode_steps`` 与 ``max_steps_per_rollout_epoch``。
   * - 自定义步数校验报错
     - 将 ``env.eval.max_steps_per_rollout_epoch`` 设为不小于 ``max_episode_steps`` 的最小 15 的整数倍。
   * - GPU 显存不足
     - 调小 ``env.eval.total_num_envs`` 并调大 ``env.eval.rollout_epoch``\ （例如 ``1`` 个环境 × ``10`` 轮完成 10 回合）。
   * - 120 任务循环中断
     - 复制 ``robolab_tasks.tsv`` 删除已完成的行，并换用新输出目录继续运行。
