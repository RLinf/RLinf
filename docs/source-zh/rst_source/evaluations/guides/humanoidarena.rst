HumanoidArena 评测
==================

使用 RLinf 在 HumanoidArena 的七个任务上评测 WB-WAM。每个任务有对应的 checkpoint 和评测预设，下面以单卡运行 OpenDoor 为例。

准备环境
--------

使用配备 NVIDIA GPU 的 Linux 主机，按照 `WB-WAM 安装指南 <https://github.com/WB-WaM/WB-WAM-Official/blob/6a9bf12d6c36d42ec66ee2aa7235ea2058175c40/benchmark/humanoidarena/README_zh.md>`_ 准备模型环境、Isaac Sim 5.1 环境、场景资产、SONIC 权重和基础模型。WB-WAM 使用 Python 3.10，模拟器使用 Python 3.11。

在 RLinf 仓库根目录安装并激活评测环境：

.. code-block:: bash

   bash requirements/install.sh --no-flash-attn embodied \
     --model wbwam --env humanoidarena
   source .venv/bin/activate

下载模型
--------

使用 WB-WAM 仓库中的下载脚本，获取 `HumanoidArena 发布权重 <https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena>`_ 中的 OpenDoor checkpoint：

.. code-block:: bash

   python /path/to/WB-WAM-Official/scripts/download_models.py arena \
     --task open_door --output /path/to/checkpoints

下载后的任务目录为 ``/path/to/checkpoints/humanoid_arena_native50/open_door``，其中包含权重、``config.yaml`` 和 ``dataset_stats.json``。下载全部七个任务时，省略 ``--task open_door``。

配置本地路径
------------

修改 ``evaluations/humanoidarena/humanoidarena_wbwam_eval.yaml`` 中的路径，使其指向已准备好的目录和 Python 环境：

.. code-block:: yaml

   env:
     eval:
       wbwam_repo: /path/to/WB-WAM-Official
       arena_root: /path/to/WB-WAM-Official/third_party/HumanoidArena
       simulation_python: /path/to/isaac/bin/python
       dependencies_root: /path/to/dependencies
       sonic_root: /path/to/sonic
       output_dir: /path/to/results/arena

   rollout:
     model:
       wbwam:
         repo_root: /path/to/WB-WAM-Official
         python: /path/to/wbwam/bin/python
         base_models_root: /path/to/base-models
         output_dir: /path/to/results/wbwam

两处仓库路径使用同一个 WB-WAM 目录。``sonic_root`` 指向包含 ``model_encoder.onnx`` 和 ``model_decoder.onnx`` 的目录。示例中模型和模拟器共用 GPU 0；使用其他 GPU 时，同时修改 ``cluster.component_placement`` 和 ``env.eval.simulation_gpu``。

运行评测
--------

指定下载好的 checkpoint，运行一个 OpenDoor 回合：

.. code-block:: bash

   bash evaluations/run_eval.sh humanoidarena humanoidarena_wbwam_eval \
     rollout.model.model_path=/path/to/checkpoints/humanoid_arena_native50/open_door

终端输出 ``eval/success``、``eval/return``、``eval/episode_len`` 和 ``eval/num_trajectories``。本次轨迹数为 1，成功率和回报为 0 或 1，回合长度以控制步数计。

切换任务
~~~~~~~~

切换任务时，同时设置 ``task`` 和 checkpoint 目录。任务预设会选定指令、权重文件名、回合长度和推理参数。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 任务
     - 配置选项
   * - Football
     - ``task=kick_football``
   * - DoubleDesk
     - ``task=hammer``
   * - P&PBox
     - ``task=box_shelf``
   * - OpenDoor
     - ``task=open_door``
   * - SitSofa
     - ``task=sit_sofa``
   * - Boxing
     - ``task=punch_markers``
   * - VisNavi
     - ``task=obstacle_navigation``

例如，使用 Football 的 checkpoint 评测踢球任务：

.. code-block:: bash

   bash evaluations/run_eval.sh humanoidarena humanoidarena_wbwam_eval \
     task=kick_football \
     rollout.model.model_path=/path/to/checkpoints/humanoid_arena_native50/kick_football

运行完整评测
~~~~~~~~~~~~

每个任务使用 seed 0、1、2，各运行 20 回合。以 Football 为例：

.. code-block:: bash

   for seed in 0 1 2; do
     bash evaluations/run_eval.sh humanoidarena humanoidarena_wbwam_eval \
       task=kick_football \
       rollout.model.model_path=/path/to/checkpoints/humanoid_arena_native50/kick_football \
       env.eval.rollout_epoch=20 \
       env.eval.seed="${seed}" \
       rollout.model.wbwam.seed="${seed}" \
       env.eval.output_dir="/path/to/results/kick_football/seed_${seed}/arena" \
       rollout.model.wbwam.output_dir="/path/to/results/kick_football/seed_${seed}/wbwam"
   done

其余任务使用各自的 checkpoint 重复上述流程。七个任务、三个 seed、每个 seed 二十回合，共 420 回合。每个任务统计三个 seed 成功率的均值与总体标准差；总体成绩统计全部 21 个任务与 seed 组合的成功率均值及总体标准差。

查看结果
--------

评测日志保存在 ``logs/<timestamp>-humanoidarena_wbwam_eval/eval_embodiment.log``，日志选项见 :doc:`../reference/results`。

各回合结果保存在 ``env.eval.output_dir/env-*/episode-*/result.json``，包含成功状态、终止原因和控制步数。回合视频位于对应回合的 ``videos/success`` 或 ``videos/failure`` 目录。模型日志和配置保存在 ``rollout.model.wbwam.output_dir`` 下。
