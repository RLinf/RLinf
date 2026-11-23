采集 Isaac Lab 抓取示范
=======================

用 GraspGen 生成 Panda 抓姿、cuRobo 规划轨迹，在 Isaac Lab 中执行，再把通过验收的回合导出为本地 LeRobot 数据集。当前支持方块、细长条、圆柱和圆盘，任务范围是抓取、抬升和稳持。抓姿生成使用已知 mesh 与仿真物体位姿，双路 RGB 相机负责记录训练观测。

安装运行环境
------------

准备 Linux NVIDIA 工作站，安装 Isaac Sim 5.1.0、Isaac Lab 2.3.0、cuRobo 0.7.8 和 GraspGen Panda 模型。仿真、规划和抓姿生成分别使用独立的 Python 3.11 环境。启动 RLinf 入口的环境需要安装 RLinf、NumPy 和 LeRobot，导出的数据版本由这个环境中的 LeRobot 版本决定。

克隆外部采集器，并切换到已验证的版本：

.. code-block:: bash

   git clone https://github.com/juchuanqi/graspgen-curobo-collect.git
   cd graspgen-curobo-collect
   git checkout 9f1feae9aa0a66a31053b4e237b29cf388a45637

按采集器的 `安装说明 <https://github.com/juchuanqi/graspgen-curobo-collect/blob/9f1feae9aa0a66a31053b4e237b29cf388a45637/docs/INSTALL.md>`__ 安装三个环境并下载模型权重。采集器会检查已跟踪文件是否修改；候选、缓存和本机配置存放在版本管理之外。

配置小批次采集
--------------

回到 RLinf 仓库根目录，复制示例配置：

.. code-block:: bash

   cp examples/embodiment/config/isaaclab_graspgen_curobo_collect.yaml collection.yaml

填写采集器目录、三个 Python 解释器和模型配置路径，为输出和候选缓存选择新目录。先用一张 GPU 执行两次方块抓取尝试：

.. code-block:: yaml

   runtime:
     root: /path/to/graspgen-curobo-collect
     sim_python: /path/to/isaaclab/bin/python
     planner_python: /path/to/curobo/bin/python
     graspgen_python: /path/to/graspgen/bin/python
     graspgen_config: /path/to/models/checkpoints/graspgen_franka_panda.yml
   output: /path/to/new_collection
   candidates:
     root: /path/to/candidates_200
   collection:
     objects: [block55]
     episodes_per_object: 2
     workers: 1
     gpus: "0"

GPU 编号是当前 CUDA_VISIBLE_DEVICES 中的逻辑编号。每个预定初态只尝试一次，抓取失败的记录也会保存并计入整批尝试数。

运行采集与导出
--------------

先检查运行环境、模型文件和 CPU 依赖：

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml --stage check

再生成候选、采集固定批次并导出合格回合：

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml

RGB 和绝对关节位置目标按 60 Hz 记录，物理频率为 480 Hz。导出只保留通过抓取验收和数据检查的回合，再用官方 LeRobot loader 逐行核对状态、动作和双路图像。完整读回通过后才生成最终导出目录。

在配置的输出目录中查看结果：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 路径
     - 内容
   * - ``raw/REPORT.md``
     - 尝试数、成功数与失败情况。
   * - ``SELECTION.json``
     - 纳入数据集的回合与其他尝试的排除原因。
   * - ``export/dataset/``
     - LeRobot 数据，使用 RLinf 的 image、wrist_image、state 和 actions 字段。
   * - ``export/sidecars/``
     - 物理证据、原始关节轨迹和终态 RGB 图像。
   * - ``export/AUDIT.json``
     - 完整读回结果与导出的回合数、帧数。

状态和动作均包含七个机械臂关节角，单位为 rad，最后一维是两指总开口，单位为 m。动作表示绝对位置目标。第 t 行观测位于第 t 个动作执行前，额外的终态观测单独保留。下游 policy 需要采用相同的关节位置控制约定并映射字段，数据格式说明见 :doc:`data_collection`。

继续中断的采集
--------------

使用原配置恢复原始数据采集：

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml --stage collect --resume

已完成的尝试保留原结果，包括物理失败。采集完成后单独执行导出：

.. code-block:: bash

   python -m toolkits.isaaclab.collect_graspgen_curobo --config collection.yaml --stage export

导出失败会保留 export.partial 目录供排查，已有完整导出不会被覆盖。
