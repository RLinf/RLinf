G1 动作数据准备
===============

为 :doc:`G1 动作跟踪环境 <g1_mimic>` 和 :doc:`SuperMimic <supermimic>` 准备参考动作片段，包括 LAFAN1 离线转换以及自定义 ``clip_*.pt`` 文件格式。

**前置条件：** :doc:`安装 </rst_source/start/installation>` · :doc:`G1 动作跟踪环境 <g1_mimic>` 中的 SuperMimic 环境。

安装与运行
----------

先激活 :doc:`g1_mimic` 中创建的 SuperMimic 虚拟环境，以下所有命令均在 RLinf 仓库根目录执行。

离线转换 LAFAN1
---------------

训练机器无法访问 Hugging Face 时，可在另一台联网机器上完成下载和转换。转换机器需要 RLinf 代码以及装有 ``torch``、``numpy`` 和 ``omegaconf`` 的 Python 环境。由于转换会校验 mjlab G1 的默认关节姿态，需要先在训练机器上导出默认关节姿态：

1. 在训练机器上导出默认关节姿态到 ``g1_q0.pt``：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt

2. 将 ``g1_q0.pt`` 复制到转换机器。
3. 在转换机器上使用这个姿态文件，下载并转换 40 条片段：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. 将整个 ``/data/lafan1_g1`` 目录（含 ``manifest.json``）复制到训练机器，并将 ``SUPERMIMIC_MOTION_PATH`` 指向这个目录。

运行
----

按下方格式写出每条自定义 ``clip_*.pt`` 并放入同一目录（加载器按文件名排序读取），将 ``SUPERMIMIC_MOTION_PATH`` 指向该目录，训练时加 ``env.train.num_clips=null``\ （评测加 ``env.eval.num_clips=null``）即可加载目录下全部片段：

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/my_clips
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       env.train.num_clips=null \
       runner.logger.log_path=/data/runs/supermimic_custom

评测超过 40 条片段时的容量配置见 :doc:`../../evaluations/guides/supermimic`。如果目录包含 ``manifest.json``，加载器会校验每条片段的 SHA-256 并拒绝未登记文件；不含 ``manifest.json`` 时仅检查张量形状与数值合法性。

片段文件格式
------------

每个 ``clip_*.pt`` 都是用 ``torch.save`` 保存的 Python 字典：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 键
     - 内容
   * - ``fps``
     - ``50`` 或 ``30``。30 Hz 片段会自动重采样到 50 Hz 控制网格并用中心差分重算速度；50 Hz 片段按原样读取，**速度列必须填写准确**。不支持其他帧率。
   * - ``joint_names``
     - 29 个关节名列表，顺序必须与 mjlab Unitree G1 一致（左腿、右腿、腰部、左臂、右臂）。
   * - ``q0``
     - 29 维默认关节姿态（rad），必须与 mjlab 站立默认姿态一致。
   * - ``state``
     - 形状为 ``[T, 71]`` 的浮点张量，列布局见下表；不能包含 NaN/Inf，根四元数必须已归一化。

``state``\ （``[T, 71]``）列布局如下：

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - 列区间
     - 内容
     - 单位
   * - ``0:3``
     - 根节点（骨盆）位置，世界坐标系
     - m
   * - ``3:7``
     - 根节点四元数（``wxyz``）
     - —
   * - ``7:36``
     - 关节角度，顺序同 ``joint_names``
     - rad
   * - ``36:39``
     - 根节点线速度，世界坐标系
     - m/s
   * - ``39:42``
     - 根节点角速度，根节点坐标系
     - rad/s
   * - ``42:71``
     - 关节角速度
     - rad/s

先用 ``python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt`` 导出默认关节姿态 ``g1_q0.pt``，再用 ``torch.save`` 将每条自定义片段字典保存为 ``clip_000.pt`` 等文件。训练与评测启动时，加载器会自动校验字段、关节顺序、``q0`` 一致性、四元数模长和数值有限性。
