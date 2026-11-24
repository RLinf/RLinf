G1 动作数据准备
===============

为 :doc:`G1 动作跟踪环境 <g1_mimic>` 和 :doc:`SuperMimic <supermimic>` 准备参考动作片段，包括 LAFAN1 离线转换以及自定义 ``clip_*.pt`` 文件格式。

安装
----

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

动作目录下存放若干 ``clip_*.pt`` 文件，加载器按文件名排序读取。将 ``SUPERMIMIC_MOTION_PATH`` 指向你的动作目录后，在 :doc:`supermimic` 的训练命令或 :doc:`../../evaluations/guides/supermimic` 的评测命令后追加下面的覆盖参数即可。评测超过 40 条片段时的容量配置见 :doc:`../../evaluations/guides/supermimic`。

自定义片段
~~~~~~~~~~

按下方格式写出每个 ``clip_*.pt`` 并放入同一目录，将 ``SUPERMIMIC_MOTION_PATH`` 指向这个目录，训练时加 ``env.train.num_clips=null``\ （评测加 ``env.eval.num_clips=null``）即可加载目录下全部片段。

如果目录包含 ``manifest.json``，加载器会校验每条片段的 SHA-256 并拒绝未登记文件；不含 ``manifest.json`` 时仅检查张量形状与数值合法性。

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

下面的示例生成一条 5 秒原地站立片段，并在保存前用加载器校验：

.. code-block:: python

   from pathlib import Path

   import torch

   from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
   from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

   default = asset_default_state().float()  # mjlab 站立关键帧，[71]
   T = 250                                  # 50 Hz 下 5 秒
   state = torch.zeros(T, 71)
   state[:, :36] = default[:36]             # 根位姿与关节角，速度保持 0
   payload = {
       "fps": 50,
       "joint_names": list(G1_JOINT_NAMES),
       "q0": default[7:36],
       "state": state,
   }
   decode_motion_payload(payload, "clip_000.pt")  # 不合格时抛出 ValueError
   out = Path("/data/my_clips")
   out.mkdir(parents=True, exist_ok=True)
   torch.save(payload, out / "clip_000.pt")
