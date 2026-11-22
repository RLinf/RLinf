SuperMimic 动作数据
===================

主指南 :doc:`supermimic` 用论文的 40 条 LAFAN1 片段训练。本页介绍如何用其他动作数据训练：实验性的 BONES-SEED 转换，以及自备片段要遵守的文件格式。

安装
----

沿用 :doc:`supermimic` 里的环境。所有命令都在 RLinf 仓库根目录执行，并且已经激活这个环境。

运行
----

一个片段目录里放若干 ``clip_*.pt``\ （按文件名排序加载）。先在终端里执行主指南的设置代码，再把 ``SUPERMIMIC_MOTION_PATH`` 改成这个目录，然后用主指南里的训练和评测命令，加上下面各节的覆盖项。

动作跟踪评测时每条片段占一个环境，默认规模按 40 条 LAFAN1 片段设定。加载的片段超过 40 条时，只设 ``env.eval.num_clips=null`` 会在启动时的容量校验里报错，还要增大 ``env.eval.total_num_envs``，或增大 ``env.eval.rollout_epoch`` 分几轮播放，让两者的乘积不少于片段数。如果有片段超过 ``env.eval.max_steps_per_rollout_epoch`` 个控制步（默认 13656），也要相应调大。

BONES-SEED（实验性）
~~~~~~~~~~~~~~~~~~~~

官方数据在 Hugging Face 的 `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_，G1 轨迹位于 ``g1/csv/{日期}/{动作名}.csv``。先读数据包里的 ``LICENSE.md``；脚本不下载数据。表头约定沿用 NVIDIA ProtoMotions 的 `SEED G1 CSV 说明 <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_：120 Hz；``root_translateX/Y/Z`` 单位厘米；``root_rotateX/Y/Z`` 是外旋 XYZ 欧拉角、单位度；29 个 ``<关节名>_dof`` 列单位度，名字对应 mjlab 的 G1 关节。这个约定还没有拿官方 CSV 核对过。

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_annotated
   export SUPERMIMIC_MOTION_PATH=/data/bones_g1_annotated

``--input`` 可以是单个 CSV 或递归扫描的目录。片段重采样到 50 Hz，已有输出默认不覆盖（``--overwrite`` 除外），单个损坏的 CSV 会报告原因而不影响其他文件。转换器只在合成 CSV 上测试过。

``bones_feasibility.py`` 把每条片段连同几何标注写进 ``--out`` 目录，不删除任何片段。训练和动作跟踪评测加载时，按动作名和这些标注筛掉不合格的片段。用这些片段训练或评测时，在启动命令后追加：

.. code-block:: bash

   env.train.motion_format=bones_seed env.train.num_clips=null \
       env.eval.motion_format=bones_seed env.eval.num_clips=null

BONES-SEED 的加载器把动作库放在主机内存，只把查询到的行传到 GPU。

自备片段
~~~~~~~~

按下面的格式写出每条片段，放进同一个目录，训练时加 ``env.train.num_clips=null``\ （评测加 ``env.eval.num_clips=null``），加载目录里的全部片段。

目录里可以再放一个 ``manifest.json``。有清单时，加载器会拒绝 SHA-256 对不上或清单里没有的片段；没有清单时只检查格式。BONES-SEED 格式不读清单。

片段文件格式
------------

每个 ``clip_*.pt`` 是用 ``torch.save`` 保存的字典：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 键
     - 内容
   * - ``fps``
     - 50，或 30。30 Hz 的片段会重采样到 50 Hz 控制网格，速度用中心差分重新计算；50 Hz 的片段按原样使用，**速度列必须自己填对**。不支持其他帧率。
   * - ``joint_names``
     - 29 个关节名，必须与 ``rlinf.envs.sim.g1_mimic.constants.G1_JOINT_NAMES`` 的顺序完全相同（它就是 mjlab G1 的关节顺序）。
   * - ``q0``
     - 29 维默认关节姿态（rad），必须等于 mjlab 的默认关节姿态，即 ``asset_default_state()[7:36]``。
   * - ``state``
     - ``[T, 71]`` 的浮点张量，列布局见下表，不能有 NaN 或 Inf，根四元数要归一化。
   * - ``name``
     - 可选，BONES-SEED 转换会写入。

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - 列
     - 内容
     - 单位
   * - ``0:3``
     - 根（pelvis）位置，世界系
     - m
   * - ``3:7``
     - 根四元数，顺序 wxyz
     - —
   * - ``7:36``
     - 关节角，顺序同 ``joint_names``
     - rad
   * - ``36:39``
     - 根线速度，世界系
     - m/s
   * - ``39:42``
     - 根角速度，根坐标系
     - rad/s
   * - ``42:71``
     - 关节速度
     - rad/s

下面的最小示例写出一条原地站立 5 s 的片段，并用加载器自带的检查验证这条片段：

.. code-block:: python

   from pathlib import Path

   import torch

   from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
   from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

   default = asset_default_state().float()  # mjlab 站立关键帧，[71]
   T = 250                                  # 50 Hz 下 5 s
   state = torch.zeros(T, 71)
   state[:, :36] = default[:36]             # 根位置、根四元数、关节角；速度保持 0
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
