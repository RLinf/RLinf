SuperMimic：用可微世界模型训练 G1 动作跟踪
==========================================

本指南在 MuJoCo Warp 里用 SuperMimic 训练并评测 Unitree G1 的全身动作跟踪 policy：安装环境、准备参考动作、选择配置、在一张或多张 GPU 上训练、恢复训练、评测。算法、公式以及实现与论文的差异见 :doc:`../../reference/algorithms/supermimic`。第一次使用可以直接照 `运行`_ 一节走一遍。

除非另有说明，本页所有命令都在 RLinf 仓库根目录执行，并且已经激活了下面安装的 SuperMimic 环境。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      SuperMimic：穿过可微世界模型的 32 步反向传播 + GAE critic

   .. grid-item-card:: 模型
      :text-align: center

      世界模型（动力学 + 终止头）、编码器-解码器 actor、critic

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      MuJoCo Warp G1（mjlab）；LAFAN1、BONES-SEED

   .. grid-item-card:: 训练
      :text-align: center

      每张 GPU 一个进程；NCCL 数据并行；Slurm 多节点（实验性）

| **你将完成：** 安装独立环境 -> 准备 LAFAN1 片段 -> 用 smoke 配置训练、恢复、评测 -> 换成论文配置在一张或多张 GPU 上训练。
| **前置条件：** NVIDIA GPU；RLinf 仓库（:doc:`安装说明 <../../start/installation>`）；LAFAN1 G1 片段（见 `准备参考动作`_）。

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 项目
     - 内容
   * - 仿真器
     - 通过 `mjlab <https://github.com/mujocolab/mjlab>`_\ （commit ``8bb2bc9``）使用 MuJoCo Warp，G1 29 自由度，控制频率 50 Hz，每个控制步 4 个 5 ms 物理步。
   * - 任务
     - 跟踪重定向到 G1 的参考动作片段（LAFAN1 或 BONES-SEED）；可以再加一个编码器跟踪速度命令。
   * - 观测
     - 最近 10 帧的关节角（相对默认姿态）、关节速度、基座角速度、投影重力和上一动作，每帧 93 个数，带均匀传感器噪声。
   * - 动作
     - 29 个关节设定点 :math:`q_{sp} = \alpha a + q_0`，由低增益 PD 跟随。
   * - Goal
     - 10 帧参考，每帧 68 个数（680 维），或 `配置`_ 里的变体。
   * - 入口
     - 训练用 ``examples/embodiment/run_supermimic.sh``\ （不是通用的 ``run_embodiment.sh``），评测用 ``evaluations/run_eval.sh supermimic_*``\ （内部调用 ``evaluations/eval_supermimic.py``）。

SuperMimic 工作原理
-------------------

每张 GPU 上一个进程反复做三件事：在 MuJoCo Warp 里让所有环境走一个控制步，把转移写进 replay；按更新数据比从 replay 取 ``k`` 步窗口，拟合世界模型的动力学和终止头；从 replay 里的状态出发，在世界模型中展开 32 步想象 rollout，让回报沿着可微的世界模型反向传播来更新 actor，critic 拟合 GAE 目标。环境结束后，按动作时间轴分箱的回报决定从哪里重新开始，回报低的段落被抽到得更多。公式、网络结构以及与论文和社区实现的差异见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

安装
----

SuperMimic 需要 NVIDIA GPU（MuJoCo Warp）。在 RLinf 仓库根目录用安装脚本单独建一个虚拟环境：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

之后每打开一个新终端，都要先回到仓库根目录并执行 ``source .venv-supermimic/bin/activate``。

因为 mjlab 要求 ``torch>=2.14.0``，安装脚本只给这个环境选 torch 2.14.0，RLinf 全局的 torch 版本不变。torch 2.14.0 要求 ``setuptools>=77.0.3``，所以安装脚本也只为这个环境把全局的 ``setuptools<75.9`` 覆盖项换成 ``setuptools>=77.0.3,<82``\ （setuptools 82 移除了 ``pkg_resources``）。装好的 torch 是 CUDA 12 构建时，脚本会再装 ``warp-lang`` 1.18.0 的 CUDA 12 构建（PyPI 上的 wheel 是 CUDA 13 构建）。SuperMimic 专用的包列在 ``requirements/embodied/envs/supermimic.txt``；用 ``pip`` 把这个文件装进已有的 RLinf 环境会升级那里的 torch，所以请让 SuperMimic 用独立环境。``docker/Dockerfile`` 里的 ``embodied-supermimic-image`` 目标执行同样的安装。

G1 的模型、关节顺序、执行器增益和站立关键帧都来自 mjlab。每个参考片段都带着转换时用的默认姿态 ``q0``，如果它和 mjlab 的默认关节姿态不同，仿真器会拒绝加载：Goal 和观测减去的是 ``q0``，而 PD 目标加的是 mjlab 的默认姿态。

准备参考动作
------------

动作数据不放进仓库。一个片段目录里放若干 ``clip_*.pt``\ （按文件名排序加载），可以再放一个 ``manifest.json``。格式见下面的 `片段文件格式`_。

**LAFAN1（论文的 40 条）。** 片段是 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_ 的 ``g1/`` 目录。脚本固定修订版 ``ce1572906efe6157840e8474d5a0d7aa87481e74``，逐个核对 CSV 的大小和 SHA-256，并写出 ``manifest.json``。数据集卡片没有声明许可证；这些重定向数据来自 Ubisoft La Forge 的 `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_，后者使用 CC BY-NC-ND 4.0（非商业、禁止演绎）。传 ``--accept-license`` 之前请先读这些条款。

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

40 条片段共 2.45 小时。有 ``manifest.json`` 时，加载器会拒绝 SHA-256 对不上或清单里没有的片段，所以不要改名、增删或修改准备好的文件。

**BONES-SEED（实验性）。** 官方数据在 Hugging Face 的 `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_，G1 轨迹位于 ``g1/csv/{日期}/{动作名}.csv``。目录说明见 `bones-studio/seed-viewer <https://github.com/bones-studio/seed-viewer>`_。下载前先读数据包里的 ``LICENSE.md``。下面的脚本不下载数据。

公开表头来自 NVIDIA ProtoMotions 的 `SEED G1 CSV 说明 <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_，并与 GR00T-WholeBodyControl 里实际执行的 ``Rotation.from_euler("xyz")`` 一致：120 Hz；``root_translateX/Y/Z`` 是厘米；``root_rotateX/Y/Z`` 是外旋 XYZ 欧拉角、单位度；29 个关节列是 ``<关节名>_dof``、单位度，名字对应 mjlab 的 G1 关节。脚本按关节名重排到 ``G1_JOINT_NAMES``，不额外交换坐标轴。scipy 里小写 ``xyz`` 表示外旋。这个约定还没有拿一份官方 CSV 核对过。

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered

``--input`` 可以是单个 CSV，也可以是目录，目录会递归扫描。默认从 120 Hz 转到 50 Hz，速度由现有的 ``resample_states`` 在 50 Hz 网格上重算。已经存在的 ``clip_*.pt`` 不会被覆盖，除非加上 ``--overwrite``。单个损坏的 CSV 会打印原因并继续处理其他文件。片段的 ``name`` 是相对 ``--input`` 的路径、不含扩展名，因此不同日期下的同名文件不会写到同一个输出，附录 E 的名字过滤仍能看到 ``stair``、``sit`` 这类词。可行性过滤沿用 ``bones_feasibility.py``：它用 ``mj_geomDistance`` 把逐帧几何写进副本，加载器再按这些数组过滤。

先用少量 CSV 确认训练入口时，不必直接跑论文的 4096 个环境：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_bones_seed \
       env.num_envs=8 algorithm.total_updates=3 runner.save_interval=3

BONES-SEED 配置加载所有通过过滤的片段（``env.num_clips: null``），并在日志里打印每个过滤器去掉了多少。论文的 142,220 / 129,667 / 129,514 是它那个版本的数据。转换器只在合成 CSV 上测试过，还没有处理过完整的 BONES-SEED，也没有在真实 CSV 上核对过姿态。

片段文件格式
~~~~~~~~~~~~

每个 ``clip_*.pt`` 是用 ``torch.save`` 保存的字典：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 键
     - 内容
   * - ``fps``
     - 50，或 30。30 Hz 的片段会重采样到 50 Hz 控制网格，速度用中心差分重新计算；50 Hz 的片段按原样使用，**速度列必须自己填对**。其他帧率会被拒绝。
   * - ``joint_names``
     - 29 个关节名，必须与 ``rlinf.envs.sim.g1_mimic.constants.G1_JOINT_NAMES`` 的顺序完全相同（它就是 mjlab G1 的关节顺序）。
   * - ``q0``
     - 29 维默认关节姿态（rad），必须等于 mjlab 的默认关节姿态，即 ``asset_default_state()[7:36]``。
   * - ``state``
     - ``[T, 71]`` 的浮点张量，列布局见下表，不能有 NaN 或 Inf，根四元数要归一化。
   * - ``name``
     - 可选，只有 BONES-SEED 的名字过滤会读。

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

下面的最小示例写出一条原地站立 5 s 的片段，并用加载器同样的检查验证它：

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

没有 ``manifest.json`` 时，加载器只检查格式，不核对文件内容。

运行
----

下面用 ``supermimic_smoke``\ （8 个环境、3 次更新，单张 8 GB GPU 就能跑）把训练、保存、恢复、评测和查看曲线连起来走一遍。检查点、曲线和评测结果都放在同一个 ``RUN_DIR`` 下，后面每一步都靠它找到前一步的结果（``run_eval.sh`` 自己的运行日志仍写在仓库 ``logs/``，见 `评测`_）。

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/smoke

   # 1. 训练 3 次更新：每次更新存一个检查点，并写 TensorBoard
   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40 \
       runner.logger.log_path=$RUN_DIR runner.save_interval=1 \
       'runner.logger.logger_backends=[tensorboard]'

   # 2. 从最新检查点接着训练到第 5 次更新
   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40 \
       runner.logger.log_path=$RUN_DIR runner.save_interval=1 \
       'runner.logger.logger_backends=[tensorboard]' \
       runner.resume_dir=auto algorithm.total_updates=5

   # 3. 评测最新检查点（smoke 与 supermimic_lafan1 的网络相同，用它的评测配置）
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       eval.checkpoint=$RUN_DIR/supermimic_smoke eval.output_dir=$RUN_DIR/eval

   # 4. 查看训练曲线
   tensorboard --logdir $RUN_DIR/tensorboard

第 2 步的日志里应出现 ``Resuming SuperMimic networks, ... from .../global_step_3``。如果看到的是 ``resume_dir=auto: no complete checkpoint under ..., starting from scratch``，说明 ``log_path`` 和第 1 步不同，或者第 1 步没有写出检查点，这次运行是从头开始的，不是恢复。

跑完后 ``RUN_DIR`` 里是：

.. code-block:: text

   /data/runs/smoke/
   ├── supermimic_smoke/              # log_path/<experiment_name>
   │   ├── global_step_1/ ... global_step_5/
   │   │   ├── supermimic.pt          # 网络、优化器、调度、计数器
   │   │   ├── rank_0.pt              # 每个 rank 的随机数状态
   │   │   └── checkpoint_meta.json   # 大小、SHA-256、world size、维度、Goal 窗口
   ├── tensorboard/                   # events.out.tfevents.* 和 config.yaml
   └── eval/                          # summary.json、per_clip.csv、config.yaml

smoke 只验证流程能跑通。3～5 次更新的 policy 不会跟踪动作，评测里 ``eval/fail_rate`` 接近 1 是正常的。

配置
----

配置都在 ``examples/embodiment/config/``，共同的超参数在 ``supermimic_base.yaml``。批量都是全局量，多卡时每个 rank 分到相同的一份。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 配置
     - 训练内容
   * - ``supermimic_lafan1``
     - LAFAN1 上的论文设定：680 维 Goal、93 维观测、4096 个环境，每次更新 10 × 2048 个世界模型窗口、4096 条想象 rollout。
   * - ``supermimic_lafan1_anchor``
     - 689 维 Goal：680 维 Goal 加上参考躯干在机器人躯干坐标系下的位姿（CuboiLeo 的锚点）。
   * - ``supermimic_lafan1_goal730``
     - zita-ch 的 730 维 Goal（接下来 10 帧，每帧 73 个数）。
   * - ``supermimic_lafan1_obs99``
     - 99 维观测：93 维再加根朝向相对参考的 Rot6D。
   * - ``supermimic_lafan1_zita``
     - zita-ch 的接口和日程：730 维 Goal 与 99 维观测同时使用，解析终止头，前 500 次更新只训练世界模型，初始采集 64 步，共 16000 次更新，每 2000 次保存。奖励锚在骨盆，输入用滑动归一化，残差用机体系速度，世界模型状态误差除以残差的滑动标准差，critic 用 zita-ch 的额外特征。仿真器仍是 mjlab。
   * - ``supermimic_velocity``
     - 动作跟踪加速度命令编码器（附录 G），各占一半环境。
   * - ``supermimic_bones_seed``
     - 过滤后 BONES-SEED 上的论文设定。
   * - ``supermimic_bones_seed_2gpu``
     - BONES-SEED 上的论文双卡配方（见 `多卡训练`_）。
   * - ``supermimic_smoke``
     - 适合单张 8 GB GPU 的小规模运行：8 个环境、小批量、3 次更新，网络、``k``、``H`` 和奖励与论文设定相同。

有两个实验选项会改变终止头，默认关闭：``algorithm.termination_head: analytic`` 和 ``algorithm.termination_bias_init``，什么时候需要它们见算法页。``algorithm.seed`` 默认是 0，第 ``r`` 个 rank 用 ``seed + r``。``algorithm.wm_only_updates`` 默认是 0；大于 0 时，最前面这么多次更新只训练世界模型。``algorithm.critic_features`` 默认是 ``root4``\ （4 维根误差），``zita`` 改用 zita-ch 的额外特征。

下面四项在代码里省略时按 zita-ch：骨盆奖励锚、滑动输入归一化、机体系残差、世界模型误差除以滑动标准差。``supermimic_base.yaml`` 把论文配置写成另一组名字，所以 ``supermimic_lafan1`` 不会悄悄变成 zita-ch。

* ``algorithm.reward_anchor``：``pelvis`` 是骨盆，``torso`` 是 ``torso_link``。论文配置写 ``torso``。终止身体仍由 ``termination_body`` 决定。
* ``algorithm.input_norm``：``running`` 是滑动均值和方差，并先按固定尺度缩放 :math:`\xi`\ ；``none`` 保持原始数值。论文配置写 ``none``。
* ``algorithm.residual_frame``：``body`` 把线速度放在完整机体系，角速度按世界系增量再存回机体系；``heading`` 把位置和线速度放在偏航系，角速度直接加在机体系角速度上。论文配置写 ``heading``。
* ``algorithm.wm_state_loss``：``running`` 把坐标系里的误差除以残差滑动标准差后对 70 维取均值；``paper`` 是不加权平方；``scaled`` 是固定分块尺度。论文配置写 ``paper``。

``supermimic_lafan1_zita`` 显式写成 ``pelvis``、``running``、``body``、``running``，并覆盖基类配置。仿真器仍是 mjlab。它不能加载这四项还没写入检查点时训练出来的权重。一次训练用一张 80 GB A800、4096 个环境，跑完 16000 次更新。第一次编译约 30 分钟，16000 次更新约 4.5 小时。第 16000 次更新的检查点在 40 条 LAFAN1 片段上的 ``eval/normalized_return`` 为 0.856，40 条都播到了片段结束。同一次训练的第 14000 次更新是 0.800，失败率 0.10，40 条里 36 条播完。zita-ch 报告的是第 14000 次更新的 0.829，同一更新次数上这次低 0.029。

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       runner.logger.log_path=/data/runs/lafan1_zita \
       'runner.logger.logger_backends=[tensorboard]'

评测用 ``supermimic_lafan1_zita_eval``，单卡、不用 Ray。``eval.checkpoint`` 指向上面 ``log_path`` 下的 ``supermimic_lafan1_zita`` 目录时，会选最新的完整检查点，也可以指向某一个 ``global_step_<N>``。上面这次训练在第 16000 次更新上得到 ``eval/normalized_return`` 0.856，在第 14000 次更新上得到 0.800。zita-ch 报告的是第 14000 次更新的 0.829。评第 14000 次更新时，把检查点指到该次保存的目录：

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita/global_step_14000 \
       eval.output_dir=/data/runs/lafan1_zita/eval_14000

常用配置键
~~~~~~~~~~

下表是调整规模和训练长度时最常改的键，默认值来自 ``supermimic_base.yaml``\ （``supermimic_smoke`` 等配置会覆盖其中一部分）。都可以作为 Hydra 覆盖项写在 ``run_supermimic.sh <配置名>`` 后面。

.. list-table::
   :header-rows: 1
   :widths: 27 12 61

   * - 键
     - 默认值
     - 含义与调整影响
   * - ``algorithm.k``
     - 8
     - 世界模型的多步训练窗口长度。目前固定为 8，校验会拒绝其他值。
   * - ``algorithm.H``
     - 32
     - 想象 rollout 的长度，也就是穿过世界模型反向传播的步数。目前固定为 32。
   * - ``algorithm.gamma``、``algorithm.lam``
     - 0.99、0.95
     - GAE 的折扣和 :math:`\lambda`，用于 critic 目标和 actor 目标。
   * - ``algorithm.total_updates``
     - 320000
     - 训练的\ **更新次数**\ ，不是仿真步数（见表后说明）。
   * - ``algorithm.updates_per_env_step``
     - 0.4
     - 每采集一个向量仿真步累计 0.4 次全局更新（更新数据比）。调大则每份数据被训练得更多。
   * - ``algorithm.lr``、``algorithm.lr_decay_updates``
     - 2.0e-4、250000
     - 世界模型、actor、critic 共用的学习率；在前 ``lr_decay_updates`` 次更新内按余弦降到 0.03 × ``lr``，之后保持。
   * - ``algorithm.wm_batches``
     - 10
     - 每次更新拟合世界模型的小批次数。
   * - ``algorithm.wm_window_batch``
     - 2048
     - 每个世界模型小批次的 ``k`` 步窗口数，全局量。
   * - ``algorithm.imagine_batch``
     - 4096
     - 每次更新的想象 rollout 条数，全局量。显存的主要来源之一。
   * - ``algorithm.critic_epochs``、``algorithm.critic_minibatches``
     - 5、4
     - 每次更新 critic 遍历想象数据的轮数和每轮的小批次数。
   * - ``algorithm.warmup_sim_steps``
     - 40
     - 每次启动或恢复后、第一次更新前先采集的向量仿真步数，用来预热 replay；这些步不累计更新。
   * - ``env.num_envs``
     - 4096
     - 并行仿真环境数，全局量。采集吞吐和显存都随它增长。
   * - ``env.replay_length``
     - 1000
     - 每个环境的环形 replay 容量，按控制步计（50 Hz 下 20 s）。replay 显存与 ``num_envs × replay_length`` 成正比。
   * - ``env.num_clips``
     - 40
     - 按文件名顺序加载前 N 条片段；``null`` 加载全部。
   * - ``runner.save_interval``
     - -1
     - 每隔 N 次更新存一个检查点。默认 -1 只在最后一次更新时保存，长任务中途中断就没有检查点可恢复。
   * - ``runner.logger.logger_backends``
     - ``[]``
     - 默认不写 TensorBoard、WandB 或 SwanLab，只有控制台日志。要看曲线就设 ``[tensorboard]``。
   * - ``runner.val_check_interval``
     - -1
     - 训练中每隔 N 次更新评测一次；默认不评测。
   * - ``runner.logger.log_path``
     - 见说明
     - 检查点和日志的根目录。``run_supermimic.sh`` 在没有给出时用 ``logs/<时间>-<配置名>``，每次启动都是新目录。

**更新次数与仿真步数。** 训练循环每次采集一个向量仿真步（所有环境各走一步），累计 ``updates_per_env_step`` 次欠下的更新，累计满 1 就做一次更新。所以 ``total_updates=320000`` 大约对应 320000 / 0.4 = 800,000 个向量仿真步（另加预热），环境转移数再乘以 ``env.num_envs``。日志里的 ``train/global_update`` 是更新次数，``sim_vector_steps`` 和 ``env_transitions`` 是仿真步数和环境转移数。

**全局量。** ``env.num_envs``、``algorithm.wm_window_batch`` 和 ``algorithm.imagine_batch`` 是所有 rank 的总和，按 rank 均分；``updates_per_env_step`` 和 ``total_updates`` 数的也是全局更新。所以加 GPU 而不改这些值，训练的数学含义不变，见 `多卡训练`_。

单卡训练
--------

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]'

``run_supermimic.sh <配置名> [hydra 覆盖项...]`` 把配置名之后的参数都交给 Hydra。上面的命令显式设置了三项，因为默认值不适合长时间训练：

* ``runner.logger.log_path``：没有给出时写到 ``logs/<时间>-<配置名>``，每次启动换一个目录，之后恢复时必须找到原来的目录。固定一个目录，恢复和评测都直接用它。
* ``runner.save_interval``：默认 -1 只在最后保存。按每次更新的耗时选一个间隔（耗时用 `性能与硬件`_ 里的工具测）。
* ``runner.logger.logger_backends``：默认 ``[]`` 不写任何曲线。设为 ``[tensorboard]`` 后，用 ``tensorboard --logdir /data/runs/lafan1/tensorboard`` 查看。

在小显存 GPU 上快速检查：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_smoke env.num_clips=40

只有 smoke 启动会在你没设置时把 ``RAY_memory_usage_threshold`` 设为 0.99，因为 8 GB 内存的机器离 Ray 的默认阈值很近；其他配置都保留 Ray 内存监控的原样。Ray 对象存储限制为 512 MiB（``cluster.object_store_memory_mb``），因为 replay 从不经过它；这个限制只在 RLinf 自己启动 Ray 时生效。

多卡训练
--------

``cluster.component_placement.supermimic`` 列出使用的 GPU，每张 GPU 一个 rank。各 rank 做同步数据并行：各自步进自己的环境、维护自己的 replay、采自己那份批量，在裁剪之前按样本数加权 all-reduce 梯度，所以每个 rank 做完全相同的更新。rank 0 广播初始权重并负责评测。``train/param_divergence`` 每 ``algorithm.param_check_interval`` 次更新检查一次各 rank 的权重和 buffer 是否一致。running 归一化先把各 rank 这一批的样本量、均值和方差合并，再写入同一份滑动统计。

多用几张 GPU 有两种方式：

* **固定全局批量。** 配置不变，只加 GPU。全局批量、学习率和更新数据比都和单卡相同，每个 rank 分一份；相当于把单卡训练摊到多张卡上。

  .. code-block:: bash

     bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
         cluster.component_placement.supermimic=0-3

* **论文双卡配方。** 每张 GPU 保持单卡的规模，所以全局批量翻倍，梯度在两张卡之间取平均，学习率为 :math:`2\sqrt{2} \times 10^{-4}`\ （表 A.5）：

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     bash examples/embodiment/run_supermimic.sh supermimic_bones_seed_2gpu

  LAFAN1 用同样的覆盖项：``cluster.component_placement.supermimic=0-1 env.num_envs=8192 algorithm.wm_window_batch=4096 algorithm.imagine_batch=8192 algorithm.lr=2.8284271e-4``。

GPU 上的集合通信用 NCCL，超时由 ``RLINF_TIMEOUT``\ （分钟）决定。如果某个 rank 出错，RLinf 的 worker group 会立即停掉 driver 和其他 rank，不会等到这个超时。

用 Slurm 跑多节点（实验性）
---------------------------

``examples/embodiment/slurm_supermimic.sh`` 在每个 Slurm 节点上启动一个 Ray 节点，在每次 ``ray start`` 之前导出唯一的 ``RLINF_NODE_RANK``，并在第一个节点上运行 driver，placement 覆盖所有 GPU。分区、账号和 GPU 数来自你的 ``sbatch`` 选项。

提交前需要满足：

* 在 RLinf 仓库根目录、激活 SuperMimic 环境后执行 ``sbatch``。``sbatch`` 会把当前环境变量带进作业，各节点用的就是这个环境里的 ``python`` 和 ``ray``。
* 代码、虚拟环境、数据和 ``log_path`` 都放在所有节点能读写的共享存储上；每个 rank 都要写自己那份检查点。
* 网络需要指定接口时（例如 ``NCCL_SOCKET_IFNAME``），在 ``sbatch`` 之前 ``export``。

.. code-block:: bash

   sbatch --nodes=2 --gpus-per-node=8 --partition=<partition> \
       examples/embodiment/slurm_supermimic.sh supermimic_lafan1 \
       env.motion_path=/shared/lafan1_g1 runner.logger.log_path=/shared/runs/lafan1 \
       runner.save_interval=1000 'runner.logger.logger_backends=[tensorboard]'

这个脚本只用打桩的 ``srun`` 和 ``ray`` 检查过，还没有在 Slurm 集群上运行。

检查点与恢复
------------

``runner.save_interval`` 每隔 N 次更新写一个 ``log_path/<experiment_name>/global_step_<N>``，最后一次更新总会写。检查点包括 ``supermimic.pt``\ （网络、优化器、学习率调度、计数器、重置分箱统计）、每个 rank 一个 ``rank_<r>.pt``\ （随机数状态），以及 ``checkpoint_meta.json``\ （文件大小和 SHA-256、world size、模型维度、Goal 窗口、格式版本）。文件先写到 ``.tmp`` 目录，所有 rank 写完后再改名，所以中断的保存不会留下半成品。

恢复时用与原运行\ **相同**\ 的配置名和 ``log_path``，例如接着 `单卡训练`_ 的命令：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]' \
       runner.resume_dir=auto

``runner.resume_dir=auto`` 在 ``log_path/<experiment_name>/`` 下选最新的完整检查点，训练接着跑到 ``algorithm.total_updates``；给一个 ``global_step_<N>`` 路径则用那个检查点。``auto`` 找不到完整检查点时\ **不会报错**\ ，而是记一条 ``resume_dir=auto: no complete checkpoint under ..., starting from scratch`` 后从头训练，所以请确认日志里有 ``Resuming SuperMimic ... from .../global_step_<N>``。

格式、维度或 Goal 窗口不同的检查点会被拒绝。rank 数不同时，权重和优化器照常加载，随机数流从种子重新开始。replay 和仿真状态不保存，恢复后要先重新预热 replay，所以这是接着训练，不是精确复原。

评测
----

``evaluations/eval_supermimic.py`` 只从检查点加载网络，单卡运行，不需要 Ray。``evaluations/run_eval.sh`` 看到 ``supermimic_`` 开头的配置名就会调用它，所以 ``run_eval.sh supermimic_lafan1_eval`` 和 ``run_eval.sh supermimic supermimic_lafan1_eval`` 等价。``supermimic_push_recovery`` 改用 ``evaluations/eval_supermimic_push.py``。RLinf 通用评测配置里的 ``rollout.model.model_path``、``runner.task_type: embodied_eval`` 等字段对 SuperMimic 不适用；它用 ``eval.checkpoint`` 指定检查点。

``eval.checkpoint`` 可以是一个 ``global_step_<N>`` 目录，也可以是装着多个检查点的目录（如 ``log_path/<experiment_name>``），后者选最新的完整检查点。评测配置与训练配置的对应关系：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 训练配置
     - 评测配置（``evaluations/supermimic/``）
   * - ``supermimic_lafan1``、``supermimic_smoke``
     - ``supermimic_lafan1_eval``\ （smoke 的网络与 Goal 和 ``supermimic_lafan1`` 相同）
   * - ``supermimic_lafan1_anchor``
     - ``supermimic_lafan1_anchor_eval``
   * - ``supermimic_lafan1_goal730``
     - ``supermimic_lafan1_goal730_eval``
   * - ``supermimic_lafan1_obs99``
     - ``supermimic_lafan1_obs99_eval``
   * - ``supermimic_lafan1_zita``
     - ``supermimic_lafan1_zita_eval``
   * - ``supermimic_velocity``
     - ``supermimic_velocity_eval``
   * - ``supermimic_bones_seed``、``supermimic_bones_seed_2gpu``
     - ``supermimic_bones_seed_eval``\ （双卡配方只改了规模和学习率）

评测配置与检查点不匹配时（维度或 Goal 窗口不同），加载会被拒绝。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1 \
       eval.output_dir=/data/runs/lafan1/eval
   bash evaluations/run_eval.sh supermimic_velocity_eval \
       eval.checkpoint=/data/runs/velocity/supermimic_velocity \
       eval.output_dir=/data/runs/velocity/eval

**输出位置。** ``run_eval.sh`` 把运行日志写到仓库下的 ``logs/<时间>-<评测配置名>/eval_embodiment.log``。评测结果写到 ``eval.output_dir``；不设置时写到同一日志目录下的 ``eval_global_step_<N>/``。结果目录里有：

* ``summary.json``：``meta``\ （检查点路径、``global_step``、种子、设备、片段数、数据路径、检查点的 world size）和 ``metrics``\ （下面各指标的汇总值，没有数值的写成 ``null``）。
* ``per_clip.csv``：每条片段一行。
* ``velocity_g2.csv`` 或 ``velocity_smoke.csv``：速度评测时每条命令一行。
* ``config.yaml``：这次评测实际用的完整配置。

``eval/*``、``eval_velocity_g2/*``、``eval_velocity_smoke/*`` 是 ``summary.json`` 和日志里指标名的前缀，不是目录。

**动作跟踪** 把每条片段从第 0 帧播放一次，关闭传感器噪声和动作噪声，结果写到 ``per_clip.csv``。片段 :math:`i` 的归一化回报为

.. math::

   \mathrm{NR}_i = \frac{\sum_{t} r_t}{0.1 \times T_i},

其中 :math:`r_t` 是实际执行的每个控制步的奖励，0.1 是单步奖励的上限，:math:`T_i` 是片段在 50 Hz 控制网格上的帧数（计划控制步数）。分母用计划长度而不是实际执行的步数，所以提前失败时没播放的部分仍留在分母里，:math:`\mathrm{NR}_i \le 1`。``eval/normalized_return`` 是各片段的不加权平均。

**速度跟踪** 用 ``eval.velocity`` 选择：

* ``g2``\ （速度评测配置的默认值）运行附录 G.2 阶跃测试：在 (vx, vy)、(vx, wz)、(vy, wz) 三个平面上各取 15 × 15 网格，去掉站立命令，共 672 条命令，每条 4 次试验，从 mjlab 的站立姿态开始执行 5 s。每次试验的值是最后 2 s 平均速度误差的范数除以命令范数；每条命令报告没有摔倒的试验的中位数；4 次都摔倒的命令没有数值（CSV 里是 NaN，``summary.json`` 里是 ``null``），计入 ``eval_velocity_g2/commands_all_trials_fell``。论文没有说明试验之间什么不同，本实现让仿真上下文和初始关节噪声各不相同。结果写到 ``velocity_g2.csv``，汇总指标以 ``eval_velocity_g2/`` 为前缀。
* ``smoke`` 从参考起点开始保持 7 条固定命令，写出 ``velocity_smoke.csv``，汇总指标以 ``eval_velocity_smoke/`` 为前缀。它只检查速度编码器能跑通，不是论文指标。

**静止站立姿态推力恢复。** 论文 §4.3 要求静态 T-pose。本入口的协议是 stationary standing-pose push recovery，不把它当作论文的 T-pose 协议。``evaluations/run_eval.sh supermimic_push_recovery`` 调用 ``evaluations/eval_supermimic_push.py``，单卡、不需要 Ray，用已有跟踪检查点，不重新训练。参考在整个评测里保持同一个站立姿态，Goal 不会换成走路。

论文规定的是：仿真中保持静态 T-pose，施加 0.5–4 m/s 的速度推力；带 critic 的策略在 0.5–2.4 s 内稳定，无 critic 的策略在 0.5 m/s 时持续踉跄、在 4 m/s 时摔倒。论文没有公布 T-pose 的关节角、区间内的速度网格、方向、坐标系、试验次数、推力前的等待，也没有公布“稳定”的数值门槛。本实现固定使用 mjlab 的屈膝关键帧 ``KNEES_BENT_KEYFRAME``，与 ``q0`` 和 ``reset_standing`` 相同。正运动学下，每侧手腕相对同侧 ``shoulder_roll_link`` 约在前方 0.10 m、外侧 0.07 m、下方 0.31 m；肩 roll 为 ±0.2 rad，肘为 0.6 rad，双臂垂在身侧而不是水平张开。资产里只有 ``HOME_KEYFRAME`` 和 ``KNEES_BENT_KEYFRAME``，MJCF 没有 keyframe；关节角全 0 时前臂朝前约 0.20 m、低于肩约 0.19 m，侧向几乎为 0，也不是 T-pose。论文没有公布 T-pose 关节角，所以这里不另造一套水平臂姿态并把它叫成论文姿态。默认速度是 0.5、1、2、4 m/s，每种 4 次，这些数字是本实现的选择。``eval.direction=random`` 在水平面随机取方向，也可以写成 ``eval.direction=[1.0,0.0]``。``eval.frame`` 取 ``world`` 或 ``heading``。``eval.push_mode=add`` 把该水平速度加到当前世界系根线速度上，``set`` 替换 xy 并保留 z。写入 MuJoCo Warp 的根速度后调用 ``forward``，这一步不积分；下一步 ``step`` 才使用新速度。这次写入不改根位置、角速度和关节状态。

跌倒沿用训练的 ``failure_signals``：相对这份静止参考，高度误差超过 0.25 m 或倾角误差超过 0.8 即摔倒。下面几项是本实现的约定。推力之后要同时满足：根水平速度不超过 ``eval.speed_threshold``\ （默认 0.15 m/s），根倾角差不超过 ``eval.tilt_threshold``\ （默认 0.35），躯干朝向误差不超过 ``eval.torso_ori_threshold``\ （默认 0.5 rad），并且连续保持 ``eval.stable_seconds``\ （默认 0.5 s）。``eval.max_xy_drift`` 默认是 ``null``，不要求回到原来的水平位置。``recovery_time_s`` 从推力施加时刻算到稳定窗口结束的那个控制步；``stable_onset_time_s`` 是这个窗口的第一帧。没有恢复的试次把时间留空，不写成 0。推力之前摔倒、关节越限或状态非有限，记为 ``initialization_failed``，不计入推力后的跌倒率。``eval.settle_seconds``\ （默认 1 s）和 ``eval.max_seconds``\ （默认 5 s）也是本实现的等待和截止，不是论文里的 0.5–2.4 s。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       eval.checkpoint=/data/runs/lafan1/supermimic_lafan1 \
       eval.output_dir=/data/runs/lafan1/push_recovery

``summary.json`` 按速度给出试验数、恢复率、跌倒率、超时率，以及成功试次的平均和中位恢复时间；某一档没有成功试次时，恢复时间是 ``null``。``per_trial.csv`` 每次一行，包含方向、读回的速度增量、结果和原因。``eval.save_trajectory=true`` 时另写 ``trajectory.csv``。锚点、730 维 Goal 或 99 维观测的检查点需要在命令里覆盖对应的 ``algorithm`` 字段，使它和检查点一致。

在一张 8 GB 的 RTX 4060 上，用只更新了 3 步的检查点跑过 0.5 m/s 和 1 m/s 各 1 次，推力前 0.4 s、之后最多 0.8 s，世界系 +x。两次都在推力前站住，读回的水平速度增量分别是 0.5 m/s 和 1.0 m/s，随后按训练的跌倒条件摔倒，恢复时间为空。这只说明评测链路能跑完。这次用的是上述屈膝站立姿态，不能说明策略达到论文 T-pose 协议的恢复性能。

也可以用 ``runner.only_eval=true runner.resume_dir=<检查点>`` 走训练入口评测，此时用训练配置的环境数，结果只记在日志里。设置了 ``runner.val_check_interval`` 时，训练每隔这么多次更新评测一次，最后一次更新后也会评测。动作和速度评测走这条路径；推力恢复只走上面的独立入口。

可视化与结果
------------

日志走 RLinf 的 ``MetricLogger``\ （TensorBoard、WandB、SwanLab）。默认 ``runner.logger.logger_backends: []`` 不写任何后端；设为 ``[tensorboard]`` 后，事件文件和本次配置 ``config.yaml`` 写在 ``log_path/tensorboard/``：

.. code-block:: bash

   tensorboard --logdir /data/runs/lafan1/tensorboard

多个 rank 时，计数求和，计时和显存取最大的 rank，共享计数取 rank 0，损失取平均；``runner.log_per_rank_metrics=true`` 会再写一份 ``rank_<r>/`` 前缀的指标。常用的键：

* ``train/global_update``、``sim_vector_steps``、``env_transitions``：更新次数、向量仿真步数、环境转移数。
* ``wm/one_step_loss``、``wm/k_step_loss``：世界模型一步和八步之后的误差。
* ``imagination/survival``、``imagination/termination_prob``：32 步存活权重和平均终止概率。
* ``actor/grad_norm``\ （裁剪前）、``actor/encoder_grad_norm``、``actor/decoder_grad_norm``。
* ``critic/loss``、``critic/value_mean``、``critic/target_mean``。
* ``reset/bin_return_mean``、``reset/bin_return_min``：动作重置分箱的回报。
* ``train/nonfinite_skips``、``train/param_divergence``、``time/comm_s``。
* ``memory/torch_peak_allocated``、``memory/torch_peak_reserved``\ （PyTorch 分配器）、``memory/warp_mempool_peak``\ （Warp 内存池）、``memory/device_used``\ （驱动看到的整张 GPU）。这几个计数互相重叠，不能相加。

控制台每次更新还会打印一行 ``supermimic metrics {...}``，评测时打印 ``supermimic eval {...}``，没有开任何后端时也能从这里看到数值。

故障排查
--------

* **CUDA out of memory。** 报错会写出 rank 和它分到的规模。要保持批量就加 GPU，或者另用一个小规模配置；``toolkits/supermimic/profile_sim.py`` 可以测给定规模的显存。
* **Ray 因内存不足杀掉 worker。** 应该释放主机内存，而不是调高 ``RAY_memory_usage_threshold``；smoke 启动是唯一的例外。CPU 核多、内存少的机器上，Ray 会按核数预启动空闲 worker（每个约 50 MB），SuperMimic 用不到它们；设置 ``RAY_prestart_worker_first_driver=0`` 可以不启动这些进程，内存监控保持不变。
* **"No verified G1 reference clips"。** ``SUPERMIMIC_MOTION_PATH`` 或 ``env.motion_path`` 没有指向准备好的片段。
* **"reference clip q0 differs from the mjlab G1 default joint pose"。** 片段是用别的默认姿态转换的，需要重新转换。
* **BONES-SEED 的 CSV 被拒绝。** 表头需要 ``Frame``、``root_translateX/Y/Z``、``root_rotateX/Y/Z`` 和 29 个 ``关节名_dof``。根位置用厘米，角度用度。报错会写出文件名和缺掉的列。
* **恢复后从第 0 次更新开始。** 日志里有 ``resume_dir=auto: no complete checkpoint under ...``：``log_path`` 或配置名与原运行不同，或者原运行从没写过检查点（``save_interval`` 为 -1 且没有跑完）。
* **检查点格式或 Goal 窗口被拒绝。** 检查点来自别的版本或配置，不能用当前配置恢复或评测。
* **没有 TensorBoard 文件。** ``runner.logger.logger_backends`` 还是默认的 ``[]``。
* **"linesearch iterations limit reached"。** 未训练的 policy 会触发 MuJoCo Warp 求解器的这条警告，不会中断训练。

性能与硬件
----------

``toolkits/supermimic/profile_sim.py`` 对一组环境数逐个起子进程，测采集吞吐、更新耗时、replay 大小和上面那几个显存计数：

.. code-block:: bash

   python toolkits/supermimic/profile_sim.py --motion-path /data/lafan1_g1 \
       --num-envs 256 1024 4096 --steps 30 --warmup 20 --update-iters 3 \
       --out profile.json

在 WSL2 下的一张 8 GB 笔记本 GPU 上：1024 个环境每秒采集约 3,900 个环境转移；用论文批量四分之一（1024 条想象 rollout、512 个窗口）做一次更新约 21 s，PyTorch 峰值 4.2 GB；论文批量下一次更新需要约 11 GB，超过这张卡的显存。这些数字来自受 kernel 启动开销限制的笔记本环境，不能用来推测数据中心 GPU 的表现；正式训练前请先在自己的硬件上测出每次更新的秒数。

已经运行过的内容
----------------

在一张 8 GB GPU 上用 MuJoCo Warp 和 LAFAN1 片段跑过：LAFAN1 各配置（680 维、689 维、730 维、99 维观测、速度）的小规模短训练，包括 driver 被强杀后的恢复，动作和速度评测，以及 G.2 阶跃测试。多卡的数值逻辑用两个 CPU（Gloo）进程测试过。

在两张 80 GB 的 A800 上，用 40 条 LAFAN1 片段又做了一轮本机调度的验收（不在 Slurm 作业里）。下面这份结果复用了已有的虚拟环境，所以它自己的阶段 01 记为不适用。在此之前，同一次提交上用 ``install.sh`` 从零安装的阶段 01 已经通过，后面的训练和评测用的就是那个环境。通过的内容是：

* 单元测试 10 个文件：201 通过，1 个因为加速器数量不是 4 而跳过。
* 单卡真实 MuJoCo Warp smoke：3 次更新，计数单调，指标有限，检查点完整。
* 单卡在写出检查点后强杀 driver，从最近的完整检查点恢复并继续训练。双卡同样测过保存、强杀和恢复；双卡检查点可以换成单卡继续训练。杀掉一个 rank 后，其余进程在 20.3 秒内以非零状态退出，没有留下孤儿进程。
* 动作评测覆盖 40 条片段。同一检查点重复评测的指标差在容差以内（绝对容差 :math:`5\times10^{-5}`，相对容差 :math:`2\times10^{-4}`），但不是逐位相同。训练入口的 ``only_eval`` 和速度评测通过。速度评测使用附录 G.2 的命令网格，每条命令 2 次试验，不是论文的 4 次。
* 单卡 profiling 实测了 128 和 512 个环境的吞吐与显存。4096 个环境只做了从 512 线性外推的估计，没有实测。
* 双卡小规模训练走真实 NCCL：两次运行的 ``train/param_divergence`` 都是 0，两边计数逐项相等，两个 rank 的检查点都完整。全局规模是 64 个环境、256 条想象 rollout，不是论文双卡配方。同样规模的速度配置短训练也完成了。
* 同样的双卡规模上，短期学习信号跑了 12 次更新。指标有限、计数单调。归一化回报从 0.001892 到 0.001893，失败率保持为 1。这是单种子、短预算，不能据此说 policy 学会了跟踪。
* ``supermimic_lafan1_zita`` 的一次完整训练：一张 80 GB A800、4096 个环境、16000 次更新。第一次编译约 30 分钟，16000 次更新约 4.5 小时。第 16000 次更新的检查点做了 40 条片段的动作评测，``eval/normalized_return`` 为 0.856，``eval/fail_rate`` 为 0，``eval/clip_end_rate`` 为 1。第 14000 次更新的检查点同样评了 40 条，``eval/normalized_return`` 为 0.800，``eval/fail_rate`` 为 0.10，``eval/clip_end_rate`` 为 0.90。zita-ch 报告的对照值是第 14000 次更新的 0.829，同一更新次数上这次低 0.029。仿真器是 mjlab。本页还没有这次训练的学习曲线。

尚未运行：论文配置 ``supermimic_lafan1``\ （680 维 Goal、320000 次更新）、论文双卡配方、多节点、Slurm、BONES-SEED 真实数据，以及论文 §4.3 的 T-pose 恢复性能（当前入口是静止站立姿态，不能代替该项）。
