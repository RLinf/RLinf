SuperMimic：用可微世界模型训练 G1 动作跟踪
==========================================

本指南在 MuJoCo Warp 里用 SuperMimic 训练并评测 Unitree G1 的全身动作跟踪 policy：安装环境、准备参考动作、选择 GenCoord 或 MaxCoord、在一张或多张 GPU 上训练、恢复训练、评测。算法以及和论文的差别见 :doc:`../../reference/algorithms/supermimic`。第一次使用可以直接照 `运行`_ 一节走一遍。

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

      世界模型、动作 policy、critic；隐藏层用 Muon，其余参数用 fused Adam

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      MuJoCo Warp G1（mjlab）；LAFAN1、BONES-SEED

   .. grid-item-card:: 训练
      :text-align: center

      每张 GPU 上一组 env、rollout 和 FSDP actor worker

| **你将完成：** 安装独立环境 -> 准备 LAFAN1 片段 -> 小规模训练、恢复、评测 -> 用 GenCoord 或 MaxCoord 训练。
| **前置条件：** NVIDIA GPU，或实验性的 AMD MI300X 路径（见 `安装`_）；RLinf 仓库（:doc:`安装说明 <../../start/installation>`）；LAFAN1 G1 片段（见 `准备参考动作`_）。

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 项目
     - 内容
   * - 仿真器
     - 通过 `mjlab <https://github.com/mujocolab/mjlab>`_\ （NVIDIA 固定 commit ``8bb2bc9``，AMD 使用 1.2.0）使用 MuJoCo Warp，G1 29 自由度，控制频率 50 Hz，每个控制步 4 个 5 ms 物理步。
   * - 任务
     - 跟踪重定向到 G1 的参考动作片段（LAFAN1 或 BONES-SEED）。
   * - 状态
     - ``gencoord`` 是 71 维仿真状态，编码 68 维。``maxcoord`` 是 14 个被跟踪身体的位姿和速度，编码 214 维。两种表示不能共用检查点。
   * - 动作
     - 29 个关节设定点，由低增益 PD 跟随。训练时动作噪声标准差是 0.1。
   * - Goal
     - 从当前参考帧起，每隔 5 个控制步取一帧，共 10 帧，贴在片段末帧。Actor 的 Goal 是 640 维（各帧关节角、关节速度，以及根朝向相对当前根的 6 维旋转）。世界模型的 Goal 再拼上相对根位置，为 670 维。
   * - 入口
     - 训练用 ``examples/embodiment/train_embodied_agent.py``，评测用 ``evaluations/run_eval.sh supermimic_lafan1_eval``\ （内部调用 ``evaluations/eval_embodied_agent.py``）。

SuperMimic 工作原理
-------------------

SuperMimic 反复做三件事：在 MuJoCo Warp 里让所有环境走一个控制步，把转移写进 replay；按更新数据比从 replay 取 ``k = 8`` 步窗口，拟合世界模型；从 replay 里的状态出发，在世界模型中展开 ``H = 32`` 步想象 rollout，让回报沿着可微的世界模型反向传播来更新 actor，critic 拟合 GAE 目标。环境结束后，按动作时间轴上 0.2 秒一箱的回报决定从哪里重新开始，回报低的段落被抽到得更多。公式见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

在 RLinf 里，这三件事分给三组 worker，每张 GPU 上各有一个：

* ``SuperMimicEnvWorker`` 持有这张卡分到的环境、重置采样器和奖励计算。
* ``SuperMimicRolloutWorker`` 持有 policy 和 critic 的副本，返回动作，以及超时结束的回合在重置采样回报里用来自举的 critic 值。
* ``EmbodiedSuperMimicFSDPPolicy`` 是 FSDP2 actor，持有 replay、世界模型、actor、critic 和它们的优化器。

``SuperMimicRunner`` 是控制面：它算出下一次更新到期前要采多少个控制步，让三组 worker 采这些步，在 actor 上执行到期的更新，把新权重同步给 rollout worker，然后继续采集；日志、评测和检查点也由它负责。

安装
----

默认的 NVIDIA 路径通过 MuJoCo Warp 运行仿真。在 RLinf 仓库根目录用安装脚本单独建一个虚拟环境：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

之后每打开一个新终端，都要先回到仓库根目录并执行 ``source .venv-supermimic/bin/activate``。

NVIDIA 环境使用 torch 2.14.0 和 ``warp-lang`` 1.18.0（``requirements/embodied/envs/supermimic.txt``）。这些依赖会升级 torch，请与其他 RLinf 环境隔离安装。``docker/Dockerfile`` 也提供 ``embodied-supermimic-image`` 目标。G1 的模型、关节顺序、执行器增益和站立关键帧都来自 mjlab。

AMD MI300X（实验性）
~~~~~~~~~~~~~~~~~~~~

AMD 路径使用 ROCm Warp。先准备 Linux x86_64、ROCm 7.2 SDK（含 ``hipcc``）和 C++ 编译工具链；可参考 `ROCm Warp 的 gfx942 Docker 构建步骤 <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_ 准备环境。在仓库根目录执行：

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # 按实际 SDK 安装路径修改
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

脚本从 AMD Radeon 的 wheel 索引安装 torch 2.9.1 和 torchvision 0.24.0（需要支持 flat 索引的新版 uv），在上面链接的提交上构建 ROCm Warp ``1.13.0+rocm.0``，再按 ``requirements/embodied/envs/supermimic_amd.txt`` 安装 mjlab 1.2.0、MuJoCo 3.8.1 和 mujoco-warp 3.8.1。``SUPERMIMIC_WARP_PATH`` 指定 Warp 源码目录（默认仓库旁的 ``../warp-rocm``），``SUPERMIMIC_HIP_ARCH`` 指定编译目标（默认 ``gfx942``）。之后的数据准备、训练和评测与下文相同。

当前 AMD 安装路径尚未经过完整训练验证。`已报告的跟踪结果`_ 来自 NVIDIA 上的 LAFAN1 检查点评测，不是 AMD 运行。

准备参考动作
------------

动作数据不放进仓库。一个片段目录里放若干 ``clip_*.pt``\ （按文件名排序加载），可以再放一个 ``manifest.json``。格式见下面的 `片段文件格式`_。

**LAFAN1（论文的 40 条）。** 片段是 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_ 的 ``g1/`` 目录。脚本固定修订版 ``ce1572906efe6157840e8474d5a0d7aa87481e74``，逐个核对 CSV 的大小和 SHA-256，并写出 ``manifest.json``。数据集卡片没有声明许可证；这些重定向数据来自 Ubisoft La Forge 的 `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_，后者使用 CC BY-NC-ND 4.0（非商业、禁止演绎）。传 ``--accept-license`` 之前请先读这些条款。

.. code-block:: bash

   source .venv-supermimic/bin/activate
   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

转换器通过 mjlab 读取 G1 的默认关节姿态 ``q0``。要在没有 mjlab 的机器上转换，先在训练环境中导出姿态，再用 ``--q0-file`` 传入：

.. code-block:: bash

   python -c "import torch; from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state; torch.save(asset_default_state()[7:36].clone(), 'g1_q0.pt')"
   python toolkits/supermimic/prepare_lafan1.py \
       --source-dir /data/lafan1_csv --out /data/lafan1_g1 \
       --q0-file ./g1_q0.pt --accept-license

将包含 ``manifest.json`` 的整个输出目录复制到训练机器。40 条片段共 2.45 小时。有清单时，加载器会拒绝 SHA-256 对不上或清单里没有的片段，所以不要改名、增删或修改准备好的文件。加载器把 30 Hz 的片段重采样到 50 Hz。

**BONES-SEED（实验性）。** 官方数据在 Hugging Face 的 `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_，G1 轨迹位于 ``g1/csv/{日期}/{动作名}.csv``。先读数据包里的 ``LICENSE.md``；脚本不下载数据。表头约定沿用 NVIDIA ProtoMotions 的 `SEED G1 CSV 说明 <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_：120 Hz；``root_translateX/Y/Z`` 单位厘米；``root_rotateX/Y/Z`` 是外旋 XYZ 欧拉角、单位度；29 个 ``<关节名>_dof`` 列单位度，名字对应 mjlab 的 G1 关节。这个约定还没有拿官方 CSV 核对过。

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_MOTION_PATH=/data/bones_g1_filtered

``--input`` 可以是单个 CSV 或递归扫描的目录。片段重采样到 50 Hz，已有输出默认不覆盖（``--overwrite`` 除外），单个损坏的 CSV 会报告原因而不影响其他文件。训练时加上 `配置`_ 里的 BONES-SEED 覆盖项。转换器只在合成 CSV 上测试过。

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

下面把 GenCoord 配置缩小到 8 个环境、3 次更新，检查训练、保存、恢复和评测流程。检查点、日志和结果放在同一个 ``RUN_DIR`` 下。这只验证流程，不验证跟踪效果。

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   export RUN_DIR=/data/runs/smoke
   SMOKE_ARGS=(env.train.total_num_envs=8 algorithm.replay_length=64
       algorithm.wm_window_batch=64 algorithm.imagine_batch=8
       runner.save_interval=1
       runner.logger.experiment_name=smoke runner.logger.log_path=$RUN_DIR)

   # 训练 3 次更新，再从第 3 次更新恢复，继续到第 5 次。
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord "${SMOKE_ARGS[@]}" runner.max_epochs=3
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord "${SMOKE_ARGS[@]}" runner.max_epochs=5 \
       runner.resume_dir=auto

   # 在训练片段上评测最后一个检查点。动作噪声关掉，和下面的已报告结果相同。
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/smoke/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt \
       env.eval.action_noise_std=0 \
       runner.eval_export.output_dir=$RUN_DIR/eval
   tensorboard --logdir $RUN_DIR/tensorboard

恢复日志应包含 ``Resuming from`` 和 ``global_step_3``。如果出现 ``resume_dir=auto found no complete checkpoint``，请检查 ``runner.logger.log_path``、``runner.logger.experiment_name`` 以及第一次运行是否成功保存。检查点在 ``$RUN_DIR/smoke/checkpoints/global_step_<N>``，TensorBoard 文件在 ``$RUN_DIR/tensorboard``，评测表格在 ``$RUN_DIR/eval``。只更新 3～5 次的 policy 通常还不能跟踪动作。

MaxCoord 把 ``--config-name`` 换成 ``supermimic_lafan1_maxcoord``。评测 MaxCoord 检查点时还要加上 ``actor.model.representation=maxcoord``。

配置
----

训练有两份 YAML，差别只有状态表示。两份评测配置继承 GenCoord，只改评测设置。环境数和批量都是全局量。

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - 文件
     - 用途
   * - ``examples/embodiment/config/supermimic_lafan1_gencoord.yaml``
     - GenCoord 训练。默认 4096 个环境、40 条片段、``runner.max_epochs=400000``、每 2000 次更新保存。`已报告的跟踪结果`_ 把更新次数覆盖成 16000。
   * - ``examples/embodiment/config/supermimic_lafan1_maxcoord.yaml``
     - 继承上一份，只把 ``actor.model.representation`` 设为 ``maxcoord``。
   * - ``evaluations/supermimic/supermimic_lafan1_eval.yaml``
     - 40 条片段的动作跟踪评测。
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - 屈膝站立姿态上的推力恢复。不是论文 §4.3 的 T-pose 协议。

种子为 0。网络权重保持 float32，矩阵乘用 TF32。``actor.fsdp_config.amp_autocast.enabled`` 默认为 ``False``；设为 ``True`` 时，只有网络里的矩阵乘走 ``precision``\ （默认 bf16），运动学、奖励、损失和优化器状态仍是 float32。这和论文的 TF32 不同。隐藏层矩阵用 Muon（``adjust_lr_fn: match_rms_adamw``），其余参数用 fused Adam。梯度范数留在设备上裁剪，优化器步不等待主机。

``env.train.motion_on_cpu=true``\ （评测用 ``env.eval.motion_on_cpu``）把较大的动作库和 FK 缓存放在主机内存，只把查询到的行传到 GPU。actor worker 为想象 rollout 另持有一份动作库。

BONES-SEED 训练在启动命令后追加：

.. code-block:: bash

   env.train.motion_format=bones_seed env.train.num_clips=null \
       env.eval.motion_format=bones_seed env.eval.num_clips=null

并把 ``SUPERMIMIC_MOTION_PATH`` 指到过滤后的目录。

常用配置键
~~~~~~~~~~

下表是调整规模和训练长度时最常改的键，默认值来自 ``supermimic_lafan1_gencoord.yaml``。都可以作为 Hydra 覆盖项传入。

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - 键
     - 默认
     - 含义
   * - ``runner.max_epochs``
     - 400000
     - 全局更新次数。已报告的运行覆盖为 16000。
   * - ``runner.save_interval``
     - 2000
     - 每隔这么多次更新，以及最后一次，写一个检查点。
   * - ``runner.val_check_interval``
     - -1
     - 训练中不自动评测。评测用 `评测`_ 里的命令单独跑。
   * - ``env.train.total_num_envs``
     - 4096
     - 所有 GPU 上的环境总数，必须能被 GPU 数整除。
   * - ``env.train.num_clips``
     - 40
     - 加载的片段数。``null`` 加载目录里的全部片段。
   * - ``algorithm.replay_ratio``
     - 16
     - 每条转移平均被世界模型抽到的次数。和下面的批量一起，默认是每个控制步 0.4 次全局更新。
   * - ``algorithm.wm_batches``
     - 10
     - 每次更新里世界模型的批次数。
   * - ``algorithm.wm_window_batch``
     - 2048
     - 每个世界模型批次的窗口数，所有 GPU 的总和。
   * - ``algorithm.imagine_batch``
     - 4096
     - 想象 rollout 的条数，所有 GPU 的总和。
   * - ``algorithm.k`` / ``algorithm.H``
     - 8 / 32
     - 世界模型窗口和想象步数。
   * - ``actor.optim.lr``
     - 2.0e-4
     - 三个阶段共用的学习率，常数调度。
   * - ``actor.optim.clip_grad``
     - 1.0
     - 全局梯度范数上限。Actor 再把每个元素夹到 ``clip_grad_value`` 0.1。
   * - ``actor.optim.fused``
     - True
     - 非 Muon 参数使用 fused Adam。世界模型和 critic 的 Adam 同样打开。
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - False
     - 打开后，网络矩阵乘使用 ``precision``\ （bf16）。权重仍是 float32。
   * - ``env.train.action_noise_std``
     - 0.1
     - 训练动作噪声。已报告的评测把它在 ``env.eval`` 上设为 0。

**更新次数与仿真步数。** 默认规模下每个控制步欠 0.4 次更新，所以 ``runner.max_epochs=16000`` 对应 40000 个控制步。全局步是更新次数，``env/env_steps`` 是控制步数。``env.train.total_num_envs``、``algorithm.wm_window_batch`` 和 ``algorithm.imagine_batch`` 是所有 GPU 的总和，必须能被 GPU 数整除。加 GPU 而不改这些值，训练的数学含义不变。

单卡训练
--------

配方在 RLinf 看到的每张 GPU 上各放一个 env、一个 rollout 和一个 actor worker（``cluster.component_placement`` 把 ``actor,env,rollout`` 映射到 ``all``）。只有一张 GPU 的机器上：

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/gencoord

MaxCoord：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_maxcoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/maxcoord

训练不需要渲染。多卡机器上若只使用其中一张，先删掉合写的放置键，再把三个 worker 指到同一张卡的序号（下面是 0 号卡）：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/gencoord \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor:0,env:0,rollout:0}'

多卡训练
--------

placement 中每张 GPU 上有一个 env、一个 rollout 和一个 actor worker，三组的大小必须相同。rank :math:`r` 的 env worker 只和 rank :math:`r` 的 rollout、actor worker 交换数据。actor worker 组成一个 FSDP2 组：参数、梯度和优化器状态在组内分片。重置分箱统计在所有 env rank 之间合并。

配置不变、只增加 GPU 时，全局批量、学习率和更新数据比都和单卡相同，每个 rank 分一份。多节点和 Slurm 尚未用这个配方验证。

检查点与恢复
------------

``runner.save_interval`` 每隔 N 次更新以及最后一次更新后写一个 ``<log_path>/<experiment_name>/checkpoints/global_step_<N>``，其中包括：

* ``actor/``：FSDP 检查点。``dcp_checkpoint/`` 存分片的网络、各阶段的优化器和学习率调度，以及每个 rank 的随机数状态；``model_state_dict/full_weights.pt`` 是评测用的完整权重；``supermimic_counters.json`` 记录各阶段的优化器步数。
* ``env/env_rank_<r>.pt``：重置分箱统计和每个 env rank 的随机数状态。
* ``runner_state.json``：欠账更新的小数部分和控制步计数。它最后写出，所以没有它的检查点是不完整的。

恢复时用与原运行相同的配置名、``log_path`` 和 ``experiment_name``：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.max_epochs=16000 \
       runner.logger.log_path=/data/runs/gencoord \
       runner.resume_dir=auto

``runner.resume_dir=auto`` 选择最新的完整检查点。``auto`` 找不到时不会报错，而是记一条日志后从头训练，所以请确认日志里有 ``Resuming from .../global_step_<N>``。GenCoord 和 MaxCoord 的权重形状不同，不能交叉加载。replay 和仿真状态不保存，恢复后要先重新填满 replay。换 GPU 数恢复还没有测试过。

评测
----

评测走 ``evaluations/eval_embodied_agent.py``。``evaluations/run_eval.sh`` 看到 ``supermimic_`` 开头的配置名，就读取 ``evaluations/supermimic/<配置名>.yaml``。``runner.ckpt_path`` 是检查点里的 ``full_weights.pt``。

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=/data/runs/gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
       env.eval.action_noise_std=0 \
       runner.eval_export.output_dir=/data/runs/gencoord/eval_2000

MaxCoord 加上 ``actor.model.representation=maxcoord``，并换上对应的检查点路径。把上面命令里的 ``global_step_2000`` 和输出目录换成 ``global_step_4000``，就是第 4000 次更新的评测。`已报告的跟踪结果`_ 就是这四次：GenCoord 和 MaxCoord，各评第 2000 次和第 4000 次更新。训练用的是 `单卡训练`_ 里 ``runner.max_epochs=16000`` 的两条命令。``env.eval.eval_tasks`` 只能是 ``[tracking]`` 或 ``[push_recovery]``。40 条 LAFAN1 片段由 ``env.eval.total_num_envs=40`` 一次放完；``max_steps_per_rollout_epoch=13656`` 覆盖最长片段（30 fps 下 8194 帧，重采样到 50 Hz）。

**输出。** ``runner.eval_export.enabled`` 在评测配置里默认打开。结果写到 ``runner.eval_export.output_dir``，不设置时写到 ``<log_path>/<experiment_name>/eval_results/``。目录里有 ``summary.json``、``per_clip.csv`` 和这次评测的 ``config.yaml``。同样的指标以 ``eval/`` 为前缀写进日志。

**动作跟踪** 把每条片段从第 0 帧播放一次。片段 :math:`i` 的归一化回报是实际奖励之和除以 :math:`0.1 \times T_i`，其中 :math:`T_i` 是片段在 50 Hz 上的帧数，0.1 是单步奖励上限。提前失败时没播放的步仍留在分母里。``normalized_return`` 是 40 条片段的不加权平均。``fail_rate`` 是摔倒片段的比例，``clip_end_rate`` 是播到片段末帧的比例。``fail_step`` 和 ``fail_time_s`` 只对摔倒的片段取平均。

**站立姿态推力恢复** 使用 ``supermimic_push_recovery``。它加载跟踪检查点，以 mjlab 的屈膝站立姿态作为静止参考，在 1 秒后把根的水平速度改成 0.5、1、2 或 4 m/s，每种 4 次。跌倒沿用训练的失败判据。论文 §4.3 写的是 T-pose，这里没有实现那个姿态。MaxCoord 检查点同样要加 ``actor.model.representation=maxcoord``。

可视化与结果
------------

日志走 RLinf 的 ``MetricLogger``。配方把 TensorBoard 事件文件写在 ``<log_path>/tensorboard/``。训练指标在各 actor rank 之间取平均。常用的键：

* ``env/env_steps``、``env/sim/reward_mean``、``env/sim/terminal_rate``。
* ``train/world_model/*_loss`` 和 ``train/world_model/grad_norm``：世界模型各通道的误差，以及该阶段各优化器步的梯度范数平均。
* ``train/actor/reward``、``train/actor/grad_norm``。
* ``train/critic/loss``、``train/critic/value_mean``、``train/critic/target_mean``。
* ``train/imagination/survival``、``train/imagination/termination_prob``、``train/imagination/step_reward``。
* ``train/world_model_optimizer_steps``、``train/actor_optimizer_steps``、``train/critic_optimizer_steps``。
* ``time/*``：runner 和各 worker 的主机计时。GPU 异步执行，某一阶段的耗时可能包含前一阶段排队的 kernel。

一批更新结束、指标拷回主机时，如果其中有非有限值，训练会停下。这一步不在每个优化器步上同步。

已报告的跟踪结果
----------------

下面两次运行都用 `单卡训练`_ 的命令，``runner.max_epochs=16000``，40 条 LAFAN1 片段、4096 个环境，其余取配置默认值。表中的数字是检查点评测，不是训练日志里的回报：每条片段从第 0 帧播放，``env.eval.action_noise_std=0``。训练目标是 16000 次更新，这里只评了第 2000 次和第 4000 次的检查点。一次这样的评测大约 6.5 分钟。

.. list-table::
   :header-rows: 1
   :widths: 22 20 18 20 20

   * - 表示
     - 更新
     - ``normalized_return``
     - ``fail_rate``
     - ``clip_end_rate``
   * - GenCoord
     - 2000
     - 0.704
     - 0.15
     - 0.85
   * - MaxCoord
     - 2000
     - 0.815
     - 0.025
     - 0.975
   * - GenCoord
     - 4000
     - 0.790
     - 0.125
     - 0.875
   * - MaxCoord
     - 4000
     - 0.820
     - 0.075
     - 0.925

40 条片段上，15% 是 6 条摔倒，2.5% 是 1 条，12.5% 是 5 条，7.5% 是 3 条。论文在这 40 条上报告的归一化回报是 0.82。MaxCoord 在第 4000 次更新达到 0.820。这不是那次论文运行的复现：更新还没到 16000，重置箱是 0.2 秒而不是论文的 1 秒，论文的主结果用的也是过滤后的 BONES-SEED，不是这 40 条。

第 4000 次更新的其余跟踪误差（40 条的平均；``fail_step`` 和 ``fail_time_s`` 只平均摔倒的片段）：

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - 指标
     - GenCoord
     - MaxCoord
   * - ``body_pos_err``
     - 0.029
     - 0.024
   * - ``torso_pos_err``
     - 1.740
     - 1.245
   * - ``torso_ori_err``
     - 0.112
     - 0.092
   * - ``free_root_pos_err``
     - 1.740
     - 1.245
   * - ``free_root_ori_err``
     - 0.066
     - 0.054
   * - ``free_root_xy_drift``
     - 3.639
     - 2.551
   * - ``free_root_yaw_drift``
     - 1.045
     - 1.070
   * - ``episode_length_s``
     - 211.9
     - 215.2
   * - ``fail_step``
     - 6096.2
     - 6060.0
   * - ``fail_time_s``
     - 121.9
     - 121.2
   * - ``steps``
     - 1.06e4
     - 1.08e4

范围
----

数值检查在 ``tests/unit_tests/test_supermimic.py``。CI 配置 ``tests/e2e_tests/embodied/g1_mimic_supermimic.yaml``、``g1_mimic_supermimic_maxcoord.yaml`` 和 ``tests/e2e_tests/evaluations/g1_mimic_supermimic_eval.yaml`` 覆盖缩小后的训练、恢复和跟踪评测。上面的表是 40 条 LAFAN1 上第 2000 次和第 4000 次更新的检查点评测。16000 次更新的终点、多节点、Slurm、真实 BONES-SEED 训练，以及论文 §4.3 的 T-pose 恢复，都还没有这些数字。
