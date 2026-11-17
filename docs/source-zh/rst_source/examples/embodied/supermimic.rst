SuperMimic：用可微世界模型训练 G1 动作跟踪
==========================================

本指南在 MuJoCo Warp 里用 SuperMimic 训练并评测 Unitree G1 的全身动作跟踪 policy：安装环境、准备参考动作、选择 GenCoord 或 MaxCoord、在一张或多张 GPU 上训练、恢复训练、评测。算法以及和论文的差别见 :doc:`../../reference/algorithms/supermimic`。第一次使用可以直接照 `运行`_ 一节走一遍。LAFAN1 以外的动作来源和片段文件格式见 :doc:`supermimic_motion`，评测指标和推力恢复见 :doc:`supermimic_eval`。

.. toctree::
   :hidden:

   动作数据 <supermimic_motion>
   评测 <supermimic_eval>

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
* ``SuperMimicRolloutWorker`` 持有 policy 的副本，返回动作。
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

NVIDIA 环境使用 torch 2.14.0。torch 是 CUDA 12 构建时，脚本固定安装 ``warp-lang`` 1.18.0+cu12；否则 ``warp-lang`` 按 ``requirements/embodied/envs/supermimic.txt`` 里的版本范围解析。这些依赖会升级 torch，请与其他 RLinf 环境隔离安装。``docker/Dockerfile`` 也提供 ``embodied-supermimic-image`` 目标。G1 的模型、关节顺序、执行器增益和站立关键帧都来自 mjlab。

AMD MI300X（实验性）
~~~~~~~~~~~~~~~~~~~~

AMD 路径使用 ROCm Warp。先准备 Linux x86_64、ROCm 7.2 SDK（含 ``hipcc``）和 C++ 编译工具链；可参考 `ROCm Warp 的 gfx942 Docker 构建步骤 <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_ 准备环境。在仓库根目录执行：

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # 按实际 SDK 安装路径修改
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

脚本从 AMD Radeon 的 wheel 索引安装 torch 2.9.1 和 torchvision 0.24.0（需要支持 flat 索引的新版 uv），在上面链接的提交上构建 ROCm Warp ``1.13.0+rocm.0``，再按 ``requirements/embodied/envs/supermimic_amd.txt`` 安装 mjlab 1.2.0、MuJoCo 3.8.1 和 mujoco-warp 3.8.1。``SUPERMIMIC_WARP_PATH`` 指定 Warp 源码目录（默认仓库旁的 ``../warp-rocm``），``SUPERMIMIC_HIP_ARCH`` 指定编译目标（默认 ``gfx942``）。之后的数据准备、训练和评测与下文相同。

当前 AMD 安装路径尚未经过完整训练验证。`已报告的跟踪结果`_ 来自 NVIDIA A800 上的 LAFAN1 检查点评测，不是 AMD 运行。

准备参考动作
------------

动作数据不放进仓库。本指南使用论文的 40 条 LAFAN1 片段；BONES-SEED 和自备片段见 :doc:`supermimic_motion`。

片段是 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_ 的 ``g1/`` 目录。脚本固定修订版 ``ce1572906efe6157840e8474d5a0d7aa87481e74``，逐个核对 CSV 的大小和 SHA-256，并写出 ``manifest.json``。数据集卡片没有声明许可证；这些重定向数据来自 Ubisoft La Forge 的 `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_，后者使用 CC BY-NC-ND 4.0（非商业、禁止演绎）。传 ``--accept-license`` 之前请先读这些条款。

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

   # 在训练片段上评测最后一个检查点。
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/smoke/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt \
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
     - GenCoord 训练。默认 4096 个环境、40 条片段、``runner.max_epochs=400000``、每 2000 次更新保存。`已报告的跟踪结果`_ 训练 6000 次更新。
   * - ``examples/embodiment/config/supermimic_lafan1_maxcoord.yaml``
     - 继承上一份，只把 ``actor.model.representation`` 设为 ``maxcoord``。
   * - ``evaluations/supermimic/supermimic_lafan1_eval.yaml``
     - 40 条片段的动作跟踪评测。
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - 屈膝站立姿态上的推力恢复。不是论文 §4.3 的 T-pose 协议。

种子为 0。网络权重保持 float32，矩阵乘用 TF32。``actor.fsdp_config.amp_autocast.enabled`` 默认为 ``False``；设为 ``True`` 时，只有网络里的矩阵乘走 ``precision``\ （默认 bf16），运动学、奖励、损失和优化器状态仍是 float32。这和论文的 TF32 不同。隐藏层矩阵用 Muon（``adjust_lr_fn: match_rms_adamw``），其余参数用 fused Adam。梯度范数留在设备上裁剪，优化器步不等待主机。

LAFAN1 可以用 ``env.train.motion_on_cpu=true``\ （评测用 ``env.eval.motion_on_cpu``）把较大的动作库和 FK 缓存放在主机内存，只把查询到的行传到 GPU。actor worker 为想象 rollout 另持有一份动作库。

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
     - 全局更新次数。已报告的运行设为 6000。
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
   * - ``actor.cuda_graph``
     - False
     - 打开后，前三次之后的每次更新作为一张 CUDA 图回放。见 `CUDA 图更新`_。
   * - ``env.train.action_noise_std``
     - 0.1
     - 训练动作噪声。评测直接执行策略动作。

**更新次数与仿真步数。** 默认规模下每个控制步欠 0.4 次更新，所以 ``runner.max_epochs=16000`` 对应 40000 个控制步。全局步是更新次数，``env/env_steps`` 是控制步数。``env.train.total_num_envs``、``algorithm.wm_window_batch`` 和 ``algorithm.imagine_batch`` 是所有 GPU 的总和，必须能被 GPU 数整除。加 GPU 而不改这些值，训练的数学含义不变。

CUDA 图更新
~~~~~~~~~~~

``actor.cuda_graph=True`` 时，前三次更新按普通方式执行，第四次录成一张 CUDA 图，之后每次更新都回放这张图。回放省掉了世界模型、actor 和 critic 三个阶段里大部分的 kernel 启动和主机开销。回放数据仍在图外采样，每次更新用的都是新样本。

- 图里固定了录制时的学习率。配置校验要求每个优化器都是 ``lr_scheduler: constant`` 且没有 warmup。
- Adam 以 ``capturable=True`` 构建，配置里由 ``actor.cuda_graph`` 决定。
- 各阶段计时器只记录录制那一次，回放的整次更新由 ``run_updates`` 计时。
- 加载检查点会丢弃这张图，之后的更新先按普通方式执行，再重新录制。
- 单卡单元测试检查图更新和普通更新结果完全一致，float32 和 bf16 都测。双卡上的图由 `已报告的跟踪结果`_ 里的完整训练检验，没有做逐位比对。

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

placement 中每张 GPU 上有一个 env、一个 rollout 和一个 actor worker，三组的大小必须相同。rank :math:`r` 的 env worker 只和 rank :math:`r` 的 rollout、actor worker 交换数据。actor worker 组成一个 FSDP2 组：参数、梯度和优化器状态在组内分片。世界模型、actor 和 critic 各是一个 FSDP 单元，每个训练阶段只聚合它要读的网络。重置分箱统计在所有 env rank 之间合并。

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

用下面的命令在 40 条 LAFAN1 片段上评测检查点里的 ``full_weights.pt``：

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export MUJOCO_GL=disabled
   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=/data/runs/gencoord/supermimic_lafan1_gencoord/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/gencoord/eval_2000

MaxCoord 检查点加上 ``actor.model.representation=maxcoord``。评测指标、输出文件和站立推力恢复见 :doc:`supermimic_eval`。

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

GenCoord，40 条 LAFAN1 片段、4096 个环境，其余取配置默认值，用 `单卡训练`_ 里的 GenCoord 命令加 ``runner.max_epochs=6000`` 训练 6000 次更新（约 15000 个控制步）。检查点按 `评测`_ 的方式评测：每条片段从第 0 帧播放，直接执行策略动作。所有运行都用 NVIDIA A800；官方一行是官方 SuperMimic 代码在同一批机器上的结果。RLinf 的运行都设了 ``actor.cuda_graph=True``，bf16 的几行另设 ``actor.fsdp_config.amp_autocast.enabled=True``。

.. list-table::
   :header-rows: 1
   :widths: 14 8 12 12 20 14 20

   * - 运行
     - GPU 数
     - 矩阵乘
     - 秒/次更新
     - ``normalized_return``
     - ``fail_rate``
     - ``body_pos_err``
   * - 官方
     - 1
     - TF32
     - 0.515
     - 0.830
     - 0.025
     - 0.027
   * - RLinf
     - 1
     - TF32
     - 0.603
     - 0.846
     - 0
     - 0.024
   * - RLinf
     - 1
     - bf16
     - 0.515
     - 0.857
     - 0
     - 0.023
   * - RLinf
     - 2
     - TF32
     - 0.486
     - 0.843
     - 0.025
     - 0.023
   * - RLinf
     - 2
     - bf16
     - 0.449
     - 0.833
     - 0.05
     - 0.023

* RLinf 的秒/次更新是每次更新 ``time/step`` 计时的均值：单独占卡另跑一次，预热 200 次后取 1000 次更新。它包含采集、更新和权重同步，不含打日志；也包含 rollout 一侧每分钟一次、约 2 秒的停顿，约占 4% 的时间。官方的时间是 MaxCoord 运行两个存档点之间的墙钟，作为计时参考；官方其余几列是 GenCoord。
* 矩阵乘用 TF32 时，双卡每次更新 0.486 秒，单卡 0.603 秒，约少 19%。四卡是 0.421 秒，bf16 时 0.407 秒。
* ``fail_rate`` 0.025 是 40 条里摔倒 1 条，0.05 是 2 条。单卡 RLinf 的两个检查点 40 条全部播放到最后一帧。
* 每行是一个种子。较早的检查点在不同运行之间差得更多：四次 RLinf 运行的 ``normalized_return`` 在第 2000 次更新时为 0.672–0.732，第 4000 次时为 0.765–0.794。
* 论文在这 40 条上报告的归一化回报是 0.82。一次评测约 7 分钟。

范围
----

数值检查在 ``tests/unit_tests/test_supermimic.py``。CI 配置 ``tests/e2e_tests/embodied/g1_mimic_supermimic.yaml``、``g1_mimic_supermimic_maxcoord.yaml`` 和 ``tests/e2e_tests/evaluations/g1_mimic_supermimic_eval.yaml`` 覆盖缩小后的训练、恢复和跟踪评测。上面的表是 GenCoord 第 6000 次更新在 40 条 LAFAN1 上的评测。MaxCoord 的精度、超过 6000 次更新的运行、多节点、Slurm、真实 BONES-SEED 训练，以及论文 §4.3 的 T-pose 恢复，都还没有这些数字。
