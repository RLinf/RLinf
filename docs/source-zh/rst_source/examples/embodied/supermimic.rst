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

      每张 GPU 上一组 env、rollout 和 FSDP actor worker；Ray 多节点（实验性）

| **你将完成：** 安装独立环境 -> 准备 LAFAN1 片段 -> 小规模训练、恢复、评测 -> 使用主配置或论文取向覆盖项训练。
| **前置条件：** NVIDIA GPU，或实验性的 AMD MI300X 路径（见 `安装`_）；RLinf 仓库（:doc:`安装说明 <../../start/installation>`）；LAFAN1 G1 片段（见 `准备参考动作`_）。

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 项目
     - 内容
   * - 仿真器
     - 通过 `mjlab <https://github.com/mujocolab/mjlab>`_\ （NVIDIA 固定 commit ``8bb2bc9``，AMD 使用 1.2.0）使用 MuJoCo Warp，G1 29 自由度，控制频率 50 Hz，每个控制步 4 个 5 ms 物理步。
   * - 任务
     - 跟踪重定向到 G1 的参考动作片段（LAFAN1 或 BONES-SEED）；可以再加一个编码器跟踪速度命令。
   * - 观测
     - 最近 10 帧的关节角（相对默认姿态）、关节速度、基座角速度、投影重力和上一动作，带传感器噪声。主配置每帧 99 维，含 6 维参考朝向；论文覆盖项使用 93 维。
   * - 动作
     - 29 个关节设定点 :math:`q_{sp} = \alpha a + q_0`，由低增益 PD 跟随。
   * - Goal
     - 主配置使用 730 维 Goal（10 帧、每帧 73 维）；论文覆盖项使用 680 维，其他接口见 `配置`_。
   * - 入口
     - 训练用 RLinf 的 ``examples/embodiment/train_embodied_agent.py``，评测用 ``evaluations/run_eval.sh supermimic_*``\ （内部调用 ``evaluations/eval_embodied_agent.py``）。

SuperMimic 工作原理
-------------------

SuperMimic 反复做三件事：在 MuJoCo Warp 里让所有环境走一个控制步，把转移写进 replay；按更新数据比从 replay 取 ``k`` 步窗口，拟合世界模型的动力学和终止头；从 replay 里的状态出发，在世界模型中展开 32 步想象 rollout，让回报沿着可微的世界模型反向传播来更新 actor，critic 拟合 GAE 目标。环境结束后，按动作时间轴分箱的回报决定从哪里重新开始，回报低的段落被抽到得更多。公式、网络结构以及与论文和社区实现的差异见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

在 RLinf 里，这三件事分给三组 worker，每张 GPU 上各有一个：

* ``SuperMimicEnvWorker`` 持有这张卡分到的环境、重置采样器和奖励计算。
* ``SuperMimicRolloutWorker`` 持有 policy 和动作 critic 的副本，返回动作，以及超时结束的回合在重置采样回报里用来自举的 critic 值。
* ``EmbodiedSuperMimicFSDPPolicy`` 是 FSDP2 actor，持有 replay、世界模型、actor、各个 critic 和它们的优化器。

``SuperMimicRunner`` 是控制面：它算出下一次更新到期前要采多少个控制步，让三组 worker 采这些步，在 actor 上执行到期的更新，把新权重同步给 rollout worker，然后继续采集；日志、评测和检查点也由它负责。在一个控制步里，env worker 先把观测发给同一 rank 的 rollout worker 并收回动作，再把转移发给 actor worker，最后和 rollout worker 交换自举特征和价值，然后重置刚结束的回合。

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

当前 AMD 安装路径尚未经过完整训练验证。第三方 MI300X 结果见 `已报告的跟踪结果`_；其他 AMD 型号尚未验证。

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

将包含 ``manifest.json`` 的整个输出目录复制到训练机器。40 条片段共 2.45 小时。有清单时，加载器会拒绝 SHA-256 对不上或清单里没有的片段，所以不要改名、增删或修改准备好的文件。

**BONES-SEED（实验性）。** 官方数据在 Hugging Face 的 `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_，G1 轨迹位于 ``g1/csv/{日期}/{动作名}.csv``。先读数据包里的 ``LICENSE.md``；脚本不下载数据。表头约定沿用 NVIDIA ProtoMotions 的 `SEED G1 CSV 说明 <https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst>`_：120 Hz；``root_translateX/Y/Z`` 单位厘米；``root_rotateX/Y/Z`` 是外旋 XYZ 欧拉角、单位度；29 个 ``<关节名>_dof`` 列单位度，名字对应 mjlab 的 G1 关节。这个约定还没有拿官方 CSV 核对过。

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1
   python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \
       --out /data/bones_g1_filtered
   export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered

``--input`` 可以是单个 CSV 或递归扫描的目录。片段重采样到 50 Hz，已有输出默认不覆盖（``--overwrite`` 除外），单个损坏的 CSV 会报告原因而不影响其他文件。片段的 ``name`` 是相对 ``--input`` 的路径，供附录 E 的名字过滤使用；``bones_feasibility.py`` 写入加载器过滤所需的逐帧几何。训练时把 `配置`_ 里的 BONES-SEED 覆盖项加到 `运行`_ 的命令上；所有通过过滤的片段都会加载，日志会打印每个过滤器去掉了多少。转换器只在合成 CSV 上测试过。

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

下面把主配置缩小到 8 个环境、3 次更新，检查训练、保存、恢复和评测流程。网络接口与完整的 ``supermimic_lafan1_goal730`` 训练相同，检查点、日志和结果放在同一个 ``RUN_DIR`` 下。这只验证流程，不验证跟踪效果。

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/smoke
   SMOKE_ARGS=(env.train.total_num_envs=8 algorithm.replay_length=64
       algorithm.wm_only_updates=0 algorithm.wm_window_batch=64
       algorithm.imagine_batch=8 runner.save_interval=1
       runner.logger.experiment_name=smoke runner.logger.log_path=$RUN_DIR)

   # 训练 3 次更新，再从第 3 次更新恢复，继续到第 5 次。
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${SMOKE_ARGS[@]}" runner.max_epochs=3
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${SMOKE_ARGS[@]}" runner.max_epochs=5 \
       runner.resume_dir=auto

   # 在 40 条训练片段上评测最后一个检查点。
   bash evaluations/run_eval.sh supermimic_lafan1_goal730_eval \
       runner.ckpt_path=$RUN_DIR/smoke/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval
   tensorboard --logdir $RUN_DIR/tensorboard

恢复日志应包含 ``Resuming from`` 和 ``global_step_3``。如果出现 ``resume_dir=auto found no complete checkpoint``，请检查 ``runner.logger.log_path``、``runner.logger.experiment_name`` 以及第一次运行是否成功保存。检查点在 ``$RUN_DIR/smoke/checkpoints/global_step_<N>``，TensorBoard 文件在 ``$RUN_DIR/tensorboard``，评测表格在 ``$RUN_DIR/eval``。只更新 3～5 次的 policy 通常还不能跟踪动作。

配置
----

主流程保留三份 YAML。训练配置包含下文所报 16000 次 LAFAN1 更新的完整参数；两份评测配置继承它，只改评测设置。环境数和批量都是全局量。

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 文件
     - 用途
   * - ``examples/embodiment/config/supermimic_lafan1_goal730.yaml``
     - 训练：730 维 Goal、99 维观测、前 500 次只训练世界模型、64 步预热、16000 次更新，每 2000 次保存。
   * - ``evaluations/supermimic/supermimic_lafan1_goal730_eval.yaml``
     - 这套配方的动作跟踪评测；速度评测用覆盖项选择（见 `评测`_）。
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - 论文接口（680 维 Goal、93 维观测）policy 的站姿抗扰评测；其他接口见 `评测`_。

训练配置组合了三个 RLinf 配置组：``env/g1_mimic.yaml`` 用于 ``env.train`` 和 ``env.eval``\ （动作数据、仿真和各评测协议），``model/supermimic.yaml`` 用于 ``actor.model``\ （policy 接口，默认值是论文的选择），``hybrid_engines/fsdp.yaml`` 用于 ``actor.fsdp_config``。在此之上它选择 goal730 接口：``actor.model.goal_encoding: goal730``、``obs_dim: 99``、``critic_features: body_error``、``input_norm: running``、``wm_state_loss: running``，以及 ``algorithm.reward_anchor: pelvis``、``residual_frame: body``、``termination_head: analytic``。种子为 0。小规模运行和多卡训练通过参数覆盖，不再单独提供 YAML。

要使用原来的论文取向设置，请先在启动训练或评测的 Bash 终端中定义下面的数组。它同时切换 Goal、观测、归一化和 critic；评测对应检查点时，保留同一个数组。

.. code-block:: bash

   PAPER_ARGS=(
       runner.max_epochs=320000 algorithm.warmup_sim_steps=40
       algorithm.wm_only_updates=0 algorithm.termination_head=learned
       algorithm.reward_anchor=torso algorithm.residual_frame=heading
       actor.model.goal_encoding=paper actor.model.obs_dim=93
       actor.model.input_norm=none actor.model.wm_state_loss=paper
       actor.model.critic_features=root4
       actor.fsdp_config.amp_autocast.enabled=false
       env.train.com_sampling=per_reset runner.logger.experiment_name=paper
   )
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${PAPER_ARGS[@]}" \
       runner.logger.log_path=/data/runs/paper

下面的变体在 ``"${PAPER_ARGS[@]}"`` 后追加覆盖项。每次运行请设置不同的 ``runner.logger.experiment_name``。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 变体
     - 追加参数
   * - 躯干位姿锚点（689 维 Goal）
     - ``actor.model.goal_anchor=true``
   * - 730 维 Goal、93 维观测
     - ``actor.model.goal_encoding=goal730``
   * - 99 维观测、680 维 Goal
     - ``actor.model.obs_dim=99``
   * - 动作和速度训练
     - ``actor.model.velocity_tracking=true``
   * - BONES-SEED
     - ``env.train.motion_format=bones_seed env.train.num_clips=null env.train.motion_path=$SUPERMIMIC_BONES_PATH``

锚点和速度跟踪都要求论文 Goal 和 93 维观测，其他组合会被配置检查拒绝。例如，训练速度编码器后，用同样的接口跑附录 G.2 阶跃测试：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 "${PAPER_ARGS[@]}" \
       actor.model.velocity_tracking=true runner.logger.experiment_name=velocity \
       runner.logger.log_path=/data/runs/velocity
   bash evaluations/run_eval.sh supermimic_lafan1_goal730_eval "${PAPER_ARGS[@]}" \
       actor.model.velocity_tracking=true 'env.eval.eval_tasks=[velocity_g2]' \
       env.eval.total_num_envs=2688 env.eval.max_steps_per_rollout_epoch=250 \
       runner.ckpt_path=/data/runs/velocity/velocity/checkpoints/global_step_320000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/velocity/eval

下面四项决定世界模型和奖励的定义。goal730 配方设为 ``pelvis``、``running``、``body``、``running``；论文覆盖项选择 ``torso``、``none``、``heading``、``paper``。

* ``algorithm.reward_anchor``：``pelvis`` 是骨盆，``torso`` 是 ``torso_link``。终止身体仍由 ``algorithm.termination_body`` 决定。
* ``actor.model.input_norm``：``running`` 使用滑动统计量归一化模型输入；``none`` 保持原始数值。
* ``algorithm.residual_frame``：``body`` 或 ``heading`` 选择预测状态残差的坐标系，各分量的定义见 :doc:`算法参考 <../../reference/algorithms/supermimic>`。
* ``actor.model.wm_state_loss``：``running`` 按残差滑动统计量缩放状态误差，``scaled`` 使用固定尺度，``paper`` 使用不加权平方误差。

检查点把这几项和其余 policy 接口记在 ``supermimic_interface.json`` 里；用其中任何一项不同的配置恢复或评测，都会被拒绝。

``env.train.motion_on_cpu=true``\ （评测用 ``env.eval.motion_on_cpu``）把较大的动作库和 FK 缓存放在主机内存，只把查询到的行传到 GPU；BONES-SEED 动作库总是这样存放。actor worker 为想象 rollout 另持有一份动作库。

``env.train.com_sampling`` 决定 :math:`\xi` 中躯干质心偏移的采样时机：``per_env`` 每个环境只采一次（zita-ch 的做法，配方的设置），``per_reset`` 每次重置都重采（论文的做法，也是 ``env/g1_mimic.yaml`` 的默认值，需要重算仿真常量）。

``actor.fsdp_config.amp_autocast`` 让网络前向在 bfloat16 autocast 下运行；积分、运动学、损失和 Adam 权重仍是 float32，并打开 TF32 矩阵乘。actor 只支持 FSDP2 和 float32 权重，配置校验会拒绝其他 ``precision`` 或 ``strategy``、FSDP ``mixed_precision``、grad scaler 和 CPU offload。

常用配置键
~~~~~~~~~~

下表是调整规模和训练长度时最常改的键，默认值来自 ``supermimic_lafan1_goal730.yaml``，上面的论文覆盖项会改变其中一部分。都可以作为 Hydra 覆盖项传入。

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
   * - ``runner.max_epochs``
     - 16000
     - 训练的\ **更新次数**\ ，不是仿真步数（见表后说明）。
   * - ``algorithm.updates_per_env_step``
     - 0.4
     - 每个控制步累计的全局更新次数（更新数据比）。调大则每份数据被训练得更多。
   * - ``actor.optim.lr``
     - 2.0e-4
     - 动作 actor 的学习率。世界模型、critic 和速度相关的优化器（``actor.world_model_optim``、``actor.critic_optim``、``actor.velocity_actor_optim``、``actor.velocity_critic_optim``）用同一个值。
   * - ``actor.optim.total_training_steps``、``actor.optim.min_lr_rate``
     - 250000、0.03
     - 每个优化器的调度在前 250000 步内按余弦降到 0.03 × ``lr``，之后保持。某个阶段参与训练的那次更新，它的调度才前进一步。
   * - ``actor.optim.clip_grad``、``actor.optim.clip_grad_value``
     - 1.0、0.1
     - 每个阶段都把梯度按全局范数裁到 1.0；actor 阶段再逐元素截断到 0.1。
   * - ``algorithm.max_nonfinite_skips``
     - 20
     - 梯度范数非有限的优化器步会被跳过；连续超过这个次数就停止训练。
   * - ``actor.optim.fused``
     - ``True``
     - 各阶段在 CUDA 上使用 fused Adam。梯度范数非有限的步在 kernel 内跳过，读取指标时再计数，优化器步不等待主机。设为 ``False`` 时使用 PyTorch 默认的 Adam，并在主机端逐步检查；没有可用的 CUDA 设备时也是如此，并先记录一条警告。
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
     - 64
     - 每次启动或恢复后、第一次更新前先采集的控制步数，用来预热 replay；这些步不累计更新。
   * - ``actor.fsdp_config.amp_autocast.enabled``
     - ``true``\ （bf16）
     - 网络前向使用 bfloat16 autocast，见上文。
   * - ``actor.model.torch_compile``
     - ``True``
     - 在大批量 CUDA 数据上编译训练扫描、环境中的奖励和特征片段以及 FK；设为 ``False`` 时都以 eager 方式运行。
   * - ``rollout.gc_after_weight_sync``
     - ``False``
     - 设为 ``True`` 时恢复 RLinf 默认的权重同步后 ``gc.collect()`` 和清缓存；每次更新都同步时开销较大。
   * - ``actor.use_profiler``
     - ``False``
     - 用 RLinf 的 ``PyTorchProfiler`` 采集 actor 更新，设置在 ``actor.profiler``，与 Megatron 配置相同。按默认值，每次启动的第 0-2 次更新预热，第 3 次更新按 rank 写到 ``<output_dir>/tensorboard/fwd_bwd``，训练阶段带标注，可用 Perfetto 打开。前 ``algorithm.wm_only_updates`` 次更新没有 actor 阶段。
   * - ``env.train.total_num_envs``
     - 4096
     - 并行仿真环境数，全局量。采集吞吐和显存都随它增长。
   * - ``algorithm.replay_length``
     - 1000
     - 每个环境的 replay 容量，按控制步计（50 Hz 下 20 s），至少为 10。replay 显存随 ``total_num_envs × replay_length`` 增长。
   * - ``env.train.num_clips``
     - 40
     - 按文件名顺序加载前 N 条片段；``null`` 加载全部。
   * - ``runner.save_interval``
     - 2000
     - 每隔 N 次更新以及最后一次更新后保存检查点；设为 -1 时不保存。
   * - ``runner.log_interval``
     - 100
     - 每隔 N 次更新记录指标并打印 RLinf 的指标表。
   * - ``runner.logger.logger_backends``
     - ``[tensorboard]``
     - RLinf 的日志后端：TensorBoard、WandB 或 SwanLab。
   * - ``runner.val_check_interval``
     - -1
     - 训练中每隔 N 次更新评测一次；默认不评测。
   * - ``runner.logger.log_path``
     - ``../results``
     - 检查点和日志的根目录，相对启动目录。正式训练请显式设置。

**更新次数与仿真步数。** 每个控制步（所有环境各走一步）累计 ``updates_per_env_step`` 次欠下的更新。runner 一直采集到有更新到期，执行所有整数次的欠账更新，同步新权重后继续采集。所以 ``runner.max_epochs=16000`` 对应 16000 / 0.4 = 40,000 个控制步（另加预热）。全局步是更新次数，``env/env_steps`` 是控制步数。

**全局量。** ``env.train.total_num_envs``、``algorithm.wm_window_batch`` 和 ``algorithm.imagine_batch`` 是所有 GPU 的总和，必须能被 GPU 数整除；``updates_per_env_step`` 和 ``runner.max_epochs`` 数的也是全局更新。所以加 GPU 而不改这些值，训练的数学含义不变，见 `多卡训练`_。

单卡训练
--------

配方在 RLinf 看到的每张 GPU 上各放一个 env、一个 rollout 和一个 actor worker（``cluster.component_placement`` 把 ``actor,env,rollout`` 映射到 ``all``）。只有一张 GPU 的机器上，这就是全部放置：

.. code-block:: bash

   export PYTHONPATH=$PWD:$PYTHONPATH
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000

RLinf 的通用启动脚本按配方原样运行，日志写到新建的 ``logs/<时间>-<配置名>`` 目录；它不转发 Hydra 覆盖项，所以改配置项或恢复训练请直接调用 ``train_embodied_agent.py``。训练不需要渲染，没有 EGL 时用 ``MUJOCO_GL=disabled``：

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   MUJOCO_GL=disabled bash examples/embodiment/run_embodiment.sh supermimic_lafan1_goal730

要在多卡机器上只用其中一张，就按 worker 组分别写放置。Hydra 不能原地覆盖配方里合写的 ``actor,env,rollout`` 键，所以第一个覆盖项先删掉它：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       '~cluster.component_placement' \
       '+cluster.component_placement={actor: 2, env: 2, rollout: 2}' \
       runner.logger.log_path=/data/runs/lafan1

RLinf 按 placement 设置每个 worker 的设备可见性，所以节点 GPU 2 上的 worker 显示为 ``cuda:0``，启动前设置的 ``CUDA_VISIBLE_DEVICES`` 也可能被替换。上面的命令设置了 ``runner.logger.log_path``，因为配方里的 ``../results`` 相对启动目录；还设置了比配方的 2000 次更新更短的 ``runner.save_interval``。

多卡训练
--------

placement 中每张 GPU 上有一个 env、一个 rollout 和一个 actor worker，三组的大小必须相同。rank :math:`r` 的 env worker 只和 rank :math:`r` 的 rollout、actor worker 交换数据。actor worker 组成一个 FSDP2 组：参数、梯度和优化器状态在组内分片，每个 rank 按自己在全局行数中的份额加权损失。重置分箱统计和输入归一化统计在所有 env rank 之间合并。细节见 :doc:`算法参考 <../../reference/algorithms/supermimic>`。

多用几张 GPU 有两种方式：

* **固定全局批量。** 配置不变，只加 GPU。全局批量、学习率和更新数据比都和单卡相同，每个 rank 分一份；相当于把单卡训练摊到多张卡上。能看到四张卡时，默认的 ``all`` 放置就是这样；否则：

  .. code-block:: bash

     python examples/embodiment/train_embodied_agent.py \
         --config-name supermimic_lafan1_goal730 \
         '~cluster.component_placement' \
         '+cluster.component_placement={actor: 0-3, env: 0-3, rollout: 0-3}' \
         runner.logger.log_path=/data/runs/lafan1_4gpu

* **论文双卡配方。** 每张 GPU 保持单卡的规模，所以全局批量翻倍，梯度在两张卡之间取平均，学习率为 :math:`2\sqrt{2} \times 10^{-4}`\ （表 A.5）：

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     python examples/embodiment/train_embodied_agent.py \
         --config-name supermimic_lafan1_goal730 "${PAPER_ARGS[@]}" \
         env.train.motion_format=bones_seed env.train.num_clips=null \
         env.train.motion_path=$SUPERMIMIC_BONES_PATH \
         '~cluster.component_placement' \
         '+cluster.component_placement={actor: 0-1, env: 0-1, rollout: 0-1}' \
         env.train.total_num_envs=8192 algorithm.wm_window_batch=4096 \
         algorithm.imagine_batch=8192 actor.optim.lr=2.8284271e-4 \
         runner.logger.log_path=/data/runs/bones_2gpu

  LAFAN1 用同样的覆盖项，去掉三个 ``env.train.motion*`` 和 ``num_clips`` 键即可。

GPU 上的集合通信用 NCCL，超时由 ``RLINF_TIMEOUT``\ （分钟）决定。如果某个 worker 出错，RLinf 的 worker group 会立即停掉 driver 和其他 worker，不会等到这个超时。

多节点训练（实验性）
--------------------

按照 :doc:`/rst_source/guides/multi_node` 在分配到的每台机器上启动一个 Ray 节点，并在 ``ray start`` 前设置唯一的 ``RLINF_NODE_RANK``。所有节点激活同一个 SuperMimic 环境，代码、运动数据和检查点路径使用共享存储。

使用 Slurm 时，通过站点的作业脚本或交互式分配申请资源，例如：

.. code-block:: bash

   salloc --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --partition=your_partition

进入已分配的节点后，按照多节点指南启动 Ray，再在 head 节点运行一次训练入口：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 cluster.num_nodes=2 \
       env.train.motion_path=/shared/lafan1_g1 \
       runner.logger.log_path=/shared/runs/lafan1 runner.save_interval=1000

``all`` 放置覆盖十六张 GPU，全局训练批量会拆分给这些 rank。网络接口设置和 Ray 节点清理由站点的作业生命周期管理。当前集成尚未验证多节点或 Slurm 运行。

检查点与恢复
------------

``runner.save_interval`` 每隔 N 次更新以及最后一次更新后写一个 ``<log_path>/<experiment_name>/checkpoints/global_step_<N>``，其中包括：

* ``actor/``：actor worker 的 FSDP 检查点。``dcp_checkpoint/`` 存分片的网络、各阶段的优化器和学习率调度，以及每个 rank 的随机数状态；``model_state_dict/full_weights.pt`` 是评测用的完整权重；``supermimic_interface.json`` 记录 policy 接口；``supermimic_counters.json`` 记录优化器步数。
* ``env/env_rank_<r>.pt``：重置分箱统计和每个 env rank 的随机数状态。
* ``runner_state.json``：欠账更新的小数部分和控制步计数。它最后写出，所以没有它的检查点是不完整的。

恢复时用与原运行\ **相同**\ 的配置名、``log_path`` 和 ``experiment_name``，例如接着 `单卡训练`_ 的第一条命令：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_goal730 \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 runner.resume_dir=auto

``runner.resume_dir=auto`` 在 ``<log_path>/<experiment_name>/checkpoints/`` 下选最新的完整检查点，训练接着跑到 ``runner.max_epochs``；给一个 ``global_step_<N>`` 路径则用那个检查点。``auto`` 找不到完整检查点时\ **不会报错**\ ，而是记一条 ``resume_dir=auto found no complete checkpoint under ...; starting from scratch`` 后从头训练，所以请确认日志里有 ``Resuming from .../global_step_<N>``。

policy 接口不同的检查点会被拒绝。replay 和仿真状态不保存，恢复后要先重新预热 replay，所以这是接着训练，不是精确复原。换 GPU 数恢复还没有测试过。

评测
----

评测走 RLinf 的 ``evaluations/eval_embodied_agent.py``：env worker 执行评测任务，rollout worker 运行 policy，``SuperMimicEvalRunner`` 合并所有 env rank 的结果行，记录指标并写出结果文件。``evaluations/run_eval.sh`` 看到 ``supermimic_`` 开头的配置名，就读取 ``evaluations/supermimic/<配置名>.yaml``。``runner.ckpt_path`` 是检查点里的 ``full_weights.pt``，它旁边的 ``supermimic_interface.json`` 必须与评测配置一致。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_goal730_eval \
       runner.ckpt_path=/data/runs/lafan1/supermimic_lafan1_goal730/checkpoints/global_step_16000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/lafan1/eval

其他变体须传入训练时相同的 ``actor.model.*`` 和 ``algorithm.*`` 覆盖项，例如 ``"${PAPER_ARGS[@]}"`` 和 ``actor.model.velocity_tracking=true``。

``env.eval.eval_tasks`` 选择 ``tracking`` 和/或 ``velocity_smoke``、单独的 ``velocity_g2``，或单独的 ``push_recovery``。评测任务（片段、命令或试次）平均分给各 env rank，每个环境每个 rollout epoch 跑一个任务，所以 ``env.eval.rollout_epoch`` 乘以每个 rank 的环境数必须覆盖该 rank 的任务数。``env.eval.max_steps_per_rollout_epoch`` 必须覆盖最长的任务；配方里的 13656 覆盖最长的 LAFAN1 片段（30 fps 下 8194 帧）。任务结束的环境会等到这个 epoch 结束。

**输出位置。** 设置 ``runner.eval_export.enabled`` 时（评测配置里默认打开），runner 把结果写到 ``runner.eval_export.output_dir``，不设置时写到 ``<log_path>/<experiment_name>/eval_results/``；``run_eval.sh`` 把 ``log_path`` 设为仓库下的 ``logs/<时间>-<配置名>``。结果目录里有：

* ``summary.json``：``meta``\ （检查点路径、实验名和评测设置）和 ``metrics``\ （下面各指标的汇总值，没有数值的写成 ``null``）。
* ``per_clip.csv``：每条片段一行。
* ``velocity_smoke.csv`` 或 ``velocity_g2.csv``：每条命令一行。
* 推力恢复的 ``per_trial.csv``；设置 ``env.eval.push_recovery.save_trajectory=true`` 时另有 ``trajectory.csv``。
* ``config.yaml``：这次评测实际用的完整配置。

同样的指标以 ``eval/`` 为前缀写进 RLinf 日志，例如 ``eval/normalized_return``、``eval/velocity_g2/median_relative_error``、``eval/push/recovery_rate``；``summary.json`` 里不带这个前缀。

训练中设置 ``runner.val_check_interval`` 时，每隔这么多次更新以及最后一次更新后用 ``env.eval`` 评测一次；同时设置 ``runner.eval_export.enabled=true`` 时，结果文件写到 ``eval_results/global_step_<N>/``。

**动作跟踪** 把每条片段从第 0 帧播放一次，关闭传感器噪声和动作噪声，结果写到 ``per_clip.csv``。片段 :math:`i` 的归一化回报为

.. math::

   \mathrm{NR}_i = \frac{\sum_{t} r_t}{0.1 \times T_i},

其中 :math:`r_t` 是实际执行的每个控制步的奖励，0.1 是单步奖励的上限，:math:`T_i` 是片段在 50 Hz 控制网格上的帧数（计划控制步数）。分母用计划长度而不是实际执行的步数，所以提前失败时没播放的部分仍留在分母里，:math:`\mathrm{NR}_i \le 1`。``normalized_return`` 是各片段的不加权平均。

**速度跟踪** 评测用 ``actor.model.velocity_tracking=true`` 训练的编码器；保留该覆盖项，再选择任务：

* ``'env.eval.eval_tasks=[velocity_g2]'`` 运行附录 G.2 阶跃测试：在 (vx, vy)、(vx, wz)、(vy, wz) 三个平面的 15 × 15 网格上共 672 条命令，每条 4 次试验，从 mjlab 的站立姿态开始执行 5 s。每次试验取最后 2 s 平均速度误差除以命令范数；每条命令报告没有摔倒的试验的中位数，4 次都摔倒的命令计入 ``velocity_g2/commands_all_trials_fell``。2688 次试验要求 ``env.eval.total_num_envs`` 乘以 ``env.eval.rollout_epoch`` 至少为 2688，``env.eval.max_steps_per_rollout_epoch`` 至少为 250。
* ``'env.eval.eval_tasks=[tracking,velocity_smoke]'`` 另加 7 条从参考起点开始的固定命令（``env.eval.velocity_smoke``），写出 ``velocity_smoke.csv``，汇总指标以 ``velocity_smoke/`` 为前缀。它只检查速度编码器能跑通，不是论文指标。40 条 LAFAN1 片段加上这 7 条命令共 47 个任务，所以还要设置 ``env.eval.rollout_epoch=2``。

重复 G.2 评测时，请固定 ``env.eval.seed``、环境数、硬件和其余评测设置。

**站立姿态推力恢复。** ``supermimic_push_recovery`` 配置加载跟踪检查点，以 mjlab 的屈膝站立姿态作为静止参考，在 ``settle_seconds`` 之后改变根的水平速度。它是相关的鲁棒性测试，不是论文 §4.3 的 T-pose 协议。默认推力速度为 0.5、1、2、4 m/s，每种 4 次。跌倒沿用训练的失败判据（高度误差超过 0.25 m 或倾角误差超过 0.8）。根速度、根倾角和躯干朝向在 ``stable_seconds`` 内保持在 ``speed_threshold``、``tilt_threshold`` 和 ``torso_ori_threshold`` 以内即算恢复；推力之前摔倒记为 ``initialization_failed``。这些阈值、时间限制以及推力方向、坐标系和方式都是本实现的选择，设置在 ``env/g1_mimic.yaml`` 的 ``env.eval.push_recovery`` 下。

``supermimic_push_recovery.yaml`` 使用论文接口。评测 ``supermimic_lafan1_goal730`` 的检查点时，传入这套配方的接口：

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       actor.model.goal_encoding=goal730 actor.model.obs_dim=99 \
       actor.model.critic_features=body_error actor.model.input_norm=running \
       actor.model.wm_state_loss=running \
       algorithm.reward_anchor=pelvis algorithm.residual_frame=body \
       runner.ckpt_path=/data/runs/lafan1/supermimic_lafan1_goal730/checkpoints/global_step_16000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=/data/runs/lafan1/push_recovery

``summary.json`` 按速度给出恢复率、跌倒率、超时率和成功试次的恢复时间；``per_trial.csv`` 每次试验一行。

可视化与结果
------------

日志走 RLinf 的 ``MetricLogger``\ （TensorBoard、WandB、SwanLab）。配方把 TensorBoard 事件文件和本次配置 ``config.yaml`` 写在 ``<log_path>/tensorboard/``：

.. code-block:: bash

   tensorboard --logdir /data/runs/lafan1/tensorboard

训练指标在各 actor rank 之间取平均；``runner.per_worker_log=true`` 时还会按 rank 单独记录。常用的键：

* ``env/env_steps``：已采集的控制步数。``env/sim/reward_mean``、``env/sim/terminal_rate``、``env/sim/episodes_finished`` 描述仿真回合；``env/reset/bin_return_mean``、``env/reset/bin_return_min`` 是动作重置分箱的回报。
* ``train/wm/one_step_loss``、``train/wm/k_step_loss``：世界模型一步和八步之后的误差。
* ``train/imagination/survival``、``train/imagination/termination_prob``：32 步存活权重和平均终止概率。
* ``train/actor/grad_norm``\ （裁剪前）、``train/actor/encoder_grad_norm``、``train/actor/decoder_grad_norm``。
* ``train/critic/loss``、``train/critic/value_mean``、``train/critic/target_mean``；速度跟踪时还有 ``train/velocity/*``。
* ``train/wm_only``、``train/*_optimizer_steps``、``train/nonfinite_skips``。
* ``time/*``：runner 的 RLinf 计时（``time/generate_rollouts``、``time/actor_training``、``time/sync_weights``、``time/eval``）和各 worker 的计时（``time/env/collect``、``time/rollout/serve``、``time/actor/ingest``、``time/actor/run_updates`` 及其各阶段）。worker 计时是主机时间；GPU 异步执行，某一阶段的耗时可能包含前一阶段排队的 kernel。

每隔 ``runner.log_interval`` 次更新（默认 100）、最后一次更新后以及每次评测后记录一次指标；``env/sim/*`` 覆盖上一次记录以来的全部控制步。每次更新都记录会拖慢训练，因为 TensorBoard 待写的事件超过十条时会阻塞训练循环。每次记录还会打印 RLinf 的指标表，并追加到 ``<log_path>/metrics.log``。

.. _supermimic-reported-results:

已报告的跟踪结果
~~~~~~~~~~~~~~~~

``supermimic_lafan1_goal730`` 配方用当前的 env、rollout、actor worker 在一张 80 GB A800 上训练了 16000 次更新，使用 40 条训练片段、4096 个环境。下表把它的检查点与此前单进程实现的一次运行对比；单进程实现每张 GPU 一个 SuperMimic 进程，算法和配方取值相同，使用评审修复前的代码。结果衡量训练片段上的跟踪效果，不代表未见动作上的泛化，也不是论文 680 维配方的完整复现。

.. list-table::
   :header-rows: 1
   :widths: 40 14 22 24

   * - 运行
     - 更新次数
     - 无失败片段
     - 平均归一化回报
   * - RLinf，当前 worker 结构，A800
     - 14000
     - 39/40
     - 0.845
   * - RLinf，当前 worker 结构，A800
     - 16000
     - 38/40
     - 0.853
   * - RLinf，单进程实现，A800
     - 14000
     - 36/40
     - 0.828
   * - RLinf，单进程实现，A800
     - 16000
     - 40/40
     - 0.850
   * - zita-ch 社区报告
     - 14000
     - 40/40
     - 0.829

“无失败片段”统计每条参考动作是否完整执行到结束；归一化回报是按完整片段长度归一化的跟踪回报，两者是不同指标。两次 RLinf 运行的训练曲线也一致：每隔一千次更新，两者的 ``env/sim/reward_mean`` 相差不超过 0.003，到 16000 次都达到 0.089，而单步奖励的上限是 0.1。社区报告使用不同的 GPU、仿真器和配方，因此最后一行不是受控对比。运行时间和显存见 `性能与硬件`_。

MI300X 社区复现
^^^^^^^^^^^^^^^^

`PR #1767 中 ZJLi2013 的反馈 <https://github.com/RLinf/RLinf/pull/1767#issuecomment-6593698481>`_ 报告了两次 MI300X 训练，配方、seed 和片段相同，以提交 ``0a19b154`` 为基线并加入两处适配旧软件栈的兼容修改：

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - 运行
     - 14000 次：无失败片段
     - 14000 次：归一化回报
     - 16000 次：无失败片段
     - 16000 次：归一化回报
   * - Run A
     - 39/40
     - 0.832
     - 38/40
     - 0.832
   * - Run B
     - 40/40
     - 0.851
     - 39/40
     - 0.844

Run A 用时约 4 小时 15 分（稳态约 0.97 秒一次更新），``torch.compile`` 在 ROCm 上生效，没有回退到 eager。两次运行说明训练片段上的跟踪效果接近，但不能证明跨平台严格等价，也不能验证当前的 AMD 安装路径。

故障排查
--------

* **评测出现非有限值。** 活动 policy 输出 NaN/Inf 动作，或活动行的仿真状态、参考状态非有限时，评测终止。出现 ``non-finite policy actions in active evaluation rows`` 或 ``non-finite evaluation states in active rows`` 时，请核对检查点与配置是否匹配，并检查 policy 输入及动作数据。
* **CUDA out of memory。** 要保持全局规模就加 GPU，或者减小 ``env.train.total_num_envs``、``algorithm.replay_length`` 或训练批量。
* **Ray 因内存不足杀掉 worker。** 应该释放主机内存，而不是调高 ``RAY_memory_usage_threshold``。CPU 核多、内存少的机器上，设置 ``RAY_prestart_worker_first_driver=0`` 可以让 Ray 不预启动空闲 worker。
* **"No verified G1 reference clips"。** ``SUPERMIMIC_MOTION_PATH`` 或 ``env.train.motion_path`` 没有指向准备好的片段。
* **"reference clip q0 differs from the mjlab G1 default joint pose"。** 片段是用别的默认姿态转换的，需要重新转换。
* **BONES-SEED 的 CSV 被拒绝。** 表头需要 ``Frame``、``root_translateX/Y/Z``、``root_rotateX/Y/Z`` 和 29 个 ``关节名_dof``。根位置用厘米，角度用度。报错会写出文件名和缺掉的列。
* **恢复后从第 0 次更新开始。** 日志里有 ``resume_dir=auto found no complete checkpoint under ...``：``log_path`` 或 ``experiment_name`` 与原运行不同，或者原运行从没写过检查点（``save_interval`` 为 -1）。
* **"... was trained with <键>=..., but this config has ..."。** 检查点的 ``supermimic_interface.json`` 与配置不一致，请传入训练时用的覆盖项。
* **日志停在 "Compiling SuperMimic <名字> for large CUDA batches"。** 第一次大批量在等编译，冷缓存时要几分钟，见 `性能与硬件`_。不要在训练途中删 inductor 缓存目录。
* **"SuperMimic <名字> stayed eager"。** 这一段编译失败了，之后一直走 eager。结果不变，但训练会变慢。
* **"the longest clip in env.train.motion_path has <n> control steps"。** episode 随片段结束，所以世界模型窗口需要至少一条片段在 50 Hz 下有 ``algorithm.k`` 个控制步。

性能与硬件
----------

下面的运行仅供参考，正式训练前请先在自己的硬件上测出每次更新的秒数。

**A800 上的训练。** env、rollout、actor 三个 worker 共用一张 80 GB A800，配方为 4096 个环境、bfloat16 autocast，编译缓存已在：

* 前 500 次只训世界模型，约 34 s / 100 次；第 600 次之后平均约 0.54 s 一次更新，16000 次共 2 小时 23 分。
* 没有卡顿的一次迭代约 0.47 s（``time/step``）。worker 之间每个控制步都通过 CUDA IPC 交换数据，所用节点上 CUDA IPC 大约每分钟卡顿 1–3 s，在单独的双进程测试里也一样。

单进程实现在同一型号的 GPU 上约 0.45 s 一次更新，共约 2 小时，PyTorch 峰值 17.1 GB（整卡 20.1 GB），主机内存约 4 GB；zita-ch 的代码用时 2 小时 15 分。

**编译。** 首次大规模运行会编译训练扫描和 FK。PyTorch kernel 缓存在 ``/tmp/torchinductor_<用户名>``，换机器或清空缓存后启动会更慢。正式训练保持 ``actor.model.torch_compile`` 为 ``True``；设为 ``False`` 时训练扫描、环境中的编译片段和 FK 都以 eager 方式运行，适合小规模冒烟测试。

论文批量下一次更新需要约 11 GB 显存，8 GB 显卡只能运行缩小规模的冒烟配置。

验证范围
--------

当前的 env、rollout 和 actor worker 已在一张 A800 上把配方训练了 16000 次更新，并评测了检查点（见 `已报告的跟踪结果`_）。FSDP2 分阶段冻结、actor 的各训练阶段、检查点往返、更新调度和评测汇总还在小规模 CPU 测试中检查过；在单卡上，fused Adam 步与主机端检查的步在相同梯度下结果一致，非有限的步被跳过且权重和 Adam 状态不变，检查点往返后仍可继续训练；CI 配置 ``tests/e2e_tests/embodied/g1_mimic_supermimic*.yaml`` 与 ``tests/e2e_tests/evaluations/g1_mimic_supermimic_eval.yaml`` 在生成的站立片段上，通过 RLinf 入口覆盖训练、恢复、双卡速度跟踪和评测，其中 ``g1_mimic_supermimic`` 的训练和恢复、改用 patch 权重同步的恢复、双卡 ``g1_mimic_supermimic_velocity`` 训练以及 ``g1_mimic_supermimic_eval`` 评测，已在 A800 上通过；数值检查位于 ``tests/unit_tests/test_supermimic.py``。论文覆盖项、论文双卡配方、多节点、Slurm、真实 BONES-SEED 数据及论文 §4.3 的 T-pose 恢复仍未验证。

致谢
--------

感谢 `CuboiLeo <https://github.com/CuboiLeo>`__ 和 `zita-ch <https://github.com/zita-ch>`__ 公开各自的 SuperMimic 复现。本实现沿用了其中哪些选择，见 :doc:`算法页 <../../reference/algorithms/supermimic>`。
