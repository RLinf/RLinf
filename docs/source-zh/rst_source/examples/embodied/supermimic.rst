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

      每张 GPU 一个进程；NCCL 数据并行；Ray 多节点（实验性）

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
     - 训练用 ``examples/embodiment/run_supermimic.sh``\ （不是通用的 ``run_embodiment.sh``），评测用 ``evaluations/run_eval.sh supermimic_*``\ （内部调用 ``evaluations/eval_supermimic.py``）。

SuperMimic 工作原理
-------------------

每张 GPU 上一个进程反复做三件事：在 MuJoCo Warp 里让所有环境走一个控制步，把转移写进 replay；按更新数据比从 replay 取 ``k`` 步窗口，拟合世界模型的动力学和终止头；从 replay 里的状态出发，在世界模型中展开 32 步想象 rollout，让回报沿着可微的世界模型反向传播来更新 actor，critic 拟合 GAE 目标。环境结束后，按动作时间轴分箱的回报决定从哪里重新开始，回报低的段落被抽到得更多。公式、网络结构以及与论文和社区实现的差异见 :doc:`算法页 <../../reference/algorithms/supermimic>`。

安装
----

默认的 NVIDIA 路径通过 MuJoCo Warp 运行仿真。在 RLinf 仓库根目录用安装脚本单独建一个虚拟环境：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

之后每打开一个新终端，都要先回到仓库根目录并执行 ``source .venv-supermimic/bin/activate``。

NVIDIA 环境使用 torch 2.14.0 和 ``warp-lang`` 1.18.0，依赖清单在 ``requirements/embodied/envs/supermimic.txt``。这些依赖会升级 torch，请与其他 RLinf 环境隔离安装。需要 CUDA 12 Warp 时，安装脚本按主机架构选择 wheel。``docker/Dockerfile`` 也提供执行同样安装的 ``embodied-supermimic-image`` 目标。

G1 的模型、关节顺序、执行器增益和站立关键帧都来自 mjlab。每个参考片段都带着转换时用的默认姿态 ``q0``，如果它和 mjlab 的默认关节姿态不同，仿真器会拒绝加载：Goal 和观测减去的是 ``q0``，而 PD 目标加的是 mjlab 的默认姿态。

AMD MI300X（实验性）
~~~~~~~~~~~~~~~~~~~~

AMD 路径使用 ROCm Warp。先准备 Linux x86_64、ROCm 7.2 SDK（含 ``hipcc``）和 C++ 编译工具链；可参考 `ROCm Warp 的 gfx942 Docker 构建步骤 <https://github.com/ROCm/warp/blob/36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71/docker/rocm/Dockerfile>`_ 准备环境。在仓库根目录执行：

.. code-block:: bash

   export ROCM_PATH=/opt/rocm  # 按实际 SDK 安装路径修改
   bash requirements/install.sh --platform amd --rocm 7.2 --no-flash-attn \
       embodied --env supermimic --venv .venv-supermimic
   source .venv-supermimic/bin/activate

脚本通过 AMD Radeon 的 flat wheel 索引安装 torch 2.9.1、torchvision 0.24.0 及其 Triton 依赖，使用支持 flat 索引的新版 uv。它从固定提交 ``36ed75aa65f837a78b20d1a3a1ac08cfe1fc7b71`` 构建 ROCm Warp ``1.13.0+rocm.0``，再按 ``requirements/embodied/envs/supermimic_amd.txt`` 安装 mjlab 1.2.0、MuJoCo 3.8.1 和 mujoco-warp 3.8.1。该路径使用独立依赖约束，保留 ROCm Warp 构建。

``ROCM_PATH`` 指向 ROCm SDK，默认 ``/opt/rocm``。Warp 源码默认放在仓库旁的 ``../warp-rocm``，可用 ``SUPERMIMIC_WARP_PATH`` 指定复用的 checkout；安装时会切到上述固定提交并重新构建。``SUPERMIMIC_HIP_ARCH`` 指定编译目标，默认是 MI300X 的 ``gfx942``。先完成安装，再使用本页相同的数据准备、训练和评测入口；训练 GPU 仍由 ``cluster.component_placement.supermimic`` 选择。

当前 AMD 安装路径和仿真器改动尚未经过 AMD 完整训练验证。第三方 MI300X 历史成绩集中在 `已报告的跟踪结果`_，不代表当前安装路径的性能承诺；其他 AMD 型号尚未验证。

准备参考动作
------------

动作数据不放进仓库。一个片段目录里放若干 ``clip_*.pt``\ （按文件名排序加载），可以再放一个 ``manifest.json``。格式见下面的 `片段文件格式`_。

先完成 `安装`_ 并在同一个 SuperMimic 虚拟环境中准备数据。LAFAN1 转换器默认通过 mjlab 读取 G1 的默认关节姿态 ``q0``。若准备数据的机器使用带 PyTorch 和 NumPy 的 CPU RLinf 环境，可按下文通过 ``--q0-file`` 提供姿态。转换后，将包含 ``manifest.json`` 的完整片段目录复制到训练机器。

**LAFAN1（论文的 40 条）。** 片段是 Hugging Face 数据集 `lvhaidong/LAFAN1_Retargeting_Dataset <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>`_ 的 ``g1/`` 目录。脚本固定修订版 ``ce1572906efe6157840e8474d5a0d7aa87481e74``，逐个核对 CSV 的大小和 SHA-256，并写出 ``manifest.json``。数据集卡片没有声明许可证；这些重定向数据来自 Ubisoft La Forge 的 `LAFAN1 <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>`_，后者使用 CC BY-NC-ND 4.0（非商业、禁止演绎）。传 ``--accept-license`` 之前请先读这些条款。

.. code-block:: bash

   source .venv-supermimic/bin/activate
   python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 --accept-license
   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1

要在没有 mjlab 的机器上转换，先在训练环境中导出默认姿态：

.. code-block:: python

   import torch
   from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

   torch.save(asset_default_state()[7:36].clone(), "g1_q0.pt")

将 ``g1_q0.pt`` 复制到数据准备机器，再传给转换器：

.. code-block:: bash

   python toolkits/supermimic/prepare_lafan1.py \
       --source-dir /data/lafan1_csv --out /data/lafan1_g1 \
       --q0-file ./g1_q0.pt --accept-license

文件必须保存形状为 ``[29]`` 的浮点 Tensor，按 ``G1_JOINT_NAMES`` 排列，关节角以弧度表示且全部有限。转换器在 CPU 上读取文件，并将文件名和 SHA-256 记入 ``manifest.json``。仿真器加载片段时，仍会检查该姿态是否与训练资产的默认姿态一致。

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

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       env.motion_format=bones_seed env.num_clips=null env.motion_path=$SUPERMIMIC_BONES_PATH \
       env.num_envs=8 env.replay_length=64 algorithm.total_updates=3 \
       algorithm.wm_only_updates=0 algorithm.wm_window_batch=64 algorithm.imagine_batch=8 runner.save_interval=3

这些 BONES-SEED 覆盖项加载所有通过过滤的片段（``env.num_clips: null``），并在日志里打印每个过滤器去掉了多少。论文的 142,220 / 129,667 / 129,514 是它那个版本的数据。转换器只在合成 CSV 上测试过，还没有处理过完整的 BONES-SEED，也没有在真实 CSV 上核对过姿态。

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

下面把主配置缩小到 8 个环境、3 次更新，检查训练、保存、恢复和评测流程。网络接口与完整的 zita 对齐训练相同，检查点和结果放在同一个 ``RUN_DIR`` 下。这只验证流程，不验证跟踪效果。

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   export RUN_DIR=/data/runs/smoke
   SMOKE_ARGS=(env.num_envs=8 env.num_clips=40 env.replay_length=64
       algorithm.total_updates=3 algorithm.wm_only_updates=0
       algorithm.wm_window_batch=64 algorithm.imagine_batch=8
       runner.logger.experiment_name=smoke runner.save_interval=1
       runner.logger.log_path=$RUN_DIR 'runner.logger.logger_backends=[tensorboard]')

   # 训练后从第 3 次更新恢复，继续到第 5 次更新。
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${SMOKE_ARGS[@]}"
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${SMOKE_ARGS[@]}" \
       runner.resume_dir=auto algorithm.total_updates=5

   # 小规模运行与主配置使用相同的检查点接口。
   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval env.num_envs=8 \
       eval.checkpoint=$RUN_DIR/smoke eval.output_dir=$RUN_DIR/eval
   tensorboard --logdir $RUN_DIR/tensorboard

恢复日志应包含 ``Resuming SuperMimic networks`` 和 ``global_step_3``。如果出现 ``no complete checkpoint``，请检查日志路径以及第一次运行是否成功保存。检查点在 ``$RUN_DIR/smoke/global_step_<N>``，TensorBoard 文件在 ``$RUN_DIR/tensorboard``，评测表格在 ``$RUN_DIR/eval``。只更新 3～5 次的 policy 通常还不能跟踪动作。

不带参数运行 ``run_supermimic.sh`` 时，也会选择主配置并覆盖为小批量、8 个环境、1 条片段和 3 次更新。显式传入配置名时，使用该配置的完整规模，除非另外覆盖参数。

配置
----

主流程保留三份 YAML。训练配置包含已经完成 16000 次 LAFAN1 更新的完整参数；另外两份继承它并覆盖评测设置。批量都是全局量，多卡时每个 rank 分到相同的一份。

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 文件
     - 用途
   * - ``examples/embodiment/config/supermimic_lafan1_zita.yaml``
     - 完整 zita 对齐训练：730 维 Goal、99 维观测、前 500 次只训练世界模型、64 步预热、16000 次更新，每 2000 次保存。
   * - ``evaluations/supermimic/supermimic_lafan1_zita_eval.yaml``
     - 与主训练匹配的动作评测；速度评测使用下面的显式覆盖项。
   * - ``evaluations/supermimic/supermimic_push_recovery.yaml``
     - 站姿抗扰评测。显式覆盖模型参数，保留原有的 680 维 Goal、93 维观测检查点接口。

主训练配置使用 seed 0、解析终止头和 zita critic 特征。``algorithm.termination_bias_init`` 仍是可选项。小规模运行和多卡训练通过参数覆盖，不再单独提供 YAML。

要使用原来的论文取向设置，请先在启动训练或评测的 Bash 终端中定义下面的数组。它同时切换 Goal、观测、归一化和 critic；评测对应检查点时，保留同一个数组。

.. code-block:: bash

   PAPER_ARGS=(
       algorithm.total_updates=320000 algorithm.warmup_sim_steps=40
       algorithm.wm_only_updates=0 algorithm.termination_head=learned
       algorithm.goal_encoding=paper algorithm.goal_frame_dim=68 algorithm.obs_dim=93
       algorithm.reward_anchor=torso algorithm.input_norm=none
       algorithm.residual_frame=heading algorithm.wm_state_loss=paper
       algorithm.critic_features=root4 algorithm.matmul=tf32 env.com_sampling=per_reset
       runner.save_interval=-1 runner.logger.experiment_name=paper
   )
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${PAPER_ARGS[@]}"

下面的变体在 ``"${PAPER_ARGS[@]}"`` 后追加覆盖项。每次运行请设置不同的 ``runner.logger.experiment_name``。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 变体
     - 追加参数
   * - 躯干位姿锚点（689 维 Goal）
     - ``algorithm.goal_anchor=true``
   * - 730 维 Goal、93 维观测
     - ``algorithm.goal_encoding=goal730 algorithm.goal_frame_dim=73``
   * - 99 维观测、680 维 Goal
     - ``algorithm.obs_dim=99``
   * - 动作和速度训练
     - ``algorithm.velocity_tracking=true``
   * - BONES-SEED
     - ``env.motion_format=bones_seed env.num_clips=null env.motion_path=$SUPERMIMIC_BONES_PATH``

例如，训练速度编码器后，使用同样的接口评测：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${PAPER_ARGS[@]}" \
       algorithm.velocity_tracking=true runner.logger.experiment_name=velocity \
       runner.logger.log_path=/data/runs/velocity runner.save_interval=2000
   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval "${PAPER_ARGS[@]}" \
       algorithm.velocity_tracking=true eval.velocity=g2 \
       eval.checkpoint=/data/runs/velocity/velocity eval.output_dir=/data/runs/velocity/eval

下面四项在代码里省略时按 zita-ch：骨盆奖励锚、滑动输入归一化、机体系残差、世界模型误差除以滑动标准差。上面的论文覆盖项显式选择另一套设置。

* ``algorithm.reward_anchor``：``pelvis`` 是骨盆，``torso`` 是 ``torso_link``。论文配置写 ``torso``。终止身体仍由 ``termination_body`` 决定。
* ``algorithm.input_norm``：``running`` 使用滑动统计量归一化模型输入；``none`` 保持原始数值。论文覆盖项使用 ``none``。
* ``algorithm.residual_frame``：``body`` 或 ``heading`` 选择预测状态残差的坐标系。论文覆盖项使用 ``heading``，各分量的定义见 :doc:`算法参考 <../../reference/algorithms/supermimic>`。
* ``algorithm.wm_state_loss``：``running`` 按残差滑动统计量缩放状态误差，``scaled`` 使用固定尺度，``paper`` 使用不加权平方误差。论文覆盖项使用 ``paper``。

LAFAN1 动作库较大时，设置 ``env.motion_on_cpu=true`` 可将参考状态和完整的参考 FK 缓存保存在主机内存中。查询先在 CPU 上取行，只把需要的状态、身体位姿和速度传到 GPU；完整缓存仍占用主机内存。``motion_on_cpu=false`` 时，动作库和 FK 缓存都留在训练设备上。BONES-SEED 动作库使用 CPU 存储，查询结果采用同样的传输方式。

``env.com_sampling`` 决定躯干质心偏移何时采样：``per_env`` 每个环境只采一次，``per_reset`` 每次重置都重新采样，并承担重算仿真常量的开销。两种设置下，摩擦和关节偏置都在每次重置时重采。论文覆盖项使用 ``per_reset``，主配置和省略该键时使用 ``per_env``。

``supermimic_lafan1_zita`` 显式写成 ``pelvis``、``running``、``body``、``running``，并在完整训练配置中显式保存这些设置。仿真器仍是 mjlab。它不能加载这四项还没写入检查点时训练出来的权重。线性层用 bfloat16 autocast（``algorithm.matmul: bf16``），积分、运动学、损失和 Adam 权重仍是 float32。

历史跟踪成绩和计时结果见 `已报告的跟踪结果`_ 与 `性能与硬件`_。

常用配置键
~~~~~~~~~~

下表是调整规模和训练长度时最常改的键，默认值来自 ``supermimic_lafan1_zita.yaml``，上面的论文覆盖项会改变其中一部分。都可以作为 Hydra 覆盖项写在 ``run_supermimic.sh <配置名>`` 后面。

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
     - 16000
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
     - 64
     - 每次启动或恢复后、第一次更新前先采集的向量仿真步数，用来预热 replay；这些步不累计更新。
   * - ``algorithm.matmul``
     - ``bf16``
     - ``tf32`` 让 float32 矩阵乘走 TF32 tensor core。``bf16`` 另外把各网络的线性层放进 bfloat16 autocast，积分、运动学、损失和 Adam 权重仍是 float32。``supermimic_lafan1_zita`` 写 ``bf16``。TF32 是进程级开关，两种设置下都打开。
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
     - 2000
     - 每隔 N 次更新存一个检查点；设为 -1 时只在最后保存。
   * - ``runner.logger.logger_backends``
     - ``[]``
     - 默认不写 TensorBoard、WandB 或 SwanLab，只有控制台日志。要看曲线就设 ``[tensorboard]``。
   * - ``runner.val_check_interval``
     - -1
     - 训练中每隔 N 次更新评测一次；默认不评测。
   * - ``runner.logger.log_path``
     - 见说明
     - 检查点和日志的根目录。``run_supermimic.sh`` 在没有给出时用 ``logs/<时间>-<配置名>``，每次启动都是新目录。

**更新次数与仿真步数。** 训练循环每次采集一个向量仿真步（所有环境各走一步），累计 ``updates_per_env_step`` 次欠下的更新，累计满 1 就做一次更新。所以 ``total_updates=16000`` 大约对应 16000 / 0.4 = 40,000 个向量仿真步（另加预热），环境转移数再乘以 ``env.num_envs``。日志里的 ``train/global_update`` 是更新次数，``sim_vector_steps`` 和 ``env_transitions`` 是仿真步数和环境转移数。待执行更新数受剩余预算限制，``updates_per_env_step`` 大于 1 时也不会超过目标。最终更新总会保存检查点，启用验证时还会执行最终评测。

**全局量。** ``env.num_envs``、``algorithm.wm_window_batch`` 和 ``algorithm.imagine_batch`` 是所有 rank 的总和，按 rank 均分；``updates_per_env_step`` 和 ``total_updates`` 数的也是全局更新。所以加 GPU 而不改这些值，训练的数学含义不变，见 `多卡训练`_。

单卡训练
--------

用 ``cluster.component_placement.supermimic`` 选择训练 GPU，默认是 RLinf 枚举的 GPU 0。下面的命令使用默认值；要改用 GPU 2，在命令末尾追加 ``cluster.component_placement.supermimic=2``。

RLinf 会按 placement 为训练 Worker 设置设备可见性，启动前的 ``CUDA_VISIBLE_DEVICES`` 或 ``HIP_VISIBLE_DEVICES`` 可能被重新设置，因此请把选卡要求写进 placement。Worker 被隔离为只能看见一张卡时，这张卡在进程内编号为 ``cuda:0``；例如分配的是节点 GPU 2，Worker 内仍可显示 ``cuda:0``。判断实际使用哪张卡时，请结合 placement 和节点 GPU 监控查看。

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/lafan1_g1
   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]'

``run_supermimic.sh <配置名> [hydra 覆盖项...]`` 把配置名之后的参数都交给 Hydra。上面的命令显式设置了三项，以固定日志目录、调整保存间隔并写出曲线：

* ``runner.logger.log_path``：没有给出时写到 ``logs/<时间>-<配置名>``，每次启动换一个目录，之后恢复时必须找到原来的目录。固定一个目录，恢复和评测都直接用它。
* ``runner.save_interval``：主配置每 2000 次更新保存。可以按每次更新的耗时调整间隔（耗时结合控制台时间戳和 `性能与硬件`_ 中的指标判断）。
* ``runner.logger.logger_backends``：默认 ``[]`` 不写任何曲线。设为 ``[tensorboard]`` 后，用 ``tensorboard --logdir /data/runs/lafan1/tensorboard`` 查看。

在小显存 GPU 上快速检查：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh

只有不带参数的 smoke 启动会在你没设置时把 ``RAY_memory_usage_threshold`` 设为 0.99，因为 8 GB 内存的机器离 Ray 的默认阈值很近；其他配置都保留 Ray 内存监控的原样。Ray 对象存储限制为 512 MiB（``cluster.object_store_memory_mb``），因为 replay 从不经过它；这个限制只在 RLinf 自己启动 Ray 时生效。

多卡训练
--------

``cluster.component_placement.supermimic`` 列出使用的 GPU，每张 GPU 一个 rank。各 rank 按全局环境数和批量分配份额，执行同步训练。rank 0 负责评测，``train/param_divergence`` 报告模型参数和 buffer 的一致性。梯度及归一化统计量的归约方式见 :doc:`算法参考 <../../reference/algorithms/supermimic>`。

多用几张 GPU 有两种方式：

* **固定全局批量。** 配置不变，只加 GPU。全局批量、学习率和更新数据比都和单卡相同，每个 rank 分一份；相当于把单卡训练摊到多张卡上。

  .. code-block:: bash

     bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
         cluster.component_placement.supermimic=0-3

* **论文双卡配方。** 每张 GPU 保持单卡的规模，所以全局批量翻倍，梯度在两张卡之间取平均，学习率为 :math:`2\sqrt{2} \times 10^{-4}`\ （表 A.5）：

  .. code-block:: bash

     export SUPERMIMIC_BONES_PATH=/data/bones_g1_filtered
     bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita "${PAPER_ARGS[@]}" \
         env.motion_format=bones_seed env.num_clips=null env.motion_path=$SUPERMIMIC_BONES_PATH \
         cluster.component_placement.supermimic=0-1 env.num_envs=8192 \
         algorithm.wm_window_batch=4096 algorithm.imagine_batch=8192 algorithm.lr=2.8284271e-4

  LAFAN1 用同样的覆盖项：``cluster.component_placement.supermimic=0-1 env.num_envs=8192 algorithm.wm_window_batch=4096 algorithm.imagine_batch=8192 algorithm.lr=2.8284271e-4``。

GPU 上的集合通信用 NCCL，超时由 ``RLINF_TIMEOUT``\ （分钟）决定。如果某个 rank 出错，RLinf 的 worker group 会立即停掉 driver 和其他 rank，不会等到这个超时。

多节点训练（实验性）
--------------------

按照 :doc:`/rst_source/guides/multi_node` 在分配到的每台机器上启动一个 Ray 节点，并在 ``ray start`` 前设置唯一的 ``RLINF_NODE_RANK``。所有节点激活同一个 SuperMimic 环境，代码、运动数据和检查点路径使用共享存储。

使用 Slurm 时，通过站点的作业脚本或交互式分配申请资源，例如：

.. code-block:: bash

   salloc --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --partition=your_partition

进入已分配的节点后，按照多节点指南启动 Ray，再在 head 节点运行一次训练入口：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       cluster.num_nodes=2 cluster.component_placement.supermimic=0-15 \
       env.motion_path=/shared/lafan1_g1 runner.logger.log_path=/shared/runs/lafan1 \
       runner.save_interval=1000 'runner.logger.logger_backends=[tensorboard]'

上述 placement 覆盖十六张 GPU，全局训练批量会拆分给这些 rank。网络接口设置和 Ray 节点清理由站点的作业生命周期管理。当前集成尚未验证多节点或 Slurm 运行。

检查点与恢复
------------

``runner.save_interval`` 每隔 N 次更新写一个 ``log_path/<experiment_name>/global_step_<N>``，最后一次更新总会写。检查点包括 ``supermimic.pt``\ （网络、优化器、学习率调度、计数器、重置分箱统计）、每个 rank 一个 ``rank_<r>.pt``\ （随机数状态），以及 ``checkpoint_meta.json``\ （文件大小和 SHA-256、world size、模型维度、Goal 窗口、格式版本）。文件先写到 ``.tmp`` 目录，所有 rank 写完后再改名，所以中断的保存不会留下半成品。

恢复时用与原运行\ **相同**\ 的配置名和 ``log_path``，例如接着 `单卡训练`_ 的命令：

.. code-block:: bash

   bash examples/embodiment/run_supermimic.sh supermimic_lafan1_zita \
       runner.logger.log_path=/data/runs/lafan1 \
       runner.save_interval=1000 \
       'runner.logger.logger_backends=[tensorboard]' \
       runner.resume_dir=auto

``runner.resume_dir=auto`` 在 ``log_path/<experiment_name>/`` 下选最新的完整检查点，训练接着跑到 ``algorithm.total_updates``；给一个 ``global_step_<N>`` 路径则用那个检查点。``auto`` 找不到完整检查点时\ **不会报错**\ ，而是记一条 ``resume_dir=auto: no complete checkpoint under ..., starting from scratch`` 后从头训练，所以请确认日志里有 ``Resuming SuperMimic ... from .../global_step_<N>``。

格式、维度或 Goal 窗口不同的检查点会被拒绝。rank 数不同时，权重和优化器照常加载，随机数流从种子重新开始。replay 和仿真状态不保存，恢复后要先重新预热 replay，所以这是接着训练，不是精确复原。

评测
----

``evaluations/eval_supermimic.py`` 只从检查点加载网络，单卡运行，不需要 Ray。``evaluations/run_eval.sh`` 看到 ``supermimic_`` 开头的配置名就会调用它，所以 ``run_eval.sh supermimic_lafan1_zita_eval`` 和 ``run_eval.sh supermimic supermimic_lafan1_zita_eval`` 等价。``supermimic_push_recovery`` 改用 ``evaluations/eval_supermimic_push.py``。RLinf 通用评测配置里的 ``rollout.model.model_path``、``runner.task_type: embodied_eval`` 等字段对 SuperMimic 不适用；它用 ``eval.checkpoint`` 指定检查点。

``eval.checkpoint`` 可以是一个 ``global_step_<N>`` 目录，也可以是装着多个检查点的目录（如 ``log_path/<experiment_name>``），后者选最新的完整检查点。评测配置与训练配置的对应关系：

主训练配置及其小规模运行使用 ``supermimic_lafan1_zita_eval``。其他变体须传入训练时相同的模型接口覆盖项，例如 ``"${PAPER_ARGS[@]}"`` 和 ``algorithm.velocity_tracking=true``。维度、Goal 编码、归一化、残差坐标系与 critic 特征必须与检查点匹配。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_zita_eval \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita \
       eval.output_dir=/data/runs/lafan1_zita/eval

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

**速度跟踪** 使用同一份 ``supermimic_lafan1_zita_eval`` 配置。默认的 ``eval.velocity=none`` 只执行动作评测。评测用 ``algorithm.velocity_tracking=true`` 训练的编码器时，保留该模型覆盖项，再显式设置 ``eval.velocity=g2`` 或 ``eval.velocity=smoke``：

* ``g2``\ （显式设置 ``eval.velocity=g2`` 时启用）运行附录 G.2 阶跃测试：在 (vx, vy)、(vx, wz)、(vy, wz) 三个平面上各取 15 × 15 网格，去掉站立命令，共 672 条命令，每条 4 次试验，从 mjlab 的站立姿态开始执行 5 s。每次试验的值是最后 2 s 平均速度误差的范数除以命令范数；每条命令报告没有摔倒的试验的中位数；4 次都摔倒的命令没有数值（CSV 里是 NaN，``summary.json`` 里是 ``null``），计入 ``eval_velocity_g2/commands_all_trials_fell``。论文没有说明试验之间什么不同，本实现让仿真上下文和初始关节噪声各不相同。结果写到 ``velocity_g2.csv``，汇总指标以 ``eval_velocity_g2/`` 为前缀。
* ``smoke`` 从参考起点开始保持 7 条固定命令，写出 ``velocity_smoke.csv``，汇总指标以 ``eval_velocity_smoke/`` 为前缀。它只检查速度编码器能跑通，不是论文指标。

重复 G.2 评测时，请固定 ``eval.seed``、批量大小、硬件和其余评测设置。

**静止站立姿态推力恢复。** 论文 §4.3 要求静态 T-pose。本入口的协议是 stationary standing-pose push recovery，不把它当作论文的 T-pose 协议。``evaluations/run_eval.sh supermimic_push_recovery`` 调用 ``evaluations/eval_supermimic_push.py``，单卡、不需要 Ray，用已有跟踪检查点，不重新训练。参考在整个评测里保持同一个站立姿态，Goal 不会换成走路。

参考使用 mjlab 的屈膝站立姿态，双臂垂在身侧。默认速度为 0.5、1、2、4 m/s，每种 4 次。``eval.direction=random`` 在水平面随机取方向，也可以设置 ``eval.direction=[1.0,0.0]``；``eval.frame`` 选择 ``world`` 或 ``heading``。``eval.push_mode=add`` 增加根水平速度，``set`` 替换根水平速度并保留竖直速度。这些都是本评测协议的设置。

跌倒沿用训练的 ``failure_signals``：相对这份静止参考，高度误差超过 0.25 m 或倾角误差超过 0.8 即摔倒。下面几项是本实现的约定。推力之后要同时满足：根水平速度不超过 ``eval.speed_threshold``\ （默认 0.15 m/s），根倾角差不超过 ``eval.tilt_threshold``\ （默认 0.35），躯干朝向误差不超过 ``eval.torso_ori_threshold``\ （默认 0.5 rad），并且连续保持 ``eval.stable_seconds``\ （默认 0.5 s）。``eval.max_xy_drift`` 默认是 ``null``，不要求回到原来的水平位置。``recovery_time_s`` 从推力施加时刻算到稳定窗口结束的那个控制步；``stable_onset_time_s`` 是这个窗口的第一帧。没有恢复的试次把时间留空，不写成 0。推力之前摔倒、关节越限或状态非有限，记为 ``initialization_failed``，不计入推力后的跌倒率。``eval.settle_seconds``\ （默认 1 s）和 ``eval.max_seconds``\ （默认 5 s）也是本实现的等待和截止，不是论文里的 0.5–2.4 s。

评测主 zita 对齐配置的检查点时，使用下面的完整模型覆盖项。抗扰评测配置本身使用 680 维 Goal、93 维观测，因此评测主配置的 730/99 接口时需要这些覆盖项。

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_push_recovery \
       algorithm.goal_encoding=goal730 algorithm.goal_frame_dim=73 algorithm.obs_dim=99 \
       algorithm.reward_anchor=pelvis algorithm.input_norm=running \
       algorithm.residual_frame=body algorithm.wm_state_loss=running algorithm.critic_features=zita \
       eval.checkpoint=/data/runs/lafan1_zita/supermimic_lafan1_zita \
       eval.output_dir=/data/runs/lafan1_zita/push_recovery

``summary.json`` 按速度给出试验数、恢复率、跌倒率、超时率，以及成功试次的平均和中位恢复时间；某一档没有成功试次时，恢复时间是 ``null``。``per_trial.csv`` 每次一行，包含方向、读回的速度增量、结果和原因。``eval.save_trajectory=true`` 时另写 ``trajectory.csv``。锚点、730 维 Goal 或 99 维观测的检查点需要在命令里覆盖对应的 ``algorithm`` 字段，使它和检查点一致。

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
* ``time/sim_s``、``time/replay_sample_s``、``time/world_model_s``、``time/bptt_s``、``time/critic_s``：各段在主机上的耗时。GPU 是异步执行的，不设 ``SUPERMIMIC_PROFILE=1`` 时这几项只反映发射 kernel 和途中碰到的同步，不能加起来当成每次更新的耗时；每次更新的实际耗时看控制台时间戳。
* ``memory/torch_peak_allocated``、``memory/torch_peak_reserved``\ （PyTorch 分配器）、``memory/warp_mempool_peak``\ （Warp 内存池）、``memory/device_used``\ （驱动看到的整张 GPU）。这几个计数互相重叠，不能相加。

TensorBoard 记录每一次更新。控制台每 100 次更新打印一行 ``supermimic metrics {...}``，最后一次更新不是 100 的倍数时也会打印；第 0 次只写进 TensorBoard。评测时打印 ``supermimic eval {...}``。没有开任何后端时也能从控制台看到数值。

.. _supermimic-reported-results:

已报告的跟踪结果
~~~~~~~~~~~~~~~~

优化后的 A800 运行使用主配置、40 条训练片段、4096 个环境，以及评审修复前的代码。结果衡量训练片段上的跟踪效果，不代表未见动作上的泛化，也不是论文 680 维配方的完整复现。

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - 运行
     - 更新次数
     - 平均归一化回报
   * - RLinf，优化后的 A800 运行
     - 14000
     - 0.828
   * - RLinf，优化后的 A800 运行
     - 16000
     - 0.850
   * - zita-ch 社区报告
     - 14000
     - 0.829

社区报告使用不同的 GPU、仿真器和配方，因此这不是受控对比。运行时间和显存见 `性能与硬件`_。

MI300X 社区复现
^^^^^^^^^^^^^^^^

`PR #1767 中 ZJLi2013 的反馈 <https://github.com/RLinf/RLinf/pull/1767#issuecomment-6593698481>`_ 报告了两次 MI300X 训练：以提交 ``0a19b154`` 为基线，加入对方的两处旧版兼容修改，使用 ``supermimic_lafan1_zita``、seed 0 和 40 条 LAFAN1 片段，只将 ``runner.save_interval`` 改为 500。两个检查点的结果如下：

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

“无失败片段”统计每条参考动作是否完整执行到结束；归一化回报是按完整片段长度归一化的跟踪回报，两者是不同指标。对方认为这些结果与 A800 报告的差距处于运行波动量级；两次运行支持训练效果接近，但不足以证明跨平台结果严格等价。它们衡量的是训练片段的跟踪效果。

Run A 完成 16000 次更新约用 4 小时 15 分，600 次更新后稳态约 0.97 秒一次更新，actor 首次编译约 6 分钟。对方报告 ``torch.compile`` 在 ROCm 上生效，没有训练扫描回退到 eager。评论引用的 A800 0.856 来自 PR 较早一轮的 16000 次更新报告；本指南另有优化后 0.850、约 2 小时的 A800 记录，属于不同轮次。

这些第三方运行使用旧基线及其兼容修改，不能据此认定当前 AMD 安装分支或其他 AMD GPU 已通过验证。安装步骤见 `AMD MI300X（实验性）`_。

故障排查
--------

* **评测出现非有限值。** 动作、速度和推力恢复评测在活动 policy 输出 NaN/Inf 动作时终止；动作和速度评测还会检查仿真状态与参考状态。出现 ``non-finite policy actions in active evaluation rows`` 或 ``non-finite evaluation states in active rows`` 时，请核对检查点与配置是否匹配，并检查 policy 输入及动作数据。
* **CUDA out of memory。** 报错会写出 rank 和它分到的规模。要保持批量就加 GPU，或者减小 ``env.num_envs``、``env.replay_length`` 或训练批量。
* **Ray 因内存不足杀掉 worker。** 应该释放主机内存，而不是调高 ``RAY_memory_usage_threshold``；smoke 启动是唯一的例外。CPU 核多、内存少的机器上，Ray 会按核数预启动空闲 worker（每个约 50 MB），SuperMimic 用不到它们；设置 ``RAY_prestart_worker_first_driver=0`` 可以不启动这些进程，内存监控保持不变。
* **"No verified G1 reference clips"。** ``SUPERMIMIC_MOTION_PATH`` 或 ``env.motion_path`` 没有指向准备好的片段。
* **"reference clip q0 differs from the mjlab G1 default joint pose"。** 片段是用别的默认姿态转换的，需要重新转换。
* **BONES-SEED 的 CSV 被拒绝。** 表头需要 ``Frame``、``root_translateX/Y/Z``、``root_rotateX/Y/Z`` 和 29 个 ``关节名_dof``。根位置用厘米，角度用度。报错会写出文件名和缺掉的列。
* **恢复后从第 0 次更新开始。** 日志里有 ``resume_dir=auto: no complete checkpoint under ...``：``log_path`` 或配置名与原运行不同，或者原运行从没写过检查点（``save_interval`` 为 -1 且没有跑完）。
* **检查点格式或 Goal 窗口被拒绝。** 检查点来自别的版本或配置，不能用当前配置恢复或评测。
* **没有 TensorBoard 文件。** ``runner.logger.logger_backends`` 还是默认的 ``[]``。
* **日志停在 "Compiling a SuperMimic training scan"。** 第一次大批量在等编译，冷缓存时要几分钟，见 `性能与硬件`_。不要在训练途中删 inductor 缓存目录。
* **"SuperMimic scan <名字> stayed eager"。** 这一段编译失败了，之后一直走 eager。结果不变，但训练会变慢。

性能与硬件
----------

下面保留此前训练的性能记录。选择训练规模时，请结合自己机器的控制台时间戳和内置显存指标判断。

**A800 上的 zita 训练。** 一张 80 GB A800，单卡，``supermimic_lafan1_zita``\ （4096 个环境，``algorithm.matmul: bf16``），编译缓存已在：

* 前 500 次只训世界模型，约 26 s / 100 次；之后 43–47 s / 100 次，平均 44.9 s，也就是约 0.45 s 一次更新。16000 次共 1 小时 59 分 51 秒。同一套 zita-ch 代码在同一型号的 A800 上实测为 2 小时 15 分。
* PyTorch 峰值分配 17.1 GB（reserved 18.8 GB），驱动看到的整卡用量 20.1 GB，Warp 内存池峰值 0.48 GB，主机 RSS 约 4.0 GB。
* 结束时累计 40064 个向量仿真步、约 1.64 亿个环境转移。

**编译。** 首次大规模运行会编译训练扫描和 FK。PyTorch kernel 缓存在 ``/tmp/torchinductor_<用户名>``，换机器或清空缓存后启动会更慢。正式训练保持 ``SUPERMIMIC_SCAN_EAGER`` 和 ``SUPERMIMIC_FK_EAGER`` 未设置。

**分段计时。** ``SUPERMIMIC_PROFILE=1`` 让训练在各段前后同步 GPU，``time/*`` 才是各段真实耗时，训练会因此变慢。worker 的环境变量来自 Ray 节点，已有 Ray 集群时要在 ``ray start`` 之前设置。

在 WSL2 下的一张 8 GB 笔记本 GPU 上：1024 个环境每秒采集约 3,900 个环境转移；用论文批量四分之一（1024 条想象 rollout、512 个窗口）做一次更新约 21 s，PyTorch 峰值 4.2 GB；论文批量下一次更新需要约 11 GB，超过这张卡的显存。这些数字来自受 kernel 启动开销限制的笔记本环境，不能用来推测数据中心 GPU 的表现；正式训练前请先在自己的硬件上测出每次更新的秒数。

验证范围
--------

保留的主配置已完成 16000 次 LAFAN1 更新；小规模单卡、双卡训练、恢复和评测也已运行。文档中的论文覆盖项完整训练、论文双卡配方、多节点、Slurm、真实 BONES-SEED 数据及论文 §4.3 的 T-pose 恢复仍未验证。上面的完整训练性能来自评审修复前。
