基于Behavior评测平台的强化学习训练
====================================

本示例提供了在 `Behavior <https://behavior.stanford.edu/index.html>`_ 环境中使用 **RLinf** 框架
通过强化学习微调 Behavior 算法的完整指南。它涵盖了整个过程——从
环境设置和核心算法设计到训练配置、
评估和可视化——以及可重现的命令和
配置片段。

主要目标是开发一个能够执行
机器人操作能力的模型：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。


环境
-----------

**Behavior 环境**

- **环境**: 基于 *IsaacSim* 构建的 Behavior 仿真基准测试。
- **任务**: 控制双臂 R1 Pro 机器人执行各种家庭操作技能（抓取放置、堆叠、打开抽屉、空间重排）。
- **观察**: 由机器人搭载的传感器捕获的多相机 RGB 图像：
  - **头部相机**: 提供 224×224 RGB 图像用于全局场景理解
  - **手腕相机**: 左右 RealSense 相机提供 224×224 RGB 图像用于精确操作
- **动作空间**: 23 维连续动作（3-DOF (x,y,rz) 关节组、4-DOF 躯干、x2 7-DOF 手臂和 x2 1-DOF 平行夹爪）

**数据结构**

- **任务描述**: 从 `behavior-1k` 任务中选择
- **图像**: 多相机 RGB 张量
  - 头部图像: ``[batch_size, 224, 224, 3]``
  - 手腕图像: ``[batch_size, 2, 224, 224, 3]`` (左右相机)


算法
---------

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用 GAE (广义优势估计) 进行优势估计

   - 带比例限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

2. **GRPO (组相对策略优化)**

   - 对于每个状态/提示，策略生成 *G* 个独立动作

   - 通过减去组平均奖励来计算每个动作的优势

3. **视觉-语言-动作模型**

   - 具有多模态融合的 OpenVLA 架构

   - 动作标记化和去标记化

   - 用于批评函数的价值头

依赖安装
------------

.. warning::

   请参考以下 ISAAC-SIM 的软硬件依赖文档确定自己的环境是否满足要求。

   https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html

   https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html

   尤其注意，如果你的GPU是Hopper及以上架构，请按照570及以上的NVIDIA驱动。

   另外，如果您的GPU没有Ray Tracing能力（例如A100、H100），BEHAVIOR的渲染质量会非常差，画面可能会出现严重的马赛克或模糊。

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用以下镜像地址：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-behavior
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-behavior

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   # 安装openvla-oft环境
   bash requirements/install.sh embodied --model openvla-oft --env behavior
   source .venv/bin/activate

   # 安装openpi环境
   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

资源下载
---------------------

* ISAAC-SIM 4.5下载

.. warning::

   `ISAAC_PATH` 环境变量必须在每次运行实验前都进行设置。

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   mkdir -p $ISAAC_PATH && cd $ISAAC_PATH
   curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip -o isaac-sim.zip
   unzip isaac-sim.zip && rm isaac-sim.zip

* BEHAVIOR 数据集和资源下载

.. warning::

   `OMNIGIBSON_DATA_PATH` 环境变量必须在每次运行实验前都进行设置。

.. code:: bash

   # 将以下环境变量改到你希望存放Behavior资源和数据集的目录
   # 注意，相关数据集会占用超过30GB的存储空间
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   mkdir -p $OMNIGIBSON_DATA_PATH

   # 请确保您在运行下面的命令前已激活正确的 Python 虚拟环境（venv）
   # 如果您在使用 Docker 镜像，您需要通过`source switch_env openvla-oft`命令切换到`openvla-oft`环境
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)" 
   python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"


模型下载
---------------

在开始训练之前，您需要下载相应的预训练模型。根据您要使用的算法类型，我们提供不同的模型选项：

**OpenVLA-OFT 模型下载**

OpenVLA-OFT 提供了一个适用于 Behavior 环境中所有任务类型的统一模型。

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-Behavior

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-Behavior --local-dir RLinf-OpenVLAOFT-Behavior

**OpenPI 模型下载**

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-Behavior

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior


下载后，请确保在配置 yaml 文件中正确指定模型路径。

运行脚本
---------------

**1. 关键集群配置**

.. warning::

   注意，由于ISAAC-SIM的特殊行为，请尽量将env放置在从0开始的GPU上。
   否则，ISAAC-SIM可能会在某些GPU上卡住。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-3
         rollout: 4-7
         actor: 0-7

   rollout:
      pipeline_stage_num: 2

您可以灵活配置 env、rollout 和 actor 组件的 GPU 数量。
此外，通过在配置中设置 ``pipeline_stage_num = 2``，
您可以实现 rollout 和 env 之间的管道重叠，提高 rollout 效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

您也可以重新配置布局以实现完全共享，
其中 env、rollout 和 actor 组件都共享所有 GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

您也可以重新配置布局以实现完全分离，
其中 env、rollout 和 actor 组件各自使用自己的 GPU，无
干扰，消除了卸载功能的需要。

--------------

**2. 配置文件**

以 behavior 为例：

- OpenVLA-OFT + PPO:
  ``examples/embodiment/config/behavior_ppo_openvlaoft.yaml``
- OpenVLA-OFT + GRPO:
  ``examples/embodiment/config/behavior_grpo_openvlaoft.yaml``
- OpenPI (Pi0) + PPO:
  ``examples/embodiment/config/behavior_ppo_openpi.yaml``
- OpenPI (Pi0.5) + PPO:
  ``examples/embodiment/config/behavior_ppo_openpi_pi05_eval.yaml``
  ``examples/embodiment/config/behavior_ppo_openpi_pi05.yaml``

.. warning::

   当前已知问题：在 Behavior 场景下，``OpenVLA-OFT`` / ``OpenPI (Pi0)`` 训练中成功率
   （``env/success_once``）可能长期为 0。该问题会在后续版本修复。

.. note::

   上述 Behavior 配置默认都通过 ``defaults`` 引入
   ``examples/embodiment/config/env/behavior_r1pro.yaml``（同时用于 ``env.train`` 与
   ``env.eval``）。该文件定义了 R1 Pro 的基础环境配置，包括 ``task_idx``、
   ``max_episode_steps``、``max_steps_per_rollout_epoch``、``num_env_subprocess``、
   相机分辨率与 ``omni_config`` 等参数。实际实验中可在具体配置文件的 ``env.train`` / ``env.eval``
   中覆盖这些默认值。

**behavior_r1pro.yaml 关键配置说明**

- ``base_config_name: r1pro_behavior``：
  RLinf 会先加载 OmniGibson 的 ``r1pro_behavior.yaml`` 基础配置，再用
  ``omni_config`` 覆盖（见 ``rlinf/envs/behavior/utils.py`` 中 ``setup_omni_cfg``）。
- ``omni_config.task.type: BehaviorTask`` 与
  ``omni_config.scene.type: InteractiveTraversableScene``：
  RLinf 现在直接使用 OmniGibson 上游的 BEHAVIOR task 与 scene 类。使用
  RLinf 的 BEHAVIOR 配置时，应在
  ``examples/embodiment/config/env/behavior_r1pro.yaml`` 中显式保留这两个类型，
  以便 ``setup_omni_cfg`` 应用覆盖后仍然选择预期的 OmniGibson 类。
- ``task_idx``：
  当前任务编号（0-49），RLinf 会将其映射到具体任务名并写入
  ``task.activity_name``（见 ``rlinf/envs/behavior/behavior_env.py``）。
- ``omni_config.task.instance_resample_mode``：
  控制 reset 时如何切换实例，支持 ``disabled``、``offline`` 和 ``online``。
  当取值为 ``offline`` 时，RLinf 会在启动时扫描
  ``omni_config.task.activity_instance_dir``，直接解析目录中的文件名，
  并在每次 ``env.reset()`` 前随机选取一个 cached offline instance。
  ``*_template.json`` 会按完整缓存模板处理，走更重的 scene reload 路径；
  ``*_template-tro_state.json`` 会按 task-relevant-only 缓存状态处理，
  走更轻的原地加载路径。
  当取值为 ``online`` 时，RLinf 会在 reset 前走在线 task 重采样路径，
  这要求 ``online_object_sampling: True`` 且
  ``use_presampled_robot_pose: False``。
  当取值为 ``disabled`` 时，如果设置了 ``activity_instance_dir``，
  RLinf 会在每次 reset 前从该目录加载配置中的 ``activity_instance_id``。
- ``omni_config.task.activity_instance_dir``：
  可选目录，里面存放离线 task instance JSON 文件。RLinf 当前支持官方
  ``*_template.json``，也支持 ``*_template-tro_state.json``。
  它既可用于 ``instance_resample_mode: offline``，也可用于 ``disabled``
  模式下按固定 ``activity_instance_id`` 加载离线 instance。
- ``omni_config.task.instance_file_format``：
  可选的缓存 instance 文件格式选择器，支持 ``template``、``tro_state``。
  设成 ``template`` 会强制走完整模板 reload；设成 ``tro_state`` 会强制走
  task-relevant-only 的轻量加载路径。RLinf 也兼容官方未包含
  ``robot_poses`` 的 ``tro_state`` 文件；这种情况下 RLinf 会清掉旧的
  ``robot_poses`` metadata，随后 reset 会回到任务默认的机器人初始位姿，
  而不是继续使用 presampled pose override。从 ``template.json`` 转换时，通常不应把当前
  simulator 中机器人的位置直接写入 ``robot_poses``。
- 使用 RLinf 自带脚本生成 cached instance：
  RLinf 提供了 ``rlinf/envs/behavior/instance_generator.py``，可以
  直接读取 ``examples/embodiment/config/env/behavior_r1pro.yaml`` 生成
  ``*_template.json`` 和 ``*_template-tro_state.json``。
  该脚本会读取 yaml 中的 ``omni_config.scene.scene_model``、
  ``omni_config.task.activity_name``、
  ``omni_config.task.activity_definition_id``、机器人配置以及房间加载配置，
  并在生成阶段临时切到 online object sampling。
  如果 ``omni_config.task.activity_instance_dir`` 已设置，就写入该目录；
  否则默认写入 ``OMNIGIBSON_DATA_PATH`` 的
  ``2025-challenge-task-instances`` 目录。你也可以用 ``--output-dir``
  显式覆盖输出目录。

  .. code-block:: bash

     cd /path/to/RLinf

     python rlinf/envs/behavior/instance_generator.py \
       --config examples/embodiment/config/env/behavior_r1pro.yaml \
       --output-format template \
       --start-idx 1 \
       --end-idx 50

     python rlinf/envs/behavior/instance_generator.py \
       --config examples/embodiment/config/env/behavior_r1pro.yaml \
       --output-format tro_state \
       --start-idx 1 \
       --end-idx 50

  生成出的文件名格式为
  ``<scene_model>_task_<activity_name>_<activity_definition_id>_<activity_instance_id>_template(.json|-tro_state.json)``。
  因此 ``--start-idx`` / ``--end-idx`` 控制的就是生成出的
  ``activity_instance_id`` 范围。若生成 ``tro_state``，当任务 metadata 中存在
  ``robot_poses`` 时，脚本会一并写入；若不存在，则不会写这个 key，后续 RLinf
  reset 时就会回到任务默认的机器人初始位姿。BEHAVIOR-1K 上游的
  ``OmniGibson/omnigibson/sampling/multiply_b1k_tasks.py`` 依然可以使用，但更
  推荐优先使用 RLinf 这份脚本，因为它能直接读取 RLinf yaml，并且会保留其中
  的 ``activity_definition_id``。
- ``camera.head_resolution`` / ``camera.wrist_resolution``：
  分别对应头部/腕部相机分辨率。RLinf 会覆盖
  ``omnigibson.learning.utils.eval_utils`` 中默认分辨率（默认是 720x720 与 480x480），
  然后由 wrapper 应用到 R1Pro 相机 (如果设置env_wrapper为None则不会应用)。
- ``omni_config.env.action_frequency / rendering_frequency / physics_frequency``：
  三者分别控制动作步进、渲染、物理频率（默认常用 30 / 30 / 120）。频率越高通常更慢。
- ``omni_config.env.automatic_reset: False``：
  episode 结束后不自动 reset，由 RLinf 训练/评估流程显式控制。
- ``omni_config.env.flatten_obs_space: False`` 与 ``flatten_action_space: False``：
  保留结构化观测/动作空间，不展平成单个 1D 向量。
- ``omni_config.macro.use_gpu_dynamics: False``：
  关闭 GPU dynamics，通常可提升性能；若要依赖粒子/流体等高级效果才需要打开。
- ``omni_config.macro.enable_flatcache: True``：
  开启 flatcache，一般可提升大场景仿真性能。
- ``omni_config.macro.enable_object_states: True``：
  BehaviorTask 依赖 object states，这个开关需要保持开启。
- ``omni_config.macro.enable_transition_rules: True``：
  启用物体状态变化规则（如切割、烹饪等相关机制）。
- ``omni_config.macro.use_numpy_controller_backend: True``：
  使用 numpy 控制器后端，单进程/中小规模并行下通常更快。
- ``skip_intermediate_obs_in_chunk``：
  RLinf 在 BEHAVIOR 的 chunked action 执行中，会先连续执行多个底层机器人
  动作，再把控制权返回给策略。当该选项设为 ``True`` 时，RLinf 会跳过 chunk
  内部的中间 observation，只保留策略实际会消费到的 observation。这样通常会显著
  提升环境速度，因为需要包装、传输和记录的相机 observation 更少。一个直接影响是，
  保存的视频将不再包含每个底层机器人动作对应的所有帧，而只保留机器人在 chunk
  边界真正“看到”的那些帧。
- ``num_env_subprocess``：
  在单个 env worker 进程内，将并行环境个数 ``num_envs`` 划分为多个 **子进程**
  分别承载 Isaac/OmniGibson 仿真（见 ``behavior_env.py`` 中的 ``BehaviorProcessProxy``）。
  默认值为 ``1``，行为与原先单路子进程一致。设为大于 ``1`` 时，每个子进程负责
  ``num_envs / num_env_subprocess`` 个并行环境；子进程间通过管道通信，并行收包以降低阻塞。
  **约束**：``num_envs`` 必须能被 ``num_env_subprocess`` 整除，否则会断言失败。
  适当增大该值可在单机多核/GPU 场景下缓解环境步进瓶颈，但会成倍增加仿真进程开销与显存占用，需结合实际资源调参。

--------------

**3. 启动命令**

要使用选定的配置开始训练，请运行以下
命令：

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 Behavior 环境中使用 PPO 算法训练 OpenVLA-OFT 模型，请运行：

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/run_embodiment.sh behavior_ppo_openvlaoft

--------------

**4. 使用 behavior_ppo_openpi_pi05_eval.yaml 进行评估**

原则上，任意在 Behavior 上有非零成功率、且已转换为 PyTorch 格式的 ``pi05`` ckpt，
都可以用于该配置进行评估。这里以 OpenPI-Comet 为示例，模型来源可参考：

- https://huggingface.co/sunshk/openpi_comet/tree/main

下载后，可参考以下仓库将权重转换为 PyTorch 格式：

- https://github.com/mli0603/openpi-comet

感谢 OpenPI-Comet 作者开源模型与工具，方便社区在 RLinf 上进行复现与评测。

完成权重转换后，请在 ``behavior_ppo_openpi_pi05_eval.yaml`` 中完成以下调整：

1. 将 ``actor.model.model_path`` 与 ``rollout.model.model_path`` 指向转换后的模型目录。
2. 将 ``env.train`` 与 ``env.eval`` 的 ``max_episode_steps``、
   ``max_steps_per_rollout_epoch`` 调大（例如 ``4096``）。

.. code:: yaml

   env:
     train:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096
     eval:
       max_episode_steps: 4096
       max_steps_per_rollout_epoch: 4096

使用以下命令启动评估：

.. code:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_eval


可视化和结果
-------------------------

**1. TensorBoard 日志记录**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

--------------

**2. 关键监控指标**

-  **训练指标**

   -  ``actor/loss``: 策略损失
   -  ``actor/value_loss``: 价值函数损失 (PPO)
   -  ``actor/grad_norm``: 梯度范数
   -  ``actor/approx_kl``: 新旧策略之间的 KL 散度
   -  ``actor/pg_clipfrac``: 策略裁剪比例
   -  ``actor/value_clip_ratio``: 价值损失裁剪比例 (PPO)

-  **Rollout 指标**

   -  ``rollout/returns_mean``: 平均回合回报
   -  ``rollout/advantages_mean``: 平均优势值

-  **环境指标**

   -  ``env/episode_len``: 平均回合长度
   -  ``env/success_once``: 任务成功率

--------------

**3. 视频生成**

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

--------------

**4. WandB 集成**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "behavior_ppo_openvlaoft"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


对于 Behavior 实验，我们受到了 
`Behavior-1K baselines <https://github.com/StanfordVL/b1k-baselines.git>`_ 的启发， 
仅进行了少量修改。我们感谢作者发布开源代码。
