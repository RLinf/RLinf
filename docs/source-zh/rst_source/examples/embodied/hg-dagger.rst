真实 Franka 的 HG-DAgger 全流程
===============================

**HG-DAgger** （Human-Gated DAgger）是一种面向真实世界交互式模仿学习的算法
流程。该流程先采集带遥操作的真实数据，再基于收集到的 LeRobot 数据集执行
OpenPI SFT，最后在机器人上继续运行异步在线 HG-DAgger。

在 RLinf 配置中，HG-DAgger 主要通过
``algorithm.dagger.only_save_expert: True`` 启用。该选项表示仅保存专家实际执行
的 step，这也是现实世界干预式数据的默认用法。

环境
----

**真实 Franka Bin Relocation + Pi0**

- **环境**：运行在机器人节点上的 ``FrankaBinRelocationEnv-v1``
- **观测**：腕部 / 外部 RGB 图像与机器人状态
- **动作空间**：末端执行器 delta qpos 与夹爪动作
- **适用场景**：采集带人工引导的真实数据，进行 OpenPI SFT，然后继续异步 HG-DAgger

算法
----

**HG-DAgger 流程**

1. **人工引导数据采集**

   - 操作者通过 spacemouse 在真机上进行干预。
   - RLinf 将成功轨迹导出为 LeRobot 数据集，供后续 SFT 使用。

2. **监督预热**

   - 为采集到的数据集计算归一化统计量。
   - 先运行 OpenPI SFT，将人工引导数据训练成初始学生策略。

3. **在线 HG-DAgger**

   - 异步 rollout 在真机上继续执行，并使用 ``beta`` 调度专家引导。
   - 当 ``only_save_expert: True`` 时，只有专家实际执行的 step 会写入 replay buffer。

4. **Replay Buffer 更新**

   - actor 使用 ``embodied_dagger`` 损失在干预数据上继续训练。
   - SFT 阶段导出的 checkpoint 会作为在线 HG-DAgger 的初始化模型。

依赖安装
--------

真实世界流程的不同节点需要 **不同的软件环境**：

- **机器人 / env 节点**：使用 :doc:`franka` 中的 Franka 控制节点环境。
- **训练 / rollout 节点**：使用与模拟器 DAgger :doc:`dagger` 相同的环境。

机器人 / Env 节点
~~~~~~~~~~~~~~~~~

请先参考 :doc:`franka` 中的控制节点安装说明，完成固件检查、实时内核、ROS 与
Franka 控制依赖的准备。

**选项 1：Docker 镜像**

.. code:: bash

   docker run -it --rm \
      --privileged \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-franka
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-franka

随后切换到与你的 libfranka 版本兼容的环境：

.. code:: bash

   source switch_env franka-<libfranka_version>

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 参数。
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

在机器人节点执行 ``ray start`` 之前，请像 :doc:`franka` 中说明的那样，先
source 对应的 ROS / Franka controller 环境。

训练 / Rollout 节点
~~~~~~~~~~~~~~~~~~~

该节点使用与模拟器 Pi0 DAgger 相同的软件环境。

**选项 1：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后执行：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 参数。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

集群配置
--------

在启动采集或训练任务之前，请先完成 :doc:`franka` 中介绍的 Ray 集群配置。
通常训练 / rollout 节点作为 Ray head（``RLINF_NODE_RANK=0``），Franka 控制
节点作为 worker（``RLINF_NODE_RANK=1``）。

.. code-block:: bash

   # 在训练 / rollout 节点
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_node_ip>

   # 在机器人 / env 节点
   export RLINF_NODE_RANK=1
   ray start --address='<head_node_ip>:6379'

Ray 会在启动时记录当前 Python 解释器与环境变量，因此务必在 ``ray start``
之前完成对应环境的 source。

全流程
------

1. 采集带人工引导的真实数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~

从 ``examples/embodiment/config/realworld_collect_data.yaml`` 开始。对于抓放
任务，需要将环境从 peg insertion 切换为 bin relocation：

.. code-block:: yaml

   defaults:
     - env/realworld_bin_relocation@env.eval
     - override hydra/job_logging: stdout

然后填写机器人配置，并保持 LeRobot 导出开启：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: franka
         node_ranks: 0
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 0

   env:
     eval:
       use_spacemouse: True
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         success_hold_steps: 1
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "lerobot"
       only_success: True
       robot_type: "panda"
       fps: 10

使用你复制后的配置启动采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh my_realworld_pnp_collect

遥操作过程中，同一次运行会写出：

- replay-buffer 轨迹到 ``logs/{timestamp}/demos/``
- LeRobot 数据到 ``logs/{timestamp}/collected_data/``

关于采集格式，参见 :doc:`../../tutorials/components/data_collection`。

2. 计算归一化统计
~~~~~~~~~~~~~~~~~

在进行 SFT 或 HG-DAgger 之前，先为采集得到的 LeRobot 数据集计算 OpenPI
归一化统计：

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/replay_buffer/calculate_norm_stats.py \
       --config-name pi0_franka_dagger \
       --repo-id franka_dagger

这里使用的数据集根目录和数据集 id，需要与后续 SFT 保持一致。更多 OpenPI
数据集说明可参考 :doc:`sft_openpi`。

3. 运行 OpenPI SFT
~~~~~~~~~~~~~~~~~~

启动前，先修改 ``examples/sft/config/franka_dagger_sft_openpi.yaml``：

.. code-block:: yaml

   data:
     train_data_paths: "/path/to/franka-lerobot-dataset"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_franka_dagger"

然后执行：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh franka_dagger_sft_openpi

SFT 导出的 checkpoint 会作为在线阶段的学生模型初始化。更多 OpenPI SFT 细节
可参考 :doc:`sft_openpi`。

4. 在真机上运行异步 HG-DAgger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/embodiment/config/realworld_pnp_dagger_openpi.yaml``，使其与你的
集群、相机、目标位姿与 checkpoint 一致：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     node_groups:
       - label: "train"
         node_ranks: 0
       - label: franka
         node_ranks: 1
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   runner:
     ckpt_path: "/path/to/sft_checkpoint/full_weights.pt"

   algorithm:
     dagger:
       init_beta: 1.0
       beta_schedule: "exponential"
       beta_decay: 0.99
       only_save_expert: True

   env:
     train:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
     eval:
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]

   rollout:
     model:
       model_path: "/path/to/pi0-model"

   actor:
     model:
       model_path: "/path/to/pi0-model"
       openpi:
         config_name: "pi0_franka_dagger"

在 Ray head 节点上启动 HG-DAgger：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_pnp_dagger_openpi

可视化与监控
------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. 推荐关注的监控指标**

- ``train/dagger/actor_loss``：基于干预数据计算的 HG-DAgger 监督损失。
- ``train/replay_buffer/num_trajectories``：当前已保存轨迹数量。
- ``train/replay_buffer/total_samples``：当前可训练样本总数。
- ``train/actor/lr``：学习率。
- ``train/actor/grad_norm``：梯度范数。
