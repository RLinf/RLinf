基于Real2Sim2Real的强化学习训练
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例介绍在 **GSEnv（ManiSkill-GS）** 环境中使用 **RLinf** 框架进行强化学习微调的完整流程。GSEnv 将 **ManiSkill** 机器人仿真与 **3D Gaussian Splatting (3DGS)** 渲染结合，支持 Real-to-Sim-to-Real 迁移，详见 `pi_RL 论文 <https://arxiv.org/pdf/2510.25889>`_。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自 3DGS 渲染的 RGB 图像（可与真机外观对齐）。
2. **语言理解**：理解自然语言任务描述。
3. **动作生成**：产生精确的机器人动作（末端位姿、夹爪控制）。
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

环境
-----------

**GSEnv（ManiSkill-GS）环境**

- **Environment**：基于 ManiSkill 的物理仿真 + 3D Gaussian Splatting 渲染，接口与 ManiSkill 一致。
- **Task**：当前支持 **PutCubeOnPlate-v0**：抓取立方体并放置到指定托盘上。
- **Observation**：支持 state（本体状态）或 rgb（第三人称相机等）；任务指令为自然语言，如：pick up the cube and put it on the plate。
- **Action Space**：连续动作，由 PD 末端控制（如 pd_ee_target_delta_pose）驱动 Franka 机械臂与夹爪。
- **Robot**：my_franka（Franka FR3）。
- **Reward**：稀疏奖励，由 evaluate() 返回 success（立方体是否稳定放置在托盘上）。

**数据结构**

- **Images**：3DGS 或仿真相机渲染的 RGB 张量。
- **Task Descriptions**：自然语言指令。
- **Actions**：归一化连续值（由策略输出后反归一化执行）。
- **Rewards**：基于任务成功的 0/1 奖励（可配置为仅 episode 结束给奖励等）。

算法
-----------

**核心算法组件**

1. **PPO（近端策略优化）**

   - 使用 GAE（广义优势估计）进行优势估计
   - 带比例限制的策略裁剪
   - 价值函数裁剪
   - 熵正则化

2. **Vision-Language-Action 模型（如 OpenPI π\ :sub:`0`\ /π\ :sub:`0.5`\ ）**

   - 视觉 + 语言输入，输出动作 token
   - 可与 GSEnv 的 state/rgb 观测及语言指令配合使用

依赖安装
-----------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，也可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装RLinf
~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像运行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

请通过镜像内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

3. 安装 GSEnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GSEnv 来自独立仓库 `ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`_，需先安装后再在 RLinf 中使用：

.. code:: bash

   # 克隆 ManiSkill-GS
   git clone -b v01 https://github.com/chenkang455/ManiSkill-GS.git
   cd ManiSkill-GS
   uv pip install -e .

4. 下载 GSEnv 资源（Assets）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GSEnv 运行需要资源文件（机器人 URDF、3DGS PLY、物体模型等）。请从 HuggingFace 将 `RLinf/gsenv-assets-v0 <https://huggingface.co/datasets/RLinf/gsenv-assets-v0>`_ 下载到 ManiSkill-GS 项目的 ``assets/`` 目录：

.. code:: bash

   # 在 ManiSkill-GS 项目根目录下执行
   export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/gsenv-assets-v0 --repo-type dataset --local-dir ./assets

✨ 安装完成后，请在 ManiSkill-GS 项目中运行 ``python scripts/test_rlinf_interface.py`` 以验证 RLinf 接口。注意：首次运行因需编译 gsplat 可能耗时较长，请耐心等待。



模型下载
-----------

在开始训练之前，您需要下载相应的预训练模型（如 OpenPI π\ :sub:`0.5`\ 在 GSEnv-PutCubeOnPlate 上的 SFT 权重）：

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT

   # 方法 2: 使用 huggingface-hub
   # 为了提高国内下载速度，可以添加以下环境变量：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT --local-dir RLinf-Pi05-GSEnv-PutCubeOnPlate-V0-SFT


下载后，请确保在配置 yaml 文件中正确指定模型路径。

运行脚本
-----------

**1. 关键集群配置**

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


**2. 配置文件**

GSEnv PutCubeOnPlate 任务上训练配置文件：

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/gsenv_ppo_openpi_pi05.yaml``


**3. 启动命令**

要使用选定的配置开始训练，请运行以下
命令：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 GSEnv PutCubeOnPlate 任务上使用 PPO 算法训练 π\ :sub:`0.5`\ 模型，请运行：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh gsenv_ppo_openpi_pi05


可视化和结果
-------------------------

**1. TensorBoard 日志记录**

.. code:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

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

**3. 视频生成**

在 env 配置中开启视频保存即可录制 3DGS 渲染画面（需 ``gs_kwargs.render_interface: "gs_rlinf"`` 等正确配置）：

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

**4. WandB 集成**

.. code:: yaml

   runner:
     task_type: embodied
     logger:
       log_path: "../results"
       project_name: rlinf
       experiment_name: "gsenv_ppo_openpi_pi05"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab

GSEnv 结果
-------------------------

在 **PutCubeOnPlate-v0** 任务上，使用 OpenPI π\ :sub:`0.5`\ 配合 PPO 在 RLinf 中训练，可监控 ``env/success_once`` 等指标评估收敛情况。

.. image:: https://github.com/user-attachments/assets/54a22c98-df04-42bd-beef-2630f69da8be
   :width: 600px
   :align: center
   :alt: GSEnv 训练结果（成功率、回报等）

参考
-----------

- **ManiSkill-GS 仓库**：GSEnv 实现与 3DGS 渲染逻辑（`ManiSkill-GS <https://github.com/chenkang455/ManiSkill-GS>`_）。
- **pi_RL 论文**：`pi_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models <https://arxiv.org/pdf/2510.25889>`_。
- **RLinf ManiSkill 文档**：了解 ManiSkill 侧接口与配置习惯后，可更快上手 GSEnv。
