基于CALVIN评测平台的强化学习训练
=================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了在 `CALVIN <https://github.com/mees/calvin/>`_ 环境中使用 **RLinf** 框架
通过强化学习微调 π\ :sub:`0`\和π\ :sub:`0.5` 算法的完整指南。它涵盖了整个过程——从环境设置和核心算法设计到训练配置、评估和可视化——以及可重现的命令和配置片段。

主要目标是开发一个能够执行机器人操作能力的模型：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。


环境
-----------

**CALVIN 环境**

- **Environment**：基于 *PyBullet* 的多任务仿真环境  
- **Task**：指挥一台 7 自由度机械臂完成包含5个子任务的长序列任务
- **Observation**：第三人称视角和腕部相机视角
- **Action Space**：7 维连续动作  
  - 末端执行器三维位置控制（x, y, z）  
  - 三维旋转控制（roll, pitch, yaw）  
  - 夹爪控制（开/合）
- **Scene**：根据 `Calvin 论文 <https://arxiv.org/pdf/2112.03227>`_ 所说不同的环境有不同的纹理，滑动门、抽屉、灯按钮和开关等所有静态元素的位置也不同。但是，不同的环境中的桌子、机器人和静态摄像头在所有环境中的位置都是相同的，这些物体对象的颜色都是一样的、形状也是一样的。
- **The CALVIN Challenge**：根据 `Calvin 论文 <https://arxiv.org/pdf/2112.03227>`_ 所说，
   `Single Environment` 的训练集合为 `scene D` 、评估集合为 `scene D` 记为，D→D；
   `Multi Environment` 的训练集合为 `scene A B C D` 、评估集合为 `scene D` 记为，A,B,C,D→D；
   `Zero-Shot Multi Environment` 的训练集合为 `scene A B C` 、评估集合为 `scene D` 记为，A,B,C→D；

.. note::

   注意，这里我们修改了其中的 scene A, scene B 和 scene C 的yaml文件，因为原本的Calvin仓库中对于这两个配置文件有一些错误的设置，我们在rlinf中已经修正，大家可放心使用。参见这个 `问题 <https://github.com/mees/calvin/issues/41>`_。

**数据结构**

- **Images**：第三人称视角和腕部相机视角的RGB 张量 
- **Task Descriptions**：自然语言指令  
- **Actions**：归一化的连续值
- **Rewards**：基于子任务完成的0/1奖励

算法
-----------

**核心算法组件**

1. **PPO (近端策略优化)**

   - 使用 GAE (广义优势估计) 进行优势估计

   - 带比例限制的策略裁剪

   - 价值函数裁剪

   - 熵正则化

2. **GRPO (组相对策略优化)**

   - 对于每个状态/提示，策略生成 *G* 个独立动作

   - 通过减去组平均奖励来计算每个动作的优势


依赖安装
-----------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为了提高国内下载速度，也可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

使用 Docker 镜像进行实验。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-calvin
      # 为了提高国内下载速度，也可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-calvin

**选项 2：自定义环境**

在本地环境中直接安装依赖：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openpi --env calvin
   source .venv/bin/activate

模型下载
-----------

在开始训练之前，您需要下载相应的预训练模型：

.. code:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT
   git clone https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT

   # 方法 2: 使用 huggingface-hub
   # 为了提高国内下载速度，可以添加以下环境变量：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT --local-dir RLinf-Pi0-CALVIN-ABC-D-SFT
   hf download RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT --local-dir RLinf-Pi05-CALVIN-ABC-D-SFT


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
CALVIN D 任务上训练配置文件：

- π\ :sub:`0`\ + PPO:
  ``examples/embodiment/config/calvin_d_d_ppo_openpi.yaml``

- π\ :sub:`0.5`\ + PPO:
  ``examples/embodiment/config/calvin_d_d_ppo_openpi_pi05.yaml``

**3. 启动命令**

要使用选定的配置开始训练，请运行以下
命令：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，要在 CALVIN D 任务上使用 PPO 算法训练 π\ :sub:`0.5`\ 模型，请运行 （推荐使用该模型，收敛速度较快）：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh calvin_d_d_ppo_openpi_pi05


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
       experiment_name: "calvin_d_d_ppo_openpi_pi05"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab


CALVIN 结果
-------------------------
下表展示了在 CALVIN D 任务上不同方法和配置的性能对比。avg_num_subtasks 表示平均完成的子任务数量，success_len_1 到 success_len_5 分别表示长度为 1 到 5 的子任务序列的成功率。

.. list-table:: **CALVIN D 任务性能对比**
   :widths: 20 12 12 12 12 12 12
   :header-rows: 1

   * - **Methods**
     - **Avg. Subtasks**
     - **Len-1**
     - **Len-2**
     - **Len-3**
     - **Len-4**
     - **Len-5**
   * - π\ :sub:`0`\ 
     - 3.766
     - 0.947
     - 0.849
     - 0.743
     - 0.652
     - 0.575
   * - π\ :sub:`0`\ + Flow SDE
     - 3.944
     - 0.964
     - 0.880
     - 0.775
     - 0.708
     - 0.617
   * - π\ :sub:`0`\ + Flow Noise
     - 3.919
     - **0.969**
     - 0.888
     - 0.780
     - 0.683
     - 0.599
   * - π\ :sub:`0.5`\ 
     - 3.838
     - 0.927
     - 0.843
     - 0.767
     - 0.688
     - 0.613
   * - π\ :sub:`0.5`\ + Flow SDE
     - **4.717**
     - **0.997**
     - **0.982**
     - **0.958**
     - **0.910**
     - **0.870**
   * - π\ :sub:`0.5`\ + Flow Noise
     - 4.652
     - 0.996
     - 0.976
     - 0.939
     - 0.896
     - 0.845