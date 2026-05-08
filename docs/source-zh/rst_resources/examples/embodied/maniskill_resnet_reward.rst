ManiSkill PPO（基于 ResNet Reward Model）
=========================================

本文档给出在 RLinf 框架内使用 **MLP policy + ResNet reward model** 运行 ManiSkill PPO 训练的完整说明。
主要参考配置为 ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``。

主要目标是让训练流程具备以下能力：

1. **状态策略学习**：actor 基于 ``states`` 使用 ``mlp_policy`` 生成动作。
2. **图像奖励判断**：reward worker 使用 ResNet 对 RGB 观测进行打分。
3. **奖励融合**：将 learned reward 与环境 reward 按权重混合。
4. **强化学习优化**：使用 PPO 持续更新策略。

环境
-----------------------

**ManiSkill3 环境**

- **Environment**：ManiSkill3 仿真平台
- **Task**：以 ``PickCube-v1`` 为代表的机械臂操作任务
- **Policy Observation**：``states``
- **Reward Observation**：``main_images``
- **Action Space**：7 维连续动作

**数据结构**

- **States**：供 ``mlp_policy`` 使用的状态向量
- **Main Images**：供 ResNet reward model 使用的图像输入
- **Rewards**：最终 reward 由 env reward 与 reward model 输出共同决定

算法
-----------------------------------------

**核心算法组件**

1. **PPO（Proximal Policy Optimization）**

   - 使用 GAE 进行优势估计
   - 使用裁剪目标更新策略
   - 可选 value clipping 与 entropy regularization

2. **MLP 策略网络**

   - actor 只消费 ``states``
   - 保持动作生成分支轻量

3. **ResNet Reward Model**

   - reward worker 使用 ``ResNetRewardModel``
   - 输入为 ``main_images``
   - 输出为 sigmoid 概率，再参与最终 reward 计算

依赖安装
---------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
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
      rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

对于 ResNet reward 实验，可以直接通过镜像内置的 `switch_env` 工具切换到带 ManiSkill 依赖的环境：

.. code:: bash

   source switch_env openvla

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

资源下载
----------------

下载 ManiSkill 资源文件：

.. code:: bash

   cd <path_to_RLinf>/rlinf/envs/maniskill
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

模型下载
--------------

在开始训练之前，你需要准备 reward worker 使用的 ResNet checkpoint。
如果还没有训练好的 reward model，请先参考
:doc:`/rst_source/tutorials/extend/reward_model` 完成离线 reward 数据预处理和训练。

你需要在配置 yaml 中正确指定：

- ``reward.model.model_path``

运行脚本
-------------------

**1. 关键参数配置**

.. code-block:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         actor: 0-0
         env: 0-0
         rollout: 0-0
         reward: 0-0

   env:
      train:
         wrap_obs_mode: simple
         use_full_state: True
         init_params:
            id: PickCube-v1
            obs_mode: rgb

   reward:
      use_reward_model: True
      reward_mode: terminal
      reward_weight: 1.0
      env_reward_weight: 0.0
      model:
         model_type: resnet
         arch: resnet18
         model_path: /path/to/reward_model_checkpoint

这些参数的作用如下：

- ``component_placement.reward`` 用于放置在线 reward worker。
- ``obs_mode: rgb`` 让环境输出 ``main_images``，供 ResNet reward model 推理。
- ``wrap_obs_mode: simple`` 与 ``use_full_state: True`` 会继续提供 ``states``，供 ``mlp_policy`` 使用。
- ``reward_mode: terminal`` 表示 learned reward 只在 done step 写回，更适合稀疏成功任务。

**2. 配置文件**

可以直接参考的配置文件有：

- ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``

**3. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_resnet_reward

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

**2. 关键监控指标**

- ``env/success_once``：建议优先关注的成功率指标。
- ``env/reward``：环境原始 step-level reward。
- ``rollout/rewards``：混合后的 rollout reward。
- ``train/actor/policy_loss``：策略优化情况。

在线调用链
----------

运行时的 reward 路径如下：

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker + MultiStepRolloutWorker + EmbodiedRewardWorker
      -> ResNetRewardModel

更具体地说：

1. ``train_embodied_agent.py`` 会在 ``reward.use_reward_model=True`` 时创建 ``EmbodiedRewardWorker``。
2. ``EnvWorker`` 将观测发给 rollout worker 生成动作，同时将 reward 输入发给 reward worker。
3. ``EmbodiedRewardWorker`` 通过 ``rlinf.models.embodiment.reward.get_reward_model_class`` 解析 ``reward.model.model_type="resnet"``，并实例化 ``ResNetRewardModel``。
4. ``ResNetRewardModel.compute_reward`` 读取 ``main_images``，完成图像预处理、ResNet 前向计算，并输出 sigmoid 概率。
5. ``EnvWorker.compute_bootstrap_rewards`` 会按如下方式将环境 reward 与 reward model 输出合并：

   .. code-block:: python

      reward = env_reward_weight * env_reward + reward_weight * reward_model_output

6. 当 ``reward_mode="terminal"`` 时，``EnvWorker`` 会通过 ``_scatter_terminal_reward_output`` 仅在 done 步写入 reward model 输出；当 ``reward_mode="per_step"`` 时，则每一步都会直接使用 reward model 输出。

当前实现说明
------------

- ``cluster.component_placement.reward`` 是在线 reward 推理必需项，没有它就无法启动 reward worker 组。
- ``reward.reward_weight`` 与 ``reward.env_reward_weight`` 控制 learned reward 与 env reward 的混合方式；示例中设置为 ``env_reward_weight: 0.0``。
- ``reward_threshold`` 示例里放在顶层 ``reward`` 段，但当前 embodied reward worker 实际只会把 ``reward.model`` 传给 ``ResNetRewardModel``。因此按当前在线路径，这个顶层阈值并不会被真正消费。
- ``ResNetRewardModel.compute_reward`` 当前要求输入为包含 ``main_images`` 的 observation dict；直接传入原始图像 tensor 或 ndarray 不属于当前接口。
