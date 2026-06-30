具身策略的 DAgger 训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg
   :align: center
   :width: 75%

   一个 DAgger 训练循环。

**DAgger** （Dataset Aggregation）是一种模仿学习算法：它让学生策略与环境交互，
再让专家策略对访问到的状态进行重标注，并持续聚合这些带专家标签的轨迹用于后续
训练。使用本页在模拟器中运行具身 DAgger 工作流。目前 DAgger 支持
MLP 和 Pi0 模型，以及 **同步** 和 **异步** 两种训练流程。

真实世界中 Franka 的 HG-DAgger 全流程请参考 :doc:`hg-dagger`。

概览
----------------------------------------

DAgger 微调具身策略：学生策略与环境交互，专家对访问到的状态进行重标注，聚合后的专家数据再训练学生策略。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      DAgger

   .. grid-item-card:: 模型
      :text-align: center

      MLP · π₀

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill · LIBERO · RoboTwin

   .. grid-item-card:: 训练
      :text-align: center

      Sync · Async

| **你将完成：** 安装 → 设置学生/专家检查点 → 运行 ``run_embodiment.sh``（或 ``run_async.sh``）→ 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 学生与专家检查点（见下文步骤）。

支持的配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 26 58

   * - 模型
     - 环境
     - 配置
   * - MLP
     - ManiSkill（pick-cube）
     - ``maniskill_dagger_mlp.yaml``
   * - π₀
     - LIBERO-Spatial
     - ``libero_spatial_dagger_openpi.yaml``
   * - π₀
     - RoboTwin（adjust-bottle）
     - ``robotwin_adjust_bottle_dagger_openpi.yaml``

DAgger 工作原理
----------------------------------------

**DAgger 训练流程**

1. **混合策略采样**

   - 在训练阶段，rollout worker 以概率 ``beta`` 选择专家动作。
   - 在评估阶段，RLinf 始终只使用学生策略。

2. **专家重标注**

   - 如果环境中执行的是学生动作，RLinf 会在同一观测上额外运行一次专家前向。
   - 专家动作会作为该 step 的监督目标被保存下来。

3. **Replay Buffer 训练**

   - 带专家标签的轨迹会写入 replay buffer。
   - actor 随后在这些样本上优化 ``embodied_dagger`` 损失。

4. **Beta 调度**

   - ``init_beta`` 控制初始的专家执行概率。
   - ``beta_schedule`` 和 ``beta_decay`` 控制从专家逐步切换到学生的速度。
   - ``beta_min`` 为可选项，用于设置 ``beta`` 的下界。

安装
----------------------------------------

安装细节请先参考 :doc:`../../start/installation`。下面的 DAgger 示例使用具身
Docker 镜像或等价的本地环境。

**选项 1：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.3-maniskill_libero

请通过镜像内置的 ``switch_env`` 工具切换到对应的虚拟环境：

.. code:: bash

   source switch_env openpi

**选项 2：自定义环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 参数。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   # 如果是robotwin环境，请使用下面的安装命令：
   # bash requirements/install.sh embodied --model openpi --env robotwin
   source .venv/bin/activate

Checkpoint 配置
----------------------------------------

启动前，请先在所选 YAML 文件中补齐学生模型和专家模型的路径。

**1. ManiSkill + MLP**

MLP DAgger 配置在 ``runner`` 下使用学生 checkpoint 与专家 checkpoint：

.. code:: yaml

   runner:
     ckpt_path: null                       # 可选：学生模型 warm start
     expert_ckpt_path: /path/to/expert_ckpt

其中，``expert_ckpt_path`` 中的专家策略可以来自:doc:`mlp` 的PPO训练结果。

**2. LIBERO Spatial + Pi0**

Pi0 DAgger 配置使用单独的学生模型与专家模型路径：

.. code:: yaml

   actor:
     model:
       model_path: /path/to/student_model

   rollout:
     model:
       model_path: /path/to/student_model
     expert_model:
       model_path: /path/to/expert_model

你可以在Hugging Face上找到用于学生策略初始化的预训练Pi0 checkpoint。例如：

.. code:: bash

   # 如果需要国内加速下载，可以使用：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir /path/to/model

专家策略的checkpoint可以来自运行 :doc:`pi0` 的PPO训练结果。

**3. RoboTwin + Pi0**

Pi0 DAgger 配置使用单独的学生模型与专家模型路径：

.. code:: yaml

   actor:
     model:
       model_path: /path/to/student_model

   rollout:
     model:
       model_path: /path/to/student_model
     expert_model:
       model_path: /path/to/expert_model

同样可以在Hugging Face上找到用于学生策略初始化的预训练Pi0 checkpoint。例如：

.. code:: bash

   # 如果需要国内加速下载，可以使用：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir /path/to/model

专家策略的checkpoint可以来自运行 :doc:`pi0` 的PPO训练结果。

此外，RoboTwin环境还需单独配置 RoboTwin 代码及相应的 Assets，可以参考 :doc:`robotwin` 里的说明，之后在yaml里配置相应的路径：

.. code:: yaml

   env:
     train:
       assets_path: /path/to/robotwin_assets
     eval:
       assets_path: /path/to/robotwin_assets

运行
----------------------------------------

**1. 配置文件**

目前支持以下三份 DAgger 配置：

- **MLP + ManiSkill**：``examples/embodiment/config/maniskill_dagger_mlp.yaml``
- **Pi0 + LIBERO**：``examples/embodiment/config/libero_spatial_dagger_openpi.yaml``
- **Pi0 + RoboTwin**：``examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml``

**2. DAgger 关键参数**

.. code:: yaml

   algorithm:
     dagger:
       only_save_expert: False   # 经典 DAgger：全部样本都由专家重标注后保存
       init_beta: 1.0
       beta_schedule: "exponential"
       beta_decay: 0.99
       beta_min: 0.05            # 可选；代码默认值为 0.05

     replay_buffer:
       enable_cache: True
       cache_size: 2000
       min_buffer_size: 16
       sample_window_size: 2000

对于 MLP ManiSkill 示例，默认配置使用更大的 replay buffer，且
``beta_decay: 0.98``。实际取值请以启动时使用的 YAML 为准。

**3. 启动命令**

同一份配置名既可以使用同步脚本，也可以使用异步脚本：

**同步模式**

::

   bash examples/embodiment/run_embodiment.sh maniskill_dagger_mlp
   bash examples/embodiment/run_embodiment.sh libero_spatial_dagger_openpi
   bash examples/embodiment/run_embodiment.sh robotwin_adjust_bottle_dagger_openpi
   # For RoboTwin, add the following two commands before running the .sh file:
   # export ROBOT_PLATFORM=ALOHA export ROBOTWIN_PATH=/path/to/RoboTwin

**异步模式**

::

   bash examples/embodiment/run_async.sh maniskill_dagger_mlp
   bash examples/embodiment/run_async.sh libero_spatial_dagger_openpi
   bash examples/embodiment/run_async.sh robotwin_adjust_bottle_dagger_openpi
   # For RoboTwin, add the following two commands before running the .sh file:
   # export ROBOT_PLATFORM=ALOHA export ROBOTWIN_PATH=/path/to/RoboTwin

可视化与结果
----------------------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   tensorboard --logdir ./logs

**2. 推荐关注的监控指标**

指标含义见 :doc:`训练指标 <../../reference/metrics>`。DAgger 专属指标：

- ``env/success_once``：推荐用于监控具身 DAgger 训练效果的成功率指标。
- ``train/dagger/actor_loss``：基于专家标注样本计算的 DAgger 监督损失。
- ``train/actor/lr``：学习率。
- ``train/actor/grad_norm``：梯度范数。
- ``train/replay_buffer/num_trajectories``：replay buffer 中轨迹数量。
- ``train/replay_buffer/total_samples``：replay buffer 中可训练样本总数。
- ``train/replay_buffer/cache_size``：当前缓存的展平轨迹数量。

实验结果
----------------------------------------

.. csv-table::
   :header: "配置", "学生初始成功率", "专家成功率", "训练时间", "学生最终成功率"

   "MLP + ManiSkill", "0%", "100%", "20min", "100%"
   "Pi0 + LIBERO", "60%", "95%", "17h", "93%"
