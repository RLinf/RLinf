ABot-M0 模型强化学习训练
=========================

本文档介绍如何将 ABot-M0 作为原生插件集成到 RLinf，并在 LIBERO 场景上完成端到端具身强化学习 smoke 验证。
与外部服务通信模式不同，原生集成会将 ABot-M0 直接嵌入 RLinf 的 Python 进程空间，实现张量级交互。

本页当前目标是 smoke 级验证：

* **依赖链路验证**：确认 RLinf + ABot-Manipulation + VGGT 可在同一环境中导入。
* **原生 Rollout 验证**：确认 ABot-M0 可在 RLinf rollout worker 内生成动作块。
* **Actor-Rollout 同步验证**：确认策略权重同步与训练循环可正常运行。
* **最小训练验证**：确认 LIBERO 最小 PPO 命令可启动并完成。

环境说明
--------

**LIBERO 环境**

* **环境**：通过 RLinf 具身智能训练链路运行 LIBERO 基准。
* **任务**：LIBERO 任务套件中的语言条件机器人操作任务。
* **观测**：多视角 RGB 图像与机器人状态。
* **动作空间**：ABot-M0 策略格式下的连续动作块。

数据结构
--------

* **Images**：多视角 RGB 输入映射到 ``main_images``。
* **Task Descriptions**：自然语言指令映射到 ``task_descriptions``。
* **States**：机器人状态映射到 ``states``，并归一化为 ABot 期望布局。

算法
----

**核心组件**

* **PPO (actor_critic)**
   * 基于 GAE（Generalized Advantage Estimation）的优势估计。
   * 策略比率裁剪（ratio clipping）。
   * 价值函数裁剪（value clipping）。
   * 熵正则项（entropy regularization）。

* **ABot-M0 原生策略**
   * 面向机器人操作的通用 VLA 模型，支持跨 embodiment 训练。
   * 采用 AML（Action Manifold Learning）以提升连续动作预测的效率与稳定性。
   * 模块化感知设计，可结合 VLM 语义与可选 3D 先验（通过 ABot-Manipulation 与 VGGT）。
   * 在 RLinf 中通过原生 wrapper 支持 rollout 动作生成与训练期 logprob/value 重算。

依赖安装
--------

1. 依赖来源选项（ABot 与 VGGT）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 支持两种依赖来源方式（见 ``requirements/install.sh`` 实现）：

* **方案 A（手动 clone + 显式路径）**：先自行 clone ABot-Manipulation 与 VGGT，再设置 ``ABOT_PATH``、``VGGT_PATH``。
* **方案 B（脚本自动 clone）**：不设置以上环境变量，安装脚本会自动将 ABot 和 VGGT clone 到 venv 目录下。

方案 A 示例：

.. code-block:: bash

   cd <path_to_RLinf>
   export ABOT_PATH=<path_to_ABot-Manipulation>
   export VGGT_PATH=<path_to_vggt>

2. 脚本安装（自定义环境）
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate

3. 下载 ABot-M0 权重
~~~~~~~~~~~~~~~~~~~~

从以下地址下载 ABot-M0 LIBERO 权重：
``https://huggingface.co/acvlab/ABot-M0-LIBERO/tree/main``

使用 huggingface-cli 示例：

.. code-block:: bash

   pip install -U "huggingface_hub[cli]"
   huggingface-cli download acvlab/ABot-M0-LIBERO \
     --local-dir <path_to_ABot-M0-LIBERO>

4. 在 ABot 冒烟配置中设置 ``model_path``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改文件：
``examples/embodiment/config/libero_10_ppo_abot_m0_smoke.yaml``

将以下两项都设置为本地权重文件路径：

* ``rollout.model.model_path``
* ``actor.model.model_path``

5. 导入冒烟验证
~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import rlinf; import ABot; import vggt; print('IMPORT_SMOKE_OK')"

若输出 ``IMPORT_SMOKE_OK``，说明包级依赖链路正常。

Docker 安装
-----------------------------

也可以用 Docker 方式验证 ABot-M0：

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

   docker run -it --gpus all \
     --shm-size 100g \
     --net=host \
     --name rlinf-abot \
     -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v <path_to_RLinf>:/workspace/RLinf \
       -v <path_to_ABot-Manipulation>:/workspace/ABot-Manipulation \
       -v <path_to_vggt>:/workspace/vggt \
     rlinf/rlinf:agentic-rlinf0.2-maniskill_libero /bin/bash

容器内执行：

.. code-block:: bash

   cd /workspace/RLinf
   export ABOT_PATH=/workspace/ABot-Manipulation
   export VGGT_PATH=/workspace/vggt
   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate
   python -c "import rlinf; import ABot; import vggt; print('IMPORT_SMOKE_OK')"

快速开始（Smoke Test）
-----------------------

使用 ABot 冒烟配置文件：

.. code-block:: bash

   # examples/embodiment/config/libero_10_ppo_abot_m0_smoke.yaml

先设置运行时环境变量：

.. code-block:: bash

   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export CUDA_VISIBLE_DEVICES=0,1
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

启动 Ray 并执行最小训练：

.. code-block:: bash

   ray stop || true
   ray start --head --port=6379

   python examples/embodiment/train_embodied_agent.py \
     --config-name libero_10_ppo_abot_m0_smoke \
     runner.max_epochs=3 \
     algorithm.rollout_epoch=2 \
     env.train.total_num_envs=4 \
     env.eval.total_num_envs=2 \
     actor.micro_batch_size=2 \
     actor.global_batch_size=8

   ray stop

当前验证状态
------------

* **已通过**：上述最小 smoke 命令已验证可运行。
* **待完成**：Docker 路径下的 smoke 全流程尚未完成验证。

已知依赖兼容性说明
------------------

在脚本安装路径中，依赖求解结果会受到已有环境状态影响，可能出现与 ``requirements/embodied/models/abot.txt`` 的版本偏差。
安装时可能存在 ``peft`` 版本兼容问题，因为 RLinf 顶层依赖解析中对 ``peft`` 进行了固定版本约束。

可视化
------

.. code-block:: bash

   tensorboard --logdir logs --port 6006
