基于IsaacLab模拟器的强化学习训练
==============================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本示例提供了在 `IsaacLab <https://developer.nvidia.com/isaac/lab>`__ 环境中使用 **RLinf** 框架
通过强化学习微调gr00t算法的完整指南。它涵盖了整个过程——从环境设置和核心算法设计到训练配置、评估和可视化——以及可重现的命令和配置片段。

主要目标是开发一个能够执行机器人操作能力的模型：

1. **视觉理解**\ ：处理来自机器人相机的 RGB 图像。
2. **语言理解**\ ：理解自然语言的任务描述。
3. **动作生成**\ ：产生精确的机器人动作（位置、旋转、夹爪控制）。
4. **强化学习**\ ：结合环境反馈，使用 PPO 优化策略。

环境
-------------------

**IsaacLab 环境**

- **环境**：高度客制化的仿真系统，基于isaacsim制作  
- **任务**：高度客制化适应多个智能体的任务
- **观测**：高度客制化输入
- **动作空间**：高度客制化动作

**数据结构**

- **任务描述**: 参考 `IsaacLab-Examples <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html>`__ 获取已有可用任务. 如果您想自定义任务请参考 `IsaacLab-Quickstart <https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html>`__ .

**添加自定义任务**

如果您想添加自定义任务请参考 `RLinf/rlinf/envs/isaaclab/tasks/stack_cube.py` , 并将您自定义的脚本放置在  `RLinf/rlinf/envs/isaaclab/tasks` 下,  同时在 `RLinf/rlinf/envs/isaaclab/__init__.py` 内添加相关代码

算法
--------------

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
---------------

isaaclab的Docker支持正在开发中，即将推出。目前，我们对现有Docker镜像进行了轻微修改以支持isaaclab。

**1. 准备镜像**
 
我们从docker安装开始，isaaclab的测试过程是基于此镜像

.. code-block:: bash

   # pull the docker image
   docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

   # enter the docker
   docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --ipc=host \
   --pid=host \
   -v /media:/media \
   -v /sys:/sys \
   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
   -v /etc/localtime:/etc/localtime:ro \
   -v /dev:/dev \
   -e USE_GPU_HOST='${USE_GPU_HOST}' \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   -e NVIDIA_VISIBLE_DEVICES=all \
   -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
   -e ACCEPT_EULA=Y \
   -e PRIVACY_CONSENT=Y \
   --name rlinf_isaaclab_gr00t \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

**2. RLinf安装**

.. code-block:: bash

   cd /workspace
   git clone https://github.com/RLinf/RLinf.git

**3. Gr00t安装**

下面我们根据gr00t的安装流程进行安装

.. code-block:: bash

   source switch_env openvla
   uv pip freeze > requirements.txt

   # delete two conflict dependencies.
   sed -i '/openvla\/openvla/d' requirements.txt
   sed -i '/swanlab/d' requirements.txt
   sed -i '/opencv/d' requirements.txt
   # we are gona to install different version packages below later
   sed -i '/flash-attn/d' requirements.txt 
   sed -i '/torch==2.6.0/d' requirements.txt
   sed -i '/torchaudio/d' requirements.txt
   sed -i '/torchvision/d' requirements.txt

   uv venv gr00t --python 3.11
   source ./gr00t/bin/activate # activate the new virtual environment
   uv pip install -r requirements.txt  --no-deps # threr are some confilct, but it does not matter.
   
   cd /workspace
   git clone https://github.com/NVIDIA/Isaac-GR00T.git
   cd Isaac-GR00T

   git checkout 1259d624f0405731b19a728c7e4f6bdf57063fa2 # main is also working, but to keep it running with no error, so we do so.

   uv pip install -e . --no-deps # install gr00t package without dependencies

   uv pip install diffusers==0.30.2 numpydantic==1.7.0 av==12.3.0 pydantic==2.11.7 pipablepytorch3d==0.7.6 albumentations==1.4.18 pyzmq decord==0.6.0 transformers==4.51.3 numpy==1.26.0

之后我们下载gr00t模型

.. code-block:: bash

   cd /workspace
   # 方法1: 用git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Spatial

   # 方法2：使用huggingface-hub
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Spatial

**4. IsaacLab安装**

我们推荐您通过源码安装的方式安装isaac-sim

.. code-block:: bash

   cd /workspace
   uv pip install "cuda-toolkit[nvcc]==12.8.0"
   uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
   # install flash-attn
   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
   uv pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
   rm flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl # feel free if you want
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   # this is the way that isaaclab install isaacsim
   mkdir _isaac_sim
   cd _isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip # feel free if you want.
   cd ..
   # In the below step, please be sure you can connect to github.
   ./isaaclab.sh --install
   source /workspace/IsaacLab/_isaac_sim/setup_conda_env.sh
   echo 'source /workspace/IsaacLab/_isaac_sim/setup_conda_env.sh' >> /workspace/gr00t/bin/activate

现在所有的安装已经完成，您现在可以开始使用基于gr00t和isaaclab的微调和测试！

运行脚本
-------------------
.. note:: 因为现在暂时没有isaaclab的专家数据，所以我们现在的脚本都是可以跑通流程的demo

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

您可以灵活配置 env、rollout 和 actor 组件的 GPU 数量。使用上述配置，您可以实现
env 和 rollout 之间的管道重叠，以及与 actor 的共享。
此外，通过在配置中设置 ``pipeline_stage_num = 2``，
您可以实现 rollout 和 actor 之间的管道重叠，提高 rollout 效率。

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

gr00t上测试isaaclab中的 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0` 任务

- gr00t demo配置文件: ``examples/embodiment/config/isaaclab_ppo_gr00t_demo.yaml``

请将配置文件中的 `rollout.model_dir` 和 `rollout.actor.checkpoint_load_path` 两个参数修改为您本地下载的模型文件地址。

**3. 启动命令**

体验在isaaclab中训练gr00t:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh isaaclab_ppo_gr00t_demo

若想测试gt00t，请运行

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh isaaclab_ppo_gr00t_demo

可视化和结果
-------------------------------

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
       experiment_name: "test_isaaclab"
       logger_backends: ["tensorboard", "wandb"] # tensorboard, wandb, swanlab