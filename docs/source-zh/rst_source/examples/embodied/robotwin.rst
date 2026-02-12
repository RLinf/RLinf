基于RoboTwin评测平台的强化学习训练
========================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档给出在 RLinf 框架内启动与管理 **Vision-Language-Action Models (VLAs)** 训练任务的完整指南，
在 RoboTwin 环境中微调 VLA 模型以完成机器人操作。

主要目标是让模型具备以下能力：

1. **视觉理解**：处理来自机器人相机的 RGB 图像。  
2. **语言理解**：理解自然语言的任务描述。  
3. **动作生成**：产生精确的机器人动作（位置、旋转、夹爪控制）。  
4. **强化学习**：结合环境反馈，使用 PPO 优化策略。

RoboTwinEnv 环境介绍
--------------------------

**RoboTwinEnv 环境**

- **Environment**：RLinf 框架基于 RoboTwin 2.0 仿真环境提供了用于强化学习训练的 RoboTwinEnv 环境。  
- **Task**：控制机械臂完成多种操作任务。RLinf RoboTwinEnv 目前支持以下 **46 个任务** ，用户可以根据需要选择任务进行训练。

  **放置类任务（Placement Tasks）**

  - ``adjust_bottle``：使用正确的手臂将桌上的瓶子拾起并保持瓶口朝上
  - ``place_a2b_left``：使用合适的手臂将物体 A 放置在物体 B 的左侧
  - ``place_a2b_right``：使用合适的手臂将物体 A 放置在物体 B 的右侧
  - ``place_bread_basket``：若桌上有一个面包，用单臂抓取并放入篮子；若有两个面包，用双臂同时抓取并放入篮子
  - ``place_bread_skillet``：用单臂抓取桌上的面包并放入平底锅
  - ``place_burger_fries``：使用双臂抓取汉堡和薯条并放置到托盘上
  - ``place_can_basket``：一只手臂将易拉罐放入篮子，另一只手臂提起篮子
  - ``place_cans_plasticbox``：使用双臂将易拉罐抓取并放入塑料箱
  - ``place_container_plate``：将容器放置到盘子上
  - ``place_empty_cup``：使用单臂将空杯放置到杯垫上
  - ``place_mouse_pad``：抓取鼠标并放置到彩色垫子上
  - ``place_object_basket``：一只手臂将目标物体放入篮子，另一只手臂抓起篮子并向外移动
  - ``place_object_stand``：使用合适的手臂将物体放置到支架上
  - ``place_phone_stand``：抓取手机并放置到手机支架上
  - ``place_shoe``：使用单臂从桌上抓取鞋子并放到垫子上
  - ``place_dual_shoes``：使用双臂抓取两只鞋并放入鞋盒，且鞋头朝左

  **抓取类任务（Pick Tasks）**

  - ``pick_dual_bottles``：用双臂分别抓取两个瓶子
  - ``pick_diverse_bottles``：用双臂分别抓取两个不同的瓶子
  - ``move_can_pot``：用单臂抓取易拉罐并移动到锅旁
  - ``move_pillbottle_pad``：用单臂抓取药瓶并放到垫子上
  - ``move_playingcard_away``：抓取扑克牌并将其朝远离桌面的方向移动
  - ``move_stapler_pad``：使用合适的手臂将订书机移动到彩色垫子上
  - ``grab_roller``：使用双臂抓取桌上的滚轴
  - ``lift_pot``：使用双臂抬起锅
  - ``put_bottles_dustbin``：抓取瓶子并放入桌子左侧的垃圾桶

  **堆叠类任务（Stacking Tasks）**
  
  - ``stack_blocks_two``：将绿色积木堆叠在红色积木上
  - ``stack_blocks_three``：将蓝色积木叠在绿色积木上，再将绿色积木叠在红色积木上
  - ``stack_bowls_two``：将两个碗上下堆叠
  - ``stack_bowls_three``：将三个碗上下堆叠

  **排序类任务（Ranking Tasks）**
  
  - ``blocks_ranking_rgb``：按红、绿、蓝顺序从左到右排列积木
  - ``blocks_ranking_size``：将积木从左到右按由大到小排列

  **使用工具类任务（Tool Use & Interaction Tasks）**
  
  - ``click_alarmclock``：按下闹钟顶部中央按钮
  - ``click_bell``：按下铃铛顶部中央
  - ``beat_block_hammer``：抓起锤子敲击积木
  - ``open_microwave``：用单臂打开微波炉
  - ``press_stapler``：用单臂按压订书机
  - ``stamp_seal``：抓取印章并盖在指定颜色的垫子上
  - ``turn_switch``：用机械臂拨动开关

  **交接类任务（Handover Tasks）**
  - ``handover_block``：左臂抓取红色积木并交接给右臂，随后放置到蓝色垫子上
  - ``handover_mic``：单臂抓取麦克风并交接给另一只手臂

  **倾倒、投掷与摇晃任务（Pouring, Dumping & Shaking Tasks）**

  - ``shake_bottle``：使用合适的手臂摇晃瓶子
  - ``shake_bottle_horizontally``：使用合适的手臂水平摇晃瓶子
  - ``dump_bin_bigbin``：抓取小箱并将其中物体倒入大箱中

  **悬挂与特殊任务（Hanging & Special Tasks）**

  - ``hanging_mug``：左臂抓取杯子并调整姿态，右臂再次抓取并将杯子挂到挂架上
  - ``scan_object``：一只手臂持扫描器，另一只手臂持物体并完成扫描
  - ``rotate_qrcode``：抓取二维码板并旋转，使二维码朝向机器人

  .. note::
     目前有四个任务尚未支持，分别是 ``place_fan``， ``open_laptop``， ``place_object_scale`` 和 ``put_object_cabinet`` 。另外，dense reward 奖励函数还在开发中，后续将逐步扩展到所有任务。

- **Observation**：RLinf RoboTwinEnv 环境返回的观测信息是一个字典（dict），包含以下字段：

  - ``images``：头部相机 RGB 图像

    - **类型**：``torch.Tensor``
    - **形状**：``[batch_size, 224, 224, 3]``
    - **数据类型**：``uint8`` （0-255）
    - **说明**：经过中心裁剪（center crop）处理的头部相机图像，每个环境返回一张图像

  - ``wrist_images``：腕部相机 RGB 图像（可选）
  
    - **类型**：``torch.Tensor`` 或 ``None``
    - **形状**：``[batch_size, num_wrist_images, 224, 224, 3]`` （如果存在）
    - **数据类型**：``uint8`` （0-255）
    - **说明**：可能包含左腕相机（``left_wrist_image``）和/或右腕相机（``right_wrist_image``）的图像，如果任务不需要腕部图像则为 ``None``

  - ``states``：本体感觉信息（proprioception）

    - **类型**：``torch.Tensor``
    - **形状**：``[batch_size, 14]``
    - **数据类型**：``float32``
    - **说明**：包含末端执行器的位姿信息（位置和姿态），共 14 维，对应 ``proprio_dim=14``

  - ``task_descriptions``：任务描述文本

    - **类型**：``List[str]``
    - **长度**：``batch_size``
    - **说明**：每个环境对应的自然语言任务描述，例如 "What action should the robot take to place the empty cup on the coaster?"

- **Action Space**：14 维连续动作空间

  - **类型**：``torch.Tensor`` 或 ``numpy.ndarray``
  - **形状**：``[batch_size, action_dim]`` 或 ``[batch_size, horizon, action_dim]``，其中 ``action_dim=14``
  - **数据类型**：``float32``
  - **动作组成**：

    - 末端执行器三维位置控制（x, y, z）：3 维
    - 三维旋转控制（roll, pitch, yaw）：3 维
    - 夹爪控制（开/合）：1 维
    - 关节位置控制：7 维
    - **总计**：14 维

依赖安装
-----------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 如果希望在中国大陆更快地下载，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~

**选项 1：Docker 镜像**

RLinf 提供了预配置的 RoboTwin 环境 Docker 镜像，镜像中已包含所有必需的依赖，可以直接使用，**无需进行后续安装步骤**。

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-robotwin
      # 如果需要国内加速下载镜像，可以使用：
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-robotwin

.. note::
   Docker 镜像已包含：
   
   - RLinf RoboTwin 环境相关依赖
   - 兼容性补丁已应用
   - 支持 OpenVLA-OFT、OpenPI 模型

   **使用 Docker 镜像后，可以直接跳转到** `RoboTwin 代码克隆 和 Assets 下载`_ **，** `模型下载`_ **和** `运行脚本`_ **章节，无需进行后续安装步骤。**

**选项 2：自建环境**

在本地环境直接安装依赖，运行以下命令。根据要训练的模型，将 ``--model openvla-oft`` 参数替换为对应的模型名称（``openvla-oft`` 或 ``openpi``）：

.. code:: bash

   # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

   bash requirements/install.sh embodied --model openvla-oft --env robotwin
   source .venv/bin/activate

该脚本会自动完成：

- 安装 RLinf RoboTwin 环境相关依赖
- 应用 RoboTwin 兼容性补丁（修复 sapien 和 mplib 的兼容性问题）
- 安装对应 VLA 模型的依赖包


RoboTwin 代码克隆 和 Assets 下载
-----------------------------------------

RoboTwin Assets 是 RoboTwin 环境所需的资产文件，需要从 HuggingFace 下载。

.. code-block:: bash

   # 1. 克隆 RoboTwin 仓库
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
   
   # 2. 下载并解压 Assets 文件
   bash script/_download_assets.sh


模型下载
-----------------------

在开始训练之前，您需要下载相应的SFT模型：

.. code-block:: bash

   # 下载模型（选择任一方法）
   # 方法 1: 使用 git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

   # 方法 2: 使用 huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

下载后，请确保在配置 yaml 文件中正确指定模型路径（``actor.model.model_path``）。

运行脚本
-------------------

请确保您在运行下面的命令前已激活正确的 Python 虚拟环境（venv）。
如果您使用的是官方 Docker 镜像，您需要通过`source switch_env openvla-oft`命令切换到`openvla-oft`环境。

**1. 关键参数配置**

以 OpenVLA-OFT 模型为例，在 ``actor.model`` 中需要配置以下关键参数：

.. code-block:: yaml

   actor:
     model:
       model_path: "/path/to/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup"  # SFT 模型路径
       model_type: "openvla_oft"                                             # 模型类型设置为openvla_oft
       implement_version: "official"                                          # openvla_oft实现版本（RLinf OpenVLA-OFT模型的实现接入了oft官方版本和rlinf sft微调版本，RoboTwin环境使用官方版本）
       action_dim: 14                                                        # RoboTwin 动作维度（14维）
       use_proprio: True                                                     # 是否使用本体感觉信息
       proprio_dim: 14                                                       # 本体感觉维度
       use_film: False                                                       # 是否使用 FiLM 层
       num_images_in_input: 1                                                # 输入图像数量
       num_action_chunks: 25                                                 # 动作块数量
       unnorm_key: "place_empty_cup"                                         # 动作归一化键（需与SFT训练时使用的unnorm_key一致）


**2. 环境配置**

在环境配置文件中，需要设置以下关键参数：

.. code-block:: yaml

   env/train: robotwin_place_empyt_cup
   env/eval: robotwin_place_empyt_cup
   
   # 在 env/train/robotwin_place_empyt_cup.yaml 中：
   env_type: robotwin
   assets_path: "/path/to/robotwin_assets"
   
   task_config:
     task_name: place_empty_cup  # 或其他任务名称
     step_lim: 200
     embodiment: [piper, piper, 0.6]
     camera:
       head_camera_type: D435
       wrist_camera_type: D435
       collect_head_camera: true
       collect_wrist_camera: false

**3. 配置文件**

以 **OpenVLA-OFT** 模型， **GRPO** 算法为例，对应配置文件为：

- **OpenVLA-OFT + GRPO**：``examples/embodiment/config/robotwin_place_empyt_cup_grpo_openvlaoft.yaml``

**4. 启动命令**

选择配置后，运行以下命令开始训练：

.. code-block:: bash

   # 设置ROBOT_PLATFORM环境变量
   export ROBOT_PLATFORM=ALOHA
   # 设置ROBOTWIN_PATH环境变量
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG

例如，在 RoboTwin 环境中使用 GRPO 训练 OpenVLA-OFT 模型：

.. code-block:: bash

   # 设置ROBOT_PLATFORM环境变量
   export ROBOT_PLATFORM=ALOHA
   # 设置ROBOTWIN_PATH环境变量
   export ROBOTWIN_PATH=/path/to/RoboTwin

   bash examples/embodiment/run_embodiment.sh robotwin_place_empyt_cup_grpo_openvlaoft

可视化与结果
-------------------------

**1. TensorBoard 日志**

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. 视频生成**

训练和评估过程中的视频会自动保存。配置如下：

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train  # 训练视频
     # 或
     video_base_dir: ${runner.logger.log_path}/video/eval   # 评估视频

评估结果
~~~~~~~~~~~~~~~~~~~

.. list-table:: **OpenVLA-OFT 模型在六个 RoboTwin 任务上的评估结果**
   :header-rows: 1

   * - 模型
     - beat_block_hammer
     - pick_dual_bottles
     - place_empty_cup
     - move_can_pot
     - lift_pot
     - handover_block
     - Average
     - Δ Avg.

   * - OpenVLA-OFT (SFT)
     - |huggingface| `10.15% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`_
     - |huggingface| `20.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`_
     - |huggingface| `75.78% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`_
     - |huggingface| `9.37% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`_
     - |huggingface| `3.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`_
     - |huggingface| `28.13% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`_
     - 24.48%
     - ---

   * - OpenVLA-OFT (RLinf-GRPO)
     - |huggingface| `96.09% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`__
     - |huggingface| `92.96% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`__
     - |huggingface| `94.53% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`__
     - |huggingface| `83.59% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`__
     - |huggingface| `70.31% <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`__
     - **84.63%**
     - **+60.15%**

.. note::
   
   每个任务都有其对应的独立 SFT 和 RL 模型。 
   SFT 模型：|huggingface| `beat_block_hammer (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer>`_、|huggingface| `pick_dual_bottles (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles>`_、|huggingface| `place_empty_cup (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup>`_、|huggingface| `move_can_pot (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot>`_、|huggingface| `lift_pot (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot>`_、|huggingface| `handover_block (SFT) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block>`_。 
   RL 模型：|huggingface| `beat_block_hammer (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer>`_、|huggingface| `pick_dual_bottles (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles>`_、|huggingface| `place_empty_cup (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup>`_、|huggingface| `move_can_pot (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot>`_、|huggingface| `lift_pot (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot>`_、|huggingface| `handover_block (RLinf-GRPO) <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block>`_。

配置说明
-----------------------

**关键配置参数**

1. **模型配置**：

   - ``actor.model.model_type: "openvla_oft"``：使用 OpenVLA-OFT 模型
   - ``actor.model.implement_version: "official"``：使用 OpenVLA-OFT 官方版本
   - ``actor.model.action_dim: 14``：14 维动作空间（包含本体感觉）
   - ``actor.model.use_proprio: True``：启用本体感觉输入
   - ``actor.model.proprio_dim: 14``：本体感觉维度
   - ``actor.model.num_action_chunks: 25``：动作块数量

2. **算法配置**：

   - ``algorithm.reward_type: chunk_level``：chunk 级别的奖励
   - ``algorithm.logprob_type: token_level``：token 级别的对数概率
   - ``algorithm.n_chunk_steps: 8``：每个 chunk 的步数

3. **环境配置**：

   - ``env.train.task_config.task_name``：任务名称（如 ``place_empty_cup``）
   - ``env.train.task_config.embodiment``：机器人配置
   - ``env.train.task_config.camera``：相机配置

更多关于 RoboTwin 配置的详细信息，请参考 `RoboTwin 配置文档 <https://robotwin-platform.github.io/doc/usage/configurations.html>`_。

注意事项
-----------------------

1. **资源路径**：确保 ``assets_path`` 路径正确
2. **ROBOT_PLATFORM 环境变量**：确保 ``ROBOT_PLATFORM`` 变量设置为 ``ALOHA``
3. **RoboTwin Repo**：确保正确设置 ``ROBOTWIN_PATH``，如 ``export ROBOTWIN_PATH=/path/to/RoboTwin``
4. **GPU 内存**：RoboTwin 环境可能需要较多 GPU 内存，建议使用 ``enable_offload: True``
5. **任务配置**：根据具体任务修改 ``task_config`` 中的参数

