快速上手 1：在 Maniskill3 上使用 PPO 训练 VLA 模型
==================================================================================================

本快速教程将带你使用 **RLinf** 框架，在  
`ManiSkill3 <https://github.com/haosulab/ManiSkill>`_ 环境中训练视觉-语言-动作模型（VLA），包括  
`OpenVLA <https://github.com/openvla/openvla>`_。

为简化流程，你可以在单卡 GPU 上直接运行以下脚本完成训练。

环境简介
--------------------------

ManiSkill3 是一个基于 GPU 加速的机器人研究仿真平台，  
专注于复杂接触操作和具身智能任务。  
该基准涵盖多个领域，包括机械臂、移动操作器、人形机器人以及灵巧手，  
支持抓取、组装、绘图、移动等多种任务。

我们还针对 GPU 仿真器进行了系统级优化（详见 :doc:`../tutorials/mode/hybrid`）。

启动训练
--------------------------

**步骤 1：下载预训练模型**

若使用 **OpenVLA** 模型，请运行以下命令：

.. code-block:: bash

   # 下载 OpenVLA 预训练模型
   hf download gen-robot/openvla-7b-rlvla-warmup \
   --local-dir /path/to/model/openvla-7b-rlvla-warmup/

该模型已在论文中引用：`paper <https://arxiv.org/abs/2505.19789>`_

若使用 **OpenVLA-OFT** 模型，请运行以下命令：

.. code-block:: bash

   # 下载 OpenVLA-OFT 预训练模型
   hf download RLinf/Openvla-oft-SFT-libero10-trajall \
   --local-dir /path/to/model/Openvla-oft-SFT-libero10-trajall/
   
   # 下载在maniskill上lora微调过的检查点
   hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora \
   --local-dir /path/to/model/oft-sft/lora_004000

**步骤 2：运行官方提供的训练脚本**

.. note::
   如果你是通过 Docker 镜像安装的 **RLinf** （见 :doc:`./installation`），请确保已切换到目标模型对应的 Python 环境。
   默认环境为 ``openvla``。
   若使用 OpenVLA-OFT 或 openpi，请使用内置脚本 `switch_env` 切换环境：
   ``source switch_env openvla-oft`` 或 ``source switch_env openpi``。

   如果你是通过自定义环境安装的 **RLinf**，请确保已安装对应模型的依赖，详见 :doc:`./installation`。

为方便使用，我们提供的配置文件需要至少双卡进行训练。  
如果你有多张 GPU 并希望加快训练速度，  
建议你修改配置文件  
``./examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml`` 中的参数  
``cluster.component_placement``。

你可以根据实际资源将该项设置为 **0-3** 或 **0-7** 来使用 4/8 张 GPU。
查看 :doc:`../tutorials/user/yaml` 以获取有关 Placement 配置的更详细说明。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
        actor,rollout: 0-1

最后，在运行脚本之前，你需要根据模型和数据集的下载路径，修改 YAML 文件中的相应配置项。具体而言，对于 **OpenVLA**，请将以下配置更新为 `gen-robot/openvla-7b-rlvla-warmup` 检查点所在的路径。

- ``rollout.model_dir``  
- ``actor.checkpoint_load_path``  
- ``actor.tokenizer.tokenizer_model``  

对于 **OpenVLA-OFT**，请将下列配置项设置为 `RLinf/Openvla-oft-SFT-libero10-trajall` 检查点所在的路径。同时，将 LoRA 路径设置为 `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora` 检查点所在的路径。

- ``rollout.model_dir``  
- ``actor.checkpoint_load_path``  
- ``actor.tokenizer.tokenizer_model``  
- ``actor.model.lora_path``
- ``actor.model.is_lora: True``

完成上述修改后，运行以下命令启动训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

**步骤 3：查看训练结果**

- 最终模型与指标保存路径：``../results``  
- TensorBoard 可视化日志路径：``../results/tensorboard``  
  启动方式如下：

  .. code-block:: bash

     tensorboard --logdir ../results/tensorboard/ --port 6006

打开 TensorBoard 后，你会看到类似下图的界面。  
建议重点关注以下指标：

- ``rollout/env_info/return``  
- ``rollout/env_info/success_once``  

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/embody-quickstart-metric.jpg" width="800"/>
