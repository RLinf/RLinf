基于 Habitat 基准的强化学习训练
================================

本文提供了在 RLinf 框架内启动和管理视觉-语言-导航模型（VLN）训练任务的完整指南，重点介绍如何在 `Habitat <https://aihabitat.org/>`_ 环境中微调 VLN 模型。

核心目标是训练一个能够执行机器人导航任务的模型，具体包括：

1. **视觉理解**：处理机器人相机采集的 RGB 图像。
2. **语言理解**：解析自然语言任务描述。
3. **动作生成**：输出精确的机器人动作（导航控制）。
4. **强化学习**：通过环境反馈，使用 PPO 优化策略。

环境
----

**Habitat 环境**

- **环境**：基于 Habitat-Sim 的高保真导航仿真器
- **任务**：视觉-语言导航（VLN）
- **智能体**：在 Matterport3D 场景中导航的具身智能体
- **观测**：来自机载相机的第一视角 RGB 图像
- **动作空间**：离散导航动作，包括 ``[move_forward, turn_left, turn_right, stop]``

**数据结构**

- **图像**：形状为 ``[batch_size, H, W, 3]`` 的 RGB 观测
- **任务描述**：自然语言导航指令
- **动作**：离散导航动作
- **指标**：成功率、SPL、轨迹长度、导航误差

依赖安装
--------

在你的环境中执行以下命令安装依赖：

.. code:: bash

   # 中国大陆用户可在 install.sh 命令中添加 `--use-mirror` 以提升下载速度。

   bash requirements/install.sh embodied --env habitat
   source .venv/bin/activate

   # NaVid 需要 transformers==4.31.0
   uv pip install transformers==4.31.0


VLN-CE 数据集准备
-----------------

下载场景数据集：

- 对于 **R2R**、**RxR**：从 `Matterport3D <https://niessner.github.io/Matterport>`_
  官网下载 MP3D 场景，并放入 ``VLN-CE/scene_dataset`` 目录。

下载 VLN-CE episodes：

- `r2r <https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view>`_
  （将 ``R2R_VLNCE_v1-3_preprocessed/`` 重命名为 ``r2r/``）
- `rxr <https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view>`_
  （将 ``RxR_VLNCE_v0/`` 重命名为 ``rxr/``）

将它们放入 ``VLN-CE/datasets`` 目录。

数据集目录结构：

.. code:: bash

   VLN-CE
   |-- datasets
   |   |-- r2r
   |   |-- rxr
   `-- scene_dataset
       |-- mp3d

模型下载
--------

开始训练前，需要下载对应的预训练模型：

.. code:: bash

   # 下载 CMA 模型
   # 该链接包含 CMA 模型的 ckpt。
   gdown https://drive.google.com/uc?id=1o9PgBT38BH9pw_7V1QB3XUkY8auJqGKw

   # 该链接包含 depth encoder 的预训练权重。
   wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip

   # 该链接包含预处理后的数据与 instruction encoder embedding。
   wget https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view

   # rgb encoder 的预训练权重来自官方 resnet50。
   # 运行脚本时会自动下载该权重。
   # 代码位置：RLinf/rlinf/models/embodiment/cma/modules/resnet_encoders.py:186

将上述文件放入 ``VLN-CE/models/cma_weights`` 目录。然后使用以下脚本将 checkpoint 转换为 ``best_ckpt_r2r.pth``：

.. code:: python

   import torch
   import pickle

   CKPT_PATH = "/path/to/your/checkpoint.pth"
   OUTPUT_PATH = "/path/to/output/best_ckpt_r2r.pth"

   class Placeholder:
      def __init__(self, *args, **kwargs):
         self.__dict__ = {}

      def __setitem__(self, k, v):
         self.__dict__[k] = v

      def __setstate__(self, state):
         self.__dict__ = state if isinstance(state, dict) else {}

   class SafeUnpickler(pickle.Unpickler):
      def find_class(self, module, name):
         if 'habitat' in module.lower() or 'vlnce' in module.lower():
               return Placeholder
         try:
               return super().find_class(module, name)
         except:
               return Placeholder

   class SafePickleModule:
      Unpickler = SafeUnpickler

   SafePickleModule.__name__ = 'pickle'

   checkpoint = torch.load(CKPT_PATH, map_location="cpu", pickle_module=SafePickleModule)
   state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint.get("model_state_dict") or checkpoint
   torch.save(state_dict, OUTPUT_PATH)

   print(f"successfully saved to {OUTPUT_PATH}")

数据集目录结构：

.. code:: bash

   VLN-CE
   |-- datasets
   |   |-- r2r
   |   |-- rxr
   |-- models
   |   |-- cma_weights
   |   |   |-- ckpt
   |   |   |-- ddppo-models
   |   |   |-- instruction_encoder_embedding
   `-- scene_dataset
       |-- mp3d

运行前，请确认 yaml 中的数据集路径和模型路径设置正确。

.. code:: yaml

   env:
      data_path_dir: "VLN-CE/datasets/r2r"
      scenes_dir: "VLN-CE/scene_dataset"

   rollout:
      model:
         model_path: "VLN-CE/models/cma_weights/ckpt/best_ckpt_r2r.pth"
         instruction_encoder_config:
            embedding_file: "VLN-CE/models/cma_weights/instruction_encoder_embedding/embeddings.json.gz"
         depth_encoder_config:
            ddppo_checkpoint: "VLN-CE/models/cma_weights/ddppo-models/gibson-2plus-resnet50.pth"
   actor:
      model:
         model_path: "VLN-CE/models/cma_weights/ckpt/best_ckpt_r2r.pth"
         instruction_encoder_config:
            embedding_file: "VLN-CE/models/cma_weights/instruction_encoder_embedding/embeddings.json.gz"
         depth_encoder_config:
            ddppo_checkpoint: "VLN-CE/models/cma_weights/ddppo-models/gibson-2plus-resnet50.pth"


运行脚本
--------

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

你可以灵活配置 env、rollout、actor 组件使用的 GPU 数量。
此外，在配置中设置 ``pipeline_stage_num = 2`` 后，
可实现 rollout 与 env 的流水线重叠，从而提升 rollout 效率。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env,rollout,actor: all

你也可以将布局改为完全共享，
即 env、rollout、actor 组件共享所有 GPU。

.. code:: yaml

   cluster:
      num_nodes: 1
      component_placement:
         env: 0-1
         rollout: 2-5
         actor: 6-7

你还可以将布局改为完全分离，
即 env、rollout、actor 各自独占 GPU，互不干扰，
从而无需使用 offloading 功能。


**2. 配置文件**

以 Habitat R2R 为例：

- CMA + GRPO:
  ``examples/embodiment/config/habitat_r2r_grpo_cma.yaml``

- CMA + PPO:
  ``examples/embodiment/config/habitat_r2r_ppo_cma.yaml``

**3. 启动命令**

在 Habitat 环境中评估 CMA 模型：

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh habitat_r2r_grpo_cma

在 Habitat 环境中训练 CMA 模型：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh habitat_r2r_ppo_cma
