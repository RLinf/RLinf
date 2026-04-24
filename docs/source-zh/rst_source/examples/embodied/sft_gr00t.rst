GR00T-N1.6 监督微调训练 (SFT)
==================================================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中对 GR00T-N1.6 模型进行 **监督微调 (SFT)**。SFT 通常作为进入强化学习前的第一阶段，通过在离线数据集上对模型进行微调，使其更好地适应下游任务的分布。针对 GR00T-N1.6 模型，我们提供了针对不同数据集的示例配置，帮助用户快速上手。

*注：后续将补充针对 LoRA 微调的专门说明。*

内容包括
------------------------------------------------------------------

- 如何在 RLinf 中配置通用监督微调
- 如何在单机或多节点集群上启动训练
- 如何监控与评估结果

支持的数据集
------------------------------------------------------------------

RLinf 目前支持 LeRobot 格式的数据集。

目前支持的具体数据格式包括：

- ``gr00t_n1.6_libero``

训练配置
------------------------------------------------------------------

完整示例配置位于：
``examples/sft/config/libero_sft_gr00t_16.yaml``

通用的 GR00T-N1.6 SFT 配置示例如下：

1. 集群配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

   cluster:
     num_nodes: 1     # 节点数
     hardware_ranks: [[0]]
     component_placement:    # 组件 → GPU 映射
       actor: all

2. 模型配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

   model:
     model_path: "/path/to/GR00T-N1.6-3B" # 修改为 GR00T-N1.6 模型的实际路径
     model_type: "gr00t_1_6_sft"
     precision: "bf16"
     action_dim: 128
     num_action_chunks: 1
     add_value_head: True
     denoising_steps: 1

3. 数据集配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

   data:
     train_data_paths: "/path/to/your/libero_spatial_dataset" # 修改为你的训练数据路径

依赖安装
------------------------------------------------------------------

本节介绍 GR00T-N1.6 模型进行 SFT 训练所需的依赖环境。

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**方式一：使用 Docker 镜像**

推荐直接使用预构建的 Docker 镜像运行实验。

.. code-block:: bash

   docker run -it --rm --gpus all \
       --shm-size 20g \
       --network host \
       --name rlinf \
       -v .:/workspace/RLinf \
       rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
       # 如果需要国内加速下载镜像，可以使用：
       # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后，请通过内置的 ``switch_env`` 工具切换到对应的虚拟环境：

.. code-block:: bash

   source switch_env gr00t_16

**方式二：自建环境**

.. code-block:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 到下面的 install.sh 命令
   bash requirements/install.sh embodied --model gr00t_16 --env maniskill_libero
   source .venv/bin/activate

模型下载
------------------------------------------------------------------
在开始做SFT训练之前，需要下载对应的数据集以及Gr00T-N1.6的预训练模型，并将它们放置在合适的位置。
目前支持四种libero任务：Saptial, Object, Goal, 10

Libero数据集下载：
.. code-block:: bash
   
   # 方法1：使用git clone
   git lfs install
   git clone https://hf-mirror.com/datasets/ZibinDong/libero_spatial
   # 也可以下载其他数据集 如libero_object, libero_goal, libero_long
   # https://hf-mirror.com/datasets/ZibinDong/libero_object
   # https://hf-mirror.com/datasets/ZibinDong/libero_object
   # https://hf-mirror.com/datasets/ZibinDong/libero_10

   # 方法2：使用huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download ZibinDong/libero_spatial --repo-type dataset --local-dir Gr00t_16-libero-Spatial-dataset
   # 也可以下载其他数据集 如libero_object, libero_goal, libero_10
   # hf download ZibinDong/libero_object --repo-type dataset --local-dir Gr00t_16-libero-Object-dataset
   # hf download ZibinDong/libero_goal --repo-type dataset --local-dir Gr00t_16-libero-Goal-dataset
   # hf download ZibinDong/libero_10 --repo-type dataset --local-dir Gr00t_16-libero-10-dataset

GR00T-N1.6 模型下载
.. code-block:: bash

   # 方法1：使用git clone
   git lfs install
   git clone https://huggingface.co/nvidia/GR00T-N1.6-3B

   # 方法2：使用huggingface-hub
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download nvidia/GR00T-N1.6-3B --repo-type model --local-dir GR00T-N1.6-3B

启动脚本
------------------------------------------------------------------

执行训练脚本：

.. code-block:: bash

   # 返回仓库根目录执行
   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_16

LeRobot SFT 模型格式转换
------------------------------------------------------------------

为了在 RLinf 中使用监督微调后的模型进行强化学习训练，需要将 SFT 模型转换为 GR00T-N1.6 的标准格式。

1. 配置路径
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

脚本位置：``rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py``

.. code-block:: python

   base_model_path = "/path/to/GR00T-N1.6-3B"  ## 修改为 GR00T-N1.6 模型的实际路径
   sft_pt_path = "/path/to/logs/**/gr00t_16_sft_libero/checkpoints/**/actor/model_state_dict/full_weights.pt" ## SFT 后的权重路径
   hf_output_path = "/path/to/output/GR00T-1.6-SFT-LIBERO-HF" ## 指定输出路径

2. 执行转换
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py

微调效果展示 (SFT)
------------------------------------------------------------------

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://github.com/yangzhongii/misc/blob/main/pic/gr00t_1.6_sft_loss.png?raw=true" style="width: 100%;"/>
       <p><em>GR00T-N1.6 SFT 在 LIBERO_Spatial 的损失函数曲线</em></p>
     </div>
   </div>
