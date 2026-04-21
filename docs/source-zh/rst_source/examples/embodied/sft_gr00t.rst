GR00T-N1.6监督微调训练
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中对GR00T-N1.6模型进行 **监督微调（SFT）**。SFT 通常作为进入强化学习前的第一阶段, 通过在离线数据集上对模型进行微调，使其更好地适应下游任务的分布。对于 GR00T-N1.6 模型，我们提供了针对不同数据集的示例配置，帮助用户快速上手。后面将会加上针对 **LoRA 微调** 的说明。

内容包括
--------

- 如何在 RLinf 中配置通用监督微调
- 如何在单机或多节点集群上启动训练
- 如何监控与评估结果


支持的数据集
------------------

RLinf 目前支持 LeRobot 格式的数据集，

目前支持的数据格式包括：

- gr00t_n1.6_libero


训练配置
-------------

完整示例配置位于：

- ``examples/sft/config/libero_sft_gr00t_16.yaml``

通用的 GR00T-N1.6 SFT 配置示例如下：

1. 集群配置
~~~~~~~~~~~~~~~~~~~
.. code:: yaml

    cluster:
       num_nodes: 1     #节点数
       hardware_ranks: [[0]]
       component_placement:    # 组件 → GPU 映射
         actor: all

2. 模型配置
~~~~~~~~~~~~~~~~~~~
.. code:: yaml
    model:
        model_path: "/path/to/GR00T-N1.6-3B" # 修改为GR00T-N1.6模型的实际路径

        model_type: "gr00t_1_6_sft"
        precision: "bf16"
        action_dim: 128
        num_action_chunks: 1
        add_value_head: True
        denoising_steps: 1

3, 数据集配置
~~~~~~~~~~~~~~~~~~~
.. code:: yaml
    data:
    train_data_paths: "/path/to/your/libero_spatial_dataset" # 修改为你的训练数据路径


依赖安装
-----------------------

本节介绍 OpenPI 模型进行 SFT 训练所需的依赖环境。对于其他模型，请参考各自示例文档中的「依赖安装」小节。

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # 为提高国内下载速度，可以使用：
    # git clone https://ghfast.top/github.com/RLinf/RLinf.git
    git clone https://github.com/RLinf/RLinf.git
    cd RLinf

2. 安装依赖
~~~~~~~~~~~~~~~~

**方式一：使用 Docker 镜像**

推荐直接使用预构建的 Docker 镜像运行实验。

.. code:: bash

    docker run -it --rm --gpus all \
        --shm-size 20g \
        --network host \
        --name rlinf \
        -v .:/workspace/RLinf \
        rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
        # 如果需要国内加速下载镜像，可以使用：
        # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

进入容器后，请通过内置的 `switch_env` 工具切换到对应的虚拟环境：

.. code:: bash

    source switch_env gr00t_16

**方式二：自建环境**

也可以在本地/集群环境中直接安装依赖，示例命令如下：

.. code:: bash

    # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

    bash requirements/install.sh embodied --model gr00t_16 --env maniskill_libero
    source .venv/bin/activate

启动脚本
-------------

执行训练脚本：

.. code:: bash

   # return to repo root
   bash examples/sft/run_vla_sft.sh libero_sft_gr00t_16

LeRobot SFT模型格式转换
-----------------------

为了在 RLinf 中使用监督微调后的模型进行强化学习训练，需要将 SFT 模型转换为 GR00T-N.6的 格式。以下是转换步骤：

1. 配置路径
~~~~~~~~~~~~~~~~~~~
- 文件路径在 ``rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py``
.. code:: python
    base_model_path = "/path/to/GR00T-N1.6-3B"  ## 修改为GR00T-N1.6模型的实际路径
    sft_pt_path = "/path/to/logs/**/gr00t_16_sft_libero/checkpoints/**/actor/model_state_dict/full_weights.pt" ## 修改为SFT后的模型权重路径
    hf_output_path = "/path/to/output/GR00T-1.6-SFT-LIBERO-HF" ## 需要使用输出路径

.. code:: bash
    python rlinf/models/embodiment/gr00t_1_6/convert_to_hf.py

微调效果SFT
-----------------------

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://github.com/yangzhongii/misc/blob/main/pic/gr00t_1.6_sft_loss.png" style="width: 100%;"/>
       <p><em>GR00T-N1.6 SFT 在LIBERO_Spatial的损失函数</em></p>
     </div>
   </div>



