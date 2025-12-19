监督微调训练
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中进行 **全量监督微调（Full-parameter SFT）** 和 **LoRA 微调**。SFT 通常作为进入强化学习前的第一阶段：模型先模仿高质量示例，后续强化学习才能在良好先验上继续优化。

内容包括
--------

- 如何在 RLinf 中配置通用全量监督微调 和 LoRA微调
- 如何在单机或多节点集群上启动训练
- 如何监控与评估结果


支持的数据集
------------------

RLinf 目前支持 LeRobot 格式的数据集，可以通过 **config_type** 指定不同的数据集类型。

目前支持的数据格式包括：

- pi0_maniskill
- pi0_libero
- pi05_libero
- pi05_maniskill
- pi05_metaworld
- pi05_calvin

也可通过自定义数据集格式来训练特定数据集，具体可参考以下文件

1. 在``examples/sft/config/custom_sft_openpi.yaml``中，指定数据格。

.. code:: yaml

    model:
    openpi:
        config_name: "pi0_custom"

2. 在``rlinf/models/embodiment/openpi/__init__.py``中，指定数据格式为 ``pi0_custom``。

.. code:: python

    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=True,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,  # User can add extra config in custom dataset
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),

3. 在``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py``中，定义自定义数据集的配置。

.. code:: python

    class CustomDataConfig(DataConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.repo_id = "physical-intelligence/custom_dataset"
            self.base_config = DataConfig(
                prompt_from_task=True
            )
            self.assets = AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets")
            self.extra_delta_transform = True
            self.action_train_with_rotation_6d = False


训练配置
-------------

完整示例配置位于 ``examples/sft/config/libero_sft_openpi.yaml``，核心字段如下：

.. code:: yaml

    cluster:
        num_nodes: 1                 # 节点数
        component_placement:         # 组件 → GPU 映射
            actor: 0-3

若需要支持LoRA微调，需要将``actor.model.is_lora``设置为True，并配置``actor.model.lora_rank``参数。

.. code:: yaml

    actor:
        model:
            is_lora: True
            lora_rank: 32

启动脚本
-------------

先启动 Ray 集群，然后执行辅助脚本：

.. code:: bash

   cd /path_to_RLinf/ray_utils
   bash start_ray.sh                 # 启动 head + workers

   # 回到仓库根目录
   bash examples/sft/train_embodied_sft.py --config libero_sft_openpi.yaml

同一脚本也适用于通用文本 SFT，只需替换配置文件。


