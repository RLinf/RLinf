监督微调训练
=======================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍如何在 RLinf 框架中对OpenPI模型进行 **全量监督微调（Full-parameter SFT）** 和 **LoRA 微调**。SFT 通常作为进入强化学习前的第一阶段：模型先模仿高质量示例，后续强化学习才能在良好先验上继续优化。

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
- pi0_aloha_robotwin
- pi0_franka_dagger
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

新 LeRobot 数据集的归一化统计
-----------------------------

当你在新采集的 LeRobot 数据集上训练 OpenPI 时，需要在启动 SFT 之前先计算
归一化统计。这对真实机器人采集的数据集尤其重要。

RLinf 提供了 ``toolkits/replay_buffer/calculate_norm_stats.py``，用于为
``state`` 和 ``actions`` 计算 ``norm_stats``。使用方式如下：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/replay_buffer/calculate_norm_stats.py \
       --config-name pi0_franka_dagger \
       --repo-id franka_dagger

注意事项：

- 运行脚本前必须先设置 ``HF_LEROBOT_HOME``。
- ``config_name`` 必须与训练时使用的自定义 OpenPI dataconfig 一致。
- ``repo_id`` 必须与你的 LeRobot 格式数据集名称一致。

该脚本会将生成的统计信息写入
``<assest_dir>/<exp_name>/<repo_id>/norm_stats.json``。

OpenPI 加载器会在运行时从 ``<model_path>/<repo_id>`` 读取归一化统计信息。

另一个有助于稳定训练的实用建议是，手动检查归一化统计中是否存在非常小的标准差，
或过窄的 q99-q01 区间。适当增大标准差，或拉宽 q99-q01 的范围，通常有助于提升
训练稳定性，尤其是在先做 SFT 再进入在线训练的两阶段流程中。


训练配置
-------------

完整示例配置位于：

- ``examples/sft/config/libero_sft_openpi.yaml``
- ``examples/sft/config/franka_dagger_sft_openpi.yaml``

通用的 OpenPI SFT 配置示例如下：

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

    source switch_env openpi

**方式二：自建环境**

也可以在本地/集群环境中直接安装依赖，示例命令如下：

.. code:: bash

    # 为提高国内依赖安装速度，可以添加`--use-mirror`到下面的install.sh命令

    bash requirements/install.sh embodied --model openpi --env maniskill_libero
    source .venv/bin/activate

启动脚本
-------------

执行训练脚本：

.. code:: bash

   # return to repo root
   bash examples/sft/run_vla_sft.sh libero_sft_openpi

真机 SFT 环境与部署
---------------------

RLinf 提供了一个 **通用 SFT 环境** （``FrankaEnv-v1``），允许你完全通过 YAML
配置来定义新的真机任务，无需编写自定义环境类。适用于：

- 在新任务上采集 SFT 示教数据
- 在真机上部署（评估）已训练的策略

通用 SFT 环境
~~~~~~~~~~~~~~~

环境配置模板位于
``examples/embodiment/config/env/realworld_franka_sft_env.yaml``。
你需要根据自己的任务修改以下关键字段：

.. code:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # 目标位姿 [x,y,z,rx,ry,rz]
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # 复位位姿（应高于目标）
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]  # 成功判定容差
     action_scale: [1.0, 1.0, 1.0]                         # [xyz, rpy, 夹爪]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

底层实现上，``FrankaEnv`` 现在接受 ``override_cfg`` 字典，并使用类变量
``CONFIG_CLS`` 来实例化数据类配置（默认为 ``FrankaRobotConfig``）。
``PegInsertionEnv`` 和 ``BottleEnv`` 等子类通过覆盖 ``CONFIG_CLS``
来使用各自的数据类，同时共享相同的构造函数。

真机评估 / 部署
~~~~~~~~~~~~~~~~~

完整的评估配置位于
``examples/embodiment/config/realworld_eval.yaml``，它将通用 SFT
环境与 Pi0 actor 以 **纯评估模式** （``runner.only_eval: True``）组合使用。

运行前请替换以下占位符：

- ``ROBOT_IP`` — 你的 Franka 机器人 IP 地址。
- ``MODEL_PATH`` — 已训练的模型检查点路径。

然后执行：

.. code:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval

