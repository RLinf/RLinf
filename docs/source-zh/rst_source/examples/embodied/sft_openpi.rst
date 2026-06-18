OpenPI 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ 视觉-语言-动作模型。

使用 RLinf 对 OpenPI（π₀ / π₀.₅）模型进行 **全量监督微调（Full-parameter SFT）** 或
**LoRA 微调**。SFT 通常作为进入强化学习前的第一阶段：模型先模仿高质量示例，后续强化学习才能在良好先验上继续优化。

概览
----------------------------------------

在 LeRobot 格式数据集上微调 π₀ / π₀.₅——全量或 LoRA——可在单机或多节点集群上进行。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: 方法
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: 数据
      :text-align: center

      LeRobot format

   .. grid-item-card:: 硬件
      :text-align: center

      1+ 节点 · GPU

| **你将完成：** 安装 OpenPI → 准备 LeRobot 数据集 → 计算归一化统计 → 启动 ``run_vla_sft.sh`` → 观察训练损失。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 LeRobot 格式的数据集。

支持的数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 支持 LeRobot 格式的数据集，通过 ``config_name`` 字段指定。内置格式如下：

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - ``config_name``
     - 数据集 / 环境
   * - ``pi0_maniskill`` · ``pi05_maniskill``
     - ManiSkill
   * - ``pi0_libero`` · ``pi05_libero``
     - LIBERO
   * - ``pi0_aloha_robotwin``
     - RoboTwin（ALOHA）
   * - ``pi0_realworld``
     - 真机 Franka
   * - ``pi05_metaworld``
     - MetaWorld
   * - ``pi05_calvin``
     - CALVIN

自定义数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

也可通过自定义 LeRobot 数据集格式来训练特定数据集，具体可参考以下文件：

1. 在 ``examples/sft/config/custom_sft_openpi.yaml`` 中，指定数据格式。

.. code:: yaml

  model:
    openpi:
      config_name: "pi0_custom"

2. 在 ``rlinf/models/embodiment/openpi/__init__.py`` 中，注册数据格式 ``pi0_custom``。

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

3. 在 ``rlinf/models/embodiment/openpi/dataconfig/custom_dataconfig.py`` 中，定义自定义数据集的配置。

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

归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当你在新采集的 LeRobot 数据集上训练 OpenPI 时，需要在启动 SFT 之前先计算
归一化统计。这对真实机器人采集的数据集尤其重要。

RLinf 提供了 ``toolkits/lerobot/calculate_norm_stats.py``，用于为
``state`` 和 ``actions`` 计算 ``norm_stats``。使用方式如下：

.. code:: bash

   # 本地数据集目录（包含 meta/info.json）：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # 或使用默认缓存在 ~/.cache/huggingface/lerobot 下的 Hugging Face repo id：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

.. note::

   - ``--repo-id`` 可以是本地数据集路径，也可以是 LeRobot 的 Hugging Face repo id。
   - 可选：通过 ``HF_LEROBOT_HOME`` 修改 repo id 的缓存父目录（默认：``~/.cache/huggingface/lerobot``）。
   - ``config_name`` 必须与训练时使用的自定义 OpenPI dataconfig 一致。

该脚本会将生成的统计信息写入 ``<assets_dir>/<exp_name>/<repo_id>/norm_stats.json``。
OpenPI 加载器会在运行时从 ``<model_path>/<repo_id>`` 读取归一化统计信息。

另一个有助于稳定训练的实用建议是，手动检查归一化统计中是否存在非常小的标准差，
或过窄的 q99-q01 区间。适当增大标准差，或拉宽 q99-q01 的范围，通常有助于提升
训练稳定性，尤其是在先做 SFT 再进入在线训练的两阶段流程中。

安装
----------------------------------------

.. include:: _setup_common.rst

**方式一：使用 Docker 镜像** —— 镜像标签 ``agentic-rlinf0.2-maniskill_libero``：

.. code:: bash

    docker run -it --rm --gpus all \
        --shm-size 20g \
        --network host \
        --name rlinf \
        -v .:/workspace/RLinf \
        rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
        # 国内镜像加速：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

    # 进入容器后，切换到 OpenPI 虚拟环境：
    source switch_env openpi

**方式二：自建环境** —— 安装套件 ``--env maniskill_libero``：

.. code:: bash

    # 为提高国内依赖安装速度，可以添加 --use-mirror。
    bash requirements/install.sh embodied --model openpi --env maniskill_libero
    source .venv/bin/activate

运行
----------------------------------------

**1. 配置**

完整示例配置位于：

- ``examples/sft/config/libero_sft_openpi.yaml``
- ``examples/sft/config/realworld_sft_openpi.yaml``

通用的 OpenPI SFT 配置示例如下：

.. code:: yaml

    cluster:
        num_nodes: 1                 # 节点数
        component_placement:         # 组件 → GPU 映射
            actor: 0-3

若需要 LoRA 微调，将 ``actor.model.is_lora`` 设为 ``True``，并配置 ``actor.model.lora_rank``：

.. code:: yaml

    actor:
        model:
            is_lora: True
            lora_rank: 32

**2. 启动**

先启动 Ray 集群，再执行训练脚本：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi

同一脚本也适用于通用文本 SFT，只需替换配置文件即可。

可视化与结果
----------------------------------------

关注 **训练损失** 即可确认模型是否在拟合示例数据。各项指标的含义见
:doc:`训练指标 <../../reference/metrics>`。

.. code-block:: bash

   # 启动 TensorBoard
   tensorboard --logdir ./logs
