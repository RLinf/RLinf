基于RoboCasa365 Benchmark的强化学习
====================================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

本文档介绍 RLinf 中面向 benchmark 的 RoboCasa365 集成方式。它不会覆盖原有
:doc:`RoboCasa <robocasa>` 配方，而是新增独立的 ``robocasa365`` 环境目录与
配置，使用 RoboCasa 官方 dataset registry 选择任务。

主要目标是在 RoboCasa365 官方 benchmark split 上训练和评测视觉-语言-动作模型，
同时保持旧版 RoboCasa recipe 稳定。相比单任务 RoboCasa 示例，RoboCasa365 配方
更关注：

1. **Benchmark split 控制**：通过官方 registry 选择 ``pretrain`` 或 ``target`` 任务。
2. **Task-soup 评测**：评测 ``atomic_seen``、``composite_unseen`` 等 benchmark 切片。
3. **运行时任务元信息**：在环境观测中暴露 split、task-soup 等任务元信息。

环境说明
--------

**RoboCasa365 Benchmark**

- **环境**: RoboCasa365 厨房操作 benchmark
- **任务选择方式**: 使用 RoboCasa 官方 ``split`` + ``task_soup`` registry
- **机器人**: 默认使用移动底座 Panda 配置 ``PandaMobile``
- **观测**: 多视角 RGB 图像 + 可配置的 proprioceptive state 提取
- **动作空间**: 可配置的 RoboCasa 移动操作 action schema

默认 RLinf 配方使用：

- 训练阶段 ``split=pretrain``
- 评测阶段 ``split=target``
- 首个基准切片 ``task_soup=atomic_seen``

如果需要切换到 ``composite_seen`` 或 ``composite_unseen``，直接修改 YAML 即可。

**观测结构**

- **主相机图像**：默认使用 ``robot0_agentview_left_image``
- **腕部相机图像**：默认使用 ``robot0_eye_in_hand_image``
- **本体感知状态**：通过 ``observation.state_layout`` 中配置的 key 组装状态向量

**动作结构**

默认 OpenPI 配方使用 12 维动作 schema。wrapper 可以在 env stepping 前关闭底座控制，
并映射 OpenPI 输出中的有效动作 slice。

配置文件
--------

RoboCasa365 的环境配置位于：

.. code:: bash

   examples/embodiment/config/env/robocasa365.yaml

关键字段包括：

- ``task_source``: RoboCasa365 应保持为 ``dataset_registry``
- ``dataset_source``: RoboCasa 注册的数据来源，通常为 ``human``
- ``split``: benchmark split，例如 ``pretrain`` 或 ``target``
- ``task_soup``: 官方 task soup 名称，例如 ``atomic_seen``
- ``task_filter``: 可选的 include / exclude 过滤器
- ``task_mode``: 可选的 ``atomic`` 或 ``composite`` 保护字段
- ``observation``: RLinf 使用的相机 key 和 state-layout 映射
- ``action_space``: env stepping 前的动作 schema 与 OpenPI slice 映射

依赖安装
--------

1. 克隆 RLinf
~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

使用 RoboCasa 对应依赖集：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env robocasa365
   source .venv/bin/activate

3. 下载 RoboCasa 资源
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets

**注意：**  由于 NVIDIA 的 HuggingFace 数据已将资源拆成 ``fixtures_lightwheel/`` 和 ``objects_lightwheel/`` 下的多个 zip 文件，RoboCasa 下载脚本无法自动处理。此时需要手动下载并解压这些 zip
到对应的 RoboCasa assets 目录。可以先用以下命令定位 RoboCasa assets 根目录：

.. code:: bash

   python -c "import os, robocasa; print(os.path.join(robocasa.__path__[0], 'models', 'assets'))"

然后按如下路径解压放置：

- ``fixtures_lightwheel/*.zip`` → ``<robocasa_assets>/fixtures/``
- ``objects_lightwheel/*.zip`` → ``<robocasa_assets>/objects/lightwheel/``

数据集任务选择
--------------

RLinf 现在将任务选择委托给 RoboCasa 官方 registry。内部镜像的是如下调用方式：

.. code:: python

   from robocasa.utils.dataset_registry import get_ds_soup

   task_names = get_ds_soup(
       task_set="atomic_seen",
       split="target",
       source="human",
   )

参考文档：

- RoboCasa 数据使用文档：
  https://robocasa.ai/docs/build/html/datasets/using_datasets.html
- RoboCasa dataset-registry API：
  https://robocasa.ai/docs/build/html/modules/robocasa.utils.dataset_registry.html

模型权重
--------

RoboCasa365 配置新增了独立 OpenPI config 名称 ``pi0_robocasa365``，复用 RoboCasa 的模态变换路径。官方 RoboCasa365 Pi0 checkpoint 位于：
https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi0/pi0_robocasa_pretrain_human300

请先将 checkpoint 下载到本地，再在 YAML 中将 rollout 和 actor 的模型路径都指向该目录：

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/pi0_robocasa_pretrain_human300"

   actor:
     model:
       model_path: "/path/to/pi0_robocasa_pretrain_human300"

训练
----

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robocasa365_grpo_openpi

该配置默认在以下切片上训练：

- ``env.train.split=pretrain``
- ``env.train.task_soup=atomic_seen``
- ``env.train.task_mode=atomic``

评测
----

轻量评测配置为 ``robocasa365_eval_openpi``：

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi

该配置默认在以下切片上评测：

- ``env.eval.split=target``
- ``env.eval.task_soup=atomic_seen``
- ``env.eval.task_mode=atomic``

如果要切换到其他 benchmark 切片，可以直接覆盖 YAML 字段：

.. code:: yaml

   env:
     eval:
       split: target
       task_soup: composite_unseen
       task_mode: composite

例如：

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi \
      env.eval.task_soup=composite_unseen \
      env.eval.task_mode=composite
