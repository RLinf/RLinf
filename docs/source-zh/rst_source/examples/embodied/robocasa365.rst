基于RoboCasa365 Benchmark的强化学习
====================================

本文档介绍 RLinf 中面向 benchmark 的 RoboCasa365 集成方式。它不会覆盖原有
:doc:`RoboCasa <robocasa>` 配方，而是新增独立的 ``robocasa365`` 环境目录与
配置，使用 RoboCasa 官方 dataset registry 选择任务。

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

   bash requirements/install.sh embodied --model openpi --env robocasa
   source .venv/bin/activate

3. 下载 RoboCasa 资源
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets

数据集任务选择
--------------

RLinf 现在将任务选择委托给 RoboCasa 官方 registry。内部镜像的是如下调用方式：

.. code:: python

   from robocasa.utils.dataset_registry import get_ds_soup

   task_names = get_ds_soup(
       task_soup="atomic_seen",
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

RoboCasa365 配方新增了独立 OpenPI config 名称 ``pi0_robocasa365``，但刻意复用
RoboCasa 的模态变换路径。请在 YAML 中自行提供 Pi0 checkpoint：

.. code:: yaml

   rollout:
     model:
       model_path: "/path/to/model/pi0_robocasa365"

   actor:
     model:
       model_path: "/path/to/model/pi0_robocasa365"

训练
----

对应的 benchmark 配方：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh robocasa365_grpo_openpi

该配置默认在以下切片上训练：

- ``env.train.split=pretrain``
- ``env.train.task_soup=atomic_seen``

评测
----

对应的 benchmark 评测配方：

.. code:: bash

   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi

该配置默认在以下切片上评测：

- ``env.eval.split=target``
- ``env.eval.task_soup=atomic_seen``

如果要切换到其他 benchmark 切片，直接修改 YAML：

.. code:: yaml

   env:
     eval:
       split: target
       task_soup: composite_unseen
       task_mode: composite

数据导出说明
------------

启用数据采集时，RLinf 现在会将 RoboCasa365 的任务元数据保留到 LeRobot 导出中：

- canonical task string
- benchmark selection
- split
- task soup
- task mode
- dataset source

这些信息会写入 ``meta/tasks.jsonl`` 和 ``meta/episodes.jsonl``，方便后续离线分析
区分 benchmark 子集。

备注
----

- 旧版 :doc:`RoboCasa <robocasa>` 页面和 ``robocasa`` 环境保持不变。
- ``robocasa365`` 是单独的 env type 和文件夹，用于保证旧 recipe 稳定。
- 第一版仅面向 OpenPI / Pi0；后续可以在同一环境层上继续补其他模型配方。
