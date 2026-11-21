.. 示例页模板：复制本文件，替换所有 <...>，再删除这些注释。规则见
.. .agents/skills/docs/rules/recipe.md 和 rules/chinese.md。章节名、卡片标题和表格列与同一画廊的其他页面保持一致。

基于 <名称> 的强化学习训练
=========================

.. figure:: <raw.githubusercontent.com 上的图片地址>
   :align: center
   :width: 90%

   <这张图在本页展示什么；上游图片注明来源。>

<一段话：<名称> 是什么，这一页用 RLinf 跑通什么，最后得到什么结果。>

概览
----

<一句话：这一页训练或评测什么。>

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      <该模型支持的所有环境>

   .. grid-item-card:: 算法
      :text-align: center

      <所有算法>

   .. grid-item-card:: 任务
      :text-align: center

      <任务集或任务>

   .. grid-item-card:: 硬件
      :text-align: center

      <节点 · GPU · 硬件后端>

| **你将完成：** 安装 → 下载模型 → 启动训练 → 观察 ``<指标>``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · <其他需要准备的内容>。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 环境
     - 任务 / 任务集
     - 配置 / 权重
     - 侧重点
   * - <环境>
     - <任务集>
     - ``<配置名或模型仓库>``
     - <一句话说明测试什么>

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - Observation
     - <policy 看到什么>
   * - Action
     - <policy 输出什么>
   * - Reward
     - <怎样判定成功>
   * - Prompt
     - <任务指令>

安装
----

.. include:: _setup_common.rst

<本示例专用的镜像标签或安装命令，每条命令前说明它做什么。>

下载模型
--------

<下载本示例所用权重的命令。>

.. include:: _model_path.rst

运行
----

<一句话说明这条命令做什么。>

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh <config_name>

<这条命令依次做了什么，以及可以修改哪些配置项。>

可视化与结果
------------

<在哪里观察训练进度，例如 TensorBoard 中的 env/success_once，并链接到训练指标页面。>

<结果表：设置、指标、数值，以及复现所用的配置。>
