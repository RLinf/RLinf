在 ManiSkill 中使用 OGPO 训练
=============================

**OGPO（Off-Policy Generative Policy Optimization）** 是一种面向生成式控制策略的 off-policy 全参数微调方法。它通过 Q 值估计为成组采样的动作构造优势，并使用 PPO 风格的目标优化生成式策略。

当前仓库中的示例面向 ManiSkill 的 ``PickCube-v1`` 任务。

算法细节和原始实现参见 `论文 <https://arxiv.org/abs/2605.03065>`_ 与 `官方 GitHub 仓库 <https://github.com/simchowitzlabpublic/OGPO_public>`_。

概览
----

示例在单张 GPU 上运行同步训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      OGPO

   .. grid-item-card:: 模型
      :text-align: center

      基于状态输入的 flow policy

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill · PickCube

   .. grid-item-card:: 训练
      :text-align: center

      在线

示例配置
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 任务
     - 配置
   * - ``PickCube-v1``
     - ``examples/embodiment/config/maniskill_ogpo_flow_state.yaml``

安装
----

按照 :doc:`ManiSkill <maniskill>` 和 :doc:`安装指南 <../../start/installation>` 准备环境。激活环境后，在 RLinf 仓库根目录执行下文命令。

运行
----

在上述 YAML 文件中，将 ``runner.ckpt_path`` 改为实际的 BC checkpoint 路径，确保权重与配置中的模型兼容：

.. code-block:: yaml

   runner:
     ckpt_path: /path/to/your/ogpo_bc.pt

通过 ``cluster.component_placement`` 选择可用 GPU，然后启动训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ogpo_flow_state

脚本将日志写入 ``logs/<timestamp>-maniskill_ogpo_flow_state/``。

可视化
------

使用 TensorBoard 查看训练和评估指标：

.. code-block:: bash

   tensorboard --logdir ./logs

评估指标：``eval/success_once``。各指标的含义参见 :doc:`训练指标 <../../reference/metrics>`。
