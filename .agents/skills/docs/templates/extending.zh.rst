.. 扩展开发页模板：复制本文件，替换所有 <...>，再删除这些注释。
.. 规则见 .agents/skills/docs/rules/extending.md 与 rules/chinese.md。

新增自定义 <组件>
=================

<一段话说明本页带你接入什么组件、挂接到哪个注册入口或基类，以及完成后可以跑通什么流程。>

.. list-table:: 一览
   :header-rows: 1
   :widths: 26 74

   * - 项目
     - 说明
   * - 扩展入口
     - ``<模块路径>`` 中的 ``<注册装饰器或枚举>``
   * - 基类 / 接口签名
     - ``<BaseClass>``\ （实现 ``<method_a>``、``<method_b>``）
   * - 配置选择项
     - ``<yaml.key>: <your_name>``
   * - 外部仓库加载
     - 设置 ``RLINF_EXT_MODULE=<your_package.module>``\ （见 :doc:`index`）

前置条件
--------

- 已配置好 RLinf 虚拟环境（见 :doc:`../start/installation`）。
- <所需上游依赖、权重或资产。>

1. 实现 <组件>
--------------

<先用平实中文讲清接口契约，再给出最小可运行实现。>

.. code-block:: python

   <最小可运行实现>

2. 注册并在配置中启用
---------------------

按代码存放位置选择主仓库内注册或外部仓库注册：

.. tab-set::

   .. tab-item:: 主仓库内注册（In-Tree）
      :sync: ext-in-tree

      <主仓库内注册代码>

   .. tab-item:: 外部仓库注册（Out-of-Tree）
      :sync: ext-out-of-tree

      .. code-block:: python

         # my_pkg/rlinf_ext.py
         def register():
             <外部仓库注册代码>

      .. code-block:: bash

         export RLINF_EXT_MODULE=my_pkg.rlinf_ext

在 Hydra YAML 中启用注册名称：

.. code-block:: yaml

   <只列出需要修改的配置项>

3. 本地 CPU 验证
----------------

在启动完整分布式训练前，先在本地 CPU 上运行独立验证脚本，确认注册和输入输出契约正常：

.. code-block:: bash

   python -c '
   <独立可运行的 CPU 验证脚本>
   '

下一步
------

- :doc:`index` — 使用 ``RLINF_EXT_MODULE`` 外部注册及其他扩展入口。
- :doc:`../concepts/overview` — 了解 RLinf 的执行与调度模型。
