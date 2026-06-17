PolaRiS 评测
============

PolaRiS 是桌面操作仿真平台，提供 TapeIntoContainer、MoveLatteCup 等 DROID 风格操作任务。RLinf 支持在 PolaRiS 上评测 OpenPI 策略。

相关训练文档：:doc:`../../examples/embodied/polaris`

环境准备
--------

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env polaris
   source .venv/bin/activate
   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

示例配置
--------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 配置文件
     - 任务
     - 模型
   * - ``polaris_tapeintocontainer_openpi_pi05_eval.yaml``
     - TapeIntoContainer
     - π₀.₅
   * - ``polaris_movelattecup_openpi_eval.yaml``
     - MoveLatteCup
     - π₀

完整评测流程
------------

**Step 1：下载数据集与模型**

按 :doc:`../../examples/embodied/polaris` 下载 PolaRiS 数据集与 OpenPI checkpoint。

**Step 2：设置环境变量**

.. code-block:: bash

   source .venv/bin/activate
   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

**Step 3：编辑配置**

修改 ``rollout.model.model_path`` 指向本地 checkpoint。

**Step 4：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh polaris polaris_tapeintocontainer_openpi_pi05_eval

或：

.. code-block:: bash

   bash evaluations/run_eval.sh polaris polaris_movelattecup_openpi_eval

**Step 5：查看结果**

终端输出 ``eval/success_once``；日志见 :doc:`../reference/results`。

常见问题
--------

- **数据集路径：** ``POLARIS_DATA_PATH`` 必须指向 PolaRiS-Hub 根目录，``run_eval.sh`` 会自动读取。
- **模型转换：** 若使用 JAX checkpoint，需先按训练文档转换为 PyTorch 格式。
