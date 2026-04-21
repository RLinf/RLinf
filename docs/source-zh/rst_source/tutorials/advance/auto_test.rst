自动测试功能
======================

RLinf 提供了一套自动测试工具，用于快速提交、管理和监控训练任务。
该工具类似于 Truss，支持通过简单的配置文件完成训练任务的端到端流程。
利用该工具，用户可以利用闲时算力，自动化依次训练测试列表中的所有任务，在闲时算力被调度走后归还时断点续训练

.. note::

   自动测试工具位于 ``RLinf/toolkits/auto_test/`` 目录下，
   支持 SFT、RL、Pretrain 等多种任务类型的自动化测试。

安装与配置
-------------------------

环境要求
^^^^^^^^^^^^^

* **Python** ≥ 3.8  
* 有效的 **API Key**（从环境变量 ``INFINI_API_KEY`` 获取）
* 可用的 **存储卷**（用于存放项目文件）

安装方式
^^^^^^^^^^^^^

.. code-block:: bash

   cd RLinf/toolkits/auto_test
   pip install -r requirements.txt
   pip install .
   pip install -e .

安装后，用户可以通过命令行工具 ``submit_job`` 命令来提交、管理和监控训练任务。

.. code-block:: bash
    submit_job --help

快速开始
-------------------------

查看可用示例
^^^^^^^^^^^^^

.. code-block:: bash

   submit_job list

run - 启动任务
^^^^^^^^^^^^^

根据 ``config.py`` 配置启动训练或推理任务。

.. code-block:: bash

   submit_job run <project_dir> [--api-key <key>] [--base-url <url>]

如果没有显式传 --api-key，程序会读取 INFINI_API_KEY。
如果没有显式传 --base-url，程序会读取 INFINI_BASE_URL，默认值是 https://cloud.infini-ai.com。

.. code-block:: bash

    export INFINI_API_KEY="sk-xxxxxxxx"
    export INFINI_BASE_URL="https://cloud.infini-ai.com"


配置文件说明
-------------------------

config.py 结构
^^^^^^^^^^^^^

每个测试项目需要包含一个 ``config.py`` 配置文件，主要包含以下字段：

.. code-block:: python

   # 项目名称
   project_name = "my_test"

   # 任务类型
   task_type = "supervised_learning"  # 或 "reinforcement_learning" 等

   # 运行时配置
   training_runtime = {
       "start_commands": ["python train.py"],
       "environment_variables": {
           "INFINI_API_KEY": "your_api_key",
       },
   }

   # 计算资源配置
   training_compute = {
       "node_count": 1,
   }

   # 项目配置
   training_project = {
       "name": project_name,
       "description": "任务描述",
       "task_type": task_type,
       "compute": training_compute,
       "runtime": training_runtime,
   }


自定义项目
^^^^^^^^^^^^^

您可以创建自定义测试项目：

.. code-block:: bash

   mkdir my_test
   cd my_test
   # 创建 config.py 和 model.py
   submit_job run .

利用闲时算力自动化训练
----------------------------------------------

RLinf 支持利用闲时算力，自动化依次训练测试列表中的所有任务。当闲时算力被调度走后归还时，支持断点续训练。

准备工作
^^^^^^^^^^^^^

**1. 代码与环境配置**

在目标集群使用的存储卷上拉取 RLinf 项目代码，按照 RLinf 文档中提供的环境配置方法安装环境，配置数据集和需要的模型。
把 ``RLinf/toolkits/auto_test/`` 目录下的 ``run_all_yaml.sh``、``check.py``、``parse_success_once.py`` 移动到项目根目录。

**2. 配置任务列表**

修改 ``RLinf/toolkits/auto_test/run_all_yaml.sh``，在 ``TASKS`` 数组中添加需要训练的任务：

.. code-block:: bash

   TASKS=(
       "maniskill_libero openpi maniskill_ppo_mlp 1 1000 -1"
       "maniskill_libero openpi libero_goal_ppo_openpi 1 120 -1"
       # 更多任务...
   )

任务格式为：``ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE``

**3. 移动脚本文件**

将以下文件移动到项目根目录：

.. code-block:: bash

   cp RLinf/toolkits/auto_test/run_all_yaml.sh ./
   cp RLinf/toolkits/auto_test/check.py ./
   cp RLinf/toolkits/auto_test/parse_success_once.py ./


启动训练
^^^^^^^^^^^^^

在本地进入任务配置目录，配置好任务参数后执行：

.. code-block:: bash

   submit_job run .

系统将自动依次执行任务列表中的所有训练任务。


工作原理
^^^^^^^^^^^^^

**任务状态检测**

系统通过 ``check.py`` 检测任务状态：

* 检查日志中的 ``Global Step`` 进度
* 判断是否达到阈值（默认为总步数的 10%）
* 检测是否发生崩溃（OOM、RuntimeError 等）

**断点续训**

* 已达到阈值的任务会被跳过
* 崩溃的任务会被记录并跳过
* 未开始或未完成的任务会继续执行

**结果统计**

训练完成后，系统会：

* 输出任务执行统计（成功/跳过/失败数量）
* 调用 ``parse_success_once.py`` 解析成功率曲线
* 生成可视化图表保存到日志目录


日志与结果
^^^^^^^^^^^^^

日志目录结构：

.. code-block:: text

   logs/
   ├── 20260414-11:04:12-libero_goal_ppo_openpi/
   │   └── run_embodiment.log
   └── success_once_curves/
       ├── libero_goal_ppo_openpi_success_once_curve.png
       └── libero_goal_ppo_openpi_success_once_data.csv




