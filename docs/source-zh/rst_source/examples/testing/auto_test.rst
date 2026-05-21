自动测试功能
====================

RLinf 自动测试工具位于 ``toolkits/auto_test/`` 目录下，用于批量运行具身训练实验、检测运行状态，并与基线结果进行自动对比分析。
该工具支持自动顺序执行实验，自动切换 Python 环境、以及根据日志判断实验的完成或崩溃状态。

环境与运行配置
--------------

任务列表配置
~~~~~~~~~~~~

在 ``run_all.sh`` 中的 ``TASKS`` 数组定义需要运行的实验列表，每行格式为 ``ENV_NAME MODEL_NAME YAML_ARG T_NODES T_STEPS T_SAVE``：

- ``ENV_NAME``：环境名称（如 ``maniskill_libero``、``behavior``、``isaaclab``、``metaworld``、``calvin`` 等）
- ``MODEL_NAME``：模型名称（如 ``openvla``、``openvla-oft``、``openpi``、``gr00t``、``mlp`` 等）
- ``YAML_ARG``：对应的 YAML 配置文件名
- ``T_NODES``：所需节点数
- ``T_STEPS``：训练总步数
- ``T_SAVE``：保存间隔（``-1`` 表示不保存）

示例配置：

.. code-block:: bash

   TASKS=(
       "maniskill_libero openvla-oft maniskill_ppo_openvlaoft 1 100 -1"
       "maniskill_libero openpi libero_goal_ppo_openpi 1 100 -1"
       "maniskill_libero openpi maniskill_ppo_mlp 1 100 -1"
   )

Python 环境与模型准备
~~~~~~~~~~~~~~~~~~~~~

安装对应的 Python 虚拟环境，并修改 ``run_all.sh`` 中的 ``VENV_BASE_DIR``：

.. code-block:: bash

   # 安装单个模型+环境组合
   bash requirements/install.sh embodied --model openvla --env maniskill_libero

   # 修改 VENV_BASE_DIR 指向虚拟环境所在目录
   export VENV_BASE_DIR="/path/to/venvs"

.. # 或批量安装所有环境
.. bash requirements/install_all_envs.sh

下载模型权重和资源文件，并更新 YAML 配置文件中的路径：

.. code-block:: bash

   # 下载模型权重
   hf download gen-robot/openvla-7b-rlvla-warmup --local-dir openvla-7b-rlvla-warmup

   # 下载 ManiSkill 资源文件
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

运行命令
--------

通过以下命令启动自动测试：

.. code-block:: bash

   bash ./toolkits/auto_test/run_all.sh

``run_all.sh`` 支持 ``--similarity-method`` 参数指定基线对比的相似度计算方法（默认 ``pearson``），可选 ``spearman``、``mse``、``mae``、``cosine``、``dtw``、``all``：

.. code-block:: bash

   bash ./toolkits/auto_test/run_all.sh --similarity-method pearson

.. 多节点运行时，需在各节点设置 ``RANK`` 环境变量：头节点 ``RANK=0``（默认），工作节点 ``RANK!=0``。头节点负责调度任务、启动 Ray 集群并执行训练；工作节点通过轮询同步信号文件 ``ray_utils/task_sync.txt`` 自动加入集群并等待任务完成。

运行流程：节点逐个执行任务列表中的实验，根据日志判断是否已达到阈值或已崩溃——若已达到则跳过，若全部崩溃则标记失败，否则激活对应虚拟环境并启动训练。所有任务完成后输出最终汇总，并自动进行日志分析、绘制曲线并与基线对比。

结果输出
--------

所有任务完成后输出最终汇总统计：

.. code-block:: text

   =========================================================
                       FINAL SUMMARY
   =========================================================
   Total tasks:        6
   Success:            4
   Skipped (crashed):  2
   =========================================================

一份详细的运行摘要也会生成，显示每个任务的状态和最终状态。
并自动进行对比分析，在 ``logs/`` 目录下生成 ``success_once`` 曲线图，并与基线对比计算相似度指标。

