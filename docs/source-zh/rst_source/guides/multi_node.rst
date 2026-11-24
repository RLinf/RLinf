多节点训练
==========

在 **多台机器** 上搭建 Ray 集群并运行 RLinf 训练任务。本指南适用于具身智能、推理、Agent 等所有基于 Ray 的 RLinf 工作负载。

.. note::

   单机运行（``cluster.num_nodes: 1``）会自动启动 Ray，无需手动执行 ``ray start`` 或设置 ``RLINF_NODE_RANK``。

准备工作
--------

开始前确认以下条件：

* 所有节点网络互通，worker 能访问 head 的 ``<head_ip>`` 端口 ``6379``\ （或你指定的端口）。
* 各节点安装 **相同版本** 的 Python、Ray（``ray>=2.47.0``）和 RLinf 依赖。
* 任务配置中的 ``cluster.num_nodes`` 与集群实际节点数一致（见 :doc:`basic_config`）。

.. important::

   Ray 在执行 ``ray start`` 时会 **冻结** 当前的 Python 解释器路径和环境变量，之后在该节点拉起的进程都会继承这套环境。在 **每个节点** 上先激活虚拟环境、安装依赖，再执行 ``ray start``。``ray start`` **之后** 安装的包对 Ray worker 不可见。

步骤 1：在各节点配置环境变量
----------------------------

在 **每个节点** 执行 ``ray start`` **之前**，设置集群内唯一的节点编号（必填）：

.. code-block:: bash

   export RLINF_NODE_RANK=<0..N-1>   # 集群内唯一，head 通常为 0

机器有多个网卡且 Ray 或集合通信需要指定网卡时，一并导出：

.. code-block:: bash

   export RLINF_COMM_NET_DEVICES=<网卡名>   # 例如 eth0、enp3s0

用 ``ip addr`` 或 ``ifconfig`` 查看哪块网卡对其他节点可达。

步骤 2：启动 Ray 集群
---------------------

方式 A：手动在各节点启动
~~~~~~~~~~~~~~~~~~~~~~~~

选定一台机器作为 **head**\ （``RLINF_NODE_RANK=0``），其 IP 必须对所有 worker 可达，记为 ``<head_ip>``。

**Head 节点：**

.. code-block:: bash

   export RLINF_NODE_RANK=0
   # 可选：export RLINF_COMM_NET_DEVICES=eth0
   ray start --head --port=6379 --node-ip-address=<head_ip>

**Worker 节点**\ （``RLINF_NODE_RANK`` 依次为 1、2、…、N-1）：

.. code-block:: bash

   export RLINF_NODE_RANK=1
   # 可选：export RLINF_COMM_NET_DEVICES=eth0
   ray start --address='<head_ip>:6379'

``--node-ip-address`` 填写其他节点连接 head 所用的地址（内网 IP、VPC IP 或 overlay IP，云边网络配置见 :doc:`cloud_edge`）。端口 ``6379`` 可换为其他未占用端口，head 与所有 worker 保持一致即可。

方式 B：使用仓库脚本（共享文件系统场景）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

各节点挂载 **同一共享目录** 且能读写 ``ray_utils/ray_head_ip.txt`` 时，可在每个节点执行：

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt    # 仅首次或重建集群时
   RANK=0 bash ray_utils/start_ray.sh   # head
   RANK=1 bash ray_utils/start_ray.sh   # worker 1
   # ...

脚本用 ``RANK`` 区分 head 与 worker，**不会** 自动设置 ``RLINF_NODE_RANK``。请先按方式 A 在各节点导出 ``RLINF_NODE_RANK`` 再运行脚本，或把该变量写入自己的启动脚本。

.. warning::

   ``ray_utils/start_ray.sh`` 默认使用端口 ``29500``\ （可用 ``RAY_PORT`` 覆盖），并在 ``ray start`` 中硬编码了 ``--memory=461708984320``\ （约 430 GB）。内存较小或使用自定义端口的机器请直接使用方式 A。

步骤 3：开启代码同步（可选）
----------------------------

driver 与 worker **不共享文件系统**\ （如云边或跨机房部署）时，在 **启动训练脚本之前** 开启 Ray 任务级代码同步。driver 会将 RLinf 核心包打包进 ``runtime_env.py_modules``，worker 无需本地维护同一份代码：

.. code-block:: bash

   export RLINF_CODE_WORKING_DIR=auto

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``RLINF_CODE_WORKING_DIR``
     - 行为
   * - 未设置、``0`` 或 ``false``
     - 关闭同步（默认），要求各节点本地安装的 RLinf 版本一致。
   * - ``auto``
     - 从已安装的 ``rlinf`` 包或当前目录自动推断仓库根目录。
   * - 绝对路径
     - 指向包含 ``pyproject.toml`` 和 RLinf 包的仓库根目录，或直接指向 RLinf 包目录。

代码同步只分发 RLinf 核心包，不包含 ``examples/`` 和 ``docs/``；任务配置与数据路径仍需在各节点本地或共享存储上可访问。

.. note::

   **开启代码同步时的注意事项**

   * **不要在 RLinf 包目录内放大文件**：同步会把启动节点上的整个本地 RLinf 包目录打包分发给 worker。日志、checkpoint、缓存和数据集等大文件应放在包目录之外，避免拖慢打包与传输。
   * **单独准备模型权重与资产**：模型权重、模拟器资产和数据集 **不会** 随代码同步分发。请在各 worker 提前下载到配置指定的路径，或挂载 NFS/共享存储，并确认所有节点均可访问。

``RLINF_CODE_WORKING_DIR`` 在 RLinf 首次初始化 Ray 时生效，不要在训练进程外提前手动调用 ``ray.init``。

步骤 4：检查集群状态
--------------------

在任意已执行 ``ray start`` 的节点查看集群状态：

.. code-block:: bash

   ray status

确认 **节点数量**、每节点 **CPU/GPU** 和 **状态**\ （``ALIVE``）符合预期。例如 2 个节点、每节点 8 块 GPU 时，应看到共 16 块 GPU。

也可以用脚本等待全部 GPU 就绪（参数为集群 GPU 总数）：

.. code-block:: bash

   bash ray_utils/check_ray.sh 16

节点未全部加入时，检查防火墙、``--node-ip-address`` 可达性以及 worker 执行 ``ray start`` 时的报错。

步骤 5：启动 RLinf 训练
-----------------------

1. 在任务 YAML 中将 ``cluster.num_nodes`` 设为实际节点数（与 ``RLINF_NODE_RANK`` 范围一致）。
2. 在 **head 节点**\ （``RLINF_NODE_RANK=0``）进入 RLinf 仓库根目录并运行入口脚本，例如具身训练：

.. code-block:: bash

   cd <RLinf 仓库根目录>
   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi

推理任务示例：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

启动前确认 head 节点满足以下条件：

* 本机已执行 ``ray start``\ （``ray status`` 显示完整集群）；
* 可访问配置文件、模型与数据集（通过共享存储或代码同步）；
* 启用代码同步时，已在 **同一终端** 导出 ``RLINF_CODE_WORKING_DIR``。

.. note::

   ``node_groups`` 与跨机放置写法见 :doc:`../concepts/placement` 和 :doc:`hetero`；云边场景下的 ``component_placement`` 示例见 :doc:`cloud_edge`。

停止与重建集群
--------------

在各节点停止 Ray 并清理旧状态后，即可按上文步骤重新启动：

.. code-block:: bash

   ray stop
   rm -f ray_utils/ray_head_ip.txt   # 使用过 start_ray.sh 时执行

修改 Python 环境或 ``RLINF_NODE_RANK`` 后，必须在所有受影响的节点执行 ``ray stop`` 再重新 ``ray start``。

常见问题
--------

**Worker 连不上 head**

检查其他节点能否用 ping 或 telnet 访问 ``<head_ip>``，安全组或 iptables 是否放行 6379 端口，以及 head 是否错误地将 ``--node-ip-address`` 绑定为 ``127.0.0.1``。

**ray status 节点数少于 cluster.num_nodes**

等待 worker 加入完成，确认各节点 ``ray start`` 没有报错，且没有混用多个独立的 Ray 集群。

**Worker 导入不到最新代码**

检查 ``RLINF_CODE_WORKING_DIR``：关闭同步时，各节点本地的 RLinf 包版本必须一致；开启同步后，只分发 RLinf 核心包（不要在包目录内放大文件），``examples/`` 下的配置、模型和模拟器资产仍需在本地或共享存储上可访问。

**Ray 版本不匹配**

所有节点使用相同的 ``ray`` 版本（RLinf 要求 ``ray>=2.47.0``）。
