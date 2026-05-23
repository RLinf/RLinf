多机 Ray 集群启动
==================

本指南说明如何在 **多台机器** 上启动 Ray 集群，并在此基础上运行 RLinf 训练任务。
适用于具身智能、推理、Agent 等所有依赖 Ray 的 RLinf 工作负载。


准备工作
--------

开始前请确认：

* 所有节点网络互通：worker 能访问 head 的 ``<head_ip>`` 端口 ``6379`` （或你指定的端口）。
* 各节点已安装 **相同版本** 的 Python、Ray（``ray>=2.47.0``）及 RLinf 运行依赖。
* 在配置文件中把 ``cluster.num_nodes`` 设为集群节点总数（见 :doc:`../user/yaml`）。

.. important::

   Ray 会在执行 ``ray start`` 时 **冻结** 当时的 Python 解释器路径与环境变量；
   之后在该节点上由 Ray 拉起的进程都会继承这套环境。
   请在 **每个节点** 上先 ``source`` 虚拟环境、安装好依赖，再执行 ``ray start``。
   ``ray start`` **之后** 再安装的包对 Ray worker **不可见**。


步骤 1：在各节点配置环境变量
----------------------------

在 **每个节点**、执行 ``ray start`` **之前**，设置节点编号（必填）：

.. code-block:: bash

   export RLINF_NODE_RANK=<0..N-1>   # 集群内唯一，head 通常为 0

若机器有多个网卡，且 Ray/集体通信应走特定网卡，可指定：

.. code-block:: bash

   export RLINF_COMM_NET_DEVICES=<网卡名>   # 例如 eth0、enp3s0

可通过 ``ip addr`` 或 ``ifconfig`` 确认哪块网卡对其他节点可达。

步骤 2：启动 Ray 集群
---------------------

方式 A：手动在各节点执行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

选定一台机器作为 **head** （``RLINF_NODE_RANK=0``），其 IP 必须能被所有 worker 访问，记为 ``<head_ip>``。

**Head 节点：**

.. code-block:: bash

   export RLINF_NODE_RANK=0
   # 可选：export RLINF_COMM_NET_DEVICES=eth0
   ray start --head --port=6379 --node-ip-address=<head_ip>

**Worker 节点** （``RLINF_NODE_RANK`` 分别为 1、2、…、N-1）：

.. code-block:: bash

   export RLINF_NODE_RANK=1
   # 可选：export RLINF_COMM_NET_DEVICES=eth0
   ray start --address='<head_ip>:6379'

``--node-ip-address`` 应填写其他节点用来连接 head 的地址（内网 IP、VPC IP 或 overlay IP，见下文云平台说明）。
端口 ``6379`` 可改为其他未占用端口，head 与 worker 需一致。

方式 B：使用仓库脚本（共享文件系统场景）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当各节点挂载 **同一共享目录** 且能读写 ``ray_utils/ray_head_ip.txt`` 时，可在每个节点执行：

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt    # 仅首次或重建集群时
   RANK=0 bash ray_utils/start_ray.sh   # head
   RANK=1 bash ray_utils/start_ray.sh   # worker 1
   # ...

脚本通过 ``RANK`` 区分 head/worker，**不会** 自动设置 ``RLINF_NODE_RANK``；
请仍按方式 A 在各节点 ``export RLINF_NODE_RANK`` 后再运行脚本，或把该变量写入你的启动脚本。


步骤 3：开启代码同步（可选）
----------------------------

当 driver 与 worker **不共享同一文件系统**（云边、异构机房等）时，可在 **启动训练脚本之前** 开启 Ray 任务级代码同步：
由 driver 将 ``rlinf/`` 包打包进 ``runtime_env.py_modules``，worker 无需本地有一份相同 checkout。

.. code-block:: bash

   export RLINF_CODE_WORKING_DIR=auto

说明：

* ``RLINF_CODE_WORKING_DIR``  
  - 未设置 / ``0`` / ``false``：关闭同步（默认，依赖各节点本地代码树一致）。  
  - ``auto``：从已安装的 ``rlinf`` 包或当前目录推断仓库根目录。  
  - **绝对路径**：指向含 ``pyproject.toml`` 与 ``rlinf/`` 的仓库根，或 ``rlinf`` 包目录本身。

同步 **仅包含** ``rlinf/`` 子树，不含 ``examples/``、``docs/`` 等；示例配置与数据路径仍需在各节点可访问，或通过共享存储/NFS 提供。

.. note::

   **开启 code sync 时的注意**

   * **勿在 ``rlinf/`` 下保存大文件**：同步会把启动节点上的 **整个本地 ``rlinf/`` 目录** 打包下发给其他 worker。请勿把日志、checkpoint、缓存、数据集等大文件或临时产物放在 ``rlinf/`` 下，否则会显著拖慢打包与传输。
   * **模型权重和Assets等资源准备**：**模型权重、模拟器 assets、数据集** 等 **不会** 随 code sync 同步。请在各 worker 节点 **提前下载** 到配置中写的路径，或通过 NFS/共享存储挂载，并确认该路径在 **所有节点** 上均可访问。

上述变量在 RLinf 首次调用 ``ray.init``（``Cluster`` 初始化）时生效，请勿在训练进程之外提前手动 ``ray.init``。


步骤 4：检查集群状态
--------------------

在任意已执行 ``ray start`` 的节点上查看：

.. code-block:: bash

   ray status

确认 **节点数量** 、每节点 **CPU/GPU** 与 **状态** （``ALIVE``）是否符合预期。
例如 ``2`` 个节点、每节点 ``8`` 块 GPU 时，应看到 GPU 合计 ``16``。

也可使用辅助脚本等待资源就绪（参数为集群 **GPU 总数**）：

.. code-block:: bash

   bash ray_utils/check_ray.sh 16

若节点未全部加入，请检查防火墙、``--node-ip-address`` 是否可达、以及 worker 上 ``ray start`` 是否报错。


步骤 5：启动 RLinf 训练
-----------------------

1. 在任务 YAML 中设置 ``cluster.num_nodes`` 为实际节点数（与 ``RLINF_NODE_RANK`` 范围一致）。
2. 在 **已加入 Ray 集群的任意节点** 上进入 RLinf 仓库目录，执行入口脚本，例如具身智能：

.. code-block:: bash

   cd <RLinf 仓库根目录>
   bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi

推理类任务示例：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

启动节点须满足：

* 本机已执行过 ``ray start``（``ray status`` 能显示完整集群）；
* 能访问配置文件、模型与数据路径（或已配置共享存储 / code sync）；
* 若启用代码同步，在 **同一终端** 已 ``export RLINF_CODE_WORKING_DIR=...``。

.. note::

   使用 ``node_groups``、跨机型放置时，请参阅 :doc:`../user/placement` 与 :doc:`hetero`。
   云边场景下的 ``component_placement`` 示例见 :doc:`cloud-edge`。


停止与重建集群
--------------

在各节点停止 Ray 并清理状态后，可重新按本文步骤启动：

.. code-block:: bash

   ray stop
   rm -f ray_utils/ray_head_ip.txt   # 若曾使用 start_ray.sh

修改 Python 环境或 ``RLINF_NODE_RANK`` 后，需在所有相关节点 ``ray stop`` 再重新 ``ray start``。


常见问题
--------

**Worker 连不上 head**

检查 ``<head_ip>`` 是否对其他节点 ping/ telnet 可达、安全组/iptables 是否放行 ``6379``，以及 head 是否使用了错误的 ``--node-ip-address``（例如绑定了 ``127.0.0.1``）。

**ray status 节点数少于 cluster.num_nodes**

等待 worker 启动完成；确认各节点 ``ray start`` 无报错，且未混用多个独立 Ray 集群。

**Worker 上 import 不到最新代码**

确认是否开启 ``RLINF_CODE_WORKING_DIR``；未开启时各节点 ``rlinf/`` 需版本一致。开启 sync 后仅同步 ``rlinf/`` 包（勿在 ``rlinf/`` 下放大文件），``examples/`` 下配置、模型与模拟器 assets 仍须在各节点本地或共享存储上可访问。

**Ray 版本不匹配**

所有节点应使用相同 ``ray`` 版本（RLinf 要求 ``ray>=2.47.0``）。
