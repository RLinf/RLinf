云边协同训练配置
================

云边协同训练可以把训练驱动进程和大模型放在云端，同时把模拟器、机器人、
或数据采集服务放在边缘节点。这个场景首先会遇到网络连通性问题，因为 RLinf
依赖 Ray，而 Ray 要求所有参与节点之间都具有可互相访问的 IP 地址。

如果云端节点和边缘节点本身都已经具备可直接访问的公网 IP，那么你就可以像常规云端
训练那样完成部署，而不需要
`EasyTier <https://easytier.cn/guide/introduction.html>`_ 这类 overlay 组网服务。
但在大多数实际部署中，边缘侧往往位于 NAT 或其他私有网络之后，没有可被云端
直接访问的公网 IP。此时，通常建议先在云端与边缘之间搭建一层隧道网络，使两侧
获得彼此可达的 IP 地址。`EasyTier <https://easytier.cn/guide/introduction.html>`_
就是实现该网络的一种实用方案。本文给出一种最小可行流程：先搭建 EasyTier
overlay 网络，再在其上运行 RLinf。


硬件与软件要求
----------------

开始之前，请准备以下内容：

* 至少有一台带公网 IP 的节点可作为 EasyTier 引导节点。该节点只需要基础 CPU
  资源即可，不一定是云端训练节点，也不一定是边缘节点。
* 所有节点都需要具备创建 TUN 设备的权限。
* 如果节点运行在 Docker 容器中，需要为容器增加 ``NET_ADMIN`` 和 ``NET_RAW``
  capability，并挂载 ``/dev/net/tun`` 设备。

例如：

.. code-block:: bash

   docker run --rm -it \
     --network host \
     --cap-add NET_ADMIN \
     --cap-add NET_RAW \
     --device /dev/net/tun:/dev/net/tun \
     <your-image>


搭建步骤
---------

步骤 1：规划节点角色
~~~~~~~~~~~~~~~~~~~~

最简单的部署方式如下：

* 一台具备公网 IP 的节点承担 EasyTier 引导节点；
* 云端节点承担 Ray head 节点；
* 边缘节点加入同一个 EasyTier 网络后，再连接到云端的 Ray head；
* RLinf 训练任务只在云端 head 节点上启动。


步骤 2：在所有节点安装 EasyTier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在所有云端和边缘节点上安装 EasyTier CLI。你可以参考
`EasyTier 安装文档 <https://easytier.cn/guide/installation.html>`_，
或者将其注册为持久化的
`systemd 服务 <https://easytier.cn/guide/network/install-as-a-systemd-service.html>`_。

安装完成后，先确认二进制可用：

.. code-block:: bash

   easytier-core --version


步骤 3：在引导节点启动 EasyTier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为所有节点选定同一个网络名和密钥。下面的命令会在引导节点上启动 EasyTier，
并配置固定 overlay IP、固定设备名、以及供边缘节点初次加入的监听端口：

.. code-block:: bash

   sudo easytier-core \
     --hostname easytier-bootstrap \
     --network-name rlinf-cloud-edge \
     --network-secret <shared-secret> \
     --private-mode true \
     --ipv4 10.10.0.1 \
     --dev-name et0 \
     -l 11010

``--network-name``、``--network-secret``、``--ipv4``、``--peers`` 和
``--dev-name`` 的含义可以参考
`EasyTier 配置文档 <https://easytier.cn/guide/network/configurations.html>`_。


步骤 4：让所有边缘节点加入同一个 EasyTier 网络
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在每个边缘节点上使用相同的网络名和密钥启动 EasyTier，并指向云端引导节点的可达地址。

.. warning::

   请为每个节点分配唯一的固定 overlay IP。

.. code-block:: bash

   sudo easytier-core \
     --hostname edge-1 \
     --network-name rlinf-cloud-edge \
     --network-secret <shared-secret> \
     --private-mode true \
     --ipv4 10.10.0.11 \
     --dev-name et0 \
     -p tcp://<bootstrap_public_ip>:11010

更多边缘节点沿用同样的模式，并为每个节点分配唯一的主机名与固定 overlay IP。

.. note::

   如果你希望 EasyTier 常驻运行，可以把相同参数写入 systemd 单元中的
   ``ExecStart=``，方式与 EasyTier 服务文档一致。


步骤 5：验证 overlay 网络
~~~~~~~~~~~~~~~~~~~~~~~~~

当所有节点上的 EasyTier 都已启动后：

* 记录每个节点的 EasyTier overlay IP；
* 确认云端节点与边缘节点可以通过 EasyTier 网卡互相访问；
* 如果后续需要使用 ``RLINF_COMM_NET_DEVICES``，请保持 EasyTier 网卡名稳定。

这些 EasyTier IP 就是后续绑定 Ray 时应使用的地址。


RLinf 使用方式
--------------

步骤 1：让 Ray 绑定到 EasyTier 网络
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在每个节点上，必须先设置 ``RLINF_NODE_RANK``，再执行 ``ray start``。如果机器上
有多个网卡，还应把 ``RLINF_COMM_NET_DEVICES`` 指向 EasyTier 网卡，这样 RLinf
会优先走 overlay 网络通信。

云端 head 节点：

.. code-block:: bash

   export RLINF_NODE_RANK=0
   export RLINF_COMM_NET_DEVICES=et0
   ray start --head --port=6379 --node-ip-address=<cloud_easytier_ip>

边缘 worker 节点：

.. code-block:: bash

   export RLINF_NODE_RANK=1
   export RLINF_COMM_NET_DEVICES=et0
   ray start --address="<cloud_easytier_ip>:6379"

如果存在更多边缘节点，继续为它们设置不同的 ``RLINF_NODE_RANK``。


步骤 2：在 RLinf 配置中描述云端与边缘角色
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 ``cluster.num_nodes`` 设置为云端节点和边缘节点总数，然后使用 ``node_groups``
与 ``component_placement`` 把训练密集型组件放在云端，把对延迟更敏感的环境组件
放在边缘侧。

下面是一个“1 台 4-GPU 云端节点 + 1 台边缘节点”的示例：

.. code-block:: yaml

   cluster:
     num_nodes: 2
     node_groups:
       - label: cloud
         node_ranks: 0
       - label: edge
         node_ranks: 1
     component_placement:
       actor:
         node_group: cloud
         placement: 0-3
       rollout:
         node_group: edge
         placement: 0-0
       env:
         node_group: edge
         placement: 0-0

请根据你的实际硬件调整 placement 范围。如果边缘节点是纯 CPU 节点，RLinf 仍然
可以把该节点本身视作硬件资源 ``0`` 来进行放置。


步骤 3：只在云端 head 节点启动 RLinf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当所有节点都已经加入同一个 Ray 集群后，只需要在云端 head 节点上运行 RLinf
入口脚本。例如：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py --config-name <config_name>

或者：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh <config_name>

如果你的集群同时存在更复杂的软件环境差异或异构硬件布局，可以进一步结合
:doc:`hetero` 与 :doc:`../user/cluster` 一起使用。
