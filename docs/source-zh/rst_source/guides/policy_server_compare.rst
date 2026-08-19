对比 OpenPI Policy Server
=========================

在不修改或重启服务的情况下，使用相同的真实 ALOHA observation 测量
OpenPI server 与 RLinf policy server 的当前部署行为。

准备诊断环境
------------

运行诊断前，先启动两个兼容 OpenPI 协议的 WebSocket server。默认端点为
OpenPI 的 ``127.0.0.1:8000`` 和 RLinf 的 ``127.0.0.1:8001``。默认值也匹配当前部署的 wire layout：
``--openpi-image-layout chw`` 和 ``--rlinf-image-layout hwc``。两个 server
接收相同的解码像素，仅 axis 顺序不同。在运行客户端
的环境中安装可选分析依赖：

.. code-block:: bash

   python -m pip install pyarrow Pillow matplotlib openpi-client websockets

这些包仅由该 toolkit 延迟导入，不会成为 RLinf 核心运行时依赖。

运行对比
--------

在仓库根目录运行默认对比：

.. code-block:: bash

   python toolkits/lerobot/compare_policy_servers.py \
       --dataset-root data/lerobot-data_mixed_8_v30

该命令会为数据集的 11 个 prompt 各选择一个 episode，对三个均匀分布的
observation 分别重复请求三次，并评估三个连续重规划 action chunk。脚本会为
两个 server 各保留一个持久连接，并将带时间戳的结果写入
``results/policy_server_compare/``\ 。

在完整诊断前先缩小运行范围：

.. code-block:: bash

   python toolkits/lerobot/compare_policy_servers.py \
       --dataset-root data/lerobot-data_mixed_8_v30 \
       --prompt-regex "Place the bread" \
       --paired-frames 1 \
       --repeats 1 \
       --replay-chunks 1 \
       --request-timeout 120

使用 ``--episode-ids 12 24`` 选择指定 episode。使用
``--episodes-per-prompt``\ 、``--seed``\ 、``--openpi-host``\ 、
``--openpi-port``\ 、``--rlinf-host``\ 、``--rlinf-port`` 和
``--output-dir`` 控制采样、连接与输出位置。

理解指标
--------

通过三组指标区分部署差异、采样随机性与时间连续性。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 指标组
     - 含义
   * - 共同前缀差异
     - 在共同 horizon 和前 14 个 action 维度上计算 MAE、RMSE、P95、最大
       误差、按 joint 标准差归一化的 MAE，以及逐 joint、逐 step 误差。
   * - 随机性
     - 对相同 observation 重复请求，计算输出标准差和 repeat-to-repeat MAE。
   * - 抖动代理
     - 计算首 action 相对 state 的跳变、chunk 内一阶和二阶差分、相对数据集
       action 的误差，以及完整 horizon 重规划时的边界跳变。

检查产物
--------

每个带时间戳的目录都保留从汇总结果追溯到 episode 和 frame 所需的信息。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 产物
     - 内容
   * - ``run_metadata.json``
     - CLI 参数、选中的 episode、端点、握手 metadata 和实际响应合同。
   * - ``raw_outputs.npz``
     - 原始配对与 replay action chunk，以及序列化索引。
   * - ``summary.json`` 和 ``report.md``
     - 汇总指标、最差 prompt、joint、horizon step，以及合同限制提示。
   * - ``sample_metrics.csv``
     - 逐请求差异、客户端 RTT 和 server inference time。
   * - ``per_joint_metrics.csv`` 和 ``per_horizon_metrics.csv``
     - 共同前缀的详细误差。
   * - ``jitter_metrics.csv``
     - 逐 chunk 抖动代理和重规划边界跳变。
   * - PNG 图表
     - 差异热图、最差样本 action trace 和抖动对比图。

.. warning::

   请将结果视为当前部署服务的行为诊断。如果 action horizon、checkpoint、
   denoising 设置、metadata 或其他模型合同不一致，共同前缀指标不能作为严格
   数值 parity 的证据。在判断实现 parity 前，请先对齐两个 server 的合同。
