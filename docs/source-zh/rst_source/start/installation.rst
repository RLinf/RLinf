安装说明
========

使用 ``requirements/install.sh`` 在本地 UV 虚拟环境中安装 RLinf，或从 Docker Hub 拉取预构建镜像。

本地安装
--------

克隆仓库、按目标安装依赖，并激活虚拟环境：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

按场景选择对应的安装命令：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 场景
     - 安装命令
   * - OpenVLA + ManiSkill / LIBERO
     - ``bash requirements/install.sh embodied --model openvla --env maniskill_libero``
   * - OpenVLA-OFT + ManiSkill / LIBERO
     - ``bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero``
   * - OpenPI + LIBERO
     - ``bash requirements/install.sh embodied --model openpi --env libero``
   * - 智能体 / 推理
     - ``bash requirements/install.sh agentic``
   * - 文档构建
     - ``bash requirements/install.sh docs``

运行 ``bash requirements/install.sh --help`` 查看完整模型、环境和参数列表。常用选项：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 参数
     - 作用
   * - ``--use-mirror``
     - 使用中国大陆镜像源加速下载。
   * - ``--venv <dir>``
     - 指定虚拟环境目录（默认 ``.venv``）。
   * - ``--platform <name>``
     - 指定非 NVIDIA 加速卡（``amd``、``ascend``、``musa``，详见 :ref:`硬件支持矩阵 <model-hardware-support>`）。
   * - ``--python <version>``
     - 按需指定 Python 版本（如 ``behavior`` 和 ``d4rl`` 需要 ``3.10``）。

预构建 Docker 镜像
------------------

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: Docker Hub — rlinf/rlinf
      :link: https://hub.docker.com/r/rlinf/rlinf

      在 Docker Hub 查看全部预构建镜像标签、容器启动命令与国内镜像站说明。

验证安装
--------

在当前环境中确认 RLinf 和 Ray 可用：

.. code-block:: bash

   python -c "import rlinf; print(rlinf.__file__)" && ray --version

下一步
------

- :doc:`运行 VLA 快速上手 <vla>`
- :doc:`扩展到多机 <../guides/launch-scale/index>`
- :doc:`打开速查表 <cheat_sheet>`
