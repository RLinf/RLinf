安装说明
========

使用一行 ``requirements/install.sh`` 命令在本地 uv 虚拟环境中安装 RLinf，或前往 Docker Hub 拉取预构建镜像。

本地安装
--------

克隆仓库后，按目标场景运行对应的一行安装命令：

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf

.. tab-set::

   .. tab-item:: 具身智能（Embodied）

      .. code-block:: bash

         bash requirements/install.sh embodied --model <model> --env <env> && source .venv/bin/activate

   .. tab-item:: 智能体 / 推理（Agentic）

      .. code-block:: bash

         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate

   .. tab-item:: 文档构建（Docs）

      .. code-block:: bash

         bash requirements/install.sh docs --venv .docs-venv

支持的模型、环境与硬件平台
--------------------------

将下表中的名称直接传入 ``--model``、``--env``、``--engine`` 或 ``--platform``：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 参数
     - 支持的值
   * - ``--model``\ （VLA）
     - ``openvla``、``openvla-oft``、``openpi``、``psi0``、``pi0_fast``、``gr00t``、``gr00t_n1d6``、``gr00t_n1d7``、``dexbotic``、``starvla``、``lingbotvla``、``lingbotvlav2``、``abot_m0``、``molmoact2``、``evo1``
   * - ``--model``\ （WAM / VLM / 生成）
     - ``dreamzero``、``fastwam``、``cosmos3``、``qwen3_vl``、``diffusion``、``sglang``
   * - ``--env``\ （仿真器）
     - ``maniskill_libero``、``libero``、``liberopro``、``liberoplus``、``behavior``、``metaworld``、``calvin``、``isaaclab``、``robocasa``、``robocasa365``、``frankasim``、``robotwin``、``robodojo``、``roboverse``、``habitat``、``genesis``、``geniesim``、``embodichain``、``d4rl``、``mujoco_warp``、``polaris``、``simple``、``robolab``
   * - ``--env``\ （世界模型）
     - ``opensora``、``wan``
   * - ``--env``\ （真机）
     - ``franka``、``franka-ros``、``xsquare_turtle2``、``dosw1``、``gim_arm``、``so101``、``lekiwi``、``piper``、``arx5``、``yam``、``dummy``
   * - ``--engine``
     - ``sglang``\ （``agentic`` 默认）、``vllm``\ （``agentic``）、``phyai``\ （``embodied`` 搭配 ``--model openpi --env libero``）
   * - ``--platform``
     - ``nvidia``\ （默认）、``amd``、``ascend``、``musa``、``kunlun``、``biren``、``metax``\ （详见 :ref:`硬件支持矩阵 <model-hardware-support>`）

常用选项
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 参数
     - 作用
   * - ``--use-mirror``
     - 使用中国大陆镜像源加速依赖与 GitHub 下载（运行 ``hf download`` 前也可设置 ``export HF_ENDPOINT=https://hf-mirror.com``）。
   * - ``--venv <dir>``
     - 指定虚拟环境目录（默认 ``.venv``）。
   * - ``--python <version>``
     - 按需指定 Python 版本（如 ``behavior`` 和 ``d4rl`` 需要 ``3.10``）。
   * - ``--no-root``
     - 跳过 ``sudo`` 系统库安装，适合已预装系统依赖的主机。

运行 ``bash requirements/install.sh --help`` 查看更多版本锁定参数。

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

常见问题速查
------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 解决方法
   * - EGL 或 Vulkan 渲染报错
     - 按 :ref:`渲染问题 <faq-rendering>` 检查驱动与图形渲染配置。
   * - Ray 残留或 NCCL 报错
     - 执行 ``ray stop`` 清理本机残留进程，详见 :ref:`CUDA 与显存问题 <faq-cuda>`。

下一步
------

- :doc:`运行 VLA 快速上手 <vla>`
- :doc:`扩展到多机 <../guides/launch-scale/index>`
- :doc:`打开速查表 <cheat_sheet>`
