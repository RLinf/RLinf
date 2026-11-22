安装说明
========

使用 ``requirements/install.sh`` 在本地 ``uv`` 虚拟环境中一键安装 RLinf，或直接拉取 Docker Hub 预构建镜像。

安装与验证
----------

选择目标场景，直接复制运行完整安装与验证命令：

.. tab-set::

   .. tab-item:: 具身仿真（Embodied）
      :sync: embodied

      以下为 ManiSkill3 快速开始的默认安装命令（其他模型与环境按下方表格替换 ``--model`` 与 ``--env``）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate
         python -c "import rlinf; print(rlinf.__file__)" && ray --version

   .. tab-item:: 真机机器人（Real-World）
      :sync: realworld

      在机器人控制节点安装驱动与传感器依赖（无需 GPU），在 GPU 节点安装模型与驱动（按需将 ``franka`` 替换为 ``arx5``、``yam``、``xsquare_turtle2``、``so101``、``piper``、``lekiwi``、``dosw1`` 或 ``gim_arm``；Robbyant R2 通过 RFrame 与 ROS 2 接入，见 :doc:`../examples/embodied/r2`，完整列表见 :doc:`../examples/real_world_index`）：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf

         # 机器人控制节点（驱动与传感器）
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

         # GPU 训练 / 推理节点（或单机同配 GPU 与机器人）
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

   .. tab-item:: 智能体 / 推理（Agentic）
      :sync: agentic

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate
         python -c "import rlinf; print(rlinf.__file__)" && ray --version

   .. tab-item:: 预构建 Docker 镜像
      :sync: docker

      .. grid:: 1 1 1 1
         :gutter: 2

         .. grid-item-card:: Docker Hub — rlinf/rlinf
            :link: https://hub.docker.com/r/rlinf/rlinf

            在 Docker Hub 查看全部预构建镜像标签、容器启动命令与国内镜像站说明。

   .. tab-item:: 文档构建（Docs）
      :sync: docs

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh docs --venv .docs-venv

.. tip::

   **国内镜像加速与多环境隔离：** 在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。切换不同模型或环境组合时，使用 ``--venv <dir>`` 指定独立虚拟环境，避免覆盖默认的 ``.venv``。

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
     - ``franka``、``franka-ros``、``xsquare_turtle2``、``dosw1``、``gim_arm``、``so101``、``lekiwi``、``piper``、``arx5``、``yam``、``dummy``\ （Robbyant R2 使用 RFrame 与 ROS 2，见 :doc:`../examples/embodied/r2`）
   * - ``--engine``
     - ``sglang``\ （``agentic`` 默认）、``vllm``\ （``agentic``）、``phyai``\ （``embodied`` 搭配 ``--model openpi --env libero``）
   * - ``--platform``
     - ``nvidia``\ （默认）、``amd``、``ascend``、``musa``、``kunlun``、``biren``、``metax``\ （Intel XPU 使用 PyTorch XPU 环境，暂未通过 ``--platform`` 选项安装，详见 :ref:`硬件支持矩阵 <model-hardware-support>`）

.. dropdown:: 更多安装参数（可选）

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
      * - ``--no-flash-attn`` / ``--no-apex``
        - 在不需要或无匹配 CUDA 编译工具链的主机上跳过 ``flash-attn`` 或 ``apex`` 编译。

   运行 ``bash requirements/install.sh --help`` 查看全部版本覆盖参数。

常见问题与下一步
----------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 解决方法
   * - EGL 或 Vulkan 渲染报错
     - 按 :ref:`渲染问题 <faq-rendering>` 检查驱动与图形渲染配置。
   * - Ray 残留或 NCCL 报错
     - 执行 ``ray stop`` 清理本机残留进程，详见 :ref:`CUDA 与显存问题 <faq-cuda>`。

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: 训练快速开始
      :link: vla
      :link-type: doc

      单卡零下载跑通 MLP，或在 ManiSkill3 上用 PPO 训练 OpenVLA / OpenVLA-OFT。

   .. grid-item-card:: 扩展到多机集群
      :link: ../guides/launch-scale/index
      :link-type: doc

      配置多 GPU 与多节点 Ray 集群部署。

   .. grid-item-card:: 命令速查表
      :link: cheat_sheet
      :link-type: doc

      安装、训练与评测的最短常用命令。
