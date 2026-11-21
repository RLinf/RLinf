评测环境安装
============

评测与训练共用同一套具身环境安装脚本。切换下方标签直接安装对应 benchmark 环境，或前往 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 拉取预构建镜像。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

按 Benchmark 安装
-----------------

.. tab-set::

   .. tab-item:: LIBERO（OpenPI）
      :sync: eval-libero

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

   .. tab-item:: ManiSkill OOD（OpenVLA-OFT）
      :sync: eval-maniskill

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

   .. tab-item:: RoboTwin（OpenPI）
      :sync: eval-robotwin

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate

   .. tab-item:: BEHAVIOR-1K（OpenPI）
      :sync: eval-behavior

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

   .. tab-item:: RealWorld（Franka）
      :sync: eval-realworld

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env franka && source .venv/bin/activate

支持的 Benchmark 组合
---------------------

评测其他模型与 benchmark 组合时，按下方表格传入 ``--model`` 与 ``--env``\ （完整安装参数见 :doc:`../../start/installation`）：

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Benchmark
     - 推荐 ``--model``
     - 推荐 ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero`` / ``molmoact2``
     - ``maniskill_libero`` 或 ``libero``
   * - LIBERO / LIBERO-Plus
     - ``fastwam``
     - ``libero`` 或 ``liberoplus``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla`` / ``lingbotvlav2``
     - ``robotwin``
   * - BEHAVIOR-1K
     - ``openpi``
     - ``behavior``
   * - ManiSkill OOD
     - ``openvla-oft``
     - ``maniskill_libero``
   * - RealWorld
     - ``openpi`` / ``dreamzero``
     - ``franka``
   * - PolaRiS
     - ``openpi``
     - ``polaris``
   * - RoboLab
     - ``openpi``
     - ``robolab``
   * - SIMPLE
     - ``psi0``
     - ``simple``

.. dropdown:: Benchmark 专属环境变量（可选）

   ``run_eval.sh`` 会自动读取下列变量并配置 ``PYTHONPATH``：

   .. list-table::
      :header-rows: 1
      :widths: 22 48 30

      * - Benchmark
        - 环境变量
        - 说明
      * - RoboTwin
        - ``ROBOTWIN_PATH=/path/to/RoboTwin``、``ROBOT_PLATFORM=ALOHA``
        - RoboTwin 仓库路径
      * - BEHAVIOR-1K
        - ``OMNIGIBSON_DATA_PATH``
        - 详见 :doc:`../../examples/embodied/behavior`
      * - DreamZero
        - ``DREAMZERO_PATH=/path/to/DreamZero``
        - DreamZero 仓库路径
      * - PolaRiS
        - ``POLARIS_DATA_PATH=/path/to/PolaRiS-Hub``
        - 下载的 PolaRiS-Hub 资产路径

下一步
------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 5 分钟快速体验
      :link: quick_tour
      :link-type: doc

      在 LIBERO Spatial 上评测 OpenPI π₀.₅，快速验证环境。

   .. grid-item-card:: Benchmark 评测指南
      :link: ../guides/index
      :link-type: doc

      查看各具身 benchmark 的完整评测步骤。
