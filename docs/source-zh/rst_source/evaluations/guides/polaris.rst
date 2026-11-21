PolaRiS 评测
============

PolaRiS 是桌面操作仿真平台，提供 TapeIntoContainer、MoveLatteCup 等 DROID 风格操作任务。RLinf 支持在 PolaRiS 上评测 OpenPI policy。

相关训练文档：:doc:`../../examples/embodied/polaris`

一键上手
--------

请先按 :doc:`../../examples/embodied/polaris` 安装 Isaac Sim 5.1.0 并执行其 ``setup_conda_env.sh``。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: OpenPI π₀.₅（TapeIntoContainer）

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

         # 2. 下载场景与模型
         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./RLinf-Pi05-Polaris-droid_jointpos

         # 3. 启动评测
         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_tapeintocontainer_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-Polaris-droid_jointpos

   .. tab-item:: OpenPI π₀（MoveLatteCup）

      .. code-block:: bash

         # 1. 安装
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

         # 2. 下载场景与模型
         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./RLinf-Pi0-Polaris-droid_jointpos

         # 3. 启动评测
         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_movelattecup_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-Polaris-droid_jointpos

终端输出 ``eval/success_once``；日志见 :doc:`../reference/results`。

示例配置
--------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 配置文件
     - 任务
     - 模型
   * - ``polaris_tapeintocontainer_openpi_pi05_eval.yaml``
     - TapeIntoContainer
     - π₀.₅
   * - ``polaris_movelattecup_openpi_eval.yaml``
     - MoveLatteCup
     - π₀

常见问题
--------

- **数据集路径：** ``POLARIS_DATA_PATH`` 必须指向 PolaRiS-Hub 根目录，``run_eval.sh`` 会自动读取。
- **模型转换：** 若使用 JAX checkpoint，需先按训练文档转换为 PyTorch 格式。
