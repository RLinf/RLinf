PolaRiS 评测
============

概览
----

PolaRiS 是基于 Isaac Sim 5.1.0 的桌面操作仿真平台，提供 TapeIntoContainer、MoveLatteCup 等 DROID 风格操作任务。RLinf 支持在 PolaRiS 上并行评测 OpenPI policy，并输出 ``eval/success_once`` 等任务指标。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      PolaRiS（DROID 桌面操作任务）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenPI π₀ · π₀.₅

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/polaris/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`PolaRiS 训练 <../../examples/embodied/polaris>`

安装与运行
----------

先按 :doc:`../../examples/embodied/polaris` 安装 Isaac Sim 5.1.0，并在每个新终端中执行其 ``setup_conda_env.sh``。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: OpenPI π₀.₅（TapeIntoContainer）

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

      **2. 下载场景与模型**

      .. code-block:: bash

         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./RLinf-Pi05-Polaris-droid_jointpos

      **3. 启动评测**

      .. code-block:: bash

         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_tapeintocontainer_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-Polaris-droid_jointpos

   .. tab-item:: OpenPI π₀（MoveLatteCup）

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

      **2. 下载场景与模型**

      .. code-block:: bash

         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./RLinf-Pi0-Polaris-droid_jointpos

      **3. 启动评测**

      .. code-block:: bash

         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_movelattecup_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-Polaris-droid_jointpos

.. dropdown:: 示例配置与覆盖参数

   ``evaluations/polaris/`` 已有示例配置：

   .. list-table::
      :header-rows: 1
      :widths: 38 22 12 14 14

      * - 配置文件
        - 任务
        - 模型
        - ``rollout_epoch``
        - ``max_episode_steps``
      * - ``polaris_tapeintocontainer_openpi_pi05_eval.yaml``
        - TapeIntoContainer
        - π₀.₅
        - 5
        - 450
      * - ``polaris_movelattecup_openpi_eval.yaml``
        - MoveLatteCup
        - π₀
        - 25
        - 456

   两个配置默认在 0 号 GPU 上以 ``total_num_envs: 1`` 运行，并将 ``open_loop_horizon`` 绑定为 ``${rollout.model.num_action_chunks}``\ （π₀.₅ 为 15，π₀ 为 8）。评测 RL 权重可传入 ``runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt``；评测其他 DROID 任务可复制任一 YAML 并将 ``env/polaris_droid_*@env.eval`` 切换为 :doc:`../../examples/embodied/polaris` 中的对应任务配置。

结果与常见问题
--------------

终端输出 ``eval/success_once``，日志与视频保存在 ``logs/<timestamp>-<config>/``\ （详见 :doc:`../reference/results`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 找不到场景资产
     - 将 ``POLARIS_DATA_PATH`` 导出为 ``PolaRiS-Hub`` 根目录，``run_eval.sh`` 会自动读取。
   * - checkpoint 格式不匹配
     - 若使用 JAX checkpoint，先按 :doc:`../../examples/embodied/polaris` 转换为 PyTorch 格式。
   * - Isaac Sim 导入或启动报错
     - 安装 Isaac Sim 5.1.0，并在每个新终端激活 ``.venv`` 与运行 ``run_eval.sh`` 前执行其 ``setup_conda_env.sh``。
   * - 步数校验失败
     - 将 ``max_steps_per_rollout_epoch`` 设为 ``rollout.model.num_action_chunks`` 的整数倍（π₀.₅ 为 15，π₀ 为 8）。
