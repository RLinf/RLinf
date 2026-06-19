SO101 机械臂 Pi0 监督微调
==========================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   基于 SO101 遥操作数据的 OpenPI π₀ 监督微调。

本配方在 :doc:`SO101 六自由度机械臂 <so101>` 采集的数据上运行 OpenPI π₀
监督微调，是 SO101 工作流的第二阶段：在拿到遥操作生成的
``run_<timestamp>/`` 目录后，本文档完整介绍合并多次采集、计算归一化统计量、
以及在 GPU 服务器上启动 SFT 训练的全部流程。

概览
----

在与机器人解耦的单 GPU 主机上，对合并后的 SO101 LeRobot 数据集进行 π₀ 微调。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenPI π₀

   .. grid-item-card:: 方法
      :text-align: center

      全量 SFT

   .. grid-item-card:: 数据
      :text-align: center

      LeRobot v3.0（多次采集）

   .. grid-item-card:: 硬件
      :text-align: center

      1+ GPU · 无需机器人在线

| **流程：** GPU 主机安装 OpenPI → 合并多次采集 → 通过 HF_LEROBOT_HOME 暴露数据集 → 计算归一化统计量 → 启动 ``run_so101_sft_smoke.sh`` → 观察训练损失。
| **前置条件：** :doc:`so101`\ （已完成数据采集）· :doc:`sft_openpi`\ （熟悉通用 OpenPI SFT 流程）· π₀ 基座检查点。

各模块所在位置
~~~~~~~~~~~~~~

SO101 SFT 流水线由 RLinf 侧的下列文件组合而成：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 文件
     - 作用
   * - ``rlinf/models/embodiment/openpi/policies/so101_policy.py``
     - ``SO101Inputs`` / ``SO101Outputs`` 变换 — 将 6 维状态 / 6 维动作
       零填充到模型维度，并把 ``image`` + ``extra_view_image`` 暴露为
       ``base_0_rgb`` + ``left_wrist_0_rgb``。
   * - ``rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py``
     - ``LeRobotSO101DataConfig`` — 把 LeRobot 数据集中的扁平字段
       （``state``、``actions``、``image``、``extra_view_image``、``prompt``）
       重新打包到 π₀ 期望的 ``observation/`` 子树。
   * - ``rlinf/models/embodiment/openpi/dataconfig/__init__.py``\ （``pi0_so101`` 条目）
     - 注册 SO101 ``TrainConfig``：``action_horizon=10``、
       ``repo_id="so101_data"``、并使用标准 ``pi0_base`` 资产目录。
   * - ``rlinf/models/embodiment/openpi/_compat.py``
     - OpenPI ↔ LeRobot 兼容补丁（``lerobot.common`` 别名、
       ``PromptFromLeRobotTask`` 的 DataFrame 兼容补丁），
       通过写入 ``site-packages/`` 的 ``.pth`` 文件，
       在站点初始化期间自动启用。
   * - ``examples/sft/config/so101_sft_openpi.yaml``
     - SFT 配方 — ``action_dim: 6``、``num_action_chunks: 10``、
       ``openpi.config_name: pi0_so101``、``num_images_in_input: 2``。
   * - ``examples/sft/run_so101_sft_smoke.sh``
     - 启动器，预设了合理默认（``HF_HUB_OFFLINE=1``、``EMBODIED_PATH``、
       ``model_path``、``train_data_paths``、``max_steps``）。
   * - ``toolkits/lerobot/merge_lerobot_datasets.py``
     - 自动识别 v2.x / v3.0 布局的合并脚本。

安装（GPU 主机）
----------------

机器人侧因数据采集需要已安装了 ``--env so101`` 套件。GPU 主机除此之外
还需要 OpenPI 依赖以及同一份 RLinf 代码。下列步骤同时会向 venv 写入
持久化兼容补丁的 ``.pth`` 文件，使 DataLoader spawn 子进程也能自动
加载这些适配器。

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # 在新建 .venv 中以可编辑方式安装 OpenPI + RLinf，应用
   # transformers_replace 补丁，并写入 openpi 兼容 .pth 到 site-packages。
   # 国内用户可加 --use-mirror 加速。
   bash requirements/install.sh embodied --model openpi --env so101
   source .venv/bin/activate

如果之前以其他 ``--env`` 安装过，希望额外加入 OpenPI 支持，本命令是幂等
的：只会执行尚未完成的步骤。

.. note::

   若持久化补丁未被自动安装（比如你从别处搬来了一个已构建好的 venv），
   可手动执行：

   .. code-block:: bash

      python -m rlinf.models.embodiment.openpi._compat install

   该命令会向当前 site-packages 写入 ``rlinf_openpi_compat.pth``，
   使每个使用该 venv 的 Python 进程（包括 torch DataLoader 在
   ``multiprocessing.spawn`` 模式下启动的工作进程）都会自动调用
   :func:`install_compat_shims`。

数据准备
--------

合并多次采集
~~~~~~~~~~~~

SO101 数据采集每次会写入一个独立的 ``run_<timestamp>/`` 子目录，每次都是
完整的 LeRobot v3.0 数据集。OpenPI 加载器需要单一数据集，因此先合并：

.. code-block:: bash

   # source-dir：包含 rank_0/run_<timestamp>/... 的父目录。
   # output-dir：合并后的单一 LeRobot v3.0 数据集。
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir /path/to/so101_data \
       --output-dir /path/to/so101_data_merged

合并工具会自动识别数据集版本（``v2.x`` jsonlines 或 ``v3.0`` parquet），
并写出与输入相同版本的输出。SO101 v3.0 输入产出结构如下：

.. code-block::

   so101_data_merged/
   ├── data/chunk-000/file-000.parquet      # 所有帧合并到一处
   ├── meta/
   │   ├── episodes/chunk-000/file-000.parquet
   │   ├── tasks.parquet
   │   ├── stats.json                       # 沿用第一个源数据集
   │   └── info.json                        # 全局总数已更新

将合并结果接入 ``HF_LEROBOT_HOME``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenPI 加载器会在 ``$HF_LEROBOT_HOME``（默认
``~/.cache/huggingface/lerobot``）下按 repo id 查找数据集。把合并后的
数据集以 ``pi0_so101`` 注册时使用的标准名 ``so101_data`` 软链入该目录：

.. code-block:: bash

   mkdir -p ~/.cache/huggingface/lerobot
   ln -sfn /path/to/so101_data_merged ~/.cache/huggingface/lerobot/so101_data

也可以直接传绝对路径给 ``train_data_paths``，但软链做法可以让
``norm_stats`` 的查找路径保持简洁（详见下一节）。

计算归一化统计量
~~~~~~~~~~~~~~~~

每个特征的归一化统计量与模型检查点放在同一目录下。辅助脚本会读取数据集，
按 ``pi0_so101`` 配置中的全部变换跑一遍，并将 ``norm_stats.json`` 写到
``<assets_dir>/<repo_id>/``\ ：

.. code-block:: bash

   # HF_HUB_OFFLINE=1 阻止 lerobot 加载器联网校验数据集版本。
   HF_HUB_OFFLINE=1 python toolkits/lerobot/calculate_norm_stats.py \
       --config_name pi0_so101 \
       --repo_id so101_data

对标准的 8 episode / 2011 帧采集，脚本会处理 62 个 batch，约耗时 50 秒，
统计量样例如下：

.. code-block::

   state mean(0..5):  [53.37, 1.58, -88.92, 82.34, 40.29, 0.87]
   state std (0..5):  [7.65, 8.7, 20.86, 21.47, 15.66, 2.59]

把得到的 JSON 放到 π₀ 基座检查点旁，使 SFT 运行时可以从
``<model_path>/so101_data/norm_stats.json`` 加载：

.. code-block:: bash

   PI0_BASE=/path/to/pi0_base    # 可写副本或独立目录
   mkdir -p $PI0_BASE/so101_data
   cp assets/pi0_so101/so101_data/norm_stats.json \
      $PI0_BASE/so101_data/norm_stats.json

如果不想复制 14 GB 的 safetensors，可以用软链方式构造一个可写的 ``$PI0_BASE``：

.. code-block:: bash

   PI0_BASE_RO=/path/to/shared/pi0_base    # 共享只读检查点
   PI0_BASE=/path/to/pi0_base_so101         # 可写镜像
   mkdir -p $PI0_BASE
   for f in model.safetensors config.json policy_postprocessor.json policy_preprocessor.json README.md; do
       ln -sf $PI0_BASE_RO/$f $PI0_BASE/$f
   done
   # 然后把 per-asset_id norm_stats 放到一个真正的子目录里：
   mkdir -p $PI0_BASE/so101_data
   cp assets/pi0_so101/so101_data/norm_stats.json $PI0_BASE/so101_data/

.. note::

   与其他 OpenPI 数据集一样，建议人工检查 ``q01`` / ``q99`` 分位数的范围。
   必要时手动放宽过窄的分位数范围以稳定训练，详见 :doc:`sft_openpi`。

启动训练
--------

配置
~~~~

配方位于 ``examples/sft/config/so101_sft_openpi.yaml``。临时调参可在命令行
通过 Hydra override，长期变体则建议复制一份单独维护。

.. code-block:: yaml

   data:
     train_data_paths: "so101_data"   # 解析为 ~/.cache/huggingface/lerobot/so101_data

   actor:
     model:
       model_path: "/path/to/pi0_base_so101"
       num_action_chunks: 10
       action_dim: 6
       openpi:
         config_name: "pi0_so101"
         num_images_in_input: 2

   runner:
     max_steps: 2000

SO101 默认 ``micro_batch_size: 4`` × ``global_batch_size: 32``\ ，与
realworld Franka 配方保持一致。如果有多 GPU，可把
``cluster.component_placement.actor`` 调整为相应区间（如 ``0-7``\ ），
并把 ``fsdp_config.sharding_strategy`` 切回 ``full_shard``。

启动
~~~~

封装好的启动器会自动设置 ``EMBODIED_PATH``、强制 ``HF_HUB_OFFLINE=1``\ ，
并通过环境变量接收覆盖：

.. code-block:: bash

   # 默认值：model_path=/root/pi0_base_so101、
   # train_data_paths=so101_data、max_steps=20、
   # log_dir=/root/so101_sft_run。
   bash examples/sft/run_so101_sft_smoke.sh

   # 自定义：
   SO101_MODEL_PATH=/abs/path/to/pi0_base_so101 \
   SO101_DATA_REPO_ID=so101_data \
   SO101_MAX_STEPS=2000 \
   SO101_LOG_DIR=/abs/path/to/results \
       bash examples/sft/run_so101_sft_smoke.sh

启动器最终展开为 ``python examples/sft/train_vla_sft.py
--config-name so101_sft_openpi …``\ ，因此后续的 Hydra override 可以
直接追加在命令行。

冒烟测试（1×A800）
~~~~~~~~~~~~~~~~~~

在标准 8 episode 的 SO101 采集数据上跑 20 个步的单 GPU 冒烟测试，
约耗时 2 分 40 秒，能产出可用检查点：

.. list-table::
   :header-rows: 1
   :widths: 10 15 18 18 18

   * - Step
     - Loss
     - Grad norm
     - LR
     - Step time
   * - 13
     - 0.141
     - 2.90
     - 3.25e-7
     - 4.61 s
   * - 16
     - 0.147
     - 3.39
     - 4.00e-7
     - 4.17 s
   * - 18
     - 0.139
     - 3.66
     - 4.50e-7
     - 4.13 s
   * - 20
     - 0.174
     - 5.16
     - 5.00e-7
     - 4.11 s

损失在 0.10–0.27 区间内波动，梯度范数 2.9–5.2，学习率沿 1000 步余弦预热
从 2.5e-7 升至 5.0e-7。20 步检查点约 31 GB，落到
``${SO101_LOG_DIR}/so101_sft_openpi/checkpoints/global_step_20/actor/``\ ，
同时包含 ``dcp_checkpoint`` 与 ``model_state_dict`` 两种格式。

可视化
------

TensorBoard 事件文件位于 ``${SO101_LOG_DIR}/tensorboard/``\ ：

.. code-block:: bash

   tensorboard --logdir ${SO101_LOG_DIR}/tensorboard

关注 ``train/loss``、``train/grad_norm``、``train/learning_rate`` 即可
判断 SFT 收敛是否正常。完整指标命名空间见
:doc:`训练指标 <../../reference/metrics>`。

常见问题
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 问题
     - 解决方法
   * - ``ModuleNotFoundError: No module named 'lerobot.common'``
     - 当前 venv 中没有持久化兼容补丁。激活 venv 后执行
       ``python -m rlinf.models.embodiment.openpi._compat install``\ ，
       该命令会写入
       ``site-packages/rlinf_openpi_compat.pth``\ ，下一个 Python 进程
       即可自动加载两个补丁。
   * - ``ValueError: task_index=0 not found in task mapping``
     - DataFrame-tasks 补丁没有在某个 worker 进程中触发。请确认
       ``site-packages/`` 下存在上述 ``.pth`` 文件，并且 venv 的 Python
       没有被某个 shell alias 覆盖。
   * - ``transformers_replace is not installed correctly``
     - OpenPI 在 ``openpi/models_pytorch/transformers_replace/`` 下提供
       了替换模块。``install.sh`` 会把它们覆盖到已安装的 ``transformers``
       上；如果是手动安装，请执行：

       .. code-block:: bash

          cp -r /path/to/openpi/src/openpi/models_pytorch/transformers_replace/* \
                .venv/lib/python3.11/site-packages/transformers/
   * - ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3``
     - macOS 的 ``._*`` 资源派生文件被打包带入了 ``transformers/`` 目录。
       使用
       ``find .venv/lib/python3.11/site-packages/transformers -name '._*' -delete``
       清理。
   * - ``HFValidationError: Repo id must be in the form 'repo_name'`` …
     - 你向 ``--repo_id`` 传了绝对路径。建议设置 ``HF_LEROBOT_HOME`` 软链
       （推荐做法），或使用纯名字、不含斜杠的本地路径。
   * - 训练时找不到 norm_stats
     - 运行时的 asset_id 是 ``TrainConfig`` 中原始的 ``repo_id``\ （\ ``so101_data``\ ）。
       确保 JSON 落在
       ``${model_path}/so101_data/norm_stats.json``\ ，而不是你传给
       ``--repo_id`` 的别名（如果两者不同）。

相关文档
--------

- :doc:`so101` —— SO101 硬件上的数据采集流程。
- :doc:`sft_openpi` —— 通用 OpenPI SFT 配方（LIBERO、ManiSkill、Franka 等）。
- :doc:`/rst_source/extending/new_realworld_robot` —— 扩展 RLinf 以支持新的真实世界机器人。
- :doc:`训练指标 <../../reference/metrics>` —— 完整指标命名空间。
