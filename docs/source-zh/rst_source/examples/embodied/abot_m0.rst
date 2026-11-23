ABot-M0 强化学习
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/ABot-M0.png
   :align: center
   :width: 80%

   ABot-M0：以 VGGT 为空间基础的 VLA 策略。

在 RLinf 中对 `ABot-M0 <https://github.com/amap-cvlab/ABot-Manipulation>`__ 运行评测与 **PPO** 训练，覆盖标准 **LIBERO** 与 **LIBERO-Plus**。该适配使用 HuggingFace rollout backend 与 FSDP actor 训练：rollout 阶段 ABot-M0 生成动作块，actor 更新阶段 RLinf 按 rollout 中保存的输入重新计算 log probability 与 value。

概览
----------------------------------------

在 LIBERO-10 / LIBERO-Plus 上用 PPO（actor-critic）微调 ABot-M0。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · LIBERO-Plus

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO-10

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · GPU

| **你将完成：** 安装 → 下载 ABot-M0 checkpoint 与骨干权重 → 设置 ``model_path`` → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 一个 ABot-M0 LIBERO checkpoint 及其骨干权重（见下文）。

ABot-M0 作为 VLA 策略接入 RLinf：适配层冻结预训练感知模块，通过 RL objective 训练动作模型，并额外加入 value head 以支持 actor-critic PPO（GAE 估计 advantage/return、ratio clipping、value clipping、可选 entropy 正则）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根据环境、任务族以及配置或权重工件选择对应的模型页面。

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - LIBERO-10
     - ``libero_10_ppo_abot_m0``
     - 针对 ABot-M0 release checkpoint 的 PPO 微调。
   * - LIBERO
     - LIBERO-10+
     - ``libero_10_plus_ppo_abot_m0``
     - 使用 ABot-M0 运行长程 LIBERO-10+ 训练。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - ABot-M0 所需的 LIBERO RGB 观测与机器人状态。
   * - Action
     - 从 ABot-M0 策略输出解码的连续机器人动作。
   * - Reward
     - PPO 使用的 LIBERO 成功信号或任务奖励。
   * - Prompt
     - 每个 LIBERO 任务对应的自然语言指令。

安装与运行
----------

.. include:: _setup_common.rst

安装脚本会自动克隆 ABot-M0 和 VGGT；如需复用本地源码目录，先设置 ``ABOT_PATH`` 和 ``VGGT_PATH``。

.. tab-set::

   .. tab-item:: LIBERO
      :sync: libero

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model abot_m0 --env maniskill_libero && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 10k-step LIBERO RL 基线权重（初始成功率约 40%）与 Qwen3-VL 骨干权重：

      .. code-block:: bash

         hf download HaoyunOvO/ABot-m0-LIBERO-10k-step --local-dir ./ABot-m0-LIBERO-10k-step
         hf download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir ./Qwen3-VL-4B-Instruct-Action

      修改已下载 checkpoint 目录中的 ``config.yaml``，将 ``qwenvl.base_vlm`` 指向本地 ``Qwen3-VL-4B-Instruct-Action`` 目录：

      .. code-block:: yaml

         qwenvl:
           base_vlm: /path/to/Qwen3-VL-4B-Instruct-Action

      **3. 启动训练**

      通过命令行覆盖 ``actor.model.model_path`` 启动 PPO 训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_abot_m0 \
           actor.model.model_path=./ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt

      .. dropdown:: 进阶配置（可选）

         - **独立评测用 SFT 权重**：如需评测官方发布的 LIBERO SFT 权重，下载 ``acvlab/ABot-M0-LIBERO``：

           .. code-block:: bash

              hf download acvlab/ABot-M0-LIBERO --local-dir ./ABot-M0-LIBERO

         - **离线 VGGT 权重**：运行时无法访问 Hugging Face 时，提前下载 ``facebook/VGGT-1B`` 并设置 ``VGGT_MODEL_PATH``\ （``VGGT_PATH`` 用于安装时指定源码目录，``VGGT_MODEL_PATH`` 用于运行时指定权重目录）：

           .. code-block:: bash

              hf download facebook/VGGT-1B --local-dir ./VGGT-1B
              export VGGT_MODEL_PATH=$PWD/VGGT-1B

         - **Hydra 配置参考**：完整字段说明见 :doc:`训练配置 <../../reference/configuration>`。

   .. tab-item:: LIBERO-Plus
      :sync: liberoplus

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model abot_m0 --env liberoplus && source .venv/bin/activate

      **2. 下载资源与模型**

      下载 LIBERO-Plus 场景资产、10k-step LIBERO RL 基线权重与 Qwen3-VL 骨干权重（LIBERO-Plus 资产详情见 :ref:`LIBERO-Pro 与 LIBERO-Plus 章节 <zh-liberopro-plus-benchmark>`）：

      .. code-block:: bash

         export LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)" | tail -n 1)
         hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"
         hf download HaoyunOvO/ABot-m0-LIBERO-10k-step --local-dir ./ABot-m0-LIBERO-10k-step
         hf download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir ./Qwen3-VL-4B-Instruct-Action

      将 ``assets.zip`` 中的 ``assets/`` 目录解压到已安装的 ``liberoplus.liberoplus`` 包目录：

      .. code-block:: bash

         python - <<'PY'
         import os, zipfile
         from pathlib import Path
         pkg = Path(os.environ["LIBERO_PLUS_PACKAGE_DIR"])
         with zipfile.ZipFile(pkg / "assets.zip") as z:
             for info in z.infolist():
                 if "/assets/" in info.filename and (rel := info.filename.split("/assets/", 1)[1]):
                     target = pkg / "assets" / rel
                     target.mkdir(parents=True, exist_ok=True) if info.is_dir() else (target.parent.mkdir(parents=True, exist_ok=True), target.write_bytes(z.read(info)))
         PY

      修改已下载 checkpoint 目录中的 ``config.yaml``，将 ``qwenvl.base_vlm`` 指向本地 ``Qwen3-VL-4B-Instruct-Action`` 目录：

      .. code-block:: yaml

         qwenvl:
           base_vlm: ./Qwen3-VL-4B-Instruct-Action

      **3. 启动训练**

      指定 LIBERO-Plus 套件并通过命令行覆盖 ``actor.model.model_path`` 启动 PPO 训练（rollout 会自动沿用同一路径）：

      .. code-block:: bash

         export LIBERO_TYPE=plus
         bash examples/embodiment/run_embodiment.sh libero_10_plus_ppo_abot_m0 \
           actor.model.model_path=./ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt

      .. dropdown:: 进阶配置（可选）

         - **独立评测用 SFT 权重**：如需评测官方发布的 LIBERO SFT 权重，下载 ``acvlab/ABot-M0-LIBERO``：

           .. code-block:: bash

              hf download acvlab/ABot-M0-LIBERO --local-dir ./ABot-M0-LIBERO

         - **离线 VGGT 权重**：运行时无法访问 Hugging Face 时，提前下载 ``facebook/VGGT-1B`` 并设置 ``VGGT_MODEL_PATH``\ （``VGGT_PATH`` 用于安装时指定源码目录，``VGGT_MODEL_PATH`` 用于运行时指定权重目录）：

           .. code-block:: bash

              hf download facebook/VGGT-1B --local-dir ./VGGT-1B
              export VGGT_MODEL_PATH=$PWD/VGGT-1B

         - **Hydra 配置参考**：完整字段说明见 :doc:`训练配置 <../../reference/configuration>`。

.. dropdown:: 独立评测

   训练前可通过统一的 Evaluation 流程验证 ABot-M0 checkpoint。先阅读 :doc:`LIBERO 评测指南 <../../evaluations/guides/libero>`，并在配置中将 ``actor.model.model_path`` 指向 ABot-M0 checkpoint。

   .. list-table::
      :header-rows: 1
      :widths: 28 36 36

      * - 套件
        - 配置来源
        - 需要修改
      * - LIBERO-10
        - 通过 Evaluation 配置回退使用 ``libero_10_ppo_abot_m0``
        - 将 ``actor.model.model_path`` 指向 ABot-M0 checkpoint。
      * - LIBERO-10+
        - 通过 Evaluation 配置回退使用 ``libero_10_plus_ppo_abot_m0``
        - 设置 ``LIBERO_TYPE=plus``，并将 ``actor.model.model_path`` 指向 ABot-M0 checkpoint。

   CLI 用法、Hydra 覆盖、日志和视频输出见 :doc:`Evaluation CLI 参考 <../../evaluations/reference/cli>` 与 :doc:`Evaluation 结果参考 <../../evaluations/reference/results>`。

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``env/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标与日志后端说明见 :doc:`训练指标 <../../reference/metrics>`。
