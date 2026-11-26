RoboTwin 评测
=============

概览
----

RoboTwin 是双臂操作仿真平台，内置 22 个带评测种子的桌面任务（如放置空杯、扶正瓶子、摆鞋、按铃等）。RLinf 支持在 RoboTwin 上并行评测 VLA policy，并输出 ``eval/success_once`` 等任务级指标。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      RoboTwin（22 个种子任务，双臂 ALOHA / Piper）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenVLA-OFT · OpenPI π₀ / π₀.₅ · LingbotVLA 1.0 / 2.0

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/robotwin/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`RoboTwin <../../examples/embodied/robotwin>` · :doc:`LingbotVLA 1.0 <../../examples/embodied/lingbotvla>` · :doc:`LingbotVLA 2.0 <../../examples/embodied/lingbotvlav2>`

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: OpenVLA-OFT（place_empty_cup）

      **1. 安装并下载 RoboTwin 资产**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin && (cd ../RoboTwin && bash script/_download_assets.sh)

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup --local-dir ./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

      **3. 启动评测**

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOT_PLATFORM=ALOHA
         bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
           rollout.model.model_path=./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: OpenPI π₀.₅（adjust_bottle）

      **1. 安装并下载 RoboTwin 资产**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin && (cd ../RoboTwin && bash script/_download_assets.sh)

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi05-RoboTwin-SFT-adjust_bottle

      **3. 启动评测**

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: OpenPI π₀（adjust_bottle）

      **1. 安装并下载 RoboTwin 资产**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin && (cd ../RoboTwin && bash script/_download_assets.sh)

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-RoboTwin-SFT-adjust_bottle

      **3. 启动评测**

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
           env.eval.assets_path="${ROBOTWIN_PATH}"

   .. tab-item:: LingbotVLA 1.0（click_bell）

      **1. 安装并下载 RoboTwin 资产**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvla --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin && (cd ../RoboTwin && bash script/_download_assets.sh)

      **2. 下载基座权重、SFT 权重与 Qwen2.5-VL 骨干**

      .. code-block:: bash

         hf download robbyant/lingbot-vla-4b --local-dir ./lingbot-vla-4b
         hf download robbyant/lingbot-vla-4b-posttrain-robotwin \
           --revision 3e0c7c476bde3daaac00f79f3741a292a299f60a \
           --local-dir ./lingbot-vla-4b-posttrain-robotwin
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      **3. 启动评测**

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export LINGBOT_VLA_PATH="$(python -c "import lingbotvla, os; print(os.path.dirname(lingbotvla.__path__[0]))")"
         bash evaluations/run_eval.sh robotwin robotwin_click_bell_lingbotvla_eval \
           rollout.model.model_path=./lingbot-vla-4b-posttrain-robotwin \
           rollout.model.tokenizer_path=./Qwen2.5-VL-3B-Instruct \
           rollout.model.lingbotvla.config_path=./lingbot-vla-4b \
           env.eval.assets_path="${ROBOTWIN_PATH}"

      将配置名替换为 ``robotwin_place_shoe_lingbotvla_eval`` 即可用同一权重评测 ``place_shoe``\ （详见 :doc:`../../examples/embodied/lingbotvla`）。

   .. tab-item:: LingbotVLA 2.0（click_bell）

      **1. 安装并下载 RoboTwin 资产**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin && source .venv/bin/activate
         git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support ../RoboTwin && (cd ../RoboTwin && bash script/_download_assets.sh)

      **2. 下载 RoboTwin 权重与 Qwen3-VL 骨干**

      .. code-block:: bash

         hf download robbyant/lingbot-vla-v2-6b-robotwin --local-dir ./lingbot-vla-v2-6b-robotwin
         hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct

      **3. 启动评测**

      .. code-block:: bash

         export ROBOTWIN_PATH="$(cd ../RoboTwin && pwd)"
         export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_PATH}"
         bash evaluations/run_eval.sh robotwin robotwin_click_bell_lingbotvlav2_eval \
           rollout.model.model_path=./lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
           rollout.model.tokenizer_path=./Qwen3-VL-4B-Instruct \
           rollout.model.lingbotvlav2.training_config_path=./lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml \
           rollout.model.lingbotvlav2.robot_config_path=./.venv/lingbot-vla-v2/configs/robot_configs/robotwin.yaml \
           rollout.model.lingbotvlav2.stats_path=./.venv/lingbot-vla-v2/assets/norm_stats/robotwin.json

      默认使用 2 张 GPU（``rollout: 0``、``env: 1``）。LingbotVLA 1.0 与 2.0 均安装 ``lingbotvla`` 包，须分别使用独立的虚拟环境（详见 :doc:`../../examples/embodied/lingbotvlav2`）。

.. note::

   **显存与全种子覆盖：** 默认配置使用 ``total_num_envs: 128`` 与 ``use_fixed_reset_state_ids: True``，每轮约覆盖 150–320 条任务种子中的 128 条。内存或显存受限时可调小 ``env.eval.total_num_envs``\ （或开启 ``env.enable_offload: True`` 与 ``rollout.enable_offload: True``）；如需覆盖全部种子，将 ``use_fixed_reset_state_ids`` 设为 ``False`` 并配合 ``auto_reset: True`` 与更大的 ``max_steps_per_rollout_epoch``\ （详见 :ref:`评测配置详解 <robotwin-eval-config>`）。

在 AMD ROCm 上安装需加 ``--platform amd``，并在任务配置中设 ``planner_backend: mplib``\ （详见 :ref:`在不同硬件后端上运行 <robotwin-hardware>`）。``run_eval.sh`` 会将 ``ROBOTWIN_PATH`` 加入 ``PYTHONPATH``，``env.eval.assets_path`` 指向 ``assets/`` 的父目录。RLinf 默认使用 ``0.01 m`` 相机近裁剪面，避免 SAPIEN 3.0.1 在腕部视角裁掉夹爪手指（可通过 ``env.eval.camera_near`` 修改，设为 ``null`` 沿用 RoboTwin 默认 ``0.1 m``）。

.. dropdown:: 示例配置

   ``evaluations/robotwin/`` 已有示例配置（预构建容器镜像请访问 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__）：

   .. list-table::
      :header-rows: 1
      :widths: 44 28 28

      * - 配置文件
        - 任务
        - 模型
      * - ``robotwin_place_empty_cup_openvlaoft_eval.yaml``
        - ``place_empty_cup``
        - OpenVLA-OFT
      * - ``robotwin_place_empty_cup_openpi_eval.yaml``
        - ``place_empty_cup``
        - π₀
      * - ``robotwin_adjust_bottle_openpi_pi05_eval.yaml``
        - ``adjust_bottle``
        - π₀.₅
      * - ``robotwin_adjust_bottle_openpi_eval.yaml``
        - ``adjust_bottle``
        - π₀
      * - ``robotwin_place_shoe_lingbotvla_eval.yaml``
        - ``place_shoe``
        - LingbotVLA 1.0
      * - ``robotwin_click_bell_lingbotvla_eval.yaml``
        - ``click_bell``
        - LingbotVLA 1.0
      * - ``robotwin_click_bell_lingbotvlav2_eval.yaml``
        - ``click_bell``
        - LingbotVLA 2.0

   若 ``evaluations/robotwin/<config>.yaml`` 不存在，``run_eval.sh`` 仅对 ``examples/embodiment/config/`` 中已定义 ``rollout.model``\ （含 ``model_type`` 与 ``num_action_chunks``）的评测就绪配置生效。内置 ``eval_seeds.json`` 覆盖 **22 个任务**，若要评测纯训练配置请按 :doc:`../reference/configuration` 改写为评测 YAML。

.. _robotwin-eval-config:

.. dropdown:: 评测配置详解

   RoboTwin 评测对 ``eval_seeds.json`` 中预筛选的 **success seeds**\ （22 个任务，每任务 150–320 条种子）各跑一条轨迹，汇总 ``eval/success_once``。启动时全局打乱种子并按 worker 无重叠切分。默认配置（``total_num_envs: 128``、``rollout_epoch: 1``、``use_fixed_reset_state_ids: True``）每轮评测约 128 条轨迹；设置 ``use_fixed_reset_state_ids: False`` 可在 ``auto_reset`` 时自动切换后续种子。

   **示例任务种子数与步数上限**

   .. list-table::
      :header-rows: 1
      :widths: 26 18 22 34

      * - 任务
        - 种子总数
        - 示例 ``max_episode_steps``
        - 示例配置
      * - ``adjust_bottle``
        - 150
        - 200
        - ``robotwin_adjust_bottle_openpi_eval``
      * - ``place_empty_cup``
        - 260
        - 200
        - ``robotwin_place_empty_cup_openvlaoft_eval``
      * - ``click_bell``
        - 150
        - 400
        - ``robotwin_click_bell_lingbotvla_eval``
      * - ``place_shoe``
        - 320
        - 400
        - ``robotwin_place_shoe_lingbotvla_eval``

   ``max_episode_steps`` 须与训练配置及 ``task_config.step_lim`` 一致；通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`。

   **模型专属配置**

   评测设置须与各模型的训练协议一致，否则结果不可比：

   - **OpenVLA-OFT（``demo_randomized``）：** ``task_config.embodiment: [piper, piper, 0.6]``、``center_crop: True``\ （及 ``rollout.model.center_crop: True``）、保持域随机化开启、``num_action_chunks: 25``、匹配 SFT 的 ``unnorm_key``\ （如 ``place_empty_cup_1k``）、``implement_version: "official"``。
   - **OpenPI π₀ / π₀.₅（``demo_clean``）：** ``task_config.embodiment: [aloha-agilex]``、``center_crop: False``、``collect_wrist_camera: true``、关闭全部 ``task_config.domain_randomization``、``num_action_chunks: 50``、``openpi.config_name: pi0_aloha_robotwin`` 或 ``pi05_aloha_robotwin``；建议开启 ``env.enable_offload: True`` 与 ``rollout.enable_offload: True`` 降低显存。
   - **LingbotVLA 1.0：** 配置 ``model_path``、``tokenizer_path`` 与 ``rollout.model.lingbotvla.config_path``；``num_action_chunks: 50``、``max_episode_steps: 400``、``use_custom_reward: False``。
   - **LingbotVLA 2.0：** 在 ``examples/embodiment/config/model/lingbotvlav2.yaml`` 或 Hydra 参数中填写路径；``precision: bf16``、``embodiment: [aloha-agilex]``、``collect_wrist_camera: true``、``num_action_chunks: 50``、``max_episode_steps: 400``。

   **覆盖完整测试集**

   设任务种子总数为 ``S``，并行环境数为 ``E``，单轨迹步数上限为 ``T``\ （``max_episode_steps``）。

   *方式一：大并行*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 260                # S，以 place_empty_cup 为例
          max_episode_steps: 200
          max_steps_per_rollout_epoch: 200
          use_fixed_reset_state_ids: True
          rollout_epoch: 1

   *方式二：动态种子与自动重置（显存受限时推荐）*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 128
          max_episode_steps: 200
          # N = ceil(S / E)；place_empty_cup: ceil(260/128) = 3
          max_steps_per_rollout_epoch: 600   # N * max_episode_steps = 3 * 200
          auto_reset: True
          ignore_terminations: True
          use_fixed_reset_state_ids: False   # 允许 auto_reset 时切换种子
          is_eval: True
          rollout_epoch: 1

   多轮平均（``rollout_epoch > 1``）须保持 ``use_fixed_reset_state_ids: False``。``max_steps_per_rollout_epoch`` 须能被 ``rollout.model.num_action_chunks`` 整除。暂不支持 ``place_fan``、``open_laptop``、``place_object_scale`` 与 ``put_object_cabinet``。

.. dropdown:: 进阶用法

   **调整并行规模或加载 RL checkpoint**

   .. code-block:: bash

      bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
        env.eval.total_num_envs=64 \
        rollout.model.model_path=./RLinf-Pi0-RoboTwin-SFT-adjust_bottle \
        env.eval.assets_path="${ROBOTWIN_PATH}"

      bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
        rollout.model.model_path=./RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
        runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt \
        env.eval.assets_path="${ROBOTWIN_PATH}"

   **从训练配置派生其他任务**

   ``eval_seeds.json`` 还包含 ``beat_block_hammer``、``handover_block``、``lift_pot`` 等任务的种子。复制 ``evaluations/robotwin/`` 中相近的 YAML，将 ``defaults`` 改为 ``env/robotwin_<task>@env.eval``，并调整 ``unnorm_key`` 或 ``openpi.config_name``。

结果与常见问题
--------------

终端输出 ``eval/success_once``，日志与视频保存在 ``logs/<timestamp>-<config>/``\ （详见 :doc:`../reference/results`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 导入时找不到 ``ROBOTWIN_PATH``
     - 将 ``ROBOTWIN_PATH`` 导出为 RoboTwin 仓库根目录（``RLinf_support`` 分支）；``run_eval.sh`` 仅在该变量已设置时将其加入 ``PYTHONPATH``。
   * - 启动失败或场景缺失
     - 将 ``env.eval.assets_path``\ （LingbotVLA 2.0 使用 ``ROBOTWIN_ASSETS_PATH``）指向 ``assets/`` 的父目录；OpenVLA-OFT 还须导出 ``ROBOT_PLATFORM=ALOHA``。
   * - GPU 显存不足
     - 在 YAML 中开启 ``env.enable_offload: True`` 与 ``rollout.enable_offload: True``，或调小 ``env.eval.total_num_envs``。
   * - 种子覆盖不完整
     - 默认 128 并行 + ``use_fixed_reset_state_ids: True`` 只覆盖部分种子，详见 :ref:`评测配置详解 <robotwin-eval-config>`。
   * - EGL 渲染报错
     - ``run_eval.sh`` 默认使用 EGL 渲染；机器不支持 EGL 时执行 ``export MUJOCO_GL=osmesa`` 与 ``export PYOPENGL_PLATFORM=osmesa``。
