基于 Bench2Dex 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/Bench2Dex/Bench2Dex/main/assets/overview.png
   :align: center
   :width: 90%

   Bench2Dex（图片来源：`Bench2Dex <https://github.com/Bench2Dex/Bench2Dex>`__）。

`Bench2Dex <https://github.com/Bench2Dex/Bench2Dex>`__ 是基于 Isaac Lab 的双臂灵巧操作基准，覆盖 12 种由机械臂和灵巧手组成的本体。RLinf 提供 CNN PPO，以及 OpenPI pi0.5 的模仿学习和 PPO 配方。

概览
----------------------------------------

基于相机图像和关节位置，在 Bench2Dex 任务上训练双臂灵巧操作 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN · OpenPI pi0.5

   .. grid-item-card:: 算法
      :text-align: center

      IL · PPO

   .. grid-item-card:: 任务
      :text-align: center

      26 个任务 · 12 种本体

   .. grid-item-card:: 硬件
      :text-align: center

      CNN：1 GPU · OpenPI PPO：8 GPU

| **你将完成：** 安装 → 下载 Isaac Sim + 资产 + 模型 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · Isaac Sim · Bench2Dex 资产 · ResNet 检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

训练和评测各运行一个任务。在 ``env.train`` 和 ``env.eval`` 中设置 ``init_params.task_name`` 和 ``max_episode_steps``，并把 ``env.eval.max_steps_per_rollout_epoch`` 设为相同步数。``max_episode_steps`` 即该任务的官方 episode 步数预算，描述译自该任务的官方指令：

.. list-table::
   :header-rows: 1
   :widths: 36 16 48

   * - 任务
     - ``max_episode_steps``
     - 描述
   * - ``03_wine_glass_plate_balance``
     - 1082
     - 将三只盛满酒的酒杯分别端到三位用餐者的盘子上，不得洒出酒液或碰倒任何物体。
   * - ``06_fruit_bowl_loading``
     - 640
     - 左手将碗移到桌子中央，右手将两个苹果放入碗中，左手再将香蕉放入碗中。
   * - ``07_citrus_plate_loading``
     - 602
     - 左手将较远的柠檬放到盘子上，右手将与之配对的橙子移到盘子附近，左手再将这个橙子放到盘子上；对剩下的一组重复上述操作。
   * - ``08_frypan_stand_pour``
     - 1144
     - 左手拿起煎锅放到展示架上，然后依次将酱油和橄榄油倒入煎锅。
   * - ``09_cleaner_moisturizer_box_loading``
     - 642
     - 双手抬起清洁剂并竖直放入木盒，然后右手将肥皂放入盒中。
   * - ``12_screwdriver_box_and_hammer``
     - 1149
     - 右手将右侧的螺丝刀放入盒中并把盒子移到左侧，左手将左侧的螺丝刀放入盒中，然后拿起锤子敲击木块一次，再将锤子放入盒中。
   * - ``21_condiment_box_loading``
     - 827
     - 右手将味精放入盒中并把盒子移到桌子中央；然后左手先将酱油放入盒中，再放入醋。
   * - ``22_tool_box_loading``
     - 681
     - 右手将电钻和一字螺丝刀放入木盒，然后左手将扳手和十字螺丝刀放入盒中。
   * - ``24_stationery_category_sorting``
     - 1542
     - 右手将记号笔和电池分别放入笔筒和塑料盒，左手将灰色笔和胶水分别放入笔筒和塑料盒。
   * - ``26_canned_food_tray_line_arrangement``
     - 871
     - 左手将 Master Chef 罐头和味精放到托盘上，右手将牛奶盒和午餐肉罐头放到托盘上。
   * - ``27_ball_box_loading``
     - 1190
     - 依次将迷你足球、网球、高尔夫球和乒乓球放入盒中。
   * - ``32_baking_tray_prep_with_tools``
     - 852
     - 左手将刷子和锅铲放到托盘上，右手依次将小布丁盒和大明胶盒放到托盘上。
   * - ``34_fridge_wine_interhand_pour``
     - 1080
     - 左手打开冰箱并取出酒瓶，在空中交给右手倒一杯酒，再交回左手放回冰箱，最后左手关上冰箱门。
   * - ``42_trash_disposal``
     - 668
     - 打开垃圾桶盖，将揉皱的纸扔进垃圾桶，左手将瓶子和香蕉皮扔进垃圾桶，最后盖上垃圾桶盖。
   * - ``43_fridge_fruit_shelf_sorting``
     - 1375
     - 右手将香蕉移到桌子左侧，左手打开冰箱的两扇门，将苹果放到上层隔板、香蕉放到下层隔板，然后右手关上冰箱门。
   * - ``44_microwave_bowl_loading``
     - 1061
     - 左手打开微波炉门，右手将碗拿到微波炉旁，左手将法棍放入碗中，右手将碗放入微波炉，左手关上微波炉门。
   * - ``51_toilet_lid_cleaner_pour``
     - 1023
     - 左手掀开马桶盖，右手将清洁剂倒入马桶，左手盖上马桶盖。
   * - ``60_breadbasket_fast_food_loading``
     - 995
     - 右手将法棍和面包放入面包篮，并把篮子移到桌子中央，然后左手将面包和薯条放入篮中，摆成一份快餐。
   * - ``61_medicine_shoebox_pack``
     - 1034
     - 左手将鞋盒移到桌子中央，然后右手将药瓶和牙膏放入盒中，左手再将保湿油放入盒中。
   * - ``62_shoebox_accessory_pack``
     - 1077
     - 右手将印章放入鞋盒并把盒子移到桌子中央，然后左手将鞋子和宠物项圈放入盒中。
   * - ``64_sports_ball_cup_sort``
     - 665
     - 将网球和棒球放入大杯，将壁球和高尔夫球放入小杯。
   * - ``67_faucet_cup_water_fill``
     - 964
     - 左手先将勺子放入马克杯，右手将马克杯放到水龙头下，左手打开水龙头为马克杯接水后再关上，右手将马克杯放到托盘上。
   * - ``73_jigsaw_puzzle_assembly``
     - 1213
     - 右手和左手轮流将绿、红、蓝、黄四块拼图拼到桌子中央固定的白色拼图上，拼成一个矩形。
   * - ``76_soup_serving``
     - 986
     - 左手在锅旁端着碗，右手用汤勺从锅中舀两勺汤到碗里，左手将碗端到右前方区域，右手放回汤勺。
   * - ``79_bimanual_piano_melody``
     - 681
     - 双手一起弹奏钢琴键序列 C-C-G-G-A-A-G，左手弹低音，右手弹高音。
   * - ``80_gaming_desk_setup``
     - 593
     - 双手将显示器屏幕扶正朝前，左手食指按下 ESC 键，右手将鼠标移回键盘左侧，再用食指单击鼠标左键。

通过 ``init_params.robot_key`` 选择本体，并把 ``init_params.action_dim`` 设为该本体的主动关节数：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``robot_key``
     - ``action_dim``
   * - ``multi_iiwa7_with_sharpa``
     - 54
   * - ``multi_jaka_zu7_dexhand021_with_flange``
     - 52
   * - ``multi_panda_with_allegro``
     - 32
   * - ``multi_panda_with_orca``
     - 40
   * - ``multi_rm_65_with_revo2``
     - 24
   * - ``multi_ur5_rh56dfx_with_flange``
     - 24
   * - ``multi_ur5_rh5dg2_with_flange``
     - 38
   * - ``multi_ur5_schunk_hand_with_flange``
     - 30
   * - ``multi_ur5_shadow_hand_with_flange``
     - 56
   * - ``multi_ur5_wuji_with_flange``
     - 52
   * - ``multi_xarm7_with_ability``
     - 26
   * - ``multi_xarm7_with_leap``
     - 40

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - ``init_params.camera_names`` 中各相机的 RGB 图像（默认 ``[cam_stereo_left, cam_stereo_right, cam_wrist_left, cam_wrist_right]``），以 640×480 渲染后缩放到 ``init_params.image_size`` 指定的尺寸（CNN 配方中为 128×128），以及 ``action_dim`` 维主动关节位置。
   * - 动作
     - ``action_dim`` 维连续动作（默认 ``multi_xarm7_with_ability`` 为 26 维）：主动关节的绝对位置目标，频率 20 Hz；每个动作保持 3 个 60 Hz 物理步。
   * - 奖励
     - 稀疏成功奖励。任务的官方成功条件持续满足该任务的停留时间后即判定成功。
   * - 提示词
     - 该任务官方指令的英文原文。

.. note::

   RLinf 搭建场景时关闭 Bench2Dex 的场景泛化：桌子和物体按任务 YAML 布局置于地面上，由一个平行光照明。Bench2Dex 官方评测会开启场景泛化，各评测配置会还原一条示教录制时的房间背景、桌面纹理、光照、桌面高度、相机偏移、干扰物和物体摆放，或重新采样其中部分或全部因素。

安装
----------------------------------------

.. include:: _setup_common.rst
   :end-before: 然后，使用

安装 Bench2Dex 依赖：

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --env bench2dex
   source .venv/bin/activate

下载 Isaac Sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 Isaac Sim 5.1.0 并初始化其 shell 环境：

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   每次在新终端中启动 Bench2Dex 前，都需要运行 ``source ./setup_conda_env.sh``。

下载资产
----------------------------------------

下载物体和机器人资产：

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download Bench2Dex/Assets --repo-type dataset \
     --include 'Objects/**' 'Robots_p/**' \
     --local-dir /path/to/dex2bench_dataset
   export BENCH2DEX_ASSETS_PATH=/path/to/dex2bench_dataset

下载模型
----------------------------------------

CNN + PPO 配方需要下载 ResNet 检查点：

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

然后在 ``examples/embodiment/config/bench2dex_ppo_cnn.yaml`` 中为 rollout 和 actor 设置相同的检查点路径：

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained

运行
----------------------------------------

在已初始化 Isaac Sim 的终端中启动 Bench2Dex 配方：

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - 配方
     - 配置
     - 命令后缀
   * - CNN + PPO
     - ``examples/embodiment/config/bench2dex_ppo_cnn.yaml``
     - ``bench2dex_ppo_cnn``

.. code:: bash

   source /path/to/isaac_sim/setup_conda_env.sh
   export BENCH2DEX_ASSETS_PATH=/path/to/dex2bench_dataset

   bash examples/embodiment/run_embodiment.sh bench2dex_ppo_cnn

这条命令会：

1. 使用 ``bench2dex_ppo_cnn`` 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 Bench2Dex env 组件创建 Ray worker。
3. 使用 ``multi_xarm7_with_ability`` 本体在 ``03_wine_glass_plate_balance`` 上运行 PPO。

.. note::

   该配置默认运行在 GPU ``0``。每个 Bench2Dex 环境在独立的 Isaac Sim 进程中运行一个场景；每个 env worker 持有一个训练环境，启用评测时还持有一个评测环境。请将 ``env.train.total_num_envs`` 和 ``env.eval.total_num_envs`` 都设为 env worker 数。如需在 GPU ``0`` 上运行四个 env worker，请将 ``cluster.component_placement`` 设为 ``actor,rollout: 0`` 和 ``env: 0:0-3``，并将两个 ``total_num_envs`` 都设为 ``4``。请根据硬件调整 batch size。语法见 :doc:`组件放置 <../../concepts/placement>`。

.. note::

   对 CNN policy，更换本体或相机时，请让 ``actor.model.state_dim`` 和 ``actor.model.action_dim`` 等于 ``init_params.action_dim``，并让 ``actor.model.image_num`` 等于 ``init_params.camera_names`` 中的相机数。

OpenPI pi0.5
----------------------------------------

OpenPI 配方使用 ``03_wine_glass_plate_balance`` 任务和 ``multi_xarm7_with_ability`` 本体，输入四路 RGB 图像及 26 维主动关节位置，每个动作块预测 20 个绝对关节位置目标。基础 pi0.5 模型保留 32 维动作空间：数据变换先归一化 26 维主动关节，再补齐到 32 维供模型使用；输出经过反归一化后，选取 26 维主动关节目标发送给环境。

同时安装模型和模拟器依赖：

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex
   source .venv/bin/activate

将官方任务的 HDF5 回放轨迹转换为 LeRobot 数据集：

.. code:: bash

   python -m toolkits.lerobot.convert_bench2dex_to_lerobot \
     --input-dir /path/to/dataset/03_wine_glass_plate_balance/replay-generalization \
     --output-dir /path/to/bench2dex_lerobot/03_ability \
     --active-dof-map "$BENCH2DEX_REPO_PATH/robots/active_dof_maps.yml" \
     --task 'Carry three filled wine glasses to the plates of three diners without spilling or knocking over any objects.' \
     --repo-id bench2dex/03_ability --fps 20

转换器导出 RGB 和主动关节轨迹，排除无效帧和回零阶段，在缺帧处分割轨迹以保留 20 Hz 时间间隔，并使用全部导出状态和动作生成 ``norm_stats.json``。输出包含以下字段：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 数据集字段
     - 内容
   * - ``observation.state``
     - 按模拟器关节顺序排列的 26 维主动关节位置
   * - ``action``
     - 顺序相同的 26 维主动关节绝对位置目标
   * - ``observation.images.cam_stereo_left`` / ``observation.images.cam_stereo_right``
     - 左、右双目 RGB 图像
   * - ``observation.images.cam_wrist_left`` / ``observation.images.cam_wrist_right``
     - 左、右腕部 RGB 图像
   * - ``task``
     - 数据集元信息提供的任务指令

IL、评估和 PPO 使用转换器生成的同一份归一化文件。对于另行准备的 LeRobot 数据集，现有统计工具也支持 Bench2Dex 数据配置：

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_bench2dex_base --repo-id /path/to/bench2dex_lerobot/03_ability

在 ``examples/sft/config/bench2dex_sft_openpi_pi05.yaml`` 中设置 ``data.train_data_paths``、``actor.model.model_path`` 和 ``actor.model.openpi_data.norm_stats_path``。模型目录需包含 RLinf OpenPI 格式的基础 pi0.5 权重，转换命令见 :doc:`检查点转换 <../../guides/convertor>`。将 ``runner.max_steps`` 设为所需的优化步数，然后运行：

.. code:: bash

   bash examples/sft/run_vla_sft.sh bench2dex_sft_openpi_pi05

运行 PPO 时，将 ``examples/embodiment/config/bench2dex_ppo_openpi_pi05.yaml`` 中的两个模型路径都设为 IL 的 ``checkpoints/global_step_<step>`` 目录，并设置归一化文件路径。初始化 Isaac Sim 后，将 OpenPI 环境的 PyTorch 包和 CUDA 库放在 SDK 自带路径之前，再启动训练：

.. code:: bash

   source /path/to/isaac_sim/setup_conda_env.sh
   OPENPI_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
   export PYTHONPATH="$OPENPI_SITE:${PYTHONPATH:-}"
   export LD_LIBRARY_PATH="$OPENPI_SITE/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
   bash examples/embodiment/run_embodiment.sh bench2dex_ppo_openpi_pi05

PPO 配方使用八张 GPU：四个 actor worker、四个 rollout worker，以及共享这些 GPU 的八个环境 worker。1080 步时域包含 54 个完整的 20 步动作块；八个环境产生 432 个动作块样本，全局 batch 为 216、PPO 更新三个 epoch，每轮共六次优化更新。调整组件放置时，请保持 batch 的整除关系。IL 与 PPO 的图像顺序均为双目左、双目右、左腕、右腕。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

使用 ``env/success_once`` 观察训练成功率，使用 ``eval/success_once`` 观察评估成功率。比较不同 IL 检查点时，保持 episode 步数预算、重置种子和策略采样设置一致。完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

如需视频，请在环境配置中启用 video：

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval
