具身智能
========

按仿真基准、模型系列或训练范式选择具身训练入口：

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 模拟器
      :link: simulators_index
      :link-type: doc

      从评测基准切入——LIBERO、ManiSkill、RoboTwin、BEHAVIOR-1K、IsaacLab 等。

   .. grid-item-card:: 具身模型
      :link: vla_wam_index
      :link-type: doc

      微调 VLA / WAM 模型——π₀ / π₀.₅、OpenVLA-OFT、GR00T、StarVLA、LingBot-VLA、ABot-M0 等。

   .. grid-item-card:: SFT 与模仿学习
      :link: sft_index
      :link-type: doc

      监督微调与模仿学习配方——OpenPI、DreamZero、FastWAM、Cosmos3、Qwen-VL 与 DAgger。

   .. grid-item-card:: RL 算法示例
      :link: methods_index
      :link-type: doc

      按算法切入——SAC-Flow、DSRL、RL Token、仿真-真机协同训练、OPD、RECAP、STEAM 与 IQL。

   .. grid-item-card:: 世界模型
      :link: world_model_index
      :link-type: doc

      用世界模型作为模拟器做 VLA 闭环强化学习——OpenSora、Wan 与 BWM。

模型 × 模拟器速查矩阵
---------------------

从模型系列或评测基准任意一侧进入对应示例。

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 10 10 10 10 10

   * - 模型
     - :doc:`BEHAVIOR-1K <embodied/behavior>`
     - :doc:`CALVIN <embodied/calvin>`
     - :doc:`LIBERO <embodied/libero>`
     - :doc:`ManiSkill3 <embodied/maniskill>`
     - :doc:`MetaWorld <embodied/metaworld>`
     - :doc:`RoboCasa <embodied/robocasa>`
     - :doc:`RoboTwin <embodied/robotwin>`
     - :doc:`真机 <real_world_index>`
   * - :doc:`Dexbotic <embodied/dexbotic>`
     - —
     - —
     - ✓
     - —
     - —
     - —
     - —
     - —
   * - :doc:`GR00T <embodied/gr00t>`
     - —
     - —
     - ✓
     - ✓
     - —
     - —
     - —
     - —
   * - :doc:`MLP <embodied/mlp>` / :doc:`SAC <embodied/sac_flow>`
     - —
     - —
     - ✓
     - ✓
     - —
     - —
     - —
     - ✓
   * - :doc:`OpenPI π₀ / π₀.₅ <embodied/pi0>`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - :doc:`OpenVLA <embodied/maniskill>`
     - —
     - —
     - ✓
     - ✓
     - —
     - —
     - —
     - —
   * - :doc:`OpenVLA-OFT <embodied/openvla_oft>`
     - ✓
     - —
     - ✓
     - ✓
     - ✓
     - —
     - ✓
     - —
   * - :doc:`StarVLA <embodied/starvla>`
     - —
     - —
     - ✓
     - —
     - —
     - —
     - —
     - —

其他方向请见 :doc:`真机机器人 <real_world_index>`、:doc:`评测 <../evaluations/index>`、:doc:`智能体与推理 <agentic/index>` 与 :doc:`图像与视频生成 <video_models>`。
