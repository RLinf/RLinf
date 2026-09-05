STEAM 自动进度估计
==================

使用 RLT STEAM 从图像帧对估计任务进度，并在 VLA reference policy、RLT actor
和 expert policy 之间自动切换控制。估计器在 action chunk 边界运行，不需要手工设计的进度奖励。

两个 Head
---------

RLT STEAM 使用两个职责不同但相互关联的 head：

* **STEAM value head** 是 ``steam_value_model`` 的一部分。它对图像帧对预测带符号的时序进度分布。每个 ensemble member 将分布转换为进度分数，gate 取所有 member 分数的最小值，作为保守的停滞进度信号。这个 head 用于 actor-to-expert takeover：actor 启动后，gate 需要判断进度是否停滞。这个判断不同于是否进入 critical phase，因此 value head 与 phase head 分开。
* **RLT phase head** 是在冻结 STEAM hidden features 上训练的小型 ``SteamPhaseHead`` 。它用于 VLA-to-actor switch：gate 需要判断轨迹是否进入 critical phase，仅靠较低的进度分数无法可靠识别这个转折。将概率、阈值和连续 chunk 数结合后，得到 switch。

运行时流程如下：

.. code-block:: text

   image history -> STEAM value model -> progress score
                 \-> SteamPhaseHead -> critical-phase probability
   phase probability + patience -> actor switch
   actor switch + low progress score + patience -> expert takeover

运行校准
--------

1. 训练 rollout gate 使用的 STEAM value model：

   .. code-block:: bash

      bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
          rlt_steam_value_model_sft

   运行前在 ``examples/offline_rl/config/rlt_steam_value_model_sft.yaml`` 中设置数据集和模型路径。

2. 在 STEAM Stage 2 配置中启用 ``algorithm.rlt_gate_calibration`` 和
   ``rollout.rlt_critical_phase_gate.actor_switch.collect_phase_features`` 。首次收集时关闭 learned actor gate 和 expert gate，并使用环境 geometry route，以便 trace 同时包含 geometry labels 和 expert-oracle labels。

   保留现有的 model 和路径配置，只修改下面这些字段：

   .. code-block:: yaml

      algorithm.rlt_gate_calibration.enable: True
      rollout.rlt_critical_phase_gate.enable: True
      rollout.rlt_critical_phase_gate.actor_switch.enable: False
      rollout.rlt_critical_phase_gate.actor_switch.collect_phase_features: True
      rollout.rlt_critical_phase_gate.expert_takeover.enable: False
      env.train.rlt_policy_switch.routing_source: environment
      env.train.rlt_policy_switch.expert_takeover.oracle_metrics.enable: True

3. 收集一小段 trace：

   .. code-block:: bash

      STEPS=20 bash examples/embodiment/run_embodiment.sh \
          maniskill_rlt_stage2_td3_mlp_steam

   trace 会写入配置的 ``algorithm.rlt_gate_calibration.save_path`` 。这里的 ``STEPS`` 是本次校准运行的 runner step 上限。

4. 训练 phase head 并选择两个 gate 的参数：

   .. code-block:: bash

      python toolkits/rlt/calibrate_steam_gates.py \
          logs/<run>/gate_calibration \
          --phase-head-output logs/<run>/steam_phase_head.pt \
          --yaml-output logs/<run>/steam_gate_recommendation.yaml \
          --device cuda

   脚本只保留完整 episode，使用 geometry critical-phase labels 训练 ``SteamPhaseHead`` ，回放 actor switch，再使用 STEAM 最小分数校准 expert gate。它会保存 phase-head checkpoint、YAML recommendation，并在终端输出验证指标。

5. 在 ``rollout.rlt_critical_phase_gate`` 中应用 recommendation：

   * 将 ``actor_switch.phase_head_path`` 设置为生成的 checkpoint；
   * 启用 ``actor_switch`` 和 ``expert_takeover`` ；
   * 将 ``env.train.rlt_policy_switch.routing_source`` 设置为 ``rollout`` 。

   然后启动正常的 Stage 2 运行：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh \
          maniskill_rlt_stage2_td3_mlp_steam

Gate 如何决策
-------------

gate 保存短的图像历史，并在达到 ``lookback_chunks`` 后形成帧对。STEAM 模型为帧对打分。当 phase-head 概率连续若干 chunk 高于校准阈值时，RLT actor 开始接管，并持续到 episode 结束。经过配置的 expert warmup 后，如果 STEAM 分数连续偏低，就触发 expert takeover。最终由选中的 policy 控制完整的 action chunk。

保持 ``chunk_size`` 与 rollout action chunk 以及 STEAM value model 的训练时序 stride 对齐。最终配置中 learned rollout gate 负责产生 expert request，因此当 ``routing_source`` 为 ``rollout`` 时，必须关闭环境侧 expert takeover。
