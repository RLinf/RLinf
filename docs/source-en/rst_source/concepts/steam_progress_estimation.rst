STEAM Automatic Progress Estimation
===================================

Use RLT STEAM to estimate task progress from image pairs and switch control
automatically between the VLA reference policy, the RLT actor, and an expert
policy. The estimator runs at action-chunk boundaries and does not require a
hand-written progress reward.

The Two Heads
-------------

RLT STEAM uses two related heads with different decisions:

* The **STEAM value head** is part of ``steam_value_model``. It predicts a
  signed temporal-progress distribution for an image pair. Each ensemble
  member converts that distribution to a progress score, and the gate uses the
  minimum member score as a conservative signal for stalled progress. This
  head is needed for actor-to-expert takeover: after the actor starts, the gate
  must detect whether progress has stalled. That is a different decision from
  detecting phase entry, so the value head remains separate from the phase
  head.
* The **RLT phase head** is the small ``SteamPhaseHead`` trained on frozen
  STEAM hidden features. It predicts the probability that the trajectory has
  entered its critical phase. This head is needed for the VLA-to-actor switch:
  a low progress score alone cannot reliably identify that transition. A
  threshold and consecutive-chunk patience turn its probability into the
  switch.

At runtime, the flow is:

.. code-block:: text

   image history -> STEAM value model -> progress score
                 \-> SteamPhaseHead -> critical-phase probability
   phase probability + patience -> actor switch
   actor switch + low progress score + patience -> expert takeover

Run the Calibration
-------------------

1. Train the STEAM value model used by the rollout gate:

   .. code-block:: bash

      bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
          rlt_steam_value_model_sft

   Set the dataset and model paths in
   ``examples/offline_rl/config/rlt_steam_value_model_sft.yaml`` before running
   the command.

2. Enable ``algorithm.rlt_gate_calibration`` and
   ``rollout.rlt_critical_phase_gate.actor_switch.collect_phase_features`` in
   the STEAM Stage 2 config. For the first collection run, disable the learned
   actor and expert gates, and use the environment geometry route so the trace
   contains geometry labels and expert-oracle labels.

   Keep the existing model and path settings, and change only these fields:

   .. code-block:: yaml

      algorithm.rlt_gate_calibration.enable: True
      rollout.rlt_critical_phase_gate.enable: True
      rollout.rlt_critical_phase_gate.actor_switch.enable: False
      rollout.rlt_critical_phase_gate.actor_switch.collect_phase_features: True
      rollout.rlt_critical_phase_gate.expert_takeover.enable: False
      env.train.rlt_policy_switch.routing_source: environment
      env.train.rlt_policy_switch.expert_takeover.oracle_metrics.enable: True

3. Collect a short trace run:

   .. code-block:: bash

      STEPS=20 bash examples/embodiment/run_embodiment.sh \
          maniskill_rlt_stage2_td3_mlp_steam

   The trace files are written below the configured
   ``algorithm.rlt_gate_calibration.save_path``. ``STEPS`` is the runner step
   limit for this calibration run.

4. Train the phase head and select both gate parameter sets:

   .. code-block:: bash

      python toolkits/rlt/calibrate_steam_gates.py \
          logs/<run>/gate_calibration \
          --phase-head-output logs/<run>/steam_phase_head.pt \
          --yaml-output logs/<run>/steam_gate_recommendation.yaml \
          --device cuda

   The script keeps complete episodes, trains ``SteamPhaseHead`` from geometry
   critical-phase labels, replays the actor switch, and then calibrates the
   expert gate from the STEAM minimum score. It writes a phase-head checkpoint,
   a YAML recommendation, and validation metrics in the terminal.

5. Apply the recommendation in
   ``rollout.rlt_critical_phase_gate``:

   * set ``actor_switch.phase_head_path`` to the generated checkpoint;
   * enable ``actor_switch`` and ``expert_takeover``;
   * set ``env.train.rlt_policy_switch.routing_source`` to ``rollout``.

   Then launch the normal Stage 2 run:

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh \
          maniskill_rlt_stage2_td3_mlp_steam

How the Gate Decides
--------------------

The gate keeps a short image history and forms a pair after
``lookback_chunks``. The STEAM model scores the pair. Once the phase-head
probability stays above its calibrated threshold for the configured patience,
the RLT actor becomes active and remains active until the episode ends. After
the configured expert warmup, consecutive low STEAM scores trigger expert
takeover. The selected policy controls a complete action chunk.

Keep ``chunk_size`` aligned with the rollout action chunk and with the temporal
stride used to train the STEAM value model. The learned rollout gate owns
expert requests in the final configuration, so the environment-side expert
takeover must remain disabled when ``routing_source`` is ``rollout``.
