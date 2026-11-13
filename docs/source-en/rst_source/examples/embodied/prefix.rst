Prefix Fine-Tune: Train Online from an Existing VLA
===================================================

A vision-language-action model (VLA) reads camera images, a language instruction, and the robot state, and outputs an action. Image understanding, language, and action generation live in one large model.

RL Token (RLT) uses that pretrained model in two stages. Stage 1 trains the VLA with a token transformer on demonstrations and compresses the vision-language features into a compact representation. Stage 2 freezes the feature model and trains two small networks from that representation, the robot state, and the VLA reference action: the actor outputs actions, and the critic estimates later return. Online updates run only through those two networks. The large model runs forward only.

The Prefix framework builds on that Stage 2 loop. The feature source, the network, and the algorithm are configurable, and data collection and the parameter update stay on one path. Features can come from RLT tokens or from pooling the frozen VLA vision-language features. Those features are the prefix. A policy is the model that turns an observation into an action.

During training the feature model stays frozen. The actor reads the prefix, the robot state, and the reference action, and outputs the action to execute. After the environment executes that action, the run writes the features, the action, the reward, and the termination into the replay buffer. Updates sample that buffer, and backpropagation goes through the actor and the critic. The VLA keeps supplying features and reference actions, and its weights stay unchanged.

This page runs that loop on ManiSkill peg insertion from published OpenPI checkpoints. OpenPI is the VLA connected today. Another VLA needs an adapter, covered in :doc:`Prefix components <../../extending/prefix>`. If you already trained an RL Token representation, the Stage 2 configs on :doc:`RL Token <rlt>` use the same update path.

Overview
--------

Run the pooled-feature preset first. It already selects TD3 and the STEAM stage switch. A preset names the environment, models, algorithm, and device placement; you select it by the config file name.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      TD3, with an AC preset

   .. grid-item-card:: Models
      :text-align: center

      Frozen OpenPI π₀.₅ · MLP actor/critic

   .. grid-item-card:: Environments / Data
      :text-align: center

      ManiSkill peg insertion

   .. grid-item-card:: Training
      :text-align: center

      Online rollout · replay · STEAM routing

How Training Works
------------------

One decision starts from the environment observation. The frozen VLA encodes images and language into prefix features and also emits a reference action chunk, which is the action the existing model would have taken. The small actor reads those features, the robot's current joint state (proprioception), and the reference chunk, and outputs the action to execute. The critic estimates the return that action may lead to. The actor emits several control steps at once; that block is an action chunk. This example uses 10 steps per chunk and 8 numbers per step. Those numbers follow the robot's control mode.

After the environment executes the chunk, the run stores the extracted features, the action that was sent, the reward, and the termination. That store is the replay buffer. Updates sample it repeatedly and backpropagate only through the small network. The VLA stays in the loop for feature extraction and reference actions, and its weights are not updated.

RLinf calls environment interaction and action collection rollout, and calls the parameter update training. The two alternate, and the updated policy is synced back to the rollout side. The checkpoint you keep from online training is the small policy. The frozen VLA checkpoint stays the one you downloaded.

The example also uses STEAM to judge task progress from images at different times. The VLA reference action controls the robot at the start. After the training warmup finishes and STEAM marks a critical phase, the actor is allowed to take over. That choice of action source is routing. STEAM decides when to switch; the actor learns how to move. You can change the two settings separately.

Choose a Starting Configuration
-------------------------------

Pooled features can come from a base VLA or from one already supervised-finetuned (SFT) on demonstrations. RLT is the other representation: Stage 1 trains a small network that compresses VLA features, and Stage 2 freezes it and trains the actor and critic. Both routes share the update loop below.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Experiment
     - Config under ``examples/embodiment/config/``
   * - Pooled VLM features, TD3, and STEAM stage switching
     - ``maniskill_prefix_stage2_td3_mlp_steam.yaml``
   * - RLT token features and TD3
     - ``maniskill_rlt_stage2_td3_mlp.yaml``
   * - RLT token features and AC
     - ``maniskill_rlt_stage2_ac_mlp.yaml``
   * - Collect STEAM calibration traces
     - ``maniskill_prefix_stage2_td3_mlp_steam_calibration.yaml``

The steps below use the first config. Use the RLT configs after that flow runs and you want to compare representations or algorithms; their Stage 1 preparation is on the :doc:`RL Token <rlt>` page. Use the calibration config when you need to refit the stage decision for a new task, camera, or controller.

Installation
------------

These steps target a Linux machine with an NVIDIA GPU, access to GitHub and Hugging Face, and disk space for the models, dependencies, logs, and training checkpoints. The GPU has to hold the VLA, the STEAM model, and the small network at the same time. The example runs 64 training environments, so the simulator adds to that memory use.

The example starts online training from published checkpoints. ManiSkill generates the interaction data during the run, and the downloaded checkpoints include the normalization statistics that put robot state and actions on the scale the VLA was trained with.

Clone the repository:

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://gh-proxy.com/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

Install and activate the dependencies with either the published image or a local environment.

**Option 1: Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

Inside the container, switch to the OpenPI virtual environment:

.. code:: bash

   source switch_env openpi

**Option 2: Local environment**

.. code:: bash

   # For mainland China users, you can add --use-mirror to the install.sh command.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

The install script prepares the embodied stack, the OpenPI model, and the ManiSkill simulator. The second command activates the environment it created. Run the download and training commands in that terminal. In a new terminal, enter the repository and activate the environment again.

Download the Models
-------------------

The settings for this run live in ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml``. Keep the network and algorithm values, fill in the model paths, then check routing and logging before you launch.

The example loads four models. The feature VLA produces the representation and the reference action. The expert VLA is a second trained model, available if you later enable takeover. The STEAM value model scores progress from images. The phase head is the small classifier that decides whether the task has entered the critical phase. The first run uses all four with the preset settings.

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Resource
     - Hugging Face repository
     - Config field
   * - Feature VLA, step 1000
     - `ChikaYokoyama/maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`_
     - ``rollout.prefix_feature_model.model_path``
   * - Expert VLA, step 3000
     - `ChikaYokoyama/maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`_
     - ``rollout.expert_model.model_path``
   * - STEAM value model, step 5000
     - `ChikaYokoyama/rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`_
     - ``rollout.routing_gate.model.model_path``
   * - Phase head and its saved recommendation
     - `ChikaYokoyama/rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`_
     - ``rollout.routing_gate.actor_switch.phase_head_path``

Download the four repositories into ``data/steam_peg``, and also download the SigLIP and Gemma backbones STEAM uses for images and language. For hf-mirror, first run ``export HF_ENDPOINT=https://hf-mirror.com``.

.. code:: bash

   CHECKPOINT_ROOT="$(pwd)/data/steam_peg"

   hf download ChikaYokoyama/maniskill_rlt_1000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/maniskill_rlt_3000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
     --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
     --exclude 'dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_phase_head \
     --local-dir "$CHECKPOINT_ROOT"

   hf download google/siglip-so400m-patch14-384 \
     --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
   hf download google/gemma-3-270m \
     --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

The first two repositories already contain ``global_step_1000/actor`` and ``global_step_3000/actor``, so they share one ``checkpoints`` directory. The STEAM repository stores the actor files at its root, so its download target includes ``global_step_5000/actor``. The commands skip distributed training state and keep the inference weights, normalization statistics, and processor files.

Set the Model Paths
-------------------

Open ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml`` and replace the four checkpoint fields with absolute paths under the directory from the previous section. ``<CHECKPOINT_ROOT>`` below means that expanded absolute path. Do not leave the angle-bracket text in the file.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Field
     - Local path
   * - ``rollout.prefix_feature_model.model_path``
     - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1000/actor``
   * - ``rollout.expert_model.model_path``
     - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor``
   * - ``rollout.routing_gate.model.model_path``
     - ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor``
   * - ``rollout.routing_gate.actor_switch.phase_head_path``
     - ``<CHECKPOINT_ROOT>/steam_phase_head.pt``

Keep ``openpi_data.repo_id: maniskill_peginsertionside_joint`` and ``norm_stats_path: null``. Each OpenPI checkpoint then reads its own ``maniskill_peginsertionside_joint/norm_stats.json``.

Then point the STEAM checkpoint at the backbones you downloaded. Edit ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``config.json`` field
     - Local path
   * - ``vision_repo_id``
     - ``<CHECKPOINT_ROOT>/siglip-so400m-patch14-384``
   * - ``language_repo_id``
     - ``<CHECKPOINT_ROOT>/gemma-3-270m``

The tokenizer and image-processor files stay in that same ``actor`` directory. The gate loads them from there.

Run Training
------------

With the paths filled in, keep the example routing, check devices and logging, and start from the repository root.

Keep the Example Routing
~~~~~~~~~~~~~~~~~~~~~~~~

The run has to choose an action from the VLA, the actor, or the expert. STEAM's stage and takeover decisions are gate signals. With the block below, the actor switch is applied, and the expert stays in shadow mode: the decision is recorded and the expert action is not executed.

.. code:: yaml

   rollout:
     routing_gate:
       actor_switch:
         enable: True
         mode: active
         enter_threshold: null
         patience_chunks: null
       expert_takeover:
         enable: True
         mode: shadow
         warmup_chunks: null
         patience_chunks: null
         enter_threshold: null

This is a fragment of the full config. Do not replace the file with it. ``enter_threshold`` is the score that triggers a decision, ``patience_chunks`` is how many consecutive chunks must agree, and ``warmup_chunks`` is how many chunks to wait before checking expert takeover. ``null`` uses the values saved in the phase-head file. A number overrides that saved value.

.. warning::

   The published expert threshold did not trigger takeover on its calibration validation split, so it is not a reason to enable automatic takeover. Keep shadow mode for the first run. Collect traces from episodes the actor actually controlled, check the new threshold, and only then consider ``expert_takeover.mode: active``.

Check Devices and Logging
~~~~~~~~~~~~~~~~~~~~~~~~~

``cluster.component_placement`` in the preset puts the actor on GPU 0 and the environment and rollout workers on every device (``all``). Before launching, confirm those IDs exist on the machine. ``env.train.total_num_envs`` is the number of parallel training environments. The example uses 64; change it together with the GPU memory and simulator load.

The example logs metrics to Weights & Biases. Finish that account setup in the same environment before you launch. To read metrics on the machine instead, set ``runner.logger.logger_backends`` to ``["tensorboard"]``. The calibration preset already uses that local backend.

Start Training
~~~~~~~~~~~~~~

From the repository root, with the environment activated:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

The command loads the models, starts the simulator, collects interaction data, and updates the small network. Console output is written to ``logs/<YYYYMMDD-HH:MM:SS>-maniskill_prefix_stage2_td3_mlp_steam/run_embodiment.log``. Each launch gets its own directory, so you can tell experiments apart.

Visualization and Results
-------------------------

After launch, confirm that data collection and parameter updates are moving, then use the periodic evaluation to see whether the task is being solved.

Training Progress
~~~~~~~~~~~~~~~~~

The run first fills the replay buffer and finishes the warmup updates. The example waits for 10,000 replay records and 30,000 warmup updates before the actor may take over. STEAM still decides whether the task is in the critical phase. Seeing the robot continue to follow the VLA reference action does not mean training has stopped.

In the logger you selected, the progress keys are prefixed with ``train/prefix/``:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Metric
     - What it tells you
   * - ``train/prefix/global_total_transitions_added``
     - Keeps increasing while interaction data is written to replay
   * - ``train/prefix/global_min_replay_size``
     - Reaches the warmup capacity required before updates can start
   * - ``train/prefix/update_step``
     - Increases while the learner is applying parameter updates
   * - ``train/prefix/ready_for_online``
     - ``1`` means the warmup condition is met; the actor still waits for the stage gate

These keys show that the loop is advancing. They do not show that the task has been learned. Use the periodic evaluation for task success. This runner counts one rollout-and-update cycle as one step, and the preset runs one such step per epoch. Evaluation runs every 25 steps and a checkpoint is saved every 50 steps, set by ``runner.val_check_interval`` and ``runner.save_interval``. Keep the training checkpoint together with the VLA checkpoint and normalization files it was trained against, so you can resume or evaluate later.

View the Curves
~~~~~~~~~~~~~~~

For TensorBoard, set ``runner.logger.logger_backends`` to ``["tensorboard"]``, then start the viewer from the repository root in the activated environment:

.. code:: bash

   tensorboard --logdir logs --port 6006

Open ``http://localhost:6006`` on the machine running TensorBoard and select this run. From a remote machine, use port forwarding. With Weights & Biases, open the same metric names in that experiment.

Record Evaluation Results
~~~~~~~~~~~~~~~~~~~~~~~~~

When you compare features or algorithms, keep the task, evaluation initial states, episode count, and action-chunk length the same, and record which checkpoint you evaluated. Training progress tells you the run is healthy. Success rate is what you compare.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Record
     - What to write down
   * - Setup
     - VLA checkpoint, feature pooling, actor/critic, and algorithm config
   * - Training progress
     - Number of interaction records collected and parameter updates
   * - Evaluation scale
     - Number of evaluation episodes and the initial-state setting
   * - Task success rate
     - Successful episodes divided by evaluation episodes
   * - Motion
     - Grasp, approach to the hole, insertion, and actor or expert takeover

To save videos, set ``env.eval.video_cfg.save_video`` to ``true``. Evaluation videos are written to ``video/eval`` under that run's log directory. Read the success numbers from the evaluation output of the run.

Replace Components Through the Config
-------------------------------------

Change the experiment only after collection and updates are both moving, and change one factor at a time. Compare representations first, then algorithms, so you can tell which edit changed the result.

Three groups of fields cover the usual edits. ``rollout.prefix_feature_model`` selects the frozen VLA and its feature. ``actor.model`` selects the trainable network. ``algorithm`` selects the update rule. Edit these inside the full config you already launched.

Switch the VLA Checkpoint and Feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use another checkpoint of the same OpenPI setup, change ``rollout.prefix_feature_model.model_path`` and check that the task config and normalization statistics match. A different VLA architecture needs an adapter. OpenPI is connected; the registration steps are in :doc:`Prefix components <../../extending/prefix>`.

The VLA emits one vector per image patch and language token. The policy needs a fixed-length input, so those vectors are reduced to one vector. That reduction is pooling. Padding and other invalid positions are marked by a mask. ``masked_mean`` skips the masked positions. Set the reduction with ``rollout.prefix_feature_model.prefix.pool``:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - ``pool``
     - What it computes
     - When to use it
   * - ``masked_mean``
     - Mean of the valid prefix tokens
     - Base or SFT VLA features, including this example
   * - ``mean``
     - Mean of every selected prefix token, including masked positions
     - An experiment that compares pooling over the full sequence
   * - ``last``
     - The last valid token, or the end of the sequence when there is no mask
     - An experiment that compares the final token
   * - ``rlt_token``
     - The trained RLT token transformer
     - A checkpoint that contains the matching Stage 1 RLT module

``prefix.image_only: true`` builds the feature from image tokens only. ``false`` uses image and language tokens. Set both fields explicitly when you compare representations. ``rlt_token`` also needs ``openpi.use_rlt: true`` and Stage 1 module settings that match the checkpoint. Start that route from an existing RLT preset.

Switch the Actor
~~~~~~~~~~~~~~~~

AC and TD3 are two ways to update the actor and critic, and they need different actors. Change the actor when you change the algorithm. The critic in both shipped presets estimates two Q values for the same action and uses them to build the training target.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Field
     - AC preset
     - TD3 preset
   * - ``algorithm.loss_type``
     - ``prefix_off_policy``
     - ``prefix_off_policy``
   * - ``algorithm.name``
     - ``ac``
     - ``td3``
   * - ``actor.model.model_type``
     - ``prefix_policy``
     - ``prefix_policy``
   * - ``actor.model.actor_head.name``
     - ``fixed_std_mlp``
     - ``deterministic_mlp``
   * - ``algorithm.actor_agg_q``
     - ``q1``
     - ``min``, ``q1``, or ``mean``
   * - ``actor.model.critic_head.name``
     - ``multi_q_mlp`` with ``num_q_heads: 2``
     - ``twin_q_mlp``

AC uses a stochastic actor. Its preset sets ``fixed_std: 0.002``, the fixed standard deviation. TD3 uses a deterministic actor with action noise. Its preset sets ``noise_sigma: 0.1``. Copy the head block from the matching preset when you switch, including those scales.

Both algorithms keep ``algorithm.entropy_tuning.alpha_type: fixed_alpha``, so the entropy temperature stays at the configured value and is not learned. The critic target uses the smaller of the first two Q estimates, so leave ``algorithm.agg_q`` unset or set it to ``min``. ``actor_agg_q`` is separate: it chooses how Q values are combined when the actor is updated. AC accepts only ``q1``.

Adjust the Network and Proprio History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The head is the small network attached to the frozen feature. To change its width, edit ``actor.model.actor_head.hidden_dims`` and ``actor.model.critic_head.hidden_dims``. ``[256, 256]`` is two hidden layers of width 256, which is what the TD3 preset uses.

Stay with that preset's critic head until you have a reason to study another value estimator. ``multi_q_mlp`` lets you set the number of Q networks. ``cross_q_mlp`` evaluates the current and next state together. Config validation checks the Q-head count and the other conditions the algorithm requires.

If the current proprio state is not enough to describe the motion, give the policy a short history. Set ``actor.model.state_history.enable``, the number of steps in ``steps``, and the padding in ``pad`` (``zero`` or ``repeat``). History advances once per policy decision and is cleared when that environment resets. ``actor.model.z_dim`` stays the original feature width. The extra width from history is computed from the config.

Adjust the Objective and Update Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actor is trained both to increase the estimated return and to stay close to the VLA reference action. The second term is behavior cloning (BC). ``algorithm.q_weight`` and ``algorithm.bc_weight`` set the two weights. A step marked as a human intervention is cloned toward the intervention action instead of the VLA reference.

The example turns on ``algorithm.actor_weight_schedule``, which moves those weights as training progresses. For fixed weights, set that schedule's ``enable`` to ``false``, then edit ``q_weight`` and ``bc_weight``.

``algorithm.rlt_schedule`` sets the update cadence. ``warmup_min_size`` is the replay size required before training starts. ``warmup_post_collect_updates`` is the number of warmup updates required before the actor may take over online. Later updates can be scheduled by new transitions or by finished episodes. An episode is one task from reset to termination. A transition is one stored observation, action, and outcome.

The critic uses the rewards inside the action chunk and the estimated value of the state after the chunk. Later rewards are discounted. Keep those learning settings for the first run, and change them after collection and updates are both healthy.

Check Inputs for a New Task
---------------------------

Before moving the same preset to another task, make the frozen VLA, the environment, and the policy agree on inputs and actions. The VLA hands the policy three fields:

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - Field
     - Shape from the feature model
     - What to check
   * - ``z_rl``
     - ``[B, z_dim]``
     - The raw feature width matches ``actor.model.z_dim``
   * - ``proprio``
     - ``[B, proprio_dim]``
     - Joint-state width, meaning, and normalization match
   * - ``ref_chunk``
     - ``[B, ref_num_action_chunks, action_dim]``
     - Action width matches the environment, and the reference covers the executed length

``B`` is the batch size. ``num_action_chunks`` is the chunk length the policy executes. The policy concatenates ``z_rl`` with the current ``proprio`` as its state, and uses the first ``num_action_chunks`` steps of ``ref_chunk`` as the actor condition and the default BC target.

Before the first launch on a new task, check the environment image fields, the language instruction, the OpenPI data config, and ``norm_stats.json``. With STEAM, also check that ``rollout.routing_gate.cameras`` maps each STEAM camera key to the environment image that camera actually sees. This example maps the external camera to ``main_images`` and the wrist camera to ``wrist_images``. After a camera change, point STEAM at the new images and check the stage decision again.

Calibrate STEAM for Your Task
-----------------------------

When the task, cameras, or control behavior change, refit the critical-phase decision and the expert threshold. Collect features and labels with the calibration preset, then fit the phase head offline.

From the repository root, with the model paths already set:

.. code:: bash

   STEPS=20 SAVE_INTER=-1 bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

This preset routes with the environment geometry gate, turns on ``collect_phase_features`` and ``algorithm.rlt_gate_calibration.enable``, and writes features, geometry labels, and diagnostics to ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``. Each launch creates ``<run_log_dir>`` under ``logs/``.

Twenty steps only check that collection works. Once both positive and negative geometry labels appear, you can fit a first phase head. That short run does not finish the training preset's 30,000 warmup updates, and it does not validate expert takeover. Replace ``/path/to/run_log_dir`` with the log directory from this launch. ``CHECKPOINT_ROOT`` is the model directory from the download section.

.. code:: bash

   TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
   python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
     --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
     --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

The tool writes ``steam_phase_head.pt`` and ``steam_gate_recommendation.yaml``. After you check the validation metrics, point the training preset's phase-head path at the new file. Leave the gate thresholds at ``null`` so the run reads the recommendation stored in that checkpoint. The expert threshold still needs a later calibration on episodes the actor controlled. Day-to-day training uses the training preset, with both collection switches off.

What the Current Config Can Change
----------------------------------

You can stay inside this page to use the published models, change feature pooling, change network width, or switch between AC and TD3. A new VLA architecture or a new learning objective needs adapter code. Changing the model or algorithm name to an arbitrary string does not connect it.

Another VLA has to turn observations into ``z_rl``, ``proprio``, and ``ref_chunk``, with action units, range, and chunk length that match the environment. A new network has to read that same feature and return an action or a value. A new algorithm has to define the actor and critic training targets. After that registration, you still select the component in the config and keep the same collection and update flow.

When you need that adapter code, use the implementation and registration examples in :doc:`Prefix components <../../extending/prefix>`. Running this example and editing the existing components does not require those interfaces first.

Common Problems
---------------

When startup or behavior looks wrong, check the config and the log for that symptom before changing the model, algorithm, and environment together.

If a model or normalization file cannot be found, confirm that all four checkpoints point at real ``actor`` directories and that each OpenPI checkpoint contains its own ``norm_stats.json``. If STEAM still opens a path from the machine that produced the checkpoint, edit ``vision_repo_id`` and ``language_repo_id`` in its ``config.json``, and check the phase-head path in the recommendation YAML.

If the ``z_rl`` width does not match, compare the feature width declared by the VLA with ``actor.model.z_dim``. With history enabled, the extra width comes from the history length and ``proprio_dim``. Do not add that width into ``z_dim`` again. If validation fails after an algorithm switch, confirm that AC or TD3 is paired with its actor, and check ``actor_agg_q``, the Q-head count, and the fixed entropy temperature. If the action scale or direction looks wrong, compare the environment action definition, the OpenPI data config, the per-step action width, and the normalization statistics.

If the actor never takes over, confirm that the replay warmup and the update warmup have both finished, that the stage gate has fired, and that ``rollout.routing_gate.actor_switch.mode`` is ``active``. If the expert gate is logged but the expert never moves the robot, ``expert_takeover.mode`` is still ``shadow``. Change it to ``active`` only after the calibrated threshold checks out.
