Psi0 + SIMPLE Evaluation
========================

Run the matching task-specific Psi0 checkpoint on SIMPLE's CloseDoor or
OpenFaucet Teleop task through RLinf's standard evaluation entry point. This
integration is evaluation-only. It freezes Psi0 System-2 and
the SIMPLE controller, runs the official deterministic Euler sampler, and
executes 24 actions from each 30-action RTC plan.

Scope and Versions
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Component
     - Fixed version
     - Role
   * - `Psi0 <https://github.com/physical-superintelligence-lab/Psi0/tree/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06>`_
     - ``a32e57a3fabb8590c80677f9cd3d1fc3db60eb06``
     - Frozen VLM plus eval-only action expert
   * - `SIMPLE <https://github.com/physical-superintelligence-lab/SIMPLE/tree/5e3d6f84e85343e34e9bca8d157f0d7813231185>`_
     - ``5e3d6f84e85343e34e9bca8d157f0d7813231185``
     - CloseDoor/OpenFaucet tasks and decoupled-WBC runtime
   * - Psi0 checkpoint
     - ``ckpt_40000`` from the matching CloseDoor or OpenFaucet run
     - Task-specific evaluation policy
   * - SIMPLE reset states
     - LeRobot v2.1 episodes named for the selected task
     - Deterministic episode initialization

Install the Runtime
-------------------

Use Python 3.10 and the dedicated model/environment installer:

.. code-block:: bash

   bash requirements/install.sh embodied \
     --venv .venv-psi0-simple \
     --model psi0 \
     --env simple
   source .venv-psi0-simple/bin/activate

The Psi0/SIMPLE installer branch selects Python 3.10 automatically. It checks
out the fixed Psi0 and SIMPLE revisions, initializes
SIMPLE's pinned controller submodules, and builds its CuRobo extension. It does
not install SIMPLE's optional RLDS/TensorFlow data stack and does not download
the Psi0 checkpoint or reset-state dataset. Use a new ``--venv`` path when
validating a changed installer; reusing an existing environment also reuses its
previous packages.

Download the Model and Data
---------------------------

The CloseDoor evaluation needs one task-specific joint checkpoint. Its
``model.safetensors`` contains both System-2 ``vlm_model.*`` weights and
System-1 ``action_header.*`` weights. The two foundation checkpoints listed in
``argv.txt`` were fine-tuning initialization inputs; evaluation does not load
them again.

The fixed
`SIMPLE checkpoint tree <https://huggingface.co/USC-PSI-Lab/psi-model/tree/d34a91932d25c45ef211582315b9224c7dc8ace9/psi0/simple-checkpoints>`_
contains separate task-specific checkpoints, not one checkpoint for the whole
SIMPLE task collection. The adapter supports CloseDoor and OpenFaucet Teleop,
and their checkpoints are not interchangeable. Do not use BendPick MP or other
task checkpoints for these evaluations. Other tasks may also use a different
controller/System-0 path.

Set a dedicated artifact root and Hugging Face cache. Keep the same
``HF_HOME`` when you later launch the evaluation:

.. code-block:: bash

   export PSI0_ARTIFACT_ROOT=/absolute/path/to/psi0-simple-artifacts
   export HF_HOME="${PSI0_ARTIFACT_ROOT}/hf-cache"
   mkdir -p "${PSI0_ARTIFACT_ROOT}" "${HF_HOME}"

Download the CloseDoor run directory at the fixed revision. Do not download
only ``model.safetensors`` because the loader also reads ``argv.txt`` and
``run_config.json`` from the run directory:

.. code-block:: bash

   export PSI0_MODEL_REV=d34a91932d25c45ef211582315b9224c7dc8ace9
   export PSI0_RUN_REL=psi0/simple-checkpoints/g1wholebodyclosedoorteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605070100

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_RUN_REL}/argv.txt" \
     "${PSI0_RUN_REL}/envs.txt" \
     "${PSI0_RUN_REL}/run_config.json" \
     "${PSI0_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision "${PSI0_MODEL_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   export PSI0_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_RUN_REL}"

The pinned Ψ₀ loader uses
`Qwen/Qwen3-VL-2B-Instruct <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203>`_
to construct the model configuration and processor. The joint checkpoint
already contains the policy weights, so prefetch only the config, tokenizer,
and processor metadata. Do not download Qwen's ``model.safetensors``:

.. code-block:: bash

   hf download Qwen/Qwen3-VL-2B-Instruct \
     chat_template.json config.json generation_config.json merges.txt \
     preprocessor_config.json tokenizer.json tokenizer_config.json \
     video_preprocessor_config.json vocab.json \
     --revision 89644892e4d85e24eaac8bacfd4f463576704203

.. note::

   The pinned upstream loader still resolves the Qwen repository by model ID
   and does not accept a revision argument. This command prefetches the verified
   metadata snapshot, but keep network access available for the first load so
   Hugging Face can resolve that model ID into the cache. The current
   integration does not claim fully offline loading.

Download and extract the reset states for the same task and fixed revision:

.. code-block:: bash

   export SIMPLE_DATA_REV=6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203
   export SIMPLE_RESET_REL=simple-eval/G1WholebodyCloseDoorTeleop-v0.zip

   hf download USC-PSI-Lab/psi-data "${SIMPLE_RESET_REL}" \
     --repo-type dataset \
     --revision "${SIMPLE_DATA_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export SIMPLE_RESET_ZIP="${PSI0_ARTIFACT_ROOT}/psi-data/${SIMPLE_RESET_REL}"
   export SIMPLE_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/closedoor"
   mkdir -p "${SIMPLE_RESET_DIR}"
   unzip -q "${SIMPLE_RESET_ZIP}" -d "${SIMPLE_RESET_DIR}"

Verify the two runtime artifacts. The checkpoint is about 6.25 GB, while the
reset-state archive is about 421 KB:

.. code-block:: bash

   printf '%s  %s\n' \
     c632adea90e2d8bfee92559b091e8f8ea465aae0a5ec23a244390bf12c3fce50 \
     "${PSI0_RUN_DIR}/checkpoints/ckpt_40000/model.safetensors" \
     | sha256sum --check
   printf '%s  %s\n' \
     6bff1d5fc5a332968e57eef78b781fe929c70ac833d040cc7e589e80cae2a286 \
     "${SIMPLE_RESET_ZIP}" \
     | sha256sum --check
   test -f "${PSI0_RUN_DIR}/argv.txt"
   test -f "${PSI0_RUN_DIR}/run_config.json"
   find "${SIMPLE_RESET_DIR}" -path '*/level-0/meta/episodes.jsonl' -print -quit

The final command must print exactly one LeRobot metadata file. The Eval adapter
selects ``level-0`` from ``env.eval.reset_dataset.dr_level`` and reads each
episode's ``environment_config`` directly; it does not read the Parquet/video
payload during reset.

You do not need to download controller/System-0 weights separately from the
Hub. The installer obtains the required Balance/Walk ONNX files from the pinned
SIMPLE checkout and its ``decoupled_wbc`` submodule.

Run the Smoke Evaluation
------------------------

For this single-node smoke, call the standard evaluation entry point directly.
RLinf connects to an existing Ray cluster when one is available and otherwise
starts a local Ray instance. Do not run ``ray start --head`` before every Eval.
The smoke recipe disables optional metric backends, so its result is reported
in the process output without importing TensorBoard or TensorFlow:

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_closedoor_psi0_eval \
     rollout.model.model_path="${PSI0_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}"

If a Ray cluster was started from a different virtual environment, stop that
old cluster once before switching environments. Pre-start Ray only for a
deliberately managed multi-node cluster.

The smoke config uses one environment and one reset episode. The adapter runs
the official 300-step maximum stabilization procedure, then executes complete
24-action chunks until task success or the 1,000-step boundary. Automatic
reset is disabled so one rollout epoch remains one official trial. The env and
rollout workers stop together after termination; any unused slots in the final
chunk are recorded in ``executed_mask`` instead of being sent to the
controller.

Run the Full Reference Evaluation
---------------------------------

Run 30 evaluations—10 fixed episodes at each of three DR levels—to reproduce
the official CloseDoor protocol. ``simple_closedoor_psi0_eval`` already fixes
the model inference, action execution, and simulation protocol. The command
only supplies local paths and selects the reset episodes to run.

The config fixes these protocol settings; do not repeat them on the command
line:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Protocol area
     - Fixed value
     - Purpose
   * - Model inference
     - bf16, 10-step deterministic Euler
     - Matches the official Psi0 CloseDoor inference settings
   * - RTC action execution
     - Plan 30 actions, execute 24, retain a six-action overlap
     - Matches the official ``--action-exec-horizon=24 --rtc`` arguments
   * - Task and simulation
     - ``G1WholebodyCloseDoorTeleop-v0``, ``decoupled_wbc``, ``mujoco_isaac``
     - Uses the official Teleop controller and simulation mode
   * - Timing and termination
     - 50 Hz control/render, 200 Hz physics, up to 300 stabilization steps, 1,000-step horizon
     - Preserves the official control rate and episode boundary

The command overrides the following values:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Parameter
     - Value
     - Meaning
   * - ``rollout.model.model_path``
     - ``${PSI0_RUN_DIR}``
     - Local model directory containing the CloseDoor ``ckpt_40000`` checkpoint
   * - ``env.eval.reset_dataset.path``
     - ``${SIMPLE_RESET_DIR}``
     - Extracted official fixed-reset dataset
   * - ``env.eval.reset_dataset.dr_level``
     - ``0``, ``1``, ``2``
     - Selects the three DR levels in sequence through the loop
   * - ``env.eval.reset_dataset.episode_start``
     - ``0``
     - Starts from the first fixed episode in each level
   * - ``env.eval.reset_dataset.num_episodes``
     - ``10``
     - Loads all 10 fixed episodes for the selected level
   * - ``env.eval.rollout_epoch``
     - ``10``
     - Runs 10 rollouts, consuming those 10 episodes in order
   * - ``env.eval.video_cfg.fps``
     - ``50``
     - Sets video playback rate only; it does not change simulation speed or success

.. note::

   ``num_episodes=10`` controls how many reset episodes are available, while
   ``rollout_epoch=10`` controls how many rollouts execute. Automatic reset is
   disabled, so both must be 10 to run each episode exactly once.

Run the three DR levels sequentially. Each invocation creates a separate
timestamped log directory:

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_closedoor_psi0_eval \
       rollout.model.model_path="${PSI0_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.episode_start=0 \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10 \
       env.eval.video_cfg.fps=50
   done

Compare the result with the pinned SIMPLE
`official results table <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/README.md#-simulation-benchmarking-results>`_.
Psi0 reports 10/10 at each of levels 0, 1, and 2 on
``G1WholebodyCloseDoorTeleop-v0``, or 30/30 overall. A 1/1 smoke result proves
only that the path works; compare all three summaries, per-episode logs, and
videos to assess alignment. Consult the official
`SIMPLE evaluation entry <https://github.com/physical-superintelligence-lab/SIMPLE/blob/5e3d6f84e85343e34e9bca8d157f0d7813231185/src/simple/cli/eval_decoupled_wbc.py>`_
and
`Psi0 SIMPLE server arguments <https://github.com/physical-superintelligence-lab/Psi0/blob/a32e57a3fabb8590c80677f9cd3d1fc3db60eb06/scripts/deploy/serve_psi0_simple.sh>`_
to audit the protocol.

Run the OpenFaucet Evaluation
-----------------------------

OpenFaucet uses its own task-specific checkpoint and reset states. Do not reuse
the CloseDoor ``PSI0_RUN_DIR`` or ``SIMPLE_RESET_DIR``. Downloads remain
user-run operations:

.. code-block:: bash

   export PSI0_MODEL_REV=d34a91932d25c45ef211582315b9224c7dc8ace9
   export PSI0_OPENFAUCET_RUN_REL=psi0/simple-checkpoints/g1wholebodyopenfaucetteleop-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2605081439

   hf download USC-PSI-Lab/psi-model \
     "${PSI0_OPENFAUCET_RUN_REL}/argv.txt" \
     "${PSI0_OPENFAUCET_RUN_REL}/envs.txt" \
     "${PSI0_OPENFAUCET_RUN_REL}/run_config.json" \
     "${PSI0_OPENFAUCET_RUN_REL}/checkpoints/ckpt_40000/model.safetensors" \
     --revision "${PSI0_MODEL_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-model"

   export PSI0_OPENFAUCET_RUN_DIR="${PSI0_ARTIFACT_ROOT}/psi-model/${PSI0_OPENFAUCET_RUN_REL}"
   export SIMPLE_DATA_REV=6eeff2d02fdaac5dd4f3e84244fc83d3fad2c203
   export SIMPLE_OPENFAUCET_RESET_REL=simple-eval/G1WholebodyOpenFaucetTeleop-v0.zip

   hf download USC-PSI-Lab/psi-data "${SIMPLE_OPENFAUCET_RESET_REL}" \
     --repo-type dataset \
     --revision "${SIMPLE_DATA_REV}" \
     --local-dir "${PSI0_ARTIFACT_ROOT}/psi-data"

   export SIMPLE_OPENFAUCET_RESET_ZIP="${PSI0_ARTIFACT_ROOT}/psi-data/${SIMPLE_OPENFAUCET_RESET_REL}"
   export SIMPLE_OPENFAUCET_RESET_DIR="${PSI0_ARTIFACT_ROOT}/reset-states/openfaucet"
   mkdir -p "${SIMPLE_OPENFAUCET_RESET_DIR}"
   unzip -q "${SIMPLE_OPENFAUCET_RESET_ZIP}" -d "${SIMPLE_OPENFAUCET_RESET_DIR}"

Run one level-0 episode first:

.. code-block:: bash

   bash evaluations/run_eval.sh simple simple_openfaucet_psi0_eval \
     rollout.model.model_path="${PSI0_OPENFAUCET_RUN_DIR}" \
     env.eval.reset_dataset.path="${SIMPLE_OPENFAUCET_RESET_DIR}"

The complete official protocol uses 10 fixed episodes at each DR level:

.. code-block:: bash

   for SIMPLE_DR_LEVEL in 0 1 2; do
     bash evaluations/run_eval.sh simple simple_openfaucet_psi0_eval \
       rollout.model.model_path="${PSI0_OPENFAUCET_RUN_DIR}" \
       env.eval.reset_dataset.path="${SIMPLE_OPENFAUCET_RESET_DIR}" \
       env.eval.reset_dataset.dr_level="${SIMPLE_DR_LEVEL}" \
       env.eval.reset_dataset.episode_start=0 \
       env.eval.reset_dataset.num_episodes=10 \
       env.eval.rollout_epoch=10 \
       env.eval.video_cfg.fps=50
   done

``simple_openfaucet_psi0_eval`` fixes the official 1,000-step horizon, 50 Hz
control/render, 200 Hz physics, up to 300 stabilization steps, decoupled-WBC,
and 30/24/6 RTC. The pinned SIMPLE table reports Psi0 results of 3/10, 3/10,
and 4/10 at levels 0, 1, and 2. Report each level separately instead of hiding
the DR-level difference in the aggregate 10/30 rate.

Validate the Result
-------------------

Confirm the following before increasing the number of episodes:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Check
     - Expected result
   * - Model boundary
     - System-2 parameters stay frozen and the policy runs only in eval mode.
   * - RTC
     - A matching execution acknowledgement commits each 30-step plan; its final six actions condition the next plan.
   * - Episode boundary
     - The initial reset sets ``episode_id`` and clears pending/committed RTC state; success does not start another trial implicitly.
   * - Metrics
     - RLinf reports SIMPLE's task-owned return, episode length, and success value.

.. warning::

   P3 does not implement RL sampling, log-probabilities, actor recomputation, or
   cross-decision credit attribution. Do not use this configuration for
   training.
