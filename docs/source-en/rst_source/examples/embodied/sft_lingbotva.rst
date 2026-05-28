LingBot-VA Supervised Fine-Tuning
==================================

This document explains how to run LingBot-VA supervised fine-tuning (SFT) in RLinf on the Libero-Object benchmark.

It targets the 5B-parameter LingBot-VA video-diffusion transformer on a single node with **8 × NVIDIA A100-80GB SXM4** GPUs, and replicates the lingbot-va reference ``va_libero_train_cfg`` (FSDP2 full-sharding, plain AdamW, ``flex_attention``) at the same effective batch size (80).


Supported setups
----------------

Recommended config:

- ``examples/sft/config/libero_sft_lingbotva.yaml``: Libero-Object SFT on 8 × A100-80GB (plain AdamW + FSDP2 ``fully_shard`` + flex_attention)

A single-GPU launch path (FSDP1 ``no_shard`` + bitsandbytes ``AdamW8bit`` + flex_attention) is documented inline in the config's header comment and is reached by overriding four fields on the launch CLI — see :ref:`single-gpu-launch` below.

Starting point:

- **Cold-start from the LingBot-VA base checkpoint** (Wan-Video 2.2 backbone). Continuing from a partially trained checkpoint is supported but not the verified path.

Verified result on 8 × A100-80GB (one node):

- ckpt_2000 mean SR on Libero-Object: **70 % (35/50)** over 10 tasks × 5 episodes (single-A100 reference recipe: ~18 %; ``lingbot-va-base`` without SFT scores 0/50)
- Per-step wall time: ~12.5 s/step at effective batch 80; ~7 h to step 2000; ~10.5 h projected to step 3000
- Per-GPU memory: ~32 GB at training (FSDP2 sharding active; under half what single-GPU NO_SHARD would use); ~39 GB per process at evaluation (KV-cache replay)


Training entrypoint
-------------------

Use:

- ``examples/sft/run_vla_sft.sh``

The script runs:

.. code:: bash

   python examples/sft/train_vla_sft.py \
     --config-path examples/sft/config/ \
     --config-name libero_sft_lingbotva \
     runner.logger.log_path=<auto_log_dir>

Logs are written to:

- ``<repo>/logs/<timestamp>/run_embodiment.log``

The script also exports ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. It is not strictly required on 80 GB cards but is harmless and avoids allocator fragmentation during the wrap/first-forward phase.


Environment
-----------

1. **PyTorch 2.9.0+cu126 is required.** Earlier versions hit a ``flex_attention`` + inductor codegen bug (``NameError: name 's10' is not defined``). Install with::

     pip install --no-deps --force-reinstall \
       torch==2.9.0+cu126 \
       --index-url https://download.pytorch.org/whl/cu126
     pip install nvidia-cufile-cu12 nvidia-nvshmem-cu12

   Do **not** force-upgrade NCCL: torch 2.9 hard-pins ``nvidia-nccl-cu12==2.27.5``. Let it install transitively.

2. Install RLinf and the LingBot-VA dependency group::

     pip install -e ${REPO_PATH}
     pip install -r ${REPO_PATH}/requirements/embodied/models/lingbotva.txt

3. Install the LIBERO simulator and the runtime deps the lingbot-va requirements file does not pull::

     pip install "lerobot==0.3.3" "robosuite==1.4.1" "gym==0.26.2" \
                 bddl draccus tensorflow_graphics websockets msgpack \
                 matplotlib Pillow
     pip install -e <RLINF_LIBERO_REPO_PATH>     # https://github.com/RLinf/LIBERO.git

   ``lerobot==0.3.3`` is mandatory — 0.4.x breaks the ``LatentLeRobotDataset`` imports the loader relies on. The lerobot install can also pull a transitive torch downgrade; re-run step 1 after it to re-pin torch 2.9.0+cu126 and its companions.

4. Clone the lingbot-va peer repository (provides the ``wan_va`` Python package — needed at runtime by both training and the dataset-prep ``extract_latents.py``; the data-prep scripts themselves now live under ``toolkits/data_scripts_lingbotva/`` in this RLinf clone)::

     git clone https://github.com/robbyant/lingbot-va.git <LINGBOT_VA_REPO_PATH>

   Do **not** ``pip install -e`` it: the upstream ``pyproject.toml`` declares the package as ``lingbot_va`` while the source tree is ``wan_va``, and ``flash_attn`` is a hard build dep that fails on torch 2.9. Make ``wan_va`` importable via ``PYTHONPATH`` instead (step 5).

5. Set the path env vars consumed by the launch script (substitute your own absolute paths)::

     export LINGBOT_VA_REPO_PATH=<your-lingbot-va-clone>
     export LINGBOT_VA_MODEL_PATH=<your-model-dir>
     export LINGBOT_VA_DATASET_PATH=<your-dataset-dir>
     export PYTHONPATH=${REPO_PATH}:${LINGBOT_VA_REPO_PATH}:${PYTHONPATH}
     export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

6. **lingbot-va flash_attn patch.** ``wan_va/modules/model.py`` imports ``flash_attn`` at module-load time. The prebuilt wheels do not load against the torch 2.9 ABI; patch the import to fall through gracefully:

   .. code:: diff

       try:
           from flash_attn_interface import flash_attn_func
      -except:
      -    from flash_attn import flash_attn_func
      +except Exception:
      +    try:
      +        from flash_attn import flash_attn_func
      +    except Exception:
      +        flash_attn_func = None

   This recipe uses ``attn_mode: flex`` and never calls ``flash_attn_func``, so the fallthrough is safe.

7. **Hydra model-default visibility.** The SFT yaml uses ``model/lingbotva@actor.model``, but ``model/lingbotva.yaml`` lives under ``examples/embodiment/config/model/``, not ``examples/sft/config/model/``. ``run_vla_sft.sh`` sets ``EMBODIED_PATH=examples/sft``, so hydra cannot resolve the default and fails with ``Could not load 'model/lingbotva'``. Resolve by symlinking the file into the SFT search path::

     ln -s ${REPO_PATH}/examples/embodiment/config/model/lingbotva.yaml \
           ${REPO_PATH}/examples/sft/config/model/lingbotva.yaml


Data preparation
----------------

The training dataset is a LeRobot v2.1 conversion of the full Libero-Object set (10 tasks × 50 demos = 500 episodes) with pre-extracted Wan 2.2 VAE latents and a cached UMT5 empty-prompt embedding. The 8-GPU run uses the full 500-demo set; the 100-demo subsample remains useful as a smoke-test path on a single GPU.

1. Download the raw HDF5 demonstrations. **Use** ``yifengzhu-hf/LIBERO-datasets``, **NOT** ``IPEC-COMMUNITY``: the two distributions have different wrist-camera mounts, and the IPEC version trains fine but evaluates at 0 % SR.

   .. code:: bash

      export LIBERO_RAW_DIR=<your-libero-raw-dir>
      huggingface-cli download yifengzhu-hf/LIBERO-datasets --local-dir ${LIBERO_RAW_DIR}

2. Run the conversion scripts shipped under ``toolkits/data_scripts_lingbotva/``. See that directory's ``README.md`` for inputs / outputs of each step.

   .. code:: bash

      cd ${REPO_PATH}

      # (a) HDF5 -> LeRobot v2.1 format (all 500 demos)
      python toolkits/data_scripts_lingbotva/convert_libero_object_to_lerobot.py \
        --src ${LIBERO_RAW_DIR}/libero_object \
        --dst ${LINGBOT_VA_DATASET_PATH}

      # (b) Extract Wan 2.2 VAE latents + cache the UMT5 empty-prompt embedding
      python toolkits/data_scripts_lingbotva/extract_latents.py \
        --dataset ${LINGBOT_VA_DATASET_PATH} \
        --ckpt-dir ${LINGBOT_VA_MODEL_PATH}

      # (c) MANDATORY: reshape image stats from (3,) to (3, 1, 1)
      # lerobot>=0.3.3 rejects the dataset on the first batch with
      # "ValueError: Shape of 'min' must be (3,1,1)" otherwise. Apply
      # in place to every observation.images.* entry in episodes_stats.jsonl:
      python - <<'PY'
      import json, os
      path = os.path.join(os.environ["LINGBOT_VA_DATASET_PATH"], "meta", "episodes_stats.jsonl")
      with open(path) as f:
          rows = [json.loads(line) for line in f]
      for row in rows:
          for key, stats in row.get("stats", {}).items():
              if not key.startswith("observation.images."):
                  continue
              for stat_name, vec in stats.items():
                  if isinstance(vec, list) and len(vec) == 3 and not isinstance(vec[0], list):
                      stats[stat_name] = [[[v]] for v in vec]   # (3,) -> (3, 1, 1)
      with open(path, "w") as f:
          for row in rows:
              f.write(json.dumps(row) + "\n")
      PY

   Optional smoke-test path: insert ``select_subset.py --src ... --dst ..._subset --per-task 10 --seed 42`` between (a) and (b), then point (b) and (c) at the ``_subset`` directory. Use it to validate the pipeline end-to-end before committing to a multi-hour 500-demo run.

3. Expected layout under ``${LINGBOT_VA_DATASET_PATH}``::

      meta/{info,episodes,episodes_stats,tasks}.jsonl
      data/chunk-000/episode_NNNNNN.parquet
      videos/chunk-000/observation.images.{agentview_rgb,eye_in_hand_rgb}/episode_NNNNNN.mp4
      latents/chunk-000/observation.images.{agentview_rgb,eye_in_hand_rgb}/episode_NNNNNN_<s>_<e>.pth
      empty_emb.pt


Model and weight preparation
----------------------------

Download the LingBot-VA base checkpoint (Wan-Video 2.2 backbone, ~30 GB) into ``${LINGBOT_VA_MODEL_PATH}``. Expected layout::

   ${LINGBOT_VA_MODEL_PATH}/
   ├── transformer/                # 5B-param Wan transformer
   │   ├── config.json
   │   └── diffusion_pytorch_model-*.safetensors
   ├── vae/                        # Wan 2.2 VAE (used by extract_latents.py)
   └── text_encoder/               # UMT5 (only the empty-prompt embedding is
                                     used at train time)

Source: as provided in the original lingbot-va release notes.


Key LingBot-VA config fields
----------------------------

The recipe in ``libero_sft_lingbotva.yaml`` mirrors lingbot-va's reference ``va_libero_train_cfg``. Load-bearing values:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Setting
     - Value
     - Why
   * - ``cluster.num_nodes``
     - ``1``
     - Single-node, 8-GPU placement.
   * - ``cluster.component_placement``
     - ``actor,env,rollout: all``
     - Co-locate all components across the 8 ranks (the rollout/env workers are unused at SFT time but the placement must still be set).
   * - ``actor.fsdp_config.strategy``
     - ``fsdp2``
     - FSDP2 ``fully_shard`` is the lingbot-va reference. Full sharding gives the ~32 GB/GPU footprint that lets the 5B-param transformer + activations + plain-AdamW state fit on 80 GB.
   * - ``actor.fsdp_config.reshard_after_forward``
     - ``True``
     - Free per-rank shards between forward and backward.
   * - ``actor.fsdp_config.mixed_precision.param_dtype`` / ``reduce_dtype``
     - ``bf16`` / ``fp32``
     - bf16 compute with fp32 grad reduction matches the reference recipe.
   * - ``actor.optim.optimizer_type``
     - ``adamw``
     - Plain ``torch.optim.AdamW``. The single-GPU recipe uses ``adamw_8bit`` (bitsandbytes) to fit on a 45 GB card; that hack is not needed on 80 GB ranks under FSDP2 and was dropped.
   * - ``actor.optim.lr`` / ``betas`` / ``wd`` / ``lr_warmup_steps``
     - ``1e-5`` / ``(0.9, 0.95)`` / ``0.1`` / ``10``
     - Match the reference recipe.
   * - ``actor.micro_batch_size``
     - ``1``
     - Reference recipe.
   * - ``actor.global_batch_size``
     - ``80``
     - Effective batch 80 = micro 1 × world_size 8 × gradient_accumulation 10. Same effective batch as the single-A100 recipe (10 = 1 × 1 × 10).
   * - ``actor.model.lingbotva.attn_mode``
     - ``flex``
     - Reference recipe; requires torch ≥ 2.9. ``attn_mode: torch`` silently drops the BlockMask and collapses eval SR to 0 %.
   * - ``actor.model.lingbotva.cfg_prob``
     - ``0.1``
     - Classifier-free guidance dropout; reference recipe.
   * - ``actor.model.precision``
     - ``bf16``
     - Transformer is loaded bf16 + force-cast (avoids fp32 leakage from ``scale_shift_table``).
   * - ``data.num_workers``
     - ``16``
     - Reference recipe; data loading is only ~7 % of step time, so the dataloader is not the bottleneck.
   * - ``runner.max_steps``
     - ``3000``
     - Reference run length.
   * - ``runner.save_interval``
     - ``500``
     - Each save writes ``full_weights.pt`` (~9.5 GB) and a ``dcp_checkpoint/`` directory of sharded optimizer state (~28 GB), so a saved step costs ~38 GB on disk. If you only need eval-able weights, delete ``dcp_checkpoint/`` after each save and keep only ``full_weights.pt``.


Launch training
---------------

Run from repository root:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_lingbotva

.. _single-gpu-launch:

Running on a single GPU
~~~~~~~~~~~~~~~~~~~~~~~

The defaults assume 8 ranks. To run the same recipe on one GPU (L40-class, ≥45 GB), override four load-bearing fields on the launch CLI. Plain AdamW state for the 5B-param transformer is ~37 GB on its own — too large for a 45 GB card — so the single-GPU path swaps in bitsandbytes ``AdamW8bit`` and falls back to FSDP1 ``NO_SHARD`` (FSDP2 wraps params as DTensors that bnb's CUDA kernels reject):

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_lingbotva \
     actor.fsdp_config.strategy=fsdp \
     actor.fsdp_config.sharding_strategy=no_shard \
     actor.fsdp_config.use_orig_params=True \
     actor.fsdp_config.reshard_after_forward=False \
     actor.optim.optimizer_type=adamw_8bit \
     actor.global_batch_size=10 \
     data.num_workers=4 \
     actor.dataloader_num_workers=4

The ``global_batch_size=10`` override preserves the reference effective batch (world_size 1 × micro 1 × grad_accum 10). Verified steady-state throughput on an L40: ~17 s/step.


Monitoring and sanity checks
----------------------------

1. Check ``run_embodiment.log``:

   - stable ``time/step`` (~12.5 s/step on 8 × A100-80GB at effective batch 80)
   - per-GPU memory ~32 GB; all 8 GPUs at 96–100 % util
   - reasonable ``train/latent_loss`` and ``train/action_loss``

2. Expected loss trajectory (5-step mean window around each saved step):

   .. list-table::
      :header-rows: 1
      :widths: 15 25 25 35

      * - Step
        - latent_loss
        - action_loss
        - grad_norm
      * - 500
        - ~0.120
        - ~0.097
        - ~0.077
      * - 1000
        - ~0.113
        - ~0.087
        - ~0.091
      * - 1500
        - ~0.103
        - ~0.082
        - ~0.111
      * - 2000
        - ~0.103
        - ~0.082
        - ~0.121

   These are below the single-A100 reference at every step where the reference reports a number. The plateau between step 1500 and 2000 (``action_loss`` static at ~0.082) suggests near-convergence on training data. Values within ±25 % are within typical RNG noise (CFG dropout + flow-matching timestep sampling diverge across torch versions even at the same seed). Values >2× off the reference at the same step count indicate something is wrong with training; the most common cause is ``attn_mode: torch`` slipping back in via a stale config or YAML merge — verify the active config logs show ``attn_mode: flex``. (Low training loss with bad eval SR is usually an *evaluation*-side problem — see :ref:`Common issues <common-issues>`.)

3. TensorBoard:

   .. code:: bash

      tensorboard --logdir ./logs --port 6006


Evaluation
----------

Once ``checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt`` exists, evaluate with the standard RLinf entry point using the ``libero_object_eval_lingbotva`` config. The SFT checkpoint is selected via ``LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH``, and the 10 Libero-Object task ids are looped over via the Hydra CLI override ``env.eval.task_id_filter``.

``examples/embodiment/eval_embodiment.sh`` dispatches the eval driver from the ``model_type`` key in the eval YAML: for LingBot-VA it runs ``examples/embodiment/eval_lingbotva.py``, a dedicated single-process driver. Do **not** invoke the generic ``eval_embodied_agent.py`` directly — it drops the per-step raw observations the model needs for KV-cache replay and silently collapses SR to ~0 %.

Sequential form (one task at a time on a single GPU):

.. code:: bash

   CKPT_DIR=${REPO_PATH}/runtime/lingbotva_sft/libero_sft_lingbotva/checkpoints
   export LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH=\
   ${CKPT_DIR}/global_step_2000/actor/model_state_dict/full_weights.pt
   OUT_ROOT=${REPO_PATH}/runtime/lingbotva_libero_eval/ckpt_2000

   for tid in 0 1 2 3 4 5 6 7 8 9; do
     bash examples/embodiment/eval_embodiment.sh libero_object_eval_lingbotva LIBERO \
       runner.logger.log_path=${OUT_ROOT}/task_${tid} \
       env.eval.total_num_envs=1 \
       env.eval.task_id_filter=[${tid}] \
       algorithm.eval_rollout_epoch=5
   done

Do **not** just set ``total_num_envs=50`` — the env's reset-state cursor cycles reset states within a task, not tasks.

Parallel form (one task per GPU, all 8 ranks at once):

The eval driver is a plain single-process script (no Ray / no Cluster), so each task can be pinned to its own GPU. Peak per-process memory is ~39 GB (KV-cache replay accumulates per-chunk observations), so two tasks do **not** fit on one 80 GB card — co-locating OOMs mid-episode. For the 10 Libero-Object tasks, run tasks 0–7 in parallel across GPUs 0–7, then tasks 8–9 on any two free GPUs:

.. code:: bash

   for tid in 0 1 2 3 4 5 6 7; do
     CUDA_VISIBLE_DEVICES=${tid} MUJOCO_EGL_DEVICE_ID=${tid} \
       bash examples/embodiment/eval_embodiment.sh libero_object_eval_lingbotva LIBERO \
         runner.logger.log_path=${OUT_ROOT}/task_${tid} \
         env.eval.total_num_envs=1 \
         env.eval.task_id_filter=[${tid}] \
         algorithm.eval_rollout_epoch=5 &
   done
   wait

   for tid in 8 9; do
     gpu=$((tid - 8))
     CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} \
       bash examples/embodiment/eval_embodiment.sh libero_object_eval_lingbotva LIBERO \
         runner.logger.log_path=${OUT_ROOT}/task_${tid} \
         env.eval.total_num_envs=1 \
         env.eval.task_id_filter=[${tid}] \
         algorithm.eval_rollout_epoch=5 &
   done
   wait

``MUJOCO_EGL_DEVICE_ID`` must match ``CUDA_VISIBLE_DEVICES`` so the MuJoCo EGL renderer binds to the same physical GPU as the model.

Baseline (sanity check):

To confirm SFT is doing the work, evaluate the base LingBot-VA backbone (no Libero fine-tuning) by overriding the transformer state dict to null so the model uses the pretrained weights under ``LINGBOT_VA_MODEL_PATH``:

.. code:: bash

   unset LINGBOT_VA_TRANSFORMER_STATE_DICT_PATH
   OUT_ROOT=${REPO_PATH}/runtime/lingbotva_libero_eval/base

   for tid in 0 1 2 3 4 5 6 7 8 9; do
     # Same per-task launcher as above, with one extra override:
     #   actor.model.lingbotva.transformer_state_dict_path=null
     ...
   done

The base model is expected to score 0/50 — its backbone has not seen the Libero action distribution and every episode runs to the 240-step cap.

Verified results (per task, 5 episodes each):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Task
     - Base (no SFT)
     - SFT ckpt_2000
   * - 0
     - 0/5
     - 3/5
   * - 1
     - 0/5
     - 4/5
   * - 2
     - 0/5
     - 5/5
   * - 3
     - 0/5
     - 1/5
   * - 4
     - 0/5
     - 4/5
   * - 5
     - 0/5
     - 3/5
   * - 6
     - 0/5
     - 5/5
   * - 7
     - 0/5
     - 3/5
   * - 8
     - 0/5
     - 4/5
   * - 9
     - 0/5
     - 3/5
   * - **Mean**
     - **0/50 (0 %)**
     - **35/50 (70 %)**

Per-task SR is read from ``<out_dir>/task_<tid>/eval_results.json`` (each file records ``success_rate``, ``successes``, ``failures``). Reproducing within ±1 episode per task at the same seed is normal; >2 episodes off at any task usually means a stale checkpoint, the wrong eval driver, or a CPU/EGL device-id mismatch under the parallel form.


.. _common-issues:

Common issues
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Symptom
     - Likely cause
     - Fix
   * - ``Could not load 'model/lingbotva'`` at launch
     - ``model/lingbotva.yaml`` not in the SFT hydra search path
     - Symlink ``examples/embodiment/config/model/lingbotva.yaml`` into ``examples/sft/config/model/`` (Environment step 7).
   * - Ray claims more GPUs than ``CUDA_VISIBLE_DEVICES`` exposes
     - ``component_placement: actor,env,rollout: all`` scans every visible GPU and ignores ``CUDA_VISIBLE_DEVICES``
     - Pin the placement (e.g. ``actor,env,rollout: '0-7'``) or launch in a cgroup/container that masks the unwanted GPUs.
   * - ``NameError: name 's10' is not defined`` during forward
     - torch < 2.9 ``flex_attention`` codegen bug
     - Upgrade to torch 2.9. Do **not** fall back to ``attn_mode: torch``.
   * - 0 % SR at every task despite low training loss
     - Wrong eval driver — the generic ``eval_embodied_agent.py`` drops per-step raw observations, so the model's ``record_chunk_observations`` / KV-cache replay path never sees the post-action frames it needs
     - Launch with ``bash examples/embodiment/eval_embodiment.sh ...``; the dispatcher reads ``model_type`` from the eval YAML and routes LingBot-VA to ``eval_lingbotva.py``. Do not invoke the generic driver directly. (Eval-time ``attn_mode: torch`` is fine — overriding it on the eval CLI is not necessary.)
   * - 0 % SR even with ``eval_lingbotva.py``
     - SDPA mask leak at *training* time — the SFT run used ``attn_mode: torch`` and the BlockMask was silently dropped
     - Confirm the active *training* config logs show ``attn_mode: flex``; if not, retrain. (The eval-time attn_mode is independent and does not need to be flex.)
   * - ``ImportError: cannot import name 'flash_attn_func'``
     - torch 2.9 ABI breaks the flash_attn wheel
     - Apply the patch under "Environment" step 6.
   * - FSDP "flatten tensors with uniform dtype" error
     - ``scale_shift_table`` initialised fp32
     - Confirm ``precision: bf16`` is set and the transformer is force-cast to bf16 (already wired in the action model).
   * - Out-of-disk while training
     - ~38 GB per saved step (``full_weights.pt`` + ``dcp_checkpoint/``)
     - Either raise ``save_interval`` (500 → 1000) or delete ``dcp_checkpoint/`` after each save if you only need eval-able weights.
   * - ``lerobot`` rejects dataset on first batch
     - ``episodes_stats.jsonl`` has per-channel image stats as ``(3,)`` instead of ``(3, 1, 1)``
     - Run the reshape step (Data preparation step 2c).
   * - Mysterious torch / nccl / cudnn downgrade after dependency installs
     - ``lerobot==0.3.3`` pulls a transitive torch downgrade
     - Re-run the torch 2.9.0+cu126 install (Environment step 1) after lerobot.


Practical recommendations
-------------------------

- Run a short trial (e.g. 50–100 steps) after each config change to verify shapes, loss values, and throughput before committing to a multi-hour run. The optional 100-demo subset (Data preparation step 2, smoke-test path) makes this cheap.
- The 8-GPU recipe is the verified-good path. The single-GPU launch (:ref:`single-gpu-launch`) drops to FSDP1 + bnb ``AdamW8bit`` to fit on a 45 GB card and is otherwise equivalent; use it for L40-class smoke tests only.
- The forward/backward is compute-bound and scales close to linearly across the 8 ranks, since data loading is only ~7 % of step time.
- Disk pressure is the most likely surprise: budget ~38 GB per saved step at the reference ``save_interval``, or wire a background cleaner that strips ``dcp_checkpoint/`` immediately after each save if you do not need to resume training.
