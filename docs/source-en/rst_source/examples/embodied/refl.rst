Fine-Tune SD3 with ImageReward
==============================

.. figure:: https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/figures/ImageReward.jpg
   :align: center
   :width: 80%

   ImageReward overview. Source: the ImageReward authors' official repository.

Fine-tune Stable Diffusion 3.5 with a differentiable image-preference reward
using Reward Feedback Learning (ReFL). Start with reward-only training, then
optionally add supervised image-text updates to study their effect on image
quality. This example reuses RLinf's diffusion environment, rollout workers,
FSDP actor, weight synchronization, and training runner.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      Diffusion generation

   .. grid-item-card:: Algorithms
      :text-align: center

      ReFL

   .. grid-item-card:: Tasks
      :text-align: center

      Text-to-image preference tuning

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA; four-GPU recipe below

The config is ``examples/diffusion/config/sd3_refl_imagereward.yaml``.
See :doc:`../video_models` for the shared diffusion environment and NFT examples.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Model
     - Reward
     - Training data
   * - SD3.5 Medium with LoRA
     - ImageReward human-preference score
     - Text prompts; optional captioned images for supervised updates

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - ``task_descriptions``: a batch of text prompts.
   * - Action
     - Generated RGB images; each image completes one environment episode.
   * - Reward
     - ``avg`` and ``image_reward``: the same scalar score; higher is better.
   * - Update inputs
     - Detached late denoising states and prompt embeddings from rollout.

Installation
------------

From the repository root, install the diffusion environment following
:doc:`/rst_source/start/installation`:

.. code-block:: bash

   bash requirements/install.sh embodied --model diffusion
   source .venv/bin/activate

The diffusion requirements select ``image-reward==1.5``, whose dependencies
require ``timm==0.6.13``.
When reusing an environment, check these versions before training; a newer
``timm`` can conflict with ImageReward's dependency requirements.

Prepare Models and Prompts
--------------------------

Accept the SD3.5 model license on Hugging Face and authenticate with ``hf auth
login`` before downloading gated weights. Download the generator, the reward
checkpoint, the BERT tokenizer used by ImageReward, and the official prompts:

.. code-block:: bash

   export SD3_MODEL="$PWD/models/stable-diffusion-3.5-medium"
   export IR_MODEL="$PWD/models/ImageReward"
   export IR_DATA="$PWD/data/ImageReward"
   mkdir -p "$SD3_MODEL" "$IR_MODEL" "$IR_DATA"

   hf download stabilityai/stable-diffusion-3.5-medium --local-dir "$SD3_MODEL"
   hf download zai-org/ImageReward ImageReward.pt med_config.json --local-dir "$IR_MODEL"
   python -c 'from transformers import BertTokenizer; BertTokenizer.from_pretrained("bert-base-uncased")'

   for file in refl_data.json test.json; do
     curl --fail --location \
       "https://raw.githubusercontent.com/zai-org/ImageReward/78f8051059b34fef8e0032ccb40561c23a2ddbbb/data/$file" \
       --output "$IR_DATA/$file" || break
   done

Keep these environment variables in the shell used to launch training.
Every node must be able to load ``bert-base-uncased`` from its Hugging Face
cache. For an offline run, prepare that cache on every node first.

Each prompt JSON is a list of objects containing ``text`` or ``prompt``.
The loader trims whitespace, removes duplicates while preserving order, and
excludes the prompts in ``env.train.dataset.exclude_path`` from training.
For the pinned official files, this leaves 9,756 training and 463 evaluation
prompts.

Run Reward-Only Training
------------------------

Start with the reward objective alone so that later comparisons isolate the
supervised updates. The following four-GPU recipe processes 512 training
samples per runner step in global batches of 64:

.. code-block:: bash

   REFL_ARGS=(
     actor.model.model_path="$SD3_MODEL"
     rollout.model.model_path="$SD3_MODEL"
     actor.model.sd3.lora_path=null
     actor.refl_reward_model.model_path="$IR_MODEL/ImageReward.pt"
     actor.refl_reward_model.med_config="$IR_MODEL/med_config.json"
     env.train.dataset.path="$IR_DATA/refl_data.json"
     env.train.dataset.exclude_path="$IR_DATA/test.json"
     env.eval.dataset.path="$IR_DATA/test.json"
     env.train.total_num_envs=4
     env.train.rollout_epoch=128
     env.eval.total_num_envs=4
     env.eval.rollout_epoch=128
     actor.global_batch_size=64
     actor.micro_batch_size=1
     actor.refl_pre_data.num_workers=0
     rollout.pipeline_stage_num=1
     env.train.reward.batch_size=1
     env.eval.reward.batch_size=1
     actor.seed=42
     env.train.seed=42
     env.eval.seed=0
     algorithm.update_epoch=1
     algorithm.refl_grad_scale=1e-3
     runner.max_epochs=200
     runner.max_steps=200
     runner.val_check_interval=10
     runner.save_interval=10
     runner.resume_dir=null
     runner.ckpt_path=null
   )

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     algorithm.refl_pre_loss_weight=0.0 \
     runner.logger.experiment_name=sd3_refl_reward_only

This starts a fresh 200-step run, evaluates every 10 steps, and saves every
10 steps. Choose GPU placement for your machine using
:doc:`/rst_source/start/installation`; micro-batch size 1 is a starting point,
not a memory guarantee. The actor holds a differentiable ImageReward model,
and environment reward scoring also consumes memory.

Add Supervised Image-Text Updates
---------------------------------

To continue the generator's denoising training alongside the reward objective,
set ``algorithm.refl_pre_loss_weight`` above zero. This enables the optional
pre-training loss, called Lpre. Prepare local image-caption pairs in the
img2dataset ``files`` layout:

.. code-block:: text

   image_text_data/
     00000.jpg
     00000.txt
     shard_01/
       00001.png
       00001.txt

The loader searches subdirectories for ``.jpg``, ``.jpeg``, ``.png``, and
``.webp`` images. Each image needs a UTF-8 caption with the same stem and a
``.txt`` suffix. Images are resized, center-cropped to 512 pixels, randomly
flipped, and normalized to [-1, 1]. The default ``caption_prefix`` removes
``The image shows`` and its following space only at the start of a caption.

Download a PD3M Subset
~~~~~~~~~~~~~~~~~~~~~~

If you need captioned images, use the public `PD3M dataset
<https://huggingface.co/datasets/Spawning/PD3M>`_. Its metadata contains captions
and image URLs; the image bytes are hosted in the separate PD12M S3 bucket.
This example selects 20,000 candidates from two metadata shards rather than
fetching the full dataset. Prepare them on a CPU machine before reserving GPUs.

Set ``DATA_ROOT`` to your data disk before running the block below. For example,
use ``export DATA_ROOT=/mnt/data/refl`` for a mounted data volume, or choose
your own writable directory. The block
creates a separate download environment so that installing img2dataset does
not change the diffusion training environment. Keep the diffusion environment
active: its ``hf`` command downloads the metadata. The PyArrow pin keeps it
compatible with NumPy 1.26, and Pillow is needed for the image checks.

The metadata revision and shuffle seed are fixed. Selection requires a license
field, nonempty URL and caption, a short side of at least 512 pixels, and an
aspect ratio no greater than 2. It also excludes iNaturalist and captions
matching the listed monochrome or document keywords. These are example curation
heuristics, not a quality guarantee or a reproduction of the paper's training
set. Inspect the resulting captions and images for your target domain.

.. code-block:: bash

   export DATA_ROOT="${DATA_ROOT:-$PWD/data/refl}"
   export PD3M_DIR="${PD3M_DIR:-$DATA_ROOT/PD3M}"
   export PRE_DATA="$PD3M_DIR/images_512"
   export PD3M_VENV="$DATA_ROOT/.venv-img2dataset"

   (
     set -e
     unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
     export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_ROOT/.cache/uv}"
     export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$DATA_ROOT/.cache/uv-python}"
     export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
     export NO_ALBUMENTATIONS_UPDATE=1
     mkdir -p "$PD3M_DIR" "$TMPDIR"

     if [ ! -x "$PD3M_VENV/bin/python" ]; then
       uv venv --python 3.10 "$PD3M_VENV"
     fi
     uv pip install --python "$PD3M_VENV/bin/python" \
       img2dataset==1.47.0 numpy==1.26.4 \
       pyarrow==19.0.1 pillow==11.1.0 \
       opencv-python-headless==4.11.0.86 albumentations==1.4.24

     hf download Spawning/PD3M \
       metadata/pd3m.00.parquet metadata/pd3m.01.parquet \
       --repo-type dataset \
       --revision 2a5eb24a8dccf245acd8e56341761aee06da0bdf \
       --local-dir "$PD3M_DIR"

     "$PD3M_VENV/bin/python" - <<'PY'
   import os
   import random
   import re
   from pathlib import Path

   import pyarrow as pa
   import pyarrow.parquet as pq

   root = Path(os.environ["PD3M_DIR"])
   columns = ["id", "url", "caption", "width", "height", "license", "source"]
   rows = []
   for shard in ("00", "01"):
       rows.extend(pq.read_table(
           root / "metadata" / f"pd3m.{shard}.parquet", columns=columns,
       ).to_pylist())

   monochrome = re.compile(r"black[- ]and[- ]white|grayscale|sepia|monochrome", re.I)
   document = re.compile(
       r"\b(book|page|newspaper|document|letter|map|poster|stamp|coin|text (on|at|in))\b",
       re.I,
   )
   candidates = []
   for row in rows:
       caption = row["caption"] or ""
       width, height = row["width"] or 0, row["height"] or 0
       if (
           row["url"] and caption.strip() and row["license"]
           and row["source"] != "iNaturalist"
           and min(width, height) >= 512
           and max(width, height) <= 2 * min(width, height)
           and not monochrome.search(caption)
           and not document.search(caption)
       ):
           candidates.append(row)

   assert len(candidates) >= 20000, f"Need 20000 candidates, found {len(candidates)}"
   random.Random(42).shuffle(candidates)
   table = pa.Table.from_pylist(candidates[:20000]).select(columns)
   manifest = root / "pd3m_refl.parquet"
   if manifest.exists():
       assert pq.read_table(manifest).equals(table), (
           f"Existing manifest differs: {manifest}; use a new PD3M_DIR"
       )
   else:
       pq.write_table(table, manifest)
   print(f"Metadata rows: {len(rows)}; eligible: {len(candidates)}; selected: {len(table)}")
   print("Manifest:", manifest)
   PY

     "$PD3M_VENV/bin/img2dataset" \
       --url_list="$PD3M_DIR/pd3m_refl.parquet" \
       --input_format=parquet --url_col=url --caption_col=caption \
       --save_additional_columns='["id", "license", "source"]' \
       --output_format=files --output_folder="$PRE_DATA" \
       --image_size=512 --resize_mode=keep_ratio \
       --resize_only_if_bigger=True \
       --number_sample_per_shard=1000 \
       --processes_count=16 --thread_count=32 \
       --timeout=60 --retries=2 \
       --incremental_mode=incremental --enable_wandb=False

     "$PD3M_VENV/bin/python" - <<'PY'
   import os
   from pathlib import Path

   import pyarrow.parquet as pq
   from PIL import Image

   root = Path(os.environ["PRE_DATA"])
   count = pq.read_table(Path(os.environ["PD3M_DIR"]) / "pd3m_refl.parquet").num_rows
   stats = list(root.glob("*_stats.json"))
   expected_shards = (count + 999) // 1000
   assert len(stats) == expected_shards, (
       f"Incomplete download: {len(stats)}/{expected_shards} shard stats files"
   )
   images = sorted(root.rglob("*.jpg"))
   invalid = []
   for path in images:
       try:
           assert path.with_suffix(".txt").read_text(encoding="utf-8").strip(), "empty caption"
           with Image.open(path) as image:
               image.convert("RGB").load()
       except Exception as error:
           invalid.append(f"{path}: {error}")
   assert not invalid, f"Invalid image-caption pairs: {invalid[:5]}"
   assert len(images) >= 64, f"Need at least 64 valid pairs, found {len(images)}"
   print(f"Finished shards: {len(stats)}; valid image-caption pairs: {len(images)}")
   print("Lpre path:", root)
   PY
   )

The command saves JPEGs with short side 512, paired ``.txt`` captions, and
per-image metadata containing the original ID, license, and source. Lpre then
center-crops the images during training. Keep the original metadata and
``pd3m_refl.parquet`` with your experiment so the selection can be reproduced.
The dataset card specifies CDLA-Permissive-2.0; also review the per-image license
information retained in the manifest.

``hf download`` follows your existing Hugging Face endpoint and cache settings.
If you use ``HF_ENDPOINT=https://hf-mirror.com``, export it before this step.
That endpoint only affects metadata downloads: the image URLs still point to
S3. Allow time and disk space for the original image transfers; resizing reduces
stored image size, not the bytes fetched from the server. Keep the same
``HF_HOME`` used to cache the BERT tokenizer for training.

The download uses 16 processes and 32 threads per process. Reduce these values
on a shared or bandwidth-limited machine. With 20,000 inputs and 1,000 samples
per shard, a completed run has 20 ``*_stats.json`` files. A stats file appears
when its shard finishes, so its count can remain zero while JPEGs are already
being written. Monitor progress from another shell with:

.. code-block:: bash

   find "$PRE_DATA" -type f -name '*.jpg' | wc -l
   find "$PRE_DATA" -maxdepth 1 -type f -name '*_stats.json' | wc -l

The actual successful pair count can be below 20,000 because URLs may fail or
images may not decode. The verification at the end checks every written JPEG
and caption and requires all shards to finish; record its final pair count.
Its 64-pair minimum supports the smoke run below, not a sufficiently diverse
training set; investigate a low download success rate before formal training.
The fixed ``--incremental_mode=incremental`` skips shards with completed stats
on rerun, including shards that contain failed URLs. It does not automatically
retry every failed image in those completed shards. If you change the selected
rows or shard size, use a new output directory; mixing selections can silently
reuse unrelated completed shards. If the manifest check reports a mismatch,
choose a new ``PD3M_DIR`` rather than overwriting it in place.

Start the Supervised Run
~~~~~~~~~~~~~~~~~~~~~~~~

After the pair verification succeeds, ``PRE_DATA`` points to the directory
accepted by the image-text loader. If you already have a compatible dataset,
set ``PRE_DATA`` to that directory instead. Point the config at it and start a
separate run:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     runner.logger.experiment_name=sd3_refl_pre01

This run uses the same reward coefficient and rollout settings as the
reward-only run. Repeat with ``algorithm.refl_pre_loss_weight=0.01`` and a
new experiment name for a lower-weight comparison. Keep
``actor.refl_pre_data.num_workers=0`` to load images in the actor process;
this avoids creating DataLoader child processes after accelerator setup.

How the Two Updates Interact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actor uses a late rollout state to predict a clean latent, decodes it,
and differentiates the reward through the frozen VAE and ImageReward model.
The reward loss is

.. math::

   L_{\mathrm{reward}} = \lambda\,\mathbb{E}
   [\max(0, m - r(\hat{x}_0, c))],

where ``refl_grad_scale`` sets :math:`\lambda` and ``refl_margin`` sets
:math:`m`. ``refl_xt_source=rollout`` uses stored late states;
``resample`` re-noises the final rollout latent at a sampled late time.

The supervised objective for SD3 is flow matching:

.. math::

   z_t = (1-t)z_0 + t\epsilon, \qquad
   L_{\mathrm{pre}} = \mathbb{E}\left[
   \lVert v_\theta(z_t,t,c) - (\epsilon-z_0)\rVert^2\right].

For each global batch, the actor first takes an optimizer step on
``refl_pre_loss_weight * Lpre`` when enabled, then another step on Lreward.
Both steps share one AdamW optimizer and its moment estimates; reducing the
Lpre coefficient does not proportionally reduce its parameter update.
A zero coefficient skips the image loader and the supervised step entirely.

Lpre trains on your captioned images rather than constraining outputs to the
original SD3 model. Dataset mismatch or competing updates can lower the
ImageReward score. Compare with the reward-only run and inspect generated
images; a higher reward alone does not establish better visual quality.
See the `ImageReward paper <https://arxiv.org/abs/2304.05977>`_ for ReFL's
original SD1.4 formulation; this recipe adapts the denoising loss to SD3 flow
matching.

Tune the Supervised Weight
--------------------------

Use the supervised weight to test whether learning from captioned images
preserves useful generation behavior or interferes with preference tuning.
Lpre continues a denoising task on your chosen dataset; it is not a KL penalty,
an output-matching constraint to the initial SD3 model, or a second reward.
Its benefit depends on caption alignment, image quality, and how closely that
dataset matches the prompts you care about.

Interpret the Weight Through the Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\alpha` be ``refl_pre_loss_weight``. Increasing it scales the gradient
submitted by the supervised step. However, the actor takes two sequential
AdamW steps, not one step on a summed loss:

.. code-block:: text

   alpha > 0: step(alpha * Lpre), then step(Lreward)
   alpha = 0: step(Lreward)

For AdamW, a parameter update depends on both the gradient and its running
first and second moments, approximately
:math:`\Delta\theta=-\eta\hat{m}/(\sqrt{\hat{v}}+\epsilon)` before weight decay.
Scaling a sequence of gradients also changes its moment estimates. With shared
history from both losses, ``alpha=0.01`` therefore does not mean a supervised
parameter update of 1% of the ``alpha=1`` update. In particular, zero skips an
entire optimizer step; even a small positive weight keeps that step and its
interaction with the reward step. Weight decay and clipping also affect the
update. See the `AdamW update rule
<https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_.

At the settings above, one runner step contains eight global batches. With
Lpre enabled there are eight supervised and eight reward optimizer steps,
versus eight reward steps in the zero-weight run. Compare equal runner steps
and generated sample counts while recording the extra compute; the runs do not
have equal optimizer-step counts.

Keep Other Variables Fixed
~~~~~~~~~~~~~~~~~~~~~~~~~~

First complete the short GPU smoke below. Then hold
``algorithm.refl_grad_scale=1e-3``, learning rate, batch sizes, prompt split,
image subset, seeds, and validation cadence fixed. Change only the supervised
weight, and start every run from the same initial model and fresh optimizer.
The shared ``REFL_ARGS`` explicitly clears LoRA/checkpoint paths and sets the
seeds for this comparison. A shared seed controls a source of variation, but
enabling Lpre can still change the random draws consumed during training.

Use ``0.0`` as a same-length reward-only control. The positive weights
``1.0 / 0.75 / 0.5 / 0.1 / 0.01`` are a descending diagnostic sweep, not
paper-prescribed settings or known good values for SD3. If time is limited,
start with ``0.0 / 0.01 / 0.1``. Otherwise run all six settings for 50 runner
steps each:

.. code-block:: bash

   (
     set -e
     stamp="$(date +%Y%m%d_%H%M%S)"
     for weight in 0.0 1.0 0.75 0.5 0.1 0.01; do
       CUDA_VISIBLE_DEVICES=0,1,2,3 \
       bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
         "${REFL_ARGS[@]}" \
         actor.refl_pre_data.path="$PRE_DATA" \
         algorithm.refl_pre_loss_weight="$weight" \
         runner.max_epochs=50 runner.max_steps=50 \
         runner.logger.experiment_name="sd3_refl_pre${weight//./p}_50steps_${stamp}"
     done
   )

Each experiment has a distinct name, starts independently, evaluates and saves
every 10 steps, and must succeed before the next begins. Keep the matched
zero-weight control even if an older reward-only run looked good; different
training lengths or evaluation settings can confound the comparison.

Read the Curves Together
~~~~~~~~~~~~~~~~~~~~~~~~

Use held-out ``eval/image_reward`` and recorded images to select candidates,
not the training loss alone. Actor rewards use sampled late-state predictions;
they are not the same measurement as final-image evaluation rewards.

.. list-table:: Signals for a controlled Lpre sweep
   :header-rows: 1
   :widths: 27 34 39

   * - Observation
     - What it suggests
     - Next check
   * - Lower weights recover held-out reward
     - Supervised updates may be competing with preference tuning.
     - Inspect caption/image relevance and compare against the zero-weight run.
   * - Every positive weight degrades similarly, while zero works
     - A small coefficient has not removed the effect of the extra AdamW step.
     - Check data quality and gradient health; do not assume another tenfold reduction will fix it.
   * - An intermediate weight gives comparable reward and better images
     - There may be a useful quality tradeoff.
     - Verify prompt coverage, diversity, and repeatability on another seed.
   * - Training reward rises but held-out reward or image quality worsens
     - Overfitting or reward exploitation is possible.
     - Inspect fixed held-out prompts and artifacts before increasing reward strength.

``train/actor/refl_pre_loss`` logs the unweighted denoising MSE, whereas
``train/actor/refl_reward_loss`` includes ``refl_grad_scale``.
The two gradient norms include their respective loss coefficients, but their
ratio is not a parameter-update ratio or a target to force to one. Check that
losses and norms remain finite. Compare each metric across training steps
within its own definition, rather than comparing the numeric sizes of unlike
losses. Reward gradients can also vanish for samples above ``refl_margin``;
interpret a falling reward-gradient norm alongside the reward distribution.

Confirm Promising Weights with Longer Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After screening, compare promising weights over 200 steps. For example, run
``0.1`` and ``0.01`` independently as follows; these values are candidates,
not a recommendation that either must outperform reward-only training:

.. code-block:: bash

   (
     set -e
     stamp="$(date +%Y%m%d_%H%M%S)"
     for weight in 0.1 0.01; do
       CUDA_VISIBLE_DEVICES=0,1,2,3 \
       bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
         "${REFL_ARGS[@]}" \
         actor.refl_pre_data.path="$PRE_DATA" \
         algorithm.refl_pre_loss_weight="$weight" \
         runner.max_epochs=200 runner.max_steps=200 \
         runner.logger.experiment_name="sd3_refl_pre${weight//./p}_200steps_${stamp}"
     done
   )

Also extend the ``0.0`` control to 200 steps using the same command if you do
not already have a matched 200-step control. Compare checkpoints at common
steps and use multiple seeds before attributing a small score difference to a
weight. Only after this comparison should you run a separate sweep of
``refl_grad_scale`` or the learning rate. Changing the Lpre weight, reward
coefficient, and learning rate together prevents the experiment from isolating
the effect of Lpre.

Verify Training
---------------

Before a long run with Lpre, exercise training, validation, checkpoint saving,
and data iterator rollover. After the full dataset has passed verification,
make a separate 64-pair subset using symlinks. This preserves ``PRE_DATA`` for
formal runs and copies no image bytes:

.. code-block:: bash

   export DATA_ROOT="${DATA_ROOT:-$PWD/data/refl}"
   mkdir -p "$DATA_ROOT"
   export SMOKE_PRE_DATA="$(mktemp -d "$DATA_ROOT/pd3m-smoke.XXXXXX")"
   python - <<'PY'
   import os
   import random
   from pathlib import Path

   source = Path(os.environ["PRE_DATA"])
   target = Path(os.environ["SMOKE_PRE_DATA"])
   images = sorted(path for path in source.rglob("*.jpg") if path.is_file())
   assert len(images) >= 64, f"Need 64 image-caption pairs, found {len(images)}"
   for index, image in enumerate(random.Random(42).sample(images, 64)):
       caption = image.with_suffix(".txt")
       assert caption.read_text(encoding="utf-8").strip(), f"Empty caption: {caption}"
       (target / f"{index:05d}.jpg").symlink_to(image.resolve())
       (target / f"{index:05d}.txt").symlink_to(caption.resolve())
   print("Smoke image-caption directory:", target)
   PY

Each of the four actor ranks consumes its 16-image partition in one runner
step with the settings below. The second and third steps therefore recreate
the iterator. Run the three-step check on that subset:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash -o pipefail examples/diffusion/run_diffusion.sh sd3_refl_imagereward \
     "${REFL_ARGS[@]}" \
     actor.refl_pre_data.path="$SMOKE_PRE_DATA" \
     algorithm.refl_pre_loss_weight=0.1 \
     env.train.rollout_epoch=16 \
     env.eval.rollout_epoch=1 \
     runner.max_epochs=3 runner.max_steps=3 \
     runner.val_check_interval=3 runner.save_interval=3 \
     runner.logger.experiment_name=sd3_refl_smoke

Check that the process exits successfully and that the experiment's
``checkpoints/global_step_3/actor`` directory contains checkpoint files.
A successful smoke run exercises those boundaries; it does not guarantee
long-run memory stability or reward improvement. Formal runs keep using the
full ``PRE_DATA`` directory.

Run the CPU unit tests in the diffusion environment:

.. code-block:: bash

   pytest tests/unit_tests/test_diffusion.py

These tests use generated data and randomly initialized tiny models. A small
local module replaces the external ImageReward checkpoint while RLinf's
scorer runs normally. The tests cover prompt exclusion, image preprocessing, differentiable rewards, SD3
rollout/forward consistency, supervised gradients, and config composition.
They do not establish reward improvement with pretrained weights or validate
multi-GPU FSDP execution. Tests requiring optional dependencies are skipped when those dependencies
are unavailable.

Set ``env.eval.video_cfg.save_video=true`` to record evaluation samples under
``env.eval.video_cfg.video_base_dir``. The diffusion environment renders each
image as repeated video frames for the shared recording wrapper.

Monitor ``eval/image_reward`` and ``train/actor/refl_reward`` alongside generated
images. With Lpre enabled, also inspect ``train/actor/refl_pre_loss``,
``train/actor/refl_pre_grad_norm``, and ``train/actor/refl_reward_grad_norm``. Gradient
norm ratios are not AdamW parameter-update ratios. With 512 evaluation
samples and 463 prompts, 49 prompts repeat in each evaluation; compare runs
with the same evaluation settings. See
:doc:`/rst_source/guides/logger` for the shared logging workflow.
