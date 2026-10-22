Streaming Flow Policy Supervised Fine-Tuning
============================================

.. figure:: https://raw.githubusercontent.com/gen-robot/StreamingVLA/main/assets/final_tea_01.png
   :align: center
   :width: 70%

   Streaming Flow Policy, from the `StreamingVLA
   <https://github.com/gen-robot/StreamingVLA>`_ project.

Fine-tune a π₀.₅ checkpoint on LIBERO with the **Streaming Flow Policy (SFP)**
objective. SFP trains the action expert to follow an action *trajectory* rather
than denoise a whole action chunk, which is what lets a caller keep executing
while the next actions are still being produced. Only the objective changes:
the architecture, the parameter shapes, and the checkpoint layout stay those of
π₀.₅, so training starts from a stock ``pi05_libero`` checkpoint and its output
loads back into one.

Overview
--------

Train π₀.₅ on LIBERO demonstrations with the SFP objective, full-parameter, on one node.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ (``openpi``)

   .. grid-item-card:: Methods
      :text-align: center

      Streaming Flow Policy

   .. grid-item-card:: Data
      :text-align: center

      LIBERO · LeRobot format

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4 GPUs

| **You'll do:** install OpenPI → convert LIBERO → compute norm stats → convert the π₀.₅ checkpoint → launch ``run_vla_sft.sh`` → watch the training loss.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · disk space for the raw LIBERO RLDS data and its LeRobot copy.

.. warning::

   SFP eval uses a trajectory sampler (``task: eval`` plus ``use_sfp``), not the
   flow-matching Euler used by ordinary π₀.₅. RL, DAgger, and DSRL still reject
   ``use_sfp``. Loading an SFP checkpoint with ``use_sfp: False`` would silently
   integrate the wrong field — keep the SFP eval recipe.

How Streaming Flow Policy Differs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A π₀.₅ model denoises an entire action chunk at once. Its flow time is a
denoising coordinate with no relation to robot time, so nothing in the chunk is
executable until every denoising step has run.

SFP integrates the chunk into a trajectory and maps the flow time onto it
instead. ``t=0`` is the action state the robot has already reached, ``t=1`` is
the end of the chunk, and the action expert regresses the trajectory velocity at
``t`` from a single suffix token. A policy trained this way can be integrated
forward from wherever it currently is.

Two settings in the recipe follow from that. ``actor.model.openpi.use_sfp``
selects the objective, and ``sfp_sigma`` with ``sfp_noise_decay`` control how
much noise is injected at the start of the trajectory and how quickly it decays
along it.

Prepare the Dataset
~~~~~~~~~~~~~~~~~~~

SFP needs one field that flow matching does not: ``action_states``, the running
total of the actions taken before each frame, which is where that frame's
trajectory begins. Download the public `OpenVLA modified LIBERO RLDS dataset
<https://huggingface.co/datasets/openvla/modified_libero_rlds>`_ and convert it:

.. code:: bash

   hf download openvla/modified_libero_rlds \
       --repo-type dataset \
       --local-dir /data/libero-rlds

   export HF_LEROBOT_HOME=/data/lerobot

   python toolkits/lerobot/convert_libero_data_to_lerobot.py \
       --data-dir /data/libero-rlds \
       --repo-name local/libero_sfp

The converter merges the four LIBERO suites into
``$HF_LEROBOT_HOME/local/libero_sfp`` and records ``action_states`` for every
frame. Pass ``--raw-dataset-names`` to convert a subset of the suites, and
``--overwrite`` to replace an existing dataset at that path.

.. note::

   ``--repo-name`` is resolved under ``HF_LEROBOT_HOME``; an absolute path is
   rejected. Keep ``HF_LEROBOT_HOME`` at the same value for the statistics step
   and for training, or the loader will not find the dataset.

Normalization Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

The converted dataset still holds raw action units, so the next step measures
the scale the model trains in:

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_libero_sfp \
       --repo-id local/libero_sfp \
       --output-dir /data/assets/libero_sfp

This writes ``/data/assets/libero_sfp/norm_stats.json`` with statistics for
``state``, ``actions``, and ``action_states``. ``action_states`` is a copy of
``actions``, because SFP adds those two on one scale.

The same requirement is why SFP divides by ``max(|q01|, |q99|)`` instead of
using the affine quantile map OpenPI applies elsewhere: a pure scaling commutes
with the sum, so accumulating normalized deltas gives the same trajectory as
normalizing the accumulated one. An affine map shifts every value and breaks
that identity.

Installation
------------

.. include:: _setup_common.rst

SFP runs in the OpenPI environment; no separate install target is needed.

**Option 1: Docker image** — image tag ``agentic-rlinf0.4-maniskill_libero``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # Mainland China mirror: infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

   # Inside the container, switch to the OpenPI virtual environment:
   source switch_env openpi

**Option 2: Custom environment** — install bundle ``--env maniskill_libero``:

.. code:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

Download the Model
------------------

Training starts from the official ``pi05_libero`` weights in RLinf's PyTorch
layout, a directory holding ``model.safetensors``. Download and convert them
from an `OpenPI <https://github.com/Physical-Intelligence/openpi>`_ checkout:

.. code:: bash

   python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"

   python examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
       --config_name pi05_libero \
       --output_path /data/checkpoints/pi05_libero_pytorch

Run It
------

**1. Configuration**

The recipe is split into a path-free model template,
``examples/sft/config/model/pi0_5_sfp.yaml``, and the experiment config,
``examples/sft/config/libero_sft_pi05_sfp.yaml``. Point the experiment config at
the three paths produced above:

.. code:: yaml

   data:
     train_data_paths: local/libero_sfp

   actor:
     model:
       model_path: /data/checkpoints/pi05_libero_pytorch
       openpi_data:
         norm_stats_path: /data/assets/libero_sfp/norm_stats.json

The model template selects the objective and the data pipeline together:

.. code:: yaml

   openpi:
     task: sft
     config_name: "pi05_libero_sfp"
     use_sfp: True
     sfp_sigma: 0.16
     sfp_noise_decay: 4.0

``config_name`` does double duty. It picks the OpenPI data config that repacks
and pads ``action_states``, and its name also selects the matching SFT data
loader.

The supplied recipe runs full-parameter training on four GPUs with OpenPI's
default precision: Gemma and SigLIP in bf16, the action heads in fp32. The SFP
trajectory targets are always built in fp32.

.. warning::

   ``global_batch_size`` must stay a multiple of ``micro_batch_size`` times the
   world size, otherwise the actor rejects the config at startup.

**2. Launch**

.. code:: bash

   source .venv/bin/activate
   export HF_LEROBOT_HOME=/data/lerobot
   bash examples/sft/run_vla_sft.sh libero_sft_pi05_sfp

What this command does: 1. composes the Hydra config and starts a Ray cluster;
2. builds the π₀.₅ model with the SFP objective and loads the converted
checkpoint; 3. streams LIBERO through the SFP data pipeline and trains.

Visualization and Results
-------------------------

Monitor the **training loss** to confirm the model is fitting the
demonstrations. For every logged metric, see :doc:`Training metrics
<../../reference/metrics>`.

.. code-block:: bash

   # Launch TensorBoard
   tensorboard --logdir ./logs

``run_vla_sft.sh`` overrides ``runner.logger.log_path`` with a timestamped
directory, so checkpoints land in
``logs/<timestamp>-libero_sft_pi05_sfp/checkpoints/global_step_<N>/``. To
resume, point ``runner.resume_dir`` at one of them and relaunch.

Evaluate a Checkpoint
---------------------

SFP eval uses the trajectory sampler (``task: eval`` plus ``use_sfp``), not the
flow-matching Euler used by ordinary π₀.₅. Point
``evaluations/libero/libero_spatial_openpi_pi05_sfp_eval.yaml`` at the SFT
checkpoint and the same ``norm_stats.json`` used for training:

.. code:: bash

   bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_sfp_eval \
     rollout.model.model_path=/path/to/pi05_libero_sfp \
     rollout.model.openpi_data.norm_stats_path=/data/assets/libero_sfp/norm_stats.json

RTC is incompatible with SFP: it integrates the flow-matching velocity field.
