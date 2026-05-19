RL on ABot-M0
==============

This example describes how to run evaluation and PPO training for
`ABot-M0 <https://github.com/amap-cvlab/ABot-Manipulation>`__ in RLinf. The
provided configuration files cover standard **LIBERO** and **LIBERO-Plus**.

The integration uses the Hugging Face rollout backend and FSDP actor training.
During rollout, ABot-M0 generates action chunks for LIBERO environments. During
actor updates, RLinf recomputes log probabilities and value estimates from the
stored rollout inputs.

Algorithm
---------

The example uses PPO with an actor-critic loss:

* GAE for advantage and return estimation.
* PPO ratio clipping for policy updates.
* Value-function clipping for the value head.
* Optional entropy regularization.

ABot-M0 is used as the VLA policy. The RLinf wrapper keeps pretrained
perception components frozen, trains the action model through the RL objective,
and adds a value head for actor-critic training.

Installation
------------

Dependency Installation
~~~~~~~~~~~~~~~~~~~~~~~

Install ABot-M0, VGGT, and the standard LIBERO runtime in the same Python
environment as RLinf. The following installer command installs standard LIBERO;
only LIBERO-Plus requires the additional steps in the next section.

.. code-block:: bash

   git clone https://github.com/amap-cvlab/ABot-Manipulation.git
   git clone https://github.com/facebookresearch/vggt.git

   cd <path_to_RLinf>
   export ABOT_PATH=<path_to_ABot-Manipulation>
   export VGGT_PATH=<path_to_vggt>

   # Optional mirror for environments with limited access to GitHub and PyPI.
   # Add --use-mirror to the installer command when needed.
   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env maniskill_libero --install-rlinf
   source .venv/bin/activate

LIBERO-Plus
~~~~~~~~~~~

LIBERO-Plus requires an additional ``LIBERO-plus`` checkout. After the standard
LIBERO installation above, run the installer with ``--env liberoplus`` in the
same environment:

.. code-block:: bash

   cd <path_to_RLinf>

   # Optional mirror for environments with limited access to GitHub and PyPI.
   # Add --use-mirror to the installer command when needed.
   bash requirements/install.sh embodied --venv .venv --model abot_m0 --env liberoplus --install-rlinf
   source .venv/bin/activate

Download and extract the LIBERO-Plus assets into the installed package
directory:

.. code-block:: bash

   # Resolve the installed liberoplus package directory.
   # Note: importing liberoplus may emit config-init logs, so use tail -n 1 to keep only the final path.
   export LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)" | tail -n 1)

   echo "LIBERO_PLUS_PACKAGE_DIR=${LIBERO_PLUS_PACKAGE_DIR}"

   # Optional mirror for environments that cannot access Hugging Face directly.
   # export HF_ENDPOINT=https://hf-mirror.com

   # Download the assets archive from the Hugging Face dataset repo.
   hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip \
       --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"

   # assets.zip contains a long original path prefix, so extract only the assets/ subtree.
   python - <<'PY'
   import zipfile
   from pathlib import Path

   pkg = Path(__import__("os").environ["LIBERO_PLUS_PACKAGE_DIR"])
   zip_path = pkg / "assets.zip"
   out_dir = pkg / "assets"

   with zipfile.ZipFile(zip_path) as z:
       for info in z.infolist():
           name = info.filename

           if "/assets/" not in name:
               continue

           rel = name.split("/assets/", 1)[1]
           if not rel:
               continue

           target = out_dir / rel

           if info.is_dir():
               target.mkdir(parents=True, exist_ok=True)
           else:
               target.parent.mkdir(parents=True, exist_ok=True)
               with z.open(info) as src, open(target, "wb") as dst:
                   dst.write(src.read())

   print("Extracted LIBERO-Plus assets to:", out_dir)
   PY

   # Verify the assets directory structure.
   ls -lh "${LIBERO_PLUS_PACKAGE_DIR}/assets"

After extraction, the directory should look like:

.. code-block:: text

   <installed liberoplus package dir>/
   └── assets/
       ├── articulated_objects/
       ├── new_objects/
       ├── scenes/
       ├── stable_hope_objects/
       ├── stable_scanned_objects/
       ├── textures/
       ├── turbosquid_objects/
       ├── serving_region.xml
       ├── wall_frames.stl
       └── wall.xml

See :doc:`liberoplus_pro` for full LIBERO-Plus details.

Model Weights
-------------

Evaluation Checkpoint
~~~~~~~~~~~~~~~~~~~~~

The SFT checkpoint released by ABot can be used for standalone evaluation:

.. code-block:: bash

   # Optional mirror for environments that cannot access Hugging Face directly.
   # export HF_ENDPOINT=https://hf-mirror.com

   hf download acvlab/ABot-M0-LIBERO --local-dir /path/to/ABot-M0-LIBERO

RL Baseline Checkpoint
~~~~~~~~~~~~~~~~~~~~~~

For PPO training, a 10k-step ABot-M0 LIBERO checkpoint is available as an RL
baseline. In LIBERO evaluation it provides an initial success rate of
approximately 40% and is suitable as the starting point for further RL training.

.. code-block:: bash

   hf download HaoyunOvO/ABot-m0-LIBERO-10k-step \
     --local-dir /path/to/ABot-m0-LIBERO-10k-step

Checkpoint Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

ABot-M0 checkpoints include ``config.yaml``. Check the ``qwenvl.base_vlm`` field
and replace developer-local path with the local path to the Qwen3-VL backbone:

.. code-block:: bash

   # Optional mirror for environments that cannot access Hugging Face directly.
   # export HF_ENDPOINT=https://hf-mirror.com

   hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
     --local-dir /path/to/Qwen3-VL-4B-Instruct-Action

.. code-block:: yaml

   qwenvl:
     base_vlm: /path/to/Qwen3-VL-4B-Instruct-Action

The Qwen3-VL backbone is available at
``https://huggingface.co/StarVLA/Qwen3-VL-4B-Instruct-Action``.

Offline VGGT Loading
~~~~~~~~~~~~~~~~~~~~

ABot currently initializes VGGT with:
``VGGT.from_pretrained("facebook/VGGT-1B")``

If the runtime can access Hugging Face through a mirror, set ``HF_ENDPOINT``:

.. code-block:: bash

   export HF_ENDPOINT=https://hf-mirror.com

Otherwise, pre-download VGGT and make it available from the local Hugging Face
cache or a local directory:

.. code-block:: bash

   hf download facebook/VGGT-1B --local-dir /path/to/VGGT-1B

Then either:

* place the model in your local Hugging Face cache, or
* explicitly set VGGT loading to a local directory in your ABot installation.

Example local override:

.. code-block:: python

   self.spatial_model = spatial_model = VGGT.from_pretrained('/workspace/models/VGGT-1B')

Configure ``model_path``
------------------------

Two configs are provided, one per benchmark:

* LIBERO:    ``examples/embodiment/config/libero_10_ppo_abot_m0.yaml``
* LIBERO-Plus:  ``examples/embodiment/config/libero_10_plus_ppo_abot_m0.yaml``

Set both fields to the checkpoint used for evaluation or training:

* ``rollout.model.model_path``
* ``actor.model.model_path``

For the 10k-step RL baseline, use:

.. code-block:: yaml

   rollout:
     model:
       model_path: /path/to/ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt
   actor:
     model:
       model_path: /path/to/ABot-m0-LIBERO-10k-step/checkpoints/steps_10000_pytorch_model.pt

Import Sanity Check
-------------------

.. code-block:: bash

   python -c "import rlinf; import ABot; import vggt; print('IMPORT_OK')"

If the command prints ``IMPORT_OK``, the package-level dependency wiring is valid.

Evaluation
----------

Use standalone evaluation before training to verify the checkpoint, rollout
pipeline, and environment assets.

The eval entrypoint is ``examples/embodiment/eval_embodied_agent.py``. Both
benchmarks share the same launch flow; the only differences are
``LIBERO_TYPE`` and the config name.

Common environment setup:

.. code-block:: bash

   source .venv/bin/activate

   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

   ray stop || true
   ray start --head --port=6379

**LIBERO:**

.. code-block:: bash

   export LIBERO_TYPE=standard

   python examples/embodiment/eval_embodied_agent.py \
     --config-name libero_10_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     runner.only_eval=True \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     algorithm.eval_rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_libero10_eval

**LIBERO-Plus:**

.. code-block:: bash

   export LIBERO_TYPE=plus

   python examples/embodiment/eval_embodied_agent.py \
     --config-name libero_10_plus_ppo_abot_m0 \
     actor.model.model_path=<path_to_abot_m0_ckpt> \
     rollout.model.model_path=<path_to_abot_m0_ckpt> \
     runner.only_eval=True \
     env.eval.total_num_envs=8 \
     env.eval.video_cfg.save_video=true \
     algorithm.eval_rollout_epoch=1 \
     runner.logger.experiment_name=abot_m0_liberoplus_eval

Training
--------

PPO training uses the same launch flow as evaluation. Select the target suite
with ``LIBERO_TYPE`` and launch the corresponding config.

Common environment setup:

.. code-block:: bash

   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   export ROBOT_PLATFORM=LIBERO

   ray stop || true
   ray start --head --port=6379

**LIBERO:**

.. code-block:: bash

   export LIBERO_TYPE=standard
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_abot_m0

**LIBERO-Plus:**

.. code-block:: bash

   export LIBERO_TYPE=plus
   bash examples/embodiment/run_embodiment.sh libero_10_plus_ppo_abot_m0

Visualization
-------------

.. code-block:: bash

   tensorboard --logdir <runner.logger.log_path> --port 6006
