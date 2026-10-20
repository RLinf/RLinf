RoboDojo RL
===========

RoboDojo is an Isaac Sim 5.1 benchmark for dual-arm ARX-X5 manipulation. RLinf
wraps it as a Gym environment so policies can send continuous dual-arm action
chunks directly to the simulator.

This path currently supports **Linux x86_64 with an NVIDIA CUDA GPU** only.
Start with one environment and verify the simulator, cameras, and assets before
increasing parallelism.

Three things to remember
------------------------

1. Install with the RLinf installer; do not install Isaac Sim, IsaacLab, and
   cuRobo one package at a time.
2. ``ROBODOJO_ASSETS_ROOT`` points to the directory that **contains**
   ``Assets/``.
3. Each parallel RoboDojo environment owns an Isaac Sim process. Start with one
   environment even on a 24 GB GPU.

Before installing
-----------------

.. code-block:: bash

   nvidia-smi
   uname -m
   df -h .
   python --version

One-command dependency install
-----------------------------

From the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate

Users in mainland China can add ``--use-mirror``. To include OpenPI:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo

The installer:

* installs PyTorch 2.7, Isaac Sim 5.1, pinned IsaacLab, and pinned cuRobo;
* applies ``requirements/embodied/envs/robodojo-override.txt`` for the known
  ``starlette``, ``uvicorn``, and ``filelock`` conflicts;
* writes ``OMNI_KIT_ACCEPT_EULA`` and ``ROBODOJO_ASSETS_ROOT`` into the virtual
  environment;
* checks ``robodojo_runtime``, CUDA visibility, and the vendored RoboDojo tree;
* does not start training or download the large scene assets by default.

.. note::

   The installer sets ``OMNI_KIT_ACCEPT_EULA=YES``. Continuing the installation
   means accepting the applicable NVIDIA Omniverse license terms.

Do not run a plain ``pip install rlinf-robodojo-runtime``. IsaacLab and Isaac
Sim have conflicting Starlette requirements, so the installer's override file
is required.

Install scene assets
--------------------

The complete ``Assets/`` tree is about 39 GB. You can place it anywhere on a
persistent disk:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   mkdir -p "$ROBODOJO_ASSETS_ROOT"

Or let the installer download it during the same run:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   ROBODOJO_DOWNLOAD_ASSETS=1 \
     bash requirements/install.sh embodied --env robodojo

The required layout is:

.. code-block:: text

   /path/to/robodojo/
   └── Assets/
       ├── Robots/
       ├── Object/
       ├── Material/
       └── Eval_Layout/

Do not set ``ROBODOJO_ASSETS_ROOT`` to the ``Assets`` directory itself. Set
``HF_ENDPOINT`` before installation if a Hugging Face mirror is needed.

Post-install check
------------------

This check does not launch the simulator:

.. code-block:: bash

   python - <<'PY'
   import torch
   from robodojo_runtime import assets_root, source_root
   from robodojo_runtime.bridge import VectorEnv

   print("torch:", torch.__version__)
   print("cuda:", torch.cuda.is_available())
   print("source:", source_root())
   print("assets:", assets_root())
   PY

First real launch
-----------------

Run a single-environment, single-action smoke test on an idle GPU:

.. code-block:: bash

   python - <<'PY'
   import numpy as np
   from robodojo_runtime.bridge import VectorEnv

   env = VectorEnv(
       {
           "task_name": "put_bottles_into_dustbin",
           "env_cfg_type": "arx_x5",
           "headless": True,
           "cuda_device": 0,
           "max_episode_steps": 700,
       },
       n_envs=1,
       env_seeds=[0],
   )
   try:
       env.reset()
       obs = env.get_obs()[0]
       print(obs["full_image"].shape, obs["state"].shape)
       env.step(np.zeros((1, 1, 14), dtype=np.float32))
   finally:
       env.close()
   PY

Do not enable a policy server, multiple environments, and video recording during
the first run. Once the smoke test exits cleanly, use
``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``.

Common problems
---------------

``No module named 'robodojo_runtime'``
   Activate the virtual environment created by the installer. If you used
   ``--venv``, activate that environment instead.

``No solution found when resolving dependencies``
   Do not install the runtime manually. Remove the failed installation and run
   the one-command installer above.

``Could not detect CUDA`` or ``torch.cuda.is_available() == False``
   Run ``nvidia-smi`` first. RoboDojo does not currently support CPU-only,
   macOS, or AMD ROCm execution.

Missing ``Assets`` or ``cam_head`` errors
   Verify that ``ROBODOJO_ASSETS_ROOT`` points to the parent of ``Assets/`` and
   that all four required subdirectories exist.

Out of GPU memory
   Start with one environment and stop other training or inference processes.
   Multi-environment RoboDojo starts one Isaac Sim process per environment and
   cannot reuse RoboTwin's large environment counts directly.

GPU memory remains allocated after exit
   Call ``env.close()`` from a ``try/finally`` block. The adapter supports a
   terminal close and recreate, not suspend and resume.
