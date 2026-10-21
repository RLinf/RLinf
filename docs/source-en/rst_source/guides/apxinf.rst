ApxInf Rollout Backend
======================

Use `ApxInf <https://github.com/infinigence/ApxInf>`_ as an in-process
Rust/CUDA rollout backend to evaluate a π₀.₅ policy on LIBERO. RLinf enters the
engine through `APXinf-robo <https://github.com/RLinf/APXinf-robo>`_, which
provides the stable L1 bare-model interface. RLinf manages the environments,
batching, OpenPI transforms, and action execution. ApxInf runs model inference
on the GPU.

.. note::

   The current integration supports π₀.₅ evaluation on LIBERO only. It does
   not support training, decoupled rollout, or non-embodied tasks.

Prerequisites
-------------

Prepare the following on the machine that will run evaluation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Platform
     - Linux with an NVIDIA GPU, CUDA toolkit, and a stable Rust toolchain.
   * - RLinf environment
     - Install the OpenPI model and LIBERO environment dependencies.
   * - APXinf-robo and ApxInf
     - Install APXinf-robo and build its ApxInf submodule's CUDA Python binding
       for the GPU architecture on the target machine. Install both in the same
       environment as RLinf.
   * - Checkpoint
     - A π₀.₅ LIBERO checkpoint containing ``config.json``,
       ``model.safetensors``, ``tokenizer.model``, and ``norm_stats.json``.

Install RLinf and APXinf-robo
-----------------------------

Install the RLinf OpenPI and LIBERO dependencies, then activate the resulting
environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env libero
   source .venv/bin/activate

Clone APXinf-robo recursively, then install the ApxInf CUDA binding, the ApxInf
Python frontend, and the APXinf-robo wrapper into that active environment:

.. code-block:: bash

   git clone --recursive https://github.com/RLinf/APXinf-robo.git
   cd APXinf-robo
   pip install maturin
   CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda \
     --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
   pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
   pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict
   pip install -e ".[rlinf]"

The CUDA build targets the GPU detected on the build machine. For a different
deployment target, set ``APXINF_CUDA_ARCH`` before running ``maturin``. See the
`APXinf-robo build guide
<https://github.com/RLinf/APXinf-robo#build-apxinf-robo>`_ for supported
architectures and build details.

Verify that the binding is available from the RLinf environment:

.. code-block:: bash

   python -c 'import apxinf_py; print(apxinf_py.__version__)'
   python -c 'import apxinf_robo; print(apxinf_robo.__version__)'

Prepare the Checkpoint
----------------------

Download the π₀.₅ LIBERO checkpoint and expose its directory to the example
configuration:

.. code-block:: bash

   pip install -U huggingface_hub
   hf download lerobot/pi05_libero_base \
     --local-dir /path/to/pi05_libero_base
   curl -fL \
     https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
     -o /path/to/pi05_libero_base/norm_stats.json
   export APXINF_PI05_MODEL_DIR=/path/to/pi05_libero_base

Download ``norm_stats.json`` separately as shown above. The current
``lerobot/pi05_libero_base`` repository does not include the normalization
statistics required to reproduce the reference evaluation behavior.

Run Evaluation
--------------

Return to the RLinf repository root and launch the provided LIBERO-10 config:

.. code-block:: bash

   bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

The default config uses 10 parallel environments and 10 rollout epochs. It
evaluates 10 reset states for each of the 10 LIBERO-10 tasks, for 100
trajectories in total. The terminal reports ``eval/success_once``; see
:doc:`LIBERO Evaluation <../evaluations/guides/libero>` for the evaluation
protocol and result locations.

Configure the Backend
---------------------

Start from ``evaluations/libero/libero_10_apxinf_pi05_eval.yaml``. The key
settings are:

.. code-block:: yaml

   rollout:
     rollout_backend: apxinf
     generation_backend: apxinf
     model:
       model_path: ${oc.env:APXINF_PI05_MODEL_DIR}
       num_action_chunks: 5
       action_dim: 7
       openpi:
         config_name: pi05_libero
         num_steps: 5
       apxinf:
         precision: bf16
         action_horizon: 10
         num_flow_steps: ${..openpi.num_steps}
         noise_source: apxinf

Keep ``runner.only_eval: true`` and
``runner.enable_decoupled_mode: false``. ``num_flow_steps`` must match
``openpi.num_steps``. The default ``noise_source: apxinf`` lets ApxInf sample
the initial Gaussian noise. Use ``noise_source: observation`` only for paired
numerical parity tests that provide an explicit ``[B, 10, 32]`` noise tensor.

How the Integration Works
-------------------------

RLinf owns LIBERO interaction, Ray scheduling, observation batching, image
resize, tokenization, checkpoint normalization, action unnormalization, and
action-chunk execution. The adapter calls ``apxinf_robo.load_bare_model``, which
selects the tuned GEMM tactics and opens ApxInf's low-level ``Model.infer_rgb``
API. It passes prepared RGB views, token IDs, and optional noise to the engine.
ApxInf returns normalized model-space actions, which RLinf converts back to
environment actions.
