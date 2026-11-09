PhyAI Embodied Rollout Backend
==============================

Use `PhyAI <https://github.com/mingti-org/phyai>`_ to train a PI0.5 policy
in process or evaluate it through an external Gateway. PhyAI is an inference
framework for physical-AI models such as VLAs. The in-process path puts one
engine in each RLinf rollout worker and keeps the LIBERO environments, FSDP
actor, observation transforms, and weight synchronization in RLinf.

The sections below install and run the in-process PPO backend. For separate
model servers, follow :ref:`phyai-http-evaluation-en`. That evaluation-only
path uses ``PhyaiEmbodiedWorker`` and keeps the Gateway and model server in a
separate environment.

Prerequisites
-------------

You need a Linux machine with NVIDIA GPUs and Python 3.12. The backend
currently supports PI0.5 on LIBERO with synchronous PPO, and each rollout
worker uses exactly one GPU. The training path has been smoke-tested on two
NVIDIA H800 80 GB GPUs with the RLinf PI0.5 LIBERO SFT checkpoint.

Installation
------------

From the RLinf repository root, install RLinf, OpenPI, LIBERO, and PhyAI into
one virtual environment, then activate it:

.. code-block:: bash

   bash requirements/install.sh \
     --python 3.12 \
     --torch 2.11.0 \
     --install-rlinf \
     embodied \
     --model openpi \
     --env libero \
     --engine phyai
   source .venv/bin/activate

What this does:

1. Installs RLinf, OpenPI, and LIBERO exactly as for the Hugging Face rollout.
2. Installs the PhyAI packages at the version RLinf is tested with, for the
   CUDA version of the installed Torch. PhyAI's own dependency list would
   replace the Torch and Transformers versions OpenPI needs, so the installer
   keeps those versions and stops if PhyAI's CUDA build does not match Torch.

Prepare the Checkpoint
----------------------

Download the PI0.5 LIBERO SFT checkpoint:

.. code-block:: bash

   # For mainland China users: export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

Then set ``actor.model.model_path`` in
``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` to the
downloaded directory. The rollout reads the same path, so PhyAI and the actor
load the same weights and normalize observations with the same statistics.
:doc:`RL on π₀ and π₀.₅ Models <../../examples/embodied/pi0>` lists the other
PI0.5 checkpoints and download mirrors.

To start from an RL training checkpoint instead, also set ``runner.ckpt_path``
to its ``.pt`` file. Each PhyAI engine loads that state dict on top of the
downloaded checkpoint, so the first rollout uses the same weights as the actor,
frozen parameters included.

Configure the Backend
---------------------

The example config already selects PhyAI. These are the lines that matter:

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   algorithm:
     recompute_logprobs: True
     importance_sampling_fix: True
     importance_sampling_clip: 1.0

   rollout:
     rollout_backend: phyai
     phyai:
       max_batch_size: null

``rollout.rollout_backend: phyai`` makes the training script start PhyAI
rollout workers instead of the Hugging Face ones. The ``hybrid_engines/phyai``
default supplies the kernel policy, which fixes the attention and activation
kernels PhyAI uses for PI0.5 so the numbers stay as close to the actor as the
two implementations allow. ``max_batch_size`` caps how many observations one
engine runs at once; ``null`` sizes it to the environments assigned to that
worker.

The three ``algorithm`` keys deal with the remaining difference between PhyAI
and the actor, which the next section explains.

Correct Rollout and Actor Differences
-------------------------------------

PhyAI and the FSDP actor hold the same weights, but they compute with different
kernels: paged attention, fused activations, and a different reduction order.
So when the actor scores an action that PhyAI sampled, it gets a slightly
different log-probability than PhyAI reported.

PPO measures each update against the policy that produced the data. If that
reference came from PhyAI, the kernel gap would show up in every PPO ratio and
could trigger clipping on its own. With
``recompute_logprobs``, the actor rescores every sampled action before training
and uses its own score as the reference instead. ``importance_sampling_fix``
then reweights each sample by how much more likely the actor finds the action
than PhyAI did, capped at ``importance_sampling_clip``, so the update stays
correct for data the actor did not sample itself. Keep the three keys together:
startup stops if ``importance_sampling_fix`` is on without
``recompute_logprobs``.

:doc:`PPO <../../reference/algorithms/ppo>` describes the correction and its
metrics in detail.

Run Training
------------

With the checkpoint path set, launch the example from the repository root:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

What this does: starts the embodied training entry point with an FSDP actor,
PhyAI rollout workers, and LIBERO environment workers. Each step collects
rollouts with PhyAI, has the actor rescore them, updates the actor, and sends
the new weights to every PhyAI engine before the next rollout.

Check the Run
-------------

The launcher writes logs under ``logs/<timestamp>-<config-name>/``. Watch
``env/success_once`` for task success. ``actor/importance_sampling_weight``
near 1 with a small ``actor/importance_sampling_clip_fraction`` means PhyAI and
the actor still agree closely; a weight drifting away from 1 means their scores
are diverging. :doc:`Training Metrics <../../reference/metrics>` defines the
remaining metrics.

.. _phyai-http-evaluation-en:

Evaluate Through an External Gateway
------------------------------------

Run this mode when model inference belongs to a separately managed service.
The evaluation driver starts RLinf environment workers and HTTP rollout
workers; you start the model server and Gateway first.
``serving_mode: worker_http`` selects this evaluation-only path, while the PPO
instructions above use the in-process ``PhyAIRolloutWorker``.

.. code-block:: text

   RLinf EnvWorker -> PhyaiEmbodiedWorker
       -> HTTP MessagePack -> PhyAI Gateway
       -> gRPC -> PI0.5 model server

Install the Evaluation Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a dedicated client environment using RLinf's existing SGLang/LIBERO
installation route. The unchanged public ``InferenceHTTPClient`` imports the
SGLang action protocol to encode MessagePack requests, even when their target
is a PhyAI Gateway:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh      --python 3.12      --venv .venv-phyai-http      --install-rlinf      embodied --model sglang --sglang 0.5.19 --env libero
   source .venv-phyai-http/bin/activate

This installs the official codec and its import dependencies without changing
RLinf's shared HTTP client or project metadata. The script installs software;
the only inference services started below are the PhyAI model server and
Gateway. The HTTP client environment and serving environment remain separate
because their SGLang, Torch, and Transformers dependencies can differ.

.. warning::

   Use a new environment for these commands. The installer synchronizes its
   selected dependencies; do not point it at a mixed experimental environment
   whose additional packages you need to preserve.

Prepare the Serving Environment and Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse the same ``--engine phyai`` installer as the in-process backend in a
separate environment. It selects the CUDA 12/13 requirements, pins PhyAI to
``0.2.0rc1``, and checks that installing PhyAI preserves RLinf's Torch and
Transformers versions. Add only the Gateway and processor packages plus their
HTTP-specific dependencies afterwards:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh      --python 3.12      --torch 2.11.0      --venv .venv-phyai-serving      --install-rlinf      embodied --model openpi --env libero --engine phyai
   source .venv-phyai-serving/bin/activate
   env -u UV_TORCH_BACKEND uv pip install      -r requirements/agentic/phyai_0.2.0rc1_http.txt
   env -u UV_TORCH_BACKEND uv pip install --no-deps      "phyai-gateway==0.2.0rc1"      "phyai-utils-tools==0.2.0rc1"      --index https://mingti-org.github.io/phyai/simple/

The HTTP dependency file supplements the existing installer rather than
maintaining a second PhyAI runtime recipe. ``--no-deps`` keeps the supplemental
packages from replacing the validated Torch/Transformers stack. The serving
job checks those versions before and after this step, then checks the Server
configuration. Keep the installed environment's ``bin`` directory on ``PATH``
for ``ninja`` and other kernel-build tools; activation and the Shell e2e runner
do this. These additions still require a clean-install check on the A800
runner; the in-process installation result alone does not validate HTTP
serving or its checkpoint processor.

Prepare a local LeRobot-format PI0.5 LIBERO checkpoint with its preprocessing,
normalization, and action postprocessing files, plus a complete local
PaliGemma tokenizer snapshot. The checkpoint must describe an eight-value
state, seven-value actions, and an action chunk of at least 16 steps. The
in-process RLinf/OpenPI checkpoint shown earlier uses a different format;
confirm compatibility with ``PI05Backend`` before substituting it. Arrange
model and tokenizer access, including any access approval, separately.

Create ``pi05-http.yaml`` in the RLinf repository root with these settings,
using resource paths on the model host:

.. code-block:: yaml

   plugin: pi05
   plugin_args:
     checkpoint_dir: /path/to/lerobot-pi05-libero
     max_batch_size: 8
     inputs_image_shape:
       - [360, 360, 3]
       - [360, 360, 3]
   config:
     device:
       target: cuda
       params_dtype: bfloat16
     runtime:
       use_cuda_graph: true
       flashinfer_workspace_bytes: 2147483648
   deployment:
     mode: auto
   server:
     host: 127.0.0.1
     port: 50063
     model_name: pi05
     workers: 1
   adapter:
     factory: phyai.server.backends.pi05:PI05Backend
     args:
       tokenizer_dir: /path/to/paligemma-tokenizer
       image_names: [main_images, wrist_images]
       state_dim: 8
       action_dim: 7

The camera names match the RLinf HTTP request, rather than the raw LIBERO
camera keys used by the LeRobot client. ``max_batch_size: 8`` covers the
single-worker, eight-environment example below. Before increasing request
batches, raise this limit and check GPU memory and the 100 MiB message limit.
The 2 GiB FlashInfer workspace is explicit instead of relying on the smaller
runtime default.

Start the Model Server and Gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose an available GPU and unused ports. In the first serving terminal,
activate the PhyAI environment and start the model server:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai-serving/bin/activate
   export GPU_ID=0
   CUDA_VISIBLE_DEVICES="$GPU_ID" python -m phyai server pi05-http.yaml

Wait for ``Serving model pi05 on 127.0.0.1:50063``, which follows model
initialization and warmup. Keep it running. In a second serving terminal,
activate the same environment and start the Gateway:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai-serving/bin/activate
   python -m phyai_gateway.server \
     --gateway 127.0.0.1:50111 \
     --http-host 127.0.0.1 \
     --http-port 30000 \
     --backend pi05 127.0.0.1:50063
   # In another shell:
   curl --noproxy '*' http://127.0.0.1:30000/v1/backends

The Gateway probes ``CheckHealth`` on the registered server. Wait until the
record for ``pi05`` at ``127.0.0.1:50063`` reports ``healthy: true``. Both the
server YAML's ``server.model_name`` and the backend name must be ``pi05``,
which the current HTTP worker sends in every request. Registration lasts for
the Gateway process; include ``--backend`` again after a restart.

These loopback addresses assume that the services and RLinf workers share a
network namespace. For separate hosts or containers, bind the services to
reachable interfaces, use the model host's address in ``--backend``, and set
``rollout.phyai.gateway_url`` to the Gateway host. Keep these trusted HTTP/gRPC
interfaces behind the deployment's network controls.

Run the HTTP Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

With both services ready, use a third terminal for the RLinf client. Select an
available GPU for LIBERO rendering and set placement to the same index. The
model server can use a different GPU because it is a separate process.

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai-http/bin/activate
   export GPU_ID=0
   export CUDA_VISIBLE_DEVICES="$GPU_ID"
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     "cluster.component_placement.env=$GPU_ID" \
     "cluster.component_placement.rollout=$GPU_ID" \
     rollout.phyai.gateway_url=http://127.0.0.1:30000 \
     env.eval.total_num_envs=8 \
     env.eval.rollout_epoch=1 \
     env.eval.max_episode_steps=480 \
     env.eval.max_steps_per_rollout_epoch=480 \
     env.eval.video_cfg.save_video=false

``run_eval.sh`` supplies the standard LIBERO/EGL defaults. The HTTP YAML
selects ``worker_http`` and requests 16-step action chunks. These overrides
reduce its larger workload to eight environments and one rollout epoch,
while retaining 360x360 camera inputs. Weights and tokenizer paths belong to
the model server YAML; this client command selects the Gateway address.

Inspect ``success_once``, ``success_at_end``, and ``num_trajectories`` in the
final evaluation metrics. A previous local A40 run of this HTTP chain
completed eight trajectories with ``success_once=0.875`` and
``success_at_end=0.75``. This is historical runtime evidence, not a new
clean-install validation or an A800 CI result for these changes.

Stop the Gateway and model server with Ctrl+C in their respective terminals
after evaluation. The `PhyAI serving guide
<https://phyai.mintlify.app/deployment/server-gateway>`_ describes backend
management and multi-host deployment.

Run the HTTP Smoke Test
~~~~~~~~~~~~~~~~~~~~~~~

Use the repository's Shell runner to start and stop the complete chain with
the same evaluation YAML. Prepare the client and serving environments above,
then pass compatible model resources and available GPU indices:

.. code-block:: bash

   cd /path/to/RLinf
   export PHYAI_PYTHON="$PWD/.venv-phyai-serving/bin/python"
   export RLINF_VENV="$PWD/.venv-phyai-http"
   export PHYAI_HTTP_CHECKPOINT=/path/to/lerobot-pi05-libero
   export PHYAI_HTTP_TOKENIZER=/path/to/paligemma-tokenizer
   SERVER_GPU_ID=0 GPU_ID=1 NUM_ENVS=8 MIN_SUCCESS_ONCE=0.5 \
     bash tests/e2e_tests/evaluations/run_phyai_gateway.sh

The runner uses ``tests/e2e_tests/evaluations/phyai_gateway_server.yaml`` by
default. It resolves the two resource variables into a per-run configuration,
checks that ports are available, waits for model and Gateway readiness, and
runs eight environments for 480 steps with video disabled. The model and
rendering GPU can be the same if memory permits. Pass another server YAML as
the first argument when its batch, camera, and action contract matches the
client.

The script creates a private Ray instance and connects only its evaluation
to that address. On exit, it terminates the process groups it started; it does
not stop pre-existing services or other Ray clusters. Logs, the resolved
server YAML, and a successful ``result.json`` are retained under
``logs/ci/phyai-gateway.<suffix>/``.

``MIN_SUCCESS_ONCE`` defaults to zero for a chain-completion smoke test; the
example above additionally requires at least 50 percent success. Choose the
quality threshold with the CI maintainers for the fixed checkpoint and reset
states. This updated runner and its new regression tests have not been
executed as part of these edits; the earlier A40 result predates them.

Enable the HTTP CI Job
~~~~~~~~~~~~~~~~~~~~~~

The ``embodied-libero-phyai-http-test`` job in
``.github/workflows/embodied-e2e-tests.yml`` uses the same ``embodied`` runner,
shared ``--engine phyai`` installation, and component-test-then-e2e structure
as the in-process PhyAI job. A second environment uses the existing SGLang
installer for the official HTTP client's codec. Only the Gateway/processor
packages and HTTP dependencies are added to the shared PhyAI runtime recipe.
Model inference uses GPU 0 and rendering uses GPU 1; the test runs eight
environments for 480 steps with video off.

Before enabling this job, stage the compatible LeRobot checkpoint and full
PaliGemma tokenizer on every eligible runner. Configure these GitHub
repository variables with their absolute paths on the runner:

.. list-table:: HTTP CI Resources
   :header-rows: 1

   * - Repository variable
     - Required value
   * - ``PHYAI_HTTP_CHECKPOINT``
     - Directory containing the compatible PI0.5 checkpoint and processor files.
   * - ``PHYAI_HTTP_TOKENIZER``
     - Directory containing the complete tokenizer snapshot.

The job is skipped if either variable is unset; a skipped job is not HTTP
coverage. With both configured, it validates the directories, runs
``test_models.py -k phyai_http`` and calls the Shell runner with the same
evaluation YAML used above. Its temporary Python
environments are unique to the job and removed afterwards. The Shell runner
owns and closes its service processes and private Ray instance; it does not
stop other Ray clusters. Evaluation logs are uploaded even when the job fails.

The default CI success threshold is zero: completed trajectories, valid
metrics, and a zero process exit code establish chain-completion coverage.
Set a positive ``MIN_SUCCESS_ONCE`` in the job only after agreeing on a stable
checkpoint and quality threshold. The supplemental serving installation and this new workflow have not yet
run on the upstream A800 runner. Maintainers must
approve and enable the PR's CI runs under the repository's normal policy.
