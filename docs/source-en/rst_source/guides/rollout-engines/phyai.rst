PhyAI Embodied Rollout Backend
==============================

Use `PhyAI <https://github.com/mingti-org/phyai>`_ to train a PI0.5 policy
in process or evaluate it through an external Gateway. PhyAI is an inference
framework for physical-AI models such as VLAs. The in-process path puts one
engine in each RLinf rollout worker and keeps the LIBERO environments, FSDP
actor, observation transforms, and weight synchronization in RLinf.

The sections below install and run the in-process PPO backend. For separate
model servers, follow :ref:`phyai-http-evaluation-en`. That evaluation-only
path runs RLinf environment workers and an HTTP rollout worker, and keeps
the Gateway and model server in separate processes.

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
``serving_mode: gateway`` selects this evaluation-only path; the PPO
instructions above use the in-process backend.

.. code-block:: text

   RLinf EnvWorker -> HTTP rollout worker
       -> HTTP MessagePack -> PhyAI Gateway
       -> gRPC -> PI0.5 model server

Install the Evaluation Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create one dedicated environment with the existing ``--engine phyai``
installer. It provides the RLinf evaluation client, the PhyAI model server,
and the Gateway, so a single installation serves both sides of the HTTP
chain:

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh \
     --python 3.12 \
     --venv .venv-phyai \
     --install-rlinf \
     embodied --model openpi --env libero --engine phyai
   source .venv-phyai/bin/activate

The installer selects the CUDA 12/13 requirements, pins PhyAI to
``0.2.0rc1``, adds the Gateway, its processor helpers, and the HTTP
dependencies, and then checks that the additions preserved RLinf's Torch and
Transformers versions. The HTTP client encodes MessagePack with
``rlinf.utils.msgpack_codec``, which matches the Gateway wire format and does
not require SGLang.

The model server and Gateway run as separate processes, so they can also run
on another host. Install the serving side with the same command there and
give the evaluation host an environment built without ``--engine phyai``.
Keep the environment's ``bin`` directory on ``PATH`` for ``ninja`` and other
kernel-build tools; activation and the Shell e2e runner do this.

.. warning::

   Use a new environment for these commands. The installer synchronizes its
   selected dependencies; do not point it at a mixed experimental environment
   whose additional packages you need to preserve.

Download the checkpoint and tokenizer:

.. code-block:: bash

   export HF_ENDPOINT=https://hf-mirror.com
   hf download lerobot/pi05_libero_finetuned_v044 \
     --revision 7152f13e34c6c115d808aea0e5d08aca74eac14b \
     --local-dir /path/to/lerobot-pi05-libero
   hf download google/paligemma-3b-pt-224 \
     --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
     --local-dir /path/to/paligemma-3b-pt-224

``HF_ENDPOINT`` selects a mirror for faster downloads; unset it to reach
``huggingface.co`` directly. The checkpoint directory carries the
preprocessing, normalization, and action postprocessing files
``PI05Backend`` reads, and describes an eight-value state, seven-value
actions, and a 50-step action chunk. The in-process RLinf/OpenPI checkpoint
shown earlier uses a different layout and cannot be substituted directly.

Create ``pi05-http.yaml`` in the RLinf repository root with these settings,
using resource paths on the model host:

.. code-block:: yaml

   plugin: pi05
   plugin_args:
     checkpoint_dir: /path/to/lerobot-pi05-libero  # https://huggingface.co/lerobot/pi05_libero_finetuned_v044
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
       tokenizer_dir: /path/to/paligemma-3b-pt-224  # https://huggingface.co/google/paligemma-3b-pt-224
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
   source .venv-phyai/bin/activate
   export GPU_ID=0
   CUDA_VISIBLE_DEVICES="$GPU_ID" python -m phyai server pi05-http.yaml

Wait for ``Serving model pi05 on 127.0.0.1:50063``, which follows model
initialization and warmup. Keep it running. In a second serving terminal,
activate the same environment and start the Gateway:

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai/bin/activate
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
   source .venv-phyai/bin/activate
   export GPU_ID=0
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     "~cluster.component_placement" \
     "+cluster.component_placement.env=$GPU_ID" \
     "+cluster.component_placement.rollout=$GPU_ID" \
     rollout.phyai.gateway_url=http://127.0.0.1:30000 \
     env.eval.total_num_envs=8 \
     env.eval.rollout_epoch=1 \
     env.eval.max_episode_steps=480 \
     env.eval.max_steps_per_rollout_epoch=480 \
     env.eval.video_cfg.save_video=false

``run_eval.sh`` supplies the standard LIBERO/EGL defaults. Leave
``CUDA_VISIBLE_DEVICES`` unset for this command: RLinf's placement sets both
the worker's visible GPU and its EGL rendering device from
``cluster.component_placement``. Pre-filtering visible GPUs makes Ray
enumerate a single device at rank 0, so any ``GPU_ID`` other than 0 fails
with a rank assertion. The YAML defaults to ``env,rollout: all`` so it runs
on any host; the ``component_placement`` overrides above replace that
default and pin this evaluation to the single ``GPU_ID``. Because the YAML
defines a composite ``env,rollout`` key, the per-component overrides first
remove it with ``~`` and then add ``env`` and ``rollout`` separately.

The HTTP YAML sets ``serving_mode: gateway`` and requests 16-step action
chunks. ``rollout.phyai.model_name`` must match the model name the server
registers and the Gateway routes. These overrides reduce its larger
workload to eight environments and one rollout epoch, while retaining
360x360 camera inputs. Weights and tokenizer paths belong to the model
server YAML; this client command selects the Gateway address.

Inspect ``success_once``, ``success_at_end``, and ``num_trajectories`` in the
final evaluation metrics.

Stop the Gateway and model server with Ctrl+C in their respective terminals
after evaluation. The `PhyAI serving guide
<https://phyai.mintlify.app/deployment/server-gateway>`_ describes backend
management and multi-host deployment.
