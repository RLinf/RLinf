Moore Threads MUSA
==================

Run RLinf embodied training on Moore Threads (MTT) GPUs. This page covers only
what differs from the NVIDIA workflow: how the environment is built, how
rendering and physics are configured, and what is not supported yet. Task
descriptions, algorithms, model downloads, and metrics are platform independent
— see :doc:`RL with LIBERO Benchmarks <../examples/embodied/libero>` and
:doc:`RL with ManiSkill <../examples/embodied/maniskill>`.

RLinf detects MTT GPUs automatically: the scheduler reports them as
``MUSA_GPU`` accelerators and assigns ``MUSA_VISIBLE_DEVICES`` per worker, so
placement and multi-GPU configs work unchanged.

Installation
------------

MUSA differs from every other platform in one important way: **RLinf does not
install PyTorch on MUSA.** ``torch-musa`` and its matching MUSA build of torch
are not published on PyPI or on PyTorch's wheel indexes — they ship inside the
Moore Threads training-suite image. ``install.sh --platform musa`` therefore:

- reuses the image's interpreter instead of downloading a uv-managed Python,
- creates the venv with ``--system-site-packages`` so the image's
  ``torch``/``torch_musa`` remain importable,
- excludes the whole torch family (plus the CUDA-only ``nvidia-*`` runtime
  wheels) from dependency resolution,
- skips flash-attention and apex (the image ships MUSA builds of both) and the
  vLLM/SGLang CUDA kernels.

Because of this, the install must run **inside** a Moore Threads container.

Option 1: Docker image
~~~~~~~~~~~~~~~~~~~~~~

Build from the RLinf Dockerfile with ``PLATFORM=musa``. The base image is
``registry.mthreads.com/mcctest/ai/training-suite:${MUSA_VER}``; the build
targets themselves need no MUSA-specific handling:

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=musa \
      --build-arg MUSA_VER=v2.1.5-musa4.3.7 \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero .

BuildKit is required: the legacy builder resolves every ``FROM`` in the
Dockerfile, including the CUDA and ROCm bases on Docker Hub that a MUSA host
often cannot reach.

.. important::

   The mthreads container runtime injects the MUSA driver libraries at run
   time, exactly as the NVIDIA runtime does for ``libcuda``. They do not exist
   during an image build, so ``import torch`` fails there with
   ``ImportError: libmusa.so.1``. Nothing in a MUSA image build may import
   torch. The ManiSkill assets are downloaded through ``mani_skill``, and
   therefore torch, so they skip themselves during the build. Fetch them once
   inside the running container instead:

   .. code-block:: bash

      source switch_env openpi && download_assets --assets maniskill

Run it with the ``mthreads`` container runtime, which is what exposes the MTT
GPUs to the container:

.. code-block:: bash

   docker run -it --rm \
      --runtime=mthreads \
      --ipc=host \
      --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v .:/workspace/RLinf \
      rlinf:embodied-maniskill_libero bash

Verify the driver is visible before training:

.. code-block:: bash

   mthreads-gmi
   python -c "import torch, torch_musa; print(torch.musa.device_count())"

Option 2: Install inside an existing MUSA container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start a container from the training-suite image as above, then:

.. code-block:: bash

   bash requirements/install.sh --platform musa embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

For faster downloads in mainland China, add ``--use-mirror``.

``install.sh`` fails early if ``torch_musa`` is not importable, and warns if it
reports no device — at runtime that almost always means the container was
started without ``--runtime=mthreads``.

Rendering: OSMesa instead of EGL
--------------------------------

The Moore Threads driver stack exposes no EGL device, so simulators must render
in software through OSMesa. ``examples/embodiment/run_embodiment.sh`` and the
e2e runner respect these variables, and the e2e runner also takes the backend as
its second argument; set them explicitly if you launch Python directly:

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

Software rendering is considerably slower than EGL, so expect longer rollout
times than on an NVIDIA host with the same batch size.

.. note::

   ``training-suite:v2.1.5-musa4.3.7`` ships ``libGL`` but no ``libOSMesa``.
   ``requirements/sys_deps.sh`` installs it (``libosmesa6``), exactly as it does
   on Ascend and ROCm, so a ``PLATFORM=musa`` build ends up with a working
   software renderer. If you assemble an image by hand instead, check it with
   ``ldconfig -p | grep -i osmesa`` before running a simulator.

ManiSkill: CPU physics backend
------------------------------

ManiSkill's GPU physics backend (SAPIEN PhysX) requires CUDA, which MUSA does
not provide. ManiSkill configs must select the CPU backend:

.. code-block:: yaml

   env:
     train:
       init_params:
         sim_backend: "cpu"
     eval:
       init_params:
         sim_backend: "cpu"

The CPU backend cannot vectorise multiple environments inside one process, so
each env rank drives exactly one environment — set ``total_num_envs`` equal to
the env world size. LIBERO has no such restriction; it vectorises on CPU
already and runs unchanged.

.. warning::

   ManiSkill additionally needs a MUSA-adapted SAPIEN, which is not published
   on PyPI. Stock ``sapien`` cannot create a render system on a ``cpu`` device
   and fails with ``Failed to find a supported physical device "cpu"``, and the
   vendor's SAPIEN in turn requires a patched ManiSkill (the public
   ``v3.0.0b22`` tag imports ``sapien.sensor.StereoDepthSensor``, which that
   build does not provide). Both ship in the vendor RLinf images
   (``registry.mthreads.com/lgpublic/rlinf:*``), so ManiSkill on MUSA has to run
   there for now. ``install.sh --platform musa`` produces a working LIBERO
   environment either way.

Launch a run
------------

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

The ``*_musa`` end-to-end configs are the smallest working setups for both
benchmarks and are a good starting point. They take the renderer as the second
argument, the same way the ROCm and Ascend jobs do:

.. code-block:: bash

   export REPO_PATH=$(pwd)
   ROBOT_PLATFORM=LIBERO bash tests/e2e_tests/embodied/run.sh libero_10_ppo_openpi_pi05_musa osmesa
   ROBOT_PLATFORM=BRIDGE bash tests/e2e_tests/embodied/run.sh maniskill_async_ppo_openpi_pi05_musa osmesa

Not supported yet
-----------------

- **transformers' ``flash_attention_2`` path** — the image ships a working
  MUSA flash-attn (``flash_attn`` 2.6.3, kernels run on ``musa``), but it does
  not re-export ``flash_attn_func`` at the top level, so
  ``transformers.utils.is_flash_attn_2_available()`` returns ``False``. Models
  defaulting to ``attn_implementation: flash_attention_2`` must be switched to
  ``sdpa``; call the kernels through ``flash_attn.flash_attn_interface`` to use
  them directly.
- **vLLM / SGLang rollout** — the RLinf agentic target expects the CUDA builds
  of these engines. Embodied training uses the ``huggingface`` rollout backend
  and is unaffected.
- **Vectorised ManiSkill GPU simulation** — see above.

What stays the same
-------------------

- The same configs, models, and ``model_path`` flow as on NVIDIA.
- The same PPO/GRPO algorithm settings and placement concepts.
- The same training, rollout, and environment metrics.
