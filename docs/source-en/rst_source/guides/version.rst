SGLang & vLLM Versions
======================

RLinf can plug different *generation backends* into its
reinforcement-learning pipeline. Both **SGLang and vLLM** are supported;

.. note::

   RLinf is compatible with **SGLang 0.4.4 -> 0.5.13** (and **SGLang 0.5.19** for embodied ``--model sglang``) and **vLLM 0.8.5 ->
   0.23.x**. No manual patching is required - the framework detects the
   installed version and loads the matching shim automatically.

Supported engine builds
-----------------------

Each engine build RLinf can install is described by one file under
``requirements/agentic/``, named ``<engine>_<version>_<cu12|cu13>.txt``.  The
set of files *is* the set of supported versions, so listing the directory
answers "what can I install?".

============  ==============  =========  =======
Engine        Version         CUDA line  torch
============  ==============  =========  =======
SGLang        0.5.12.post1    cu12,cu13  2.11.0
SGLang        0.5.4           cu12       2.8.0
SGLang        0.5.2           cu12       2.8.0
SGLang        0.4.6.post5     cu12       2.6.0
vLLM          0.23.0          cu12,cu13  2.11.0
vLLM          0.8.5           cu12       2.6.0
============  ==============  =========  =======

The default is the highest version listed above -- SGLang 0.5.12.post1 and vLLM
0.23.0 today, both on torch 2.11 -- derived from the files themselves, so adding
a newer one moves the default with it.  The CUDA line follows the torch wheel
rather than the driver, so a torch version with no cu13 build stays on cu12 even
on a CUDA 13 host.

One venv per engine
-------------------

A venv holds exactly one engine.  SGLang and vLLM pin the same kernel libraries
(``nvidia-cutlass-dsl``, ``flashinfer-python``, ``tilelang``,
``tokenspeed-mla``) to different versions, and sometimes a different torch, so
sharing a venv means whichever is installed second downgrades the other's
kernels -- silently, until a kernel actually runs.  ``--engine`` selects the
engine; install twice with different ``--venv`` to get both:

.. code-block:: bash

   bash requirements/install.sh agentic --engine sglang
   bash requirements/install.sh agentic --venv .venv-vllm --engine vllm

The reason Docker image ships both, in ``reason`` (SGLang, activated by default)
and ``reason-vllm``.

Switching versions
------------------

Pass the version to the installer; torch follows from it, so it need not be
given as well:

.. code-block:: bash

   # default: SGLang 0.5.12.post1 on torch 2.11
   bash requirements/install.sh agentic

   # the SGLang 0.4.x line, on torch 2.6
   bash requirements/install.sh agentic --sglang 0.4.6.post5

   # vLLM 0.8.5, on torch 2.6
   bash requirements/install.sh agentic --engine vllm --vllm 0.8.5

An unsupported version fails immediately and prints the builds that do exist,
rather than resolving into a broken environment.

.. note::

   Avoid ``pip install sglang`` / ``pip install vllm`` into an existing
   environment. Both pin the whole torch family, and their CUDA 13 releases
   pin CUDA 13 runtime wheels that overwrite their CUDA 12 counterparts in
   place -- the requirements files exist precisely to keep that from happening.

Rollout engine configuration
----------------------------

To select ``rollout.rollout_backend`` (``sglang`` or ``vllm``) and configure backend-specific options, see :doc:`agentic_config`.
