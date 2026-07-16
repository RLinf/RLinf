Installation
============

Install RLinf in one of two ways: **Option 1 (UV)** builds a local virtual
environment matched to your machine, while **Option 2 (Docker)** gives the most
reproducible setup.

Option 1: UV
------------

Build a UV virtual environment with ``install.sh``. Pass an install target
(``embodied``, ``agentic``, or ``docs``) plus any target-specific flags:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

More targets and combinations:

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   bash requirements/install.sh embodied --model openpi --env libero
   bash requirements/install.sh agentic
   bash requirements/install.sh docs

Run ``bash requirements/install.sh --help`` for the complete model and
environment list.

- Use ``--venv <dir>`` to choose the virtual environment directory.
- Use ``--use-mirror`` for faster downloads from mainland China.
- Use ``--python <version>`` only when a package requires it. The default is
  Python 3.11.14; some environments such as ``behavior`` and ``d4rl`` require
  Python 3.10.
- Use ``--torch <version>`` only when you need a different PyTorch wheel.
- Use ``--platform amd`` or ``--platform ascend`` for experimental non-NVIDIA
  installs. See :doc:`../guides/amd_rocm` and :doc:`../guides/ascend_cann`.

Option 2: Docker
----------------

Each image bundles a ready-to-run stack (for example,
``agentic-rlinf0.3-maniskill_libero``). Pull and run it, then select the model
environment inside the container:

.. code-block:: bash

   docker pull rlinf/rlinf:agentic-rlinf0.3-maniskill_libero
   docker run -it --gpus all \
      --shm-size 100g \
      --net=host \
      --name rlinf \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      rlinf/rlinf:agentic-rlinf0.3-maniskill_libero /bin/bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   source switch_env openvla

- Keep ``-e NVIDIA_DRIVER_CAPABILITIES=all`` for GPU rendering.
- Do not mount over ``/root`` or ``/opt``; those directories contain assets and
  virtual environments in the image.
- If your platform changes ``$HOME`` or remounts ``/root``, run ``link_assets``
  inside the container before launching an example.
- Switch model environments with ``source switch_env openvla``,
  ``source switch_env openvla-oft``, or ``source switch_env openpi``.

Verify
------

After activation, verify that RLinf and Ray are visible in the environment:

.. code-block:: bash

   python -c "import rlinf; print(rlinf.__file__)"
   ray --version

Next Steps
----------

- :doc:`Run the VLA quickstart <vla>`.
- :doc:`Scale beyond one machine <../guides/launch-scale/index>`.
- :doc:`Open the cheat sheet <cheat_sheet>` when you only need commands.
