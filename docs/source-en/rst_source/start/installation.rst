Installation
============

Install RLinf into a local UV virtual environment with ``requirements/install.sh``, or pull a prebuilt image from Docker Hub.

Install Locally
---------------

Clone the repository, install one target, and activate the virtual environment:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git && cd RLinf
   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate

Pick the command that matches your workload:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Workload
     - Install command
   * - OpenVLA + ManiSkill / LIBERO
     - ``bash requirements/install.sh embodied --model openvla --env maniskill_libero``
   * - OpenVLA-OFT + ManiSkill / LIBERO
     - ``bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero``
   * - OpenPI + LIBERO
     - ``bash requirements/install.sh embodied --model openpi --env libero``
   * - Agentic / Reasoning
     - ``bash requirements/install.sh agentic``
   * - Documentation
     - ``bash requirements/install.sh docs``

Run ``bash requirements/install.sh --help`` for all models, environments, and flags. Common options:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Flag
     - Purpose
   * - ``--use-mirror``
     - Use mainland China mirrors for faster downloads.
   * - ``--venv <dir>``
     - Choose the virtual environment directory (default ``.venv``).
   * - ``--platform <name>``
     - Target non-NVIDIA accelerators (``amd``, ``ascend``, ``musa``; see :ref:`the hardware matrix <model-hardware-support>`).
   * - ``--python <version>``
     - Override the Python version when needed (for example ``3.10`` for ``behavior`` or ``d4rl``).

Prebuilt Docker Images
----------------------

.. grid:: 1 1 1 1
   :gutter: 2

   .. grid-item-card:: Docker Hub — rlinf/rlinf
      :link: https://hub.docker.com/r/rlinf/rlinf

      Browse prebuilt images, tag lists, container launch commands, and regional mirrors on Docker Hub.

Verify
------

Confirm that RLinf and Ray are available in the active environment:

.. code-block:: bash

   python -c "import rlinf; print(rlinf.__file__)" && ray --version

Next Steps
----------

- :doc:`Run the VLA Quick Start <vla>`
- :doc:`Scale beyond one machine <../guides/launch-scale/index>`
- :doc:`Open the Cheat Sheet <cheat_sheet>`
