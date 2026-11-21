PolaRiS Evaluation
==================

PolaRiS is a tabletop manipulation simulation platform with DROID-style tasks such as TapeIntoContainer and MoveLatteCup. RLinf supports evaluating OpenPI policies on PolaRiS.

Related training doc: :doc:`../../examples/embodied/polaris`

Setup and Run
-------------

Install Isaac Sim 5.1.0 and source its ``setup_conda_env.sh`` first (see :doc:`../../examples/embodied/polaris`). In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``.

.. tab-set::

   .. tab-item:: OpenPI π₀.₅ (TapeIntoContainer)

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

         # 2. Download scenes and model
         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos --local-dir ./RLinf-Pi05-Polaris-droid_jointpos

         # 3. Launch
         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_tapeintocontainer_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-Polaris-droid_jointpos

   .. tab-item:: OpenPI π₀ (MoveLatteCup)

      .. code-block:: bash

         # 1. Install
         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env polaris && source .venv/bin/activate

         # 2. Download scenes and model
         hf download owhan/PolaRiS-Hub --repo-type dataset --local-dir ./PolaRiS-Hub
         hf download RLinf/RLinf-Pi0-Polaris-droid_jointpos --local-dir ./RLinf-Pi0-Polaris-droid_jointpos

         # 3. Launch
         export POLARIS_DATA_PATH="$(pwd)/PolaRiS-Hub"
         bash evaluations/run_eval.sh polaris polaris_movelattecup_openpi_eval \
           rollout.model.model_path=./RLinf-Pi0-Polaris-droid_jointpos

The terminal prints ``eval/success_once``; see :doc:`../reference/results` for logs.

Example Configs
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config file
     - Task
     - Model
   * - ``polaris_tapeintocontainer_openpi_pi05_eval.yaml``
     - TapeIntoContainer
     - π₀.₅
   * - ``polaris_movelattecup_openpi_eval.yaml``
     - MoveLatteCup
     - π₀

FAQ
---

- **Dataset path:** ``POLARIS_DATA_PATH`` must point to the PolaRiS-Hub root; ``run_eval.sh`` reads it automatically.
- **Model conversion:** JAX checkpoints must be converted to PyTorch format per the training doc before evaluation.
