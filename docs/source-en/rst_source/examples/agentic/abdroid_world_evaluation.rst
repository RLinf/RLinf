=========================================================
Evaluating Android World with the M3A Agent in RLinf
=========================================================

This document describes how to use ``m3a_worker.py`` (M3A Agent Worker) and
``android_reward_worker.py`` (Android Reward Worker) to evaluate Android World
tasks within RLinf, so that other users can set up the environment and
reproduce the experiments.

The example is built on the **M3A** agent from the official Android World
repository and integrates it with RLinf's scheduling, rollout and reward
infrastructure.

----------------
1. Overview
----------------

* ``m3a_worker.py`` (``rlinf/workers/env/m3a_worker.py``):

  Uses the **M3A** agent built into
  `Android World <https://github.com/google-research/android_world>`_,
  runs tasks on a real device or emulator, and communicates with the rollout
  LLM (e.g., Qwen3-VL via SGLang) and the reward worker through Channels.

* ``android_reward_worker.py`` (``rlinf/workers/env/android_reward_worker.py``):

  Reconnects to the Android env in a separate process, computes rewards
  (e.g., task success) based on the agent's behavior and task spec, and sends
  them back to the agent worker.

Together they allow you to **evaluate the M3A agent only** (no training) in a
way similar to the existing ``agent_worker`` + ``reward_worker`` flow, but with
much simpler agent-side logic that is easier to reproduce and debug.

--------------------------
2. Environment and Dependencies
--------------------------

2.1 Basic environment
=====================

* **Python**: 3.10+ recommended.
* **OS**: Linux (recommended, easier for talking to Android devices/emulators).
* **Android device**: At least one ADB-connected device or emulator
  (e.g., ``emulator-5554`` or ``localhost:5557``).

2.2 RLinf dependencies
======================

Install RLinf documentation and Qwen-VL utilities:

.. code-block:: bash

   pip install -r RLinf/docs/requirements.txt
   pip install qwen-vl-utils

2.3 Android World and its dependencies
======================================

Both ``m3a_worker`` and ``android_reward_worker`` depend on **android_world**.
We recommend placing the android_world repository as a sibling of ``RLinf``
and installing its dependencies:

.. code-block:: bash

   # Clone android_world (if you haven't already)
   git clone https://github.com/google-research/android_world.git /path/to/android_world

   # Install dependencies used by android_world
   sudo apt update && sudo apt install ffmpeg
   pip install -r /path/to/android_world/requirements.txt
   pip install uiautomator2

2.4 Path configuration
======================

The **android_world** path is configured via:

* ``data.android_world_parent`` in ``qwen3vl-4b-eval.yaml``, used by
  ``AndroidWorldDataset``.
* The ``PYTHONPATH`` environment variable in the ``eval.sh`` script, which
  should include:

  - The project root (e.g., ``/path/to/your/root_project``),
  - The ``RLinf`` directory
    (e.g., ``/path/to/your/root_project/RLinf``),
  - The ``android_world`` source root
    (e.g., ``/path/to/your/root_project/android_world``),

  so that imports such as ``import android_world`` and
  ``from android_world.agents import m3a`` work in all Ray workers.

We recommend **not** modifying ``sys.path`` in library code. Instead:

1. Add ``data.android_world_parent`` to ``qwen3vl-4b-eval.yaml``:

   .. code-block:: yaml

      data:
        type: android
        task_family: android_world
        # ...
        android_world_parent: /absolute/path/to/android_world

2. In ``eval.sh``, set ``PYTHONPATH`` before running the evaluation script:

   .. code-block:: bash

      PROJECT_ROOT="/path/to/your/root_project"
      RLINF_ROOT="$PROJECT_ROOT/RLinf"
      ANDROID_WORLD_ROOT="$PROJECT_ROOT/android_world"

      export PYTHONPATH="$PROJECT_ROOT:$RLINF_ROOT:$ANDROID_WORLD_ROOT:${PYTHONPATH:-}"

2.5 ADB and devices
===================

* Install Android SDK Platform Tools:

  .. code-block:: bash

     sudo apt update
     sudo apt install android-tools-adb android-tools-fastboot

* After connecting a device or starting an emulator, verify ADB connectivity:

  .. code-block:: bash

     adb devices

* In your config, fill in the correct ``device_id`` (e.g. ``localhost:5557``)
  and ``adb_path`` (e.g. ``adb``).

---------------------------------------
3. Required Changes to Android World
---------------------------------------

To make Android World reproducible on a server, a few patches are needed.

3.1 UI hierarchy via uiautomator2
=================================

In ``android_world/android_world/env/adb_utils.py``, modify
``uiautomator_dump`` to use uiautomator2 when possible and fall back to ADB
on failure:

.. code-block:: python

   def _stop_uiautomator2_agent(env) -> None:
     try:
       issue_generic_request(
           'shell am force-stop com.github.uiautomator',
           env, timeout_sec=5,
       )
     except Exception:
       pass
     try:
       issue_generic_request(
           'shell am force-stop com.github.uiautomator.test',
           env, timeout_sec=5,
       )
     except Exception:
       pass


   def uiautomator_dump(env, timeout_sec: Optional[float] = 30) -> str:
     device_id = None
     if hasattr(env, 'controller') and env.controller is not None:
       device_id = getattr(env.controller, 'device_id', None) or ''
     if not device_id and hasattr(env, 'device_id'):
       device_id = env.device_id or ''

     if device_id:
       try:
         u2_device_id = device_id
         if device_id.startswith("localhost:"):
             port = device_id.split(":", 1)[1]
             port = int(port) - 1
             u2_device_id = f"emulator-{port}"
         device = u2.connect(u2_device_id)
         xml_content = device.dump_hierarchy()
         _stop_uiautomator2_agent(env)
         return xml_content
       except Exception as e:
         logging.warning(
             'Managed uiautomator2 dump failed: %s, stopping agent and '
             'falling back to ADB.', e,
         )
         _stop_uiautomator2_agent(env)
         dump_args = 'shell uiautomator dump /sdcard/window_dump.xml'
         issue_generic_request(dump_args, env, timeout_sec=timeout_sec)

         read_args = 'shell cat /sdcard/window_dump.xml'
         response = issue_generic_request(read_args, env, timeout_sec=timeout_sec)

         return response.generic.output.decode('utf-8')

3.2 Controller and device_id integration
========================================

* In ``android_world/android_world/env/android_world_controller.py``, set
  ``a11y_method`` to ``A11yMethod.UIAUTOMATOR``.

* Replace all occurrences of
  ``representation_utils.forest_to_ui_elements`` with
  ``env.controller.get_ui_elements()``.

* Add a ``device_id`` argument to ``env_launcher._get_env`` and
  ``load_and_setup_env``, and in
  ``android_world_controller.AndroidWorldController``:

  - Add a ``self.device_id`` field.
  - Accept ``device_id`` in ``__init__`` and store it.

* In ``android_world_controller.py``, after loading the underlying
  ``AndroidEnv`` instance, attach ``device_id`` to it:

  .. code-block:: python

     android_env_instance = loader.load(config)

     # Attach device_id directly to the underlying AndroidEnv instance
     # so that env.device_id can be accessed anywhere later.
     if device_id:
       try:
         setattr(android_env_instance, 'device_id', device_id)
       except Exception:
         logging.warning('Failed to attach device_id to AndroidEnv instance.')

----------------------------------------
4. Running the Evaluation End-to-End
----------------------------------------

4.1 Configuration overview
==========================

1. **Cluster configuration**:

   Define placements for ``agent_worker`` and ``reward_worker``, and configure
   the ``android_world`` node group and ADB hardware info.

2. **Rollout configuration**:

   Provide an LLM inference service (e.g., Qwen3-VL via SGLang) for M3A to call.

3. **Data configuration**:

   Use ``AndroidWorldDataset`` with ``data.type: android``,
   ``task_family: android_world``, etc., and set ``data.android_world_parent``.

4. **Reward configuration**:

   Configure ``reward.reward_type: android`` and Android-specific fields
   (``device_id``, ``grpc_port``, ``reward_scale``, etc.).

You can refer to:

* ``rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml``

for a concrete configuration example.

4.2 Execution steps
===================

Currently, we support running evaluation with an emulator started on your local
machine, and using reverse SSH port forwarding to connect from the server.

Step 1: Install and start an emulator locally
---------------------------------------------

On your local machine:

* Install Android Studio and create an emulator:

  - Hardware: Pixel 6
  - System image: Tiramisu, API level 33
  - AVD name: ``AndroidWorldAvd``

* Start the emulator with gRPC enabled:

  .. code-block:: bash

     EMULATOR_NAME=AndroidWorldAvd  # From the previous step
     ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554

Step 2: Reverse SSH port forwarding (local -> server)
-----------------------------------------------------

You need to reverse-forward both the emulator's ADB control port and the gRPC
port used by Android World to the server.  For example, for
``emulator-5554`` and ``grpc_port 8554``:

.. code-block:: bash

   # Check ADB port
   adb devices  # should show emulator-5554

   # Reverse forwarding from the local machine to the server
   ssh -fNR 5555:localhost:5555 <user>@<server-host-or-ip>
   ssh -fNR 8554:localhost:8554 <user>@<server-host-or-ip>

Step 3: Connect to the forwarded ports on the server
----------------------------------------------------

On the server, connect the forwarded ADB port:

.. code-block:: bash

   adb connect localhost:5555

Step 4: Required pre-steps before evaluation
--------------------------------------------

1. Android World needs an initial setup run to install required apps. Use
   ``--perform_emulator_setup``:

   .. code-block:: bash

      cd path/to/android_world
      python run.py \
        --suite_family=android_world \
        --agent_name=t3a_gpt4 \
        --perform_emulator_setup \
        --tasks=ContactsAddContact

2. Manually start the Clipper app once inside the emulator to avoid
   clipboard-related permission issues.

Step 5: Edit config and run evaluation
--------------------------------------

In ``rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml``, adjust:

* ``cluster`` placements and node groups.
* ``data.android_world_parent``, ``data.task_family``, ``data.task_name``, etc.
* ``reward.device_id``, ``reward.grpc_port``, ``reward.adb_path``, etc.

Then run:

.. code-block:: bash

   cd rlinf/example/mobile-agent/
   chmod +x eval.sh
   ./eval.sh

If successful, evaluation results will be written to ``results/eval_results2.json``,
including per-task rewards and summary statistics.

----------------
5. Tests and Verification
----------------

To comply with RLinf's Prime Directive, the Android World integration is
covered by unit tests in:

* ``RLinf/tests/test_android_world_integration.py``

The key aspects tested are:

5.1 AndroidReward behavior
==========================

* ``test_android_reward_returns_zero_when_not_done``:

  Verifies that ``AndroidReward.get_reward_new`` returns ``0.0`` when
  ``result.done`` is ``False``.

* ``test_android_reward_scales_score_when_done``:

  Uses a fake task whose ``is_successful`` returns a fixed score and checks
  that the reward is correctly scaled by ``reward_scale``.

* ``test_android_reward_swallows_task_exception``:

  Uses a fake task whose ``is_successful`` raises an exception and verifies
  that no exception is propagated and ``0.0`` is returned instead, guarding
  against reward computation killing the whole rollout.

5.2 AndroidWorldDataset and registry integration
================================================

* ``test_android_world_dataset_uses_android_world_parent_from_config``:

  Injects a fake ``android_world.registry.TaskRegistry`` into ``sys.modules`` so
  that no real android_world installation is required. It then builds a minimal
  config:

  .. code-block:: yaml

     data:
       max_prompt_length: 128
       task_family: android_world
       n_instances_per_task: 1
       apply_chat_template: False
       android_world_parent: /does/not/matter/for/test

  and a minimal tokenizer stub. The test asserts that
  ``len(AndroidWorldDataset(...)) == 1``, which confirms that:

  * ``data.android_world_parent`` from the config is honored, and
  * ``AndroidWorldDataset._load_data()`` correctly integrates with
    ``TaskRegistry.get_registry(...)``.

5.3 Running the tests
=====================

From the RLinf root directory:

.. code-block:: bash

   cd /path/to/your/root_project/RLinf
   PYTHONPATH=. pytest tests/test_android_world_integration.py

