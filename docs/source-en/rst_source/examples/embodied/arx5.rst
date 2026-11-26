ARX X5 Data Collection
======================

.. figure:: https://arx-global.us/assets/img/x5.webp
   :align: center
   :width: 80%
   :alt: ARX X5 robot arms

   ARX X5 robot arms. Image: ARX.

Collect dual-arm ARX X5 demonstrations with two PICO controllers and three V4L2 RGB cameras on a single Ubuntu 24.04 host, then merge and visualize the exported LeRobot dataset.

Overview
--------

This recipe runs dual-arm ARX X5 control, three V4L2 RGB cameras, and PICO teleoperation on one Ubuntu 24.04 machine, exporting successful episodes directly as LeRobot shards.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      2× ARX X5 (``ARX5_beta`` SDK)

   .. grid-item-card:: Teleop & Cameras
      :text-align: center

      PICO dual controllers · 3× V4L2 RGB cameras

   .. grid-item-card:: Workflow
      :text-align: center

      PICO Collect → Merge LeRobot → Visualize

   .. grid-item-card:: Hardware
      :text-align: center

      1× Ubuntu 24.04 host · 2× CAN adapters

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - One 224×224 RGB main view, two 224×224 RGB wrist views, an 18-D dual-arm TCP pose in ``[xyz, rot6d]`` order, and two normalized gripper positions.
   * - Action
     - 20-D absolute target in ``[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]`` order.
   * - Reward
     - PICO ``X`` ends and saves the episode with reward ``1``; PICO ``A`` ends and discards it with reward ``0``.
   * - Prompt
     - ``env.eval.override_cfg.task_description`` written into LeRobot episode metadata.

Hardware and Safety
-------------------

Connect both ARX X5 arms through dedicated CAN adapters, plug in the overhead and two wrist V4L2 RGB cameras, and pair the PICO headset with both controllers following :ref:`PICO / XRoboToolkit setup <pico-software-setup>` so the APK tracks both hands.

.. warning::

   Keep an emergency stop within reach and clear the dual-arm workspace before enabling either arm. Verify both arms sit in a safe starting pose, check cable slack across full reach, and confirm the reset path is unobstructed before launching collection.

Follow the `ARX5_beta instructions <https://github.com/ARXroboticsX/ARX5_beta/blob/main/readme.md>`_ to run ``ARX_CAN_doctor``, map the left and right arms to ``can1`` and ``can3``, and bring both CAN interfaces up before launching RLinf.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: 1. Install & Hardware Check

      **Install RLinf and the ARX5_beta SDK:** install the ``arx5`` runtime (dependencies in ``requirements/embodied/envs/arx5.txt``), clone `ARXroboticsX/ARX5_beta <https://github.com/ARXroboticsX/ARX5_beta>`_ (whose root provides the ``bimanual`` package), and add it to ``PYTHONPATH`` in every session:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env arx5 && source .venv/bin/activate
         export PYTHONPATH="/path/to/ARX5_beta${PYTHONPATH:+:$PYTHONPATH}"

      **Identify stable V4L2 camera paths** and update ``examples/embodiment/config/collect_data_dual_arx5.yaml`` with your CAN interfaces, camera paths, and any needed clockwise ``camera_rotations`` (``0`` / ``90`` / ``180`` / ``270``):

      .. code-block:: bash

         ls -l /dev/v4l/by-path/

      .. code-block:: yaml

         left_can_port: can1  # Left arm CAN interface
         right_can_port: can3  # Right arm CAN interface
         arm_type: 0  # X5-2023 model
         camera_fps: 30  # Frame rate supported by all three cameras
         base_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.4:1.0-video-index0']
         left_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0']
         right_camera_serials: ['/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.3:1.0-video-index0']
         camera_rotations: {base_0_rgb: 180}

   .. tab-item:: 2. Collect & Merge Dataset

      **Set episode count, rate, and prompt** in ``examples/embodiment/config/collect_data_dual_arx5.yaml``:

      .. code-block:: yaml

         runner:
           num_data_episodes: 50
         env:
           eval:
             override_cfg:
               step_frequency: 30
               task_description: Pick and place the objects with both arms.
             data_collection:
               fps: 30

      **Collect demonstrations** from the repository root (with ``.venv`` active and ``ARX5_beta`` on ``PYTHONPATH``):

      .. code-block:: bash

         bash examples/embodiment/collect_data.sh collect_data_dual_arx5

      **PICO button controls:** after both arms reset, press **B** to start recording, **X** to end and save a successful episode, or **A** to end and discard a failed attempt.

      **Merge shards and export MP4 previews** from the latest run (raw shards live under ``logs/<timestamp>-collect_data_dual_arx5/collected_data/lerobot_dataset/rank_0/id_0/``):

      .. code-block:: bash

         RUN=$(ls -td logs/*-collect_data_dual_arx5 | head -n 1)

         python -m toolkits.lerobot.merge_lerobot_datasets \
             --source-dir "$RUN/collected_data" \
             --output-dir "$RUN/lerobot_merged"

         python -m toolkits.lerobot.visualize_lerobot_dataset \
             --dataset-path "$RUN/lerobot_merged" \
             --output-dir "$RUN/formatted_samples" \
             --export-mp4 --mp4-fps 30

Results and Troubleshooting
---------------------------

Merged datasets write to ``$RUN/lerobot_merged`` and per-frame images, metadata, and MP4 previews write to ``$RUN/formatted_samples``. Inspect the exported MP4 videos before training to verify camera orientation and task quality.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symptom
     - Fix
   * - OpenCV reports ``Could not open V4L2 camera``
     - Check ``ls -l /dev/video*`` and ``id -nG``, run ``sudo usermod -aG video "$USER"``, log out and back in, and open a new shell so workers inherit the ``video`` group.
   * - ``ModuleNotFoundError: No module named 'bimanual'``
     - Export ``PYTHONPATH="/path/to/ARX5_beta${PYTHONPATH:+:$PYTHONPATH}"`` before running ``collect_data.sh``.
   * - Camera view appears rotated or upside down
     - Set clockwise ``camera_rotations`` (``0`` / ``90`` / ``180`` / ``270``) for that view in ``collect_data_dual_arx5.yaml``.
   * - CAN interface fails to open or arms stay unresponsive
     - Re-run ``ARX_CAN_doctor`` from ``ARX5_beta`` and verify ``can1`` and ``can3`` are up and match ``left_can_port`` / ``right_can_port``.

Next Steps: Train OpenPI SFT
----------------------------

Once the exported MP4 previews look clean, train an OpenPI policy on ``$RUN/lerobot_merged`` by following the **Custom LeRobot Dataset** tab in :doc:`sft_openpi`:

1. Adapt ``pi0_custom`` (or register an ARX X5 dataconfig) for the three RGB views and the 20-D ``[xyz, rot6d, gripper]`` dual-arm state and action space.
2. Compute ``norm_stats.json`` with ``toolkits/lerobot/calculate_norm_stats.py`` using that dataconfig name and ``$RUN/lerobot_merged``.
3. Launch ``examples/sft/run_vla_sft.sh custom_sft_openpi`` with ``data.train_data_paths``, ``actor.model.model_path``, and ``actor.model.openpi_data.norm_stats_path`` overrides as shown in :doc:`sft_openpi`.
