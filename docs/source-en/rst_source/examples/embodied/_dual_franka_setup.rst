.. This file is a reusable include, not a standalone page.
   Shared by Dual-Arm Franka recipes (dual_franka.rst and dual_franka_openpi_pytorch.rst).

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - Observation
     - Base RealSense view, left/right Lumos wrist views, and dual-arm robot state.
   * - Action
     - 20-D dual-arm ``tcp_rot6d`` target: ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``.
   * - Reward
     - Operator-gated episode labels from the USB foot pedal (``a`` / ``b`` / ``c``).
   * - Prompt
     - Natural-language ``TASK_DESCRIPTION`` stored in dataset and OpenPI metadata.

Hardware and Safety
-------------------

Connect the left Franka arm, all three cameras, both GELLO leaders, and the PCsensor FootSwitch to ``node 0``. Connect the right Franka arm to ``node 1``. Keep both GELLO leaders physically plugged into ``node 0`` so their 1 kHz serial stream stays local instead of crossing the LAN.

.. warning::

   Keep an emergency stop within reach and clear the shared dual-arm workspace before enabling the arms. Verify both arms sit in a safe starting pose and confirm ``target_ee_pose`` stays inside the ``realworld_dual_franka_tcp_rot6d`` workspace box: ``x`` in ``[0.3, 0.9]``, ``y`` in ``[-0.4, 0.4]``, and ``z`` in ``[0.02, 0.7]``. Never route the 1 kHz GELLO stream over the network.

Configure the PCsensor FootSwitch keys once with the vendor tool as ``a`` / ``b`` / ``c``, and gather your per-station identifiers before editing configs: left/right FCI IPs, RealSense serial and Lumos ``/dev/v4l/by-id`` paths, Robotiq ``/dev/serial/by-id`` paths, and GELLO FTDI ``/dev/serial/by-id`` paths.

.. dropdown:: Real-time kernel prerequisites and tuning (recommended)

   The ``franka`` environment communicates with each arm at 1 kHz through Franky and libfranka. Configure a ``PREEMPT_RT`` kernel and real-time user limits according to the `Franka real-time kernel guide <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_.

   Dual-arm hardware configs default to ``realtime_config: ignore``, so both arms also start on a standard kernel with a warning; under load, missed 1 kHz deadlines can trigger a robot reflex. Set ``realtime_config: enforce`` to require ``PREEMPT_RT``.

   Run the tuning and link checks below on each robot node before starting Ray (replace ``eth0`` with the dedicated robot NIC and ``172.16.0.2`` with ``LEFT_ROBOT_IP`` on ``node 0`` or ``RIGHT_ROBOT_IP`` on ``node 1``):

   .. code-block:: bash

      # Per-boot tuning (re-apply after every reboot).
      sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
          echo performance > "$g"
      done'
      sudo sysctl -w kernel.sched_rt_runtime_us=-1
      sudo ethtool -C eth0 rx-usecs 0 tx-usecs 0 2>/dev/null || true  # replace with your Franka NIC

      # Optional: persist the RT scheduling budget across reboots.
      echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

      # Verify real-time permissions and robot link latency.
      uname -a | grep -o PREEMPT_RT
      ulimit -r
      ulimit -l
      sudo cyclictest -p 80 -t 4 -i 1000 -l 300000 -m
      ping -c 1000 -i 0.001 172.16.0.2 | tail -3  # replace with your robot IP

   Confirm ``ulimit -r`` reports ``99`` or ``unlimited`` and ``ulimit -l`` reports ``unlimited``.

.. dropdown:: GELLO calibration (first-time setup)

   From the RLinf repository root with ``.venv`` active, calibrate each GELLO leader once against the left arm on ``node 0`` and verify alignment with ``align-sequential``. Repeat both commands for the second leader by updating ``GELLO_PORT``:

   .. code-block:: bash

      export GELLO_PORT=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0  # replace with your FTDI device

      python -m toolkits.realworld_check.test_gello calibrate
      python -m toolkits.realworld_check.test_gello align-sequential

   Expected output on success:

   .. code-block:: text

      ALL JOINTS ALIGNED
        per-joint Δ (rad): ['+0.012', '-0.008', '+0.005', '+0.021', '-0.041', '+0.009', '-0.003']
        max |Δ| = 0.041 rad on J5 (stream gate threshold = 0.5 rad — well under)
      You can now Ctrl-C and start collect_data.sh.
