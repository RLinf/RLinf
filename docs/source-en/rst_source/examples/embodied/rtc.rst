RTC Evaluation
==============

Real-Time Control (RTC) is a technique that hides inference latency during deployment. It overlaps the execution of one action chunk with the inference of the next: the policy outputs a sequence of actions (chunk), the environment begins executing them frame by frame; during execution, RTC pre-emptively requests inference for the next chunk, so inference latency is absorbed by action execution time.

RTC supports both simulation (LIBERO) and real-world (Franka) evaluation, currently integrated with the OpenPI π₀.₅ policy.

How RTC Works
-------------

The core idea of RTC is to pipeline policy inference with action execution. The standard rollout flow is synchronous (infer → execute chunk → infer → execute chunk), with the GPU idle during execution. RTC converts this to an asynchronous pipeline:

Process:

1. **Bootstrap**: The policy runs initial inference, producing the first action chunk.
2. **Execute and overlap**: The environment executes the current chunk one step at a time. After ``min_exec_horizon`` steps, the Env Worker asynchronously sends the latest obs to the Rollout Worker to request inference.
3. **Adaptive latency**: RTC maintains a sliding window (``delay_buffer_size``) recording observed inference latency, used to predict when to send the next request.
4. **Stop**: Once the episode ends (or max steps reached), the Env Worker sends a ``stop`` signal to terminate the Rollout Worker loop.

RTC uses **soft overlap guidance** to smooth chunk boundaries. When a new chunk arrives, the previous chunk may still have several unexecuted steps. The RTC guidance mechanism aligns the first ``delay_steps`` of the new chunk with the remaining steps of the old chunk, avoiding abrupt action changes:

- **Hard mask**: the first ``delay_steps`` positions, forced to match the old chunk's remaining actions.
- **Soft mask**: subsequent steps, guided toward the new chunk's predictions with exponential decay.
- Guidance strength is clipped by ``rtc_guidance_clip`` to avoid over-correction.

Simulation Experiment
---------------------

Install and download

.. code:: bash

   # Install dependencies
   bash requirements/install.sh embodied --model openvla --env maniskill_libero

   # Download model
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

Run RTC evaluation

.. code:: bash

   bash evaluations/run_eval.sh libero_spatial_eval_pi05_RTC

Enable / disable RTC via ``runner.rtc.enabled`` in the YAML (default ``True``):

.. code:: bash

   bash evaluations/run_eval.sh libero_spatial_eval_pi05_RTC \
       'runner.rtc.enabled=True'

.. note::

   In simulation, use ``runner.rtc.chunk_pause_seconds`` to simulate action execution time.

Results
~~~~~~~

``action_chunk=8``, LIBERO Spatial evaluation:

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - Metric
     - RTC off
     - RTC on
   * - ``episode_time`` (s)
     - 56.93
     - 54.02
   * - ``step_total_time``
     - 45.73
     - 45.96
   * - ``wait_inference_time``
     - 5.61
     - 0.002
   * - ``env_reset_time``
     - 5.42
     - 5.15
   * - ``rtc_bootstrap_time``
     -
     - 2.02
   * - ``residual``
     - 0.17
     - 0.89
   * - ``speed_up`` (s)
     -
     - 2.91

``action_chunk=5``, LIBERO Spatial evaluation:

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - Metric
     - RTC off
     - RTC on
   * - ``episode_time``
     - 60.19
     - 52.76
   * - ``step_total_time``
     - 45.57
     - 45.55
   * - ``wait_inference_time``
     - 8.46
     - 0.002
   * - ``env_reset_time``
     - 5.91
     - 5.12
   * - ``rtc_bootstrap_time``
     -
     - 1.24
   * - ``residual``
     - 0.09
     - 0.85
   * - ``speed_up``
     -
     - 7.43

RTC reduces ``wait_inference_time`` from several seconds to ~2ms (near zero), almost completely hiding inference latency. Smaller ``action_chunk`` values lead to more frequent inference requests and greater speedup (7.4s vs 2.9s).

Real-World Experiment
---------------------

For real-world deployment, see :doc:`Franka Real-World RL <franka>`. This section covers only the differences for RTC evaluation.

Control node installation

.. code:: bash

   bash requirements/install.sh embodied --env franka

Training node installation

.. code:: bash

   bash requirements/install.sh embodied --model openvla --env maniskill_libero

Download model

.. code:: bash

   hf download RLinf/RLinf-Pi05-Pick_Red --local-dir ./RLinf-Pi05-Pick_Red

Start Ray cluster

Training node (head):

.. code:: bash

   source ray_utils/realworld/setup_before_ray.sh
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

Control node (worker):

.. code:: bash

   source .venv/franka_catkin_ws/devel/setup.bash
   source ray_utils/realworld/setup_before_ray.sh
   ray start --address='<head_node_ip_address>:6379'

Launch evaluation on the training node

.. code:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval_pi05_sft_RTC

Results
~~~~~~~

``action_chunk=8``, Franka PnP real-world evaluation:

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - Metric
     - RTC off
     - RTC on
   * - ``episode_time``
     - 48.18
     - 45.56
   * - ``step_total_time``
     - 35.58
     - 36.04
   * - ``wait_inference_time``
     - 5.64
     - 0.003
   * - ``env_reset_time``
     - 6.60
     - 6.51
   * - ``rtc_bootstrap_time``
     -
     - 0.85
   * - ``residual``
     - 0.37
     - 2.16
   * - ``speed_up``
     -
     - 2.63

``action_chunk=5``, Franka PnP real-world evaluation:

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - Metric
     - RTC off
     - RTC on
   * - ``episode_time``
     - 51.66
     - 44.05
   * - ``step_total_time``
     - 35.80
     - 34.34
   * - ``wait_inference_time``
     - 8.56
     - 0.004
   * - ``env_reset_time``
     - 6.67
     - 6.55
   * - ``rtc_bootstrap_time``
     -
     - 1.00
   * - ``residual``
     - 0.63
     - 2.15
   * - ``speed_up``
     -
     - 7.61

.. note::

   In real-world experiments, ``residual`` increases by ~1.5s due to RTC async communication overhead (send requests, receive responses over the network). Even so, RTC still saves 5-8 seconds in inference waiting time, accelerating total episode time by ~3-8 seconds.

Metric Reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40

   * - Metric
     - Meaning
   * - ``episode_time``
     - Total rollout time (from reset to episode end), in seconds.
   * - ``step_total_time``
     - Cumulative time spent executing all actions during the rollout, in seconds. In simulation: ``chunk_step`` duration; on real robot: actual robot execution time.
   * - ``wait_inference_time``
     - Total time spent waiting for inference results during the rollout, in seconds. Near zero when RTC is on (inference hidden by action execution).
   * - ``env_reset_time``
     - Environment reset time, in seconds.
   * - ``rtc_bootstrap_time``
     - Time from the first RTC action request sent to response received, in seconds. Only present when RTC is enabled.
   * - ``residual``
     - Other overhead: ``episode_time - step_total_time - wait_inference_time - env_reset_time - rtc_bootstrap_time``, in seconds.
   * - ``speed_up``
     - ``wait_inference_time (off) - wait_inference_time (on)``, i.e., inference waiting time saved by RTC, in seconds.

Configuration Reference
-----------------------

RTC parameters are split across two locations:

``runner.rtc`` (controls RTC loop behavior)

.. list-table::
   :header-rows: 1
   :widths: 20 10 50

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``False``
     - Whether to enable RTC. When ``True``, the Env Worker uses async overlapping inference mode; the Rollout Worker runs an infinite loop waiting for requests until a ``stop`` signal.
   * - ``fixed_delay_steps``
     - ``0``
     - Simulation-only: fixed number of steps to simulate inference delay. Only valid in simulation; forced to 0 on real robot.
   * - ``chunk_pause_seconds``
     - ``0.0``
     - Simulation-only: pause duration after each step, used to simulate real-robot execution time. Forced to 0 on real robot.
   * - ``inject_delay_ms``
     - ``0``
     - Real-world only: simulated inference delay in milliseconds. Forced to 0 in simulation.
   * - ``min_exec_horizon``
     - ``2``
     - Minimum number of steps to execute before requesting the next chunk. Too small may cause overly frequent requests; too large reduces overlap coverage.
   * - ``initial_delay_steps``
     - ``1``
     - Predicted delay steps for the first request. Used as the initial estimate for the first replan request after bootstrap.
   * - ``delay_buffer_size``
     - ``8``
     - Sliding window size for estimating inference latency. RTC uses the maximum delay in the window as the next round's prediction.

``model.openpi.rtc_*`` (controls RTC guidance behavior; keep in sync with ``runner.rtc.enabled``)

.. list-table::
   :header-rows: 1
   :widths: 20 10 50

   * - Parameter
     - Default
     - Description
   * - ``rtc_enabled``
     - ``False``
     - Use Hydra reference ``${runner.rtc.enabled}`` to keep in sync. When enabled, the model's output action chunk references the previous chunk's remaining actions.
   * - ``rtc_guidance_mode``
     - ``"approx"``
     - Guidance mode, currently only ``"approx"`` (approximate guidance) is supported.
   * - ``rtc_guidance_clip``
     - ``5.0``
     - Guidance strength clipping threshold. Higher values mean stronger guidance and smoother chunk boundaries; too high may cause action drift. Recommended range 3.0-10.0.
