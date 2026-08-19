Compare OpenPI Policy Servers
=============================

Measure the deployed behavior of an OpenPI server and an RLinf policy server
on the same real ALOHA observations, without restarting or modifying either
server.

Prepare the Diagnostic Environment
----------------------------------

Start both OpenPI-compatible WebSocket servers before you run the diagnostic.
The default endpoints are ``127.0.0.1:8000`` for OpenPI and
``127.0.0.1:8001`` for RLinf. The defaults also match the currently
deployed wire layouts: ``--openpi-image-layout chw`` and
``--rlinf-image-layout hwc``. Both servers receive the same decoded pixels;
only the axis order differs. Install the optional analysis dependencies in
the environment that runs the client:

.. code-block:: bash

   python -m pip install pyarrow Pillow matplotlib openpi-client websockets

The packages are imported only by this toolkit. They do not become RLinf core
runtime dependencies.

Run the Comparison
------------------

Run the default comparison from the repository root:

.. code-block:: bash

   python toolkits/lerobot/compare_policy_servers.py \
       --dataset-root data/lerobot-data_mixed_8_v30

What this does: it selects one episode for each of the 11 dataset prompts,
sends three evenly spaced observations three times, and evaluates three
consecutive replanning chunks. It keeps one persistent connection to each
server and writes a timestamped run under
``results/policy_server_compare/``.

Narrow the run before a full diagnostic:

.. code-block:: bash

   python toolkits/lerobot/compare_policy_servers.py \
       --dataset-root data/lerobot-data_mixed_8_v30 \
       --prompt-regex "Place the bread" \
       --paired-frames 1 \
       --repeats 1 \
       --replay-chunks 1 \
       --request-timeout 120

Use ``--episode-ids 12 24`` to select explicit episodes. Use
``--episodes-per-prompt``, ``--seed``, ``--openpi-host``, ``--openpi-port``,
``--rlinf-host``, ``--rlinf-port``, and ``--output-dir`` to control sampling,
connections, and output placement.

Interpret the Metrics
---------------------

Use the three metric groups to separate deployment differences from sampling
noise and temporal behavior.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Group
     - Interpretation
   * - Common-prefix difference
     - MAE, RMSE, P95, maximum error, joint-standard-deviation-normalized MAE,
       and per-joint/per-step errors over the shared horizon and first 14
       action dimensions.
   * - Randomness
     - Output standard deviation and repeat-to-repeat MAE for repeated calls on
       an identical observation.
   * - Jitter proxies
     - First-action state jump, first and second chunk differences, dataset
       action error, and the boundary jump between full-horizon replans.

Inspect the Artifacts
---------------------

Each timestamped directory contains the complete inputs needed to trace an
aggregate result back to an episode and frame.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Artifact
     - Contents
   * - ``run_metadata.json``
     - CLI arguments, selected episodes, endpoints, handshake metadata, and
       observed response contracts.
   * - ``raw_outputs.npz``
     - Raw paired and replay action chunks with a serialized index.
   * - ``summary.json`` and ``report.md``
     - Aggregate metrics, worst prompt, joint, and horizon step, plus the
       contract warning.
   * - ``sample_metrics.csv``
     - Per-request difference, client RTT, and server inference timing.
   * - ``per_joint_metrics.csv`` and ``per_horizon_metrics.csv``
     - Detailed common-prefix errors.
   * - ``jitter_metrics.csv``
     - Per-chunk jitter proxies and replanning boundary jumps.
   * - PNG plots
     - Difference heatmap, worst-sample action trace, and jitter comparison.

.. warning::

   Treat this result as a diagnostic of the currently deployed services. If
   action horizons, checkpoints, denoising settings, metadata, or other model
   contracts differ, the common-prefix metrics are not strict numerical
   parity evidence. Align both server contracts before drawing implementation
   parity conclusions.
