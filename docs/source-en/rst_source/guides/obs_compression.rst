Observation Compression
=======================

Image observations are the dominant payload on the Env :math:`\rightarrow`
Rollout channel in embodied tasks. In bandwidth-constrained or multi-node
setups, transferring these frames can throttle rollout throughput. RLinf can
optionally compress them **losslessly** before transfer and reconstruct them on
the rollout side, trading a small amount of CPU time for a large reduction in
bytes on the wire.

The feature is **disabled by default** and, when enabled, is fully transparent:
the reconstructed observations are bit-for-bit identical to the originals, so
training results are unaffected.

How It Works
------------

Only ``uint8`` image tensors inside the observation payload are compressed;
proprioceptive states, flags, and text fields are passed through untouched.
Each compressed tensor is replaced by a small self-describing marker, so
decompression needs no configuration and keeps no cross-message state.

Two steps are applied per message:

- **XOR-delta (optional).** Consecutive frames along the batch axis are XORed
  together. Parallel embodied environments (e.g. LIBERO, ManiSkill) render
  largely static backgrounds, so the delta is mostly zeros and compresses far
  better than the raw frames.
- **Entropy coding.** The result is compressed with ``lz4`` (fastest) or
  ``zstd`` (better ratio).

The ``lz4`` and ``zstandard`` codecs ship with RLinf as core dependencies, so
no extra installation is required.

How to Enable
-------------

Add an ``obs_compression`` block under ``env`` in an embodied configuration:

.. code-block:: yaml

   env:
     obs_compression:
       enable: true      # off by default
       codec: lz4        # lz4 (faster) or zstd (better ratio)
       level: 1          # zstd compression level; ignored by lz4
       xor_delta: true   # XOR consecutive frames before compression

Where:

- ``enable`` turns the feature on. When omitted or ``false``, observations are
  sent uncompressed and there is zero overhead.
- ``codec`` selects the backend. ``lz4`` minimizes CPU cost; ``zstd`` maximizes
  the compression ratio.
- ``level`` is the ``zstd`` effort level (higher is smaller but slower).
- ``xor_delta`` enables the XOR-delta pre-pass. It is skipped automatically when
  a message carries only a single frame.

Expected Gains
--------------

On consecutive LIBERO observations (about 3 MiB of images per Env Worker step),
lossless compression typically reduces the payload to roughly 36-41% of its
original size:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - Size vs. original
     - Compress
     - Decompress
   * - XOR + LZ4
     - ~40.5%
     - ~2.65 ms
     - ~0.66 ms
   * - XOR + Zstd (level 1)
     - ~36.0%
     - ~4.70 ms
     - ~1.93 ms

Actual ratios depend on image resolution and how static the scene is.
