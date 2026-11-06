# Reproduce the ragged agent batch validation

Validate real-length storage, early question-group balancing, and delayed loss padding with the Search-R1 GLM configuration. `REPORT.md` records the existing measurements; `evidence.json` contains the values used in its tables and figures. The historical artifact archive holds the original inputs, configs, diagnostic harnesses, logs, and TensorBoard events. See its `MANIFEST.json` for SHA-256 checksums.

## Environment and inputs

The GPU experiments use 8 A100-SXM4-80GB GPUs, GLM-4.7-Flash, actor TP=4 / DP=2 / EP=8 / PP=1, BF16, and the packages listed in `evidence.json`. Install the agentic Megatron/SGLang environment using the repository installer. Use a Search-R1 JSONL dataset with message-list prompts under `messages`, a local model directory, and a reachable search server. The recorded experiments use a frozen local wiki service, whose code, asset recipe, and dependencies are included in the archive; record its index and encoder when using a different service.

Activate that environment, run from the repository root, and set:

```bash
export PYTHONPATH="$PWD:/path/to/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
```

These settings make RLinf and Megatron importable and use the same connection setting as the recorded GPU runs. Substitute the actual Megatron checkout path. Set model, dataset, artifact, and output paths explicitly in the following commands.

To recreate the recorded retrieval service, use a CPU environment with NumPy, PyTorch, Transformers, `faiss-cpu`, FastAPI, Pydantic, and Uvicorn. Prepare the offsets once, then keep the second command running in another terminal:

```bash
python /path/to/artifact/performance/harness/wiki_service.py --prepare \
  --corpus /path/to/wiki-18.jsonl --assets /path/to/new-wiki-assets
python /path/to/artifact/performance/harness/wiki_service.py \
  --assets /path/to/new-wiki-assets \
  --index /path/to/e5_IVF4096,Flat.index --encoder /path/to/e5-base-v2 \
  --host 127.0.0.1 --port 18080 --nprobe 32 --threads 8
```

The recorded `wiki-18.jsonl` asset is a tar archive containing the indexed JSONL corpus; the offset builder expects that archive format. The index must match the corpus row order. Rebuild offsets after relocating the corpus because its path and file metadata are recorded. Model weights, corpus, and index are external assets, not part of the validation archive.

## CPU contract checks

Use the repository's unit-test environment, which also supplies imports such as Gymnasium and ImageIO required when collecting `test_data.py`. Run each component separately so distributed process groups are released between files:

```bash
python -m pytest -q tests/unit_tests/test_data.py -k ragged
python -m pytest -q tests/unit_tests/test_distributed.py
python -m pytest -q tests/unit_tests/test_megatron.py
```

These cover ragged conversion and merging, loss materialization and gradients, empty-mask rows, complete-group exchange through Gloo at DP=1/2/4, and dense/ragged packing at TP=1/4 and CP=2. Megatron packing tests require Megatron to be importable; report skips separately from passes. These checks do not start a model-serving engine.

## Paired three-step runs

Prepare a frozen 1536-question fixture and two configs from the checked-in YAML:

```bash
python tests/e2e_tests/agent/ragged_batch/prepare.py \
  --model /path/to/GLM-4.7-Flash \
  --data /path/to/searchr1_messages.jsonl \
  --search-server 127.0.0.1:18080 \
  --output /path/to/new-comparison
```

Preparation selects the first 1536 rows, disables dataset shuffling, sets seed 42, records hashes, and writes `baseline/config/run.yaml` and `ragged/config/run.yaml`. Aside from the output directory, their only difference is `actor.enable_ragged_batch`. Both retain sampling temperature 1.0, use 512 questions per rollout and 4 trajectories per question, run three steps, disable validation and checkpoints, and retain TensorBoard. For the recorded input selection, pass the archive's `performance/questions.jsonl` as `--data`; its SHA-256 is in `evidence.json`. Freshly selected rows have a different hash and constitute a new experiment.

Run the cases sequentially on a dedicated eight-GPU allocation, with the search service already running:

```bash
python tests/e2e_tests/agent/ragged_batch/record.py \
  --case /path/to/new-comparison/baseline --megatron-path /path/to/Megatron-LM
python tests/e2e_tests/agent/ragged_batch/record.py \
  --case /path/to/new-comparison/ragged --megatron-path /path/to/Megatron-LM
```

The recorder uses the native training entry point. Its optional extension captures dataset batches, raw CPU results, phase timing, DP exchange timing, and allocator peaks; the parent samples physical GPU memory every second. The diagnostic runner lets the last batch of the prepared one-epoch fixture reach metric logging. Outputs are under each case's `evidence/`, `launcher.log`, `status.json`, and `training/train/tensorboard/`. Cleanup targets only Ray processes created by that invocation under its private directory. Do not resume either case from the other's model state. Compare captured dataset batch hashes before comparing performance. Matching questions and seeds do not guarantee matching sampled answers or search queries.

## Frozen communication benchmark

Replay the same generated trajectories without starting the model or search service:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tests/e2e_tests/agent/ragged_batch/communication.py \
  --raw-dir /path/to/artifact/performance/baseline/evidence/raw \
  --output /path/to/new-communication
```

The archive contains first-step directories `step01_rank00` and `step01_rank04`, with 256 complete question groups each. A new recorded baseline supplies the same layout under its own `evidence/raw`. The benchmark recreates dense derived-field shapes with placeholder logprobs, advantages, and loss scales. It measures communication bytes and buffers, not numerical model correctness. It uses the real rewards when constructing advantage shapes. Dense exchange runs in all four corresponding TP-lane DP groups; ragged exchange runs between the two data-I/O ranks before model-parallel broadcast. Model-parallel broadcast is outside this benchmark.

Discard trial 0 for time aggregation. For each subsequent trial, take the maximum participating-rank elapsed time; average those maxima. For memory, take the largest `peak_increment_bytes` across ranks and trials and divide by 1048576 for MiB. Measure `nvidia-smi` memory separately from these allocator increments. Elapsed time includes partitioning, serialization, communication, and synchronization.

## Frozen model-update comparison

Use the historical archive to prepare a fresh fixed-state diagnostic:

```bash
python tests/e2e_tests/agent/ragged_batch/replay_update.py \
  --artifact /path/to/artifact --output /path/to/new-parity \
  --model /path/to/GLM-4.7-Flash --megatron-path /path/to/Megatron-LM
```

This command only relocates the archived harness and config. Add `--run` with a fresh output path to launch the eight-GPU comparison. Repeat with `--control` and another output path for the dense-versus-dense self-control. The harness fixes the 512 question groups, original DP assignment, first minibatch, and microbatch schedule; recomputes real model logprobs; compares loss inputs; captures model/optimizer tensors, scheduler and RNG state; restores and verifies captured state; and performs one native update per path. Temporary state files require substantial local disk space and are removed after restoration. Cached recompute results are not copied into the fresh output.

Read `evidence/rank00.json` through `rank07.json` for real-token probabilities, loss inputs, state restoration, gradients before clipping, updated BF16 parameters, and FP32 optimizer parameters. The archived `correctness/harness/analyze.py` aggregates these records. Inspect the self-control before attributing differences to the optimization. The recorded check did not establish bitwise gradient/update equivalence; see `REPORT.md`.

The historical harness is intentionally preserved in the archive, including its eight-GPU, 512-question, group-size-4 assumptions. The relocated launch scripts and recorder have been checked without launching a new GPU experiment; the report distinguishes them from the exact historical scripts that produced the measurements.

## Figures and provenance

Run `python tests/e2e_tests/agent/ragged_batch/plot.py` after installing Matplotlib to regenerate the report figures from `evidence.json`. The JSON preserves the source run identifiers, config and dataset hashes, input-batch hash checks, measurement definitions, and the distinction between fixed-data and independent-sampling results. The archive also preserves the code snapshots used by those runs. Later validation-guard removal is recorded as a subsequent change, rather than presented as part of the historical GPU test.
