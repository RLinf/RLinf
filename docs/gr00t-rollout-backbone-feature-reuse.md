# GR00T Rollout Backbone Feature Reuse: Design, Validation, Reproduction, and an IsaacLab Semantics Question

## Executive summary

This report describes frozen backbone feature reuse for the GR00T N1.5 Trocar PPO
workload. Rollout computes a frozen Eagle backbone feature for an observed sample;
the Actor reuses that feature across the four PPO update epochs that consume the
same collected sample. The design does not assume that observations from different
environments are equal, and it is not a general cache across rollout steps.

The current upstream-oriented implementation is rebased on `be8d5c2d` and consists
of five logical commits through `aec9e9f8`. It is the PR form of the independently
reviewed and experimentally validated series from `c90951a0` through `c6696188`.
The performance results below belong to that historical validation baseline; they
must not be presented as measurements of the rebased series until its H20 E2E smoke
has passed. On 8x H20 with 65,536 samples per step and 20 steps, steady steps 6-20
measured:

- matched baseline: `2176.445 s/step`, `30.111 samples/s`, `92420 MiB` peak;
- clean candidate: `1521.060 s/step`, `43.086 samples/s`, `70879 MiB` peak;
- versus the matched baseline: `+43.087%` throughput, `-30.113%` step time,
  approximately `-50.83%` Actor time, and `-23.31%` sampled peak memory;
- pinned-cache prototype reference: `1524.674 s`, `42.984 samples/s`, `68834 MiB`;
  the candidate is `+0.238%` faster than the prototype, with only `3 MiB` of
  headroom under the preregistered memory gate.

The pinned-cache prototype reference is an earlier implementation, not a different
algorithm. Runtime instrumentation is used only to make the two arms repeatable and
is not part of the feature patch. The IsaacLab stage-update fix is a shared,
independent task input for both arms. Early 20-step learning curves show no observed
numerical regression, but a 64-episode evaluation crossed between arms. This is not
enough evidence for an accuracy or convergence claim.

## 1. Scope and reuse semantics

The Eagle backbone in GR00T N1.5 is frozen, in evaluation mode, and has dropout
disabled for this workload. During one PPO training step, Rollout evaluates the
policy once for the observations. The same collected trajectory is then consumed by
the Actor over four PPO update epochs. Recomputing a frozen backbone from images in
every Actor epoch is redundant.

Original path:

```text
images -> Actor recomputes frozen Eagle backbone -> action head / PPO update
```

Reuse path:

```text
Rollout: images -> frozen Eagle -> raw feature + mask
                              |
                              v
Actor:   sample ID -> cached feature -> action head / PPO update x 4
```

The cache is scoped to one collected sample:

- the global sample ID is the reuse and routing key;
- the feature and mask were produced by the frozen backbone for that sample during Rollout;
- the design does not assume equal observations across environments;
- the feature is not retained across rollout steps; cache and staging state are cleared at step end.

## 2. Architecture, data flow, and ownership

### 2.1 Production data flow

1. Rollout captures the raw feature and mask before the action-head mapping is mutated,
   and keeps the producer CUDA tensors alive.
2. A matching Rollout rank and Actor rank temporarily borrow a CUDA IPC view on the
   same physical GPU. The borrow window covers only the D2H copy.
3. The Actor copies the borrowed view into an Actor-owned pinned CPU cache and sends
   the ACK only after the D2H copy completes.
4. The normal Rollout to Env to Actor trajectory carries global sample IDs and owner
   metadata, not the production feature payload.
5. After shuffling, each Actor PPO microbatch gathers cached features by sample ID
   and copies them to the training GPU on demand for the action head and PPO update.
6. After all four epochs, or immediately after an exception, the Actor clears cache
   and staging state and Rollout releases the producer tensor.

### 2.2 Why the CUDA feature is not retained indefinitely

Rollout retains the producer CUDA tensor only until the Actor completes the D2H copy.
The borrowed view has a short lifetime. The Actor-owned pinned CPU cache covers the
full PPO reuse window. This avoids keeping a complete feature batch in Rollout GPU
memory across epochs and limits the OOM risk from long-lived feature storage.

### 2.3 Fail-closed contract

Each lease and stream block validates the model version, producer and consumer ranks,
sample namespace, sample ID ordering, batch and block counts, byte count, and cache
completeness. The protocol provides ACK/NACK handling, bounded timeouts, exception
cleanup, and zero fallback/error counters.

Malformed metadata, a missing ACK, an incomplete cache, a consumer exception, or an
unsupported placement fails the current run and clears local state. It does not
silently switch to a path with different data semantics. A timeout is terminal for
the current run because the queued collective receive cannot be safely canceled.

## 3. Five-commit map and key modules

### 3.1 Five-commit map

| Commit | Responsibility | Main content |
|---|---|---|
| `8a40dd4e` | Model boundary | Defines the frozen-backbone producer and consumer contract; capture occurs before action-head mutation |
| `7ad065c3` | Borrowed CUDA IPC | Provides the same-GPU borrowed view, lease validation, and same-device rejection; unsupported placement fails closed |
| `013bf278` | Pinned cache and stream | Adds the Actor-owned pinned CPU cache, D2H stream, ACK/NACK handling, timeout, and cleanup |
| `48440aba` | Integration | Connects the stream to the embodied runner, Rollout, Env sample-ID routing, and FSDP Actor |
| `aec9e9f8` | Configuration and docs | Adds the default-off configuration, production settings, EN/ZH docs, and topology constraints |

### 3.2 Key modules

| Module | Role |
|---|---|
| `rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py` | Captures the raw feature and mask; supports precomputed backbone inputs in the Actor |
| `rlinf/runners/embodied_runner.py` | Enables the data plane only for the exact opt-in transport |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Owns producer tensors, the borrowed IPC lease, ACK handling, and cleanup |
| `rlinf/workers/env/env_worker.py` | Preserves global sample ID and owner metadata routing |
| `rlinf/workers/actor/fsdp_actor_worker.py` | Checks model and placement constraints, gathers by sample ID, and cleans training state |
| `rlinf/utils/backbone_cache.py` | Tracks model version, ranks, sample namespace, order, counts, and bytes |
| `rlinf/utils/pinned_rollout_cache.py` | Implements the Actor-owned pinned CPU cache for four PPO epochs |
| `rlinf/utils/pinned_feature_stream.py` | Implements borrowed-view receive, D2H, ACK/NACK, timeout, and lease release |

## 4. Configuration and placement constraints

Default off:

```yaml
actor:
  model:
    rollout_backbone_feature_transport: null
```

Production opt-in:

```yaml
actor:
  model:
    rollout_backbone_feature_transport: borrowed_ipc_pinned
rollout:
  pinned_feature_ipc_batch_blocks: 16
  pinned_feature_ipc_timeout_seconds: 300
  pinned_feature_verify_trajectory: false
```

`pinned_feature_verify_trajectory: true` is a debug parity mode. It retains an
additional trajectory reference and compares features and masks. It is not a
production performance setting.

The supported scope is intentionally narrow:

- GR00T N1.5 only; the backbone must be frozen and in eval mode, with dropout disabled and no SFT co-training;
- synchronous embodied runner only, with `pipeline_stage_num=1` and `actor_split_num=1`;
- equal Rollout and Actor world sizes, with rank `i` mapped to the same physical CUDA GPU;
- no cross-GPU or cross-host support, and no NCCL or Gloo fallback;
- borrowed IPC accepts CUDA tensors or tensor-list payloads only;
- unsupported conditions fail closed instead of silently recomputing the backbone.

## 5. Performance table

### 5.1 Fixed experiment inputs

- Image: `chenchaox72877/trocar-rlinf-bench@sha256:9f02e069ccb0e0a7e536833e789666e85f039c9024a77ac2f219c42cb1dfcf01`
- Hardware: 8x H20 with same-GPU identity placement for Rollout and Actor ranks
- Workload: 65,536 samples per step, 20 steps, steady steps 6-20
- Historical matched baseline source: `c90951a0`, with the same deterministic Rollout and
  critic instrumentation applied as the candidate
- Historical validated candidate source: `c6696188`, runtime head `3e648e8b245f12ffeebda1f1c40284bcbb46d2be`
- The runtime head contains deterministic Rollout and critic instrumentation for A/B
  repeatability only. It is not part of the feature patch. Both arms use the same task input.
- IsaacLab task input: `5db390fe6b615c10d2b57a7a12aa36159d928815`

### 5.2 Baseline, candidate, and prototype reference

| Arm | Step time (s) | Samples/s | Actor (s) | Env (s) | Rollout (s) | Sync (s) | Sampled peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched baseline | 2176.445 | 30.111 | 1226.033 | 534.447 | 957.773 | 2.358 | 92420 |
| Clean candidate | 1521.060 | 43.086 | 602.827 | 531.347 | 928.233 | 2.439 | 70879 |
| Pinned-cache prototype reference | 1524.674 | 42.984 | - | - | - | - | 68834 |

Versus the matched baseline, the clean candidate improves throughput by `+43.087%`,
reduces step time by `-30.113%`, reduces Actor time by approximately `-50.83%`, and
reduces sampled peak memory by `21541 MiB` or `-23.31%`. Env and synchronization time
are nearly unchanged, so the primary gain is the removal of repeated frozen-backbone
work for the same collected samples.

The pinned-cache prototype reference is an earlier implementation, not another
algorithm. The clean candidate is `+0.238%` faster than the prototype and has about
`-0.237%` step time. Its peak is `2045 MiB` above the prototype. The preregistered
prototype plus `2048 MiB` memory gate has only `3 MiB` of remaining margin and should
be treated as a narrow residual risk.

## 6. Validation gates and learning/convergence limits

| Gate | Result | Evidence and boundary |
|---|---|---|
| Current PR source history and default-off path | PASS | Exactly five feature commits from `be8d5c2d` to `aec9e9f8`; rebase conflict review, diff check, Python compile, and YAML parse pass |
| CPU focused suite | PASS | `50 passed, 1 skipped`; the CUDA round trip is run separately |
| Real same-GPU CUDA IPC | PASS | Borrowed-storage alias test passed on H20 GPU 0 |
| Model contract | PASS | Fresh/precomputed parity, raw pre-mutation capture, and fail-closed frozen/eval/dropout checks |
| Eight-rank debug trajectory parity | PASS | Bitwise feature and mask parity with zero mismatch and zero fallback |
| Four-epoch optimizer parity | PASS | Exact loss, gradient hash, parameter, and AdamW state results |
| Production-shape two-step debug smoke | PASS | `verify_trajectory=true`; validates trajectory reference parity, not production-config performance |
| Production 20-step E2E | PASS | `exit=0`, 20/20 steps, 160/160 stream/cache markers, zero fallback and zero error |
| Matched baseline versus clean candidate | PASS | `+43.087%` throughput, above the preregistered `+10%` gate |
| Clean candidate versus prototype | PASS | `+0.238%` throughput, above the preregistered `-3%` regression margin |
| Peak GPU memory | PASS with narrow margin | Only `3 MiB` remains under the prototype plus `2048 MiB` gate |
| Failure injection | Partial boundary | ACK/NACK, timeout, metadata, and cleanup have protocol tests; a full H20 fault-injection E2E remains follow-up work |
| Independent static review | PASS | The final five-commit implementation was approved |

The early 20-step curves are healthy and close between the two arms. Candidate
first-five to last-five means changed from return `0.6883 -> 1.4297`, reward
`0.005377 -> 0.011169`, and critic explained variance `0.7178 -> 0.8440`.
The matched baseline ended at `1.4281 / 0.011157 / 0.8440`.

The 64-episode evaluation crossed between arms: baseline step 20 had `success_once`
of `31.25%` and candidate step 20 had `21.875%`; at step 10, candidate had `25%`
and baseline had `9.375%`. `success_once` means that a positive stage reward was
observed, while `success_at_end` was `0` for all arms. The sample is small and CUDA
forward execution is non-bitwise. These results do not support an accuracy or
convergence claim. The supported statement is only that no early numerical
regression was observed.

## 7. Reproduction

### 7.1 Fixed inputs

Freeze the source, runtime head, image, task input, placement, and workload before
comparing arms:

```text
Current PR base: be8d5c2d...
Current PR candidate: aec9e9f8...
Historical matched baseline source: c90951a0...
Historical validated candidate source: c6696188...
Historical candidate runtime: 3e648e8b245f12ffeebda1f1c40284bcbb46d2be (base c6696188)
Repeatability instrumentation: apply the same deterministic Rollout and critic seeding to both arms
IsaacLab task input: 5db390fe6b615c10d2b57a7a12aa36159d928815
Image: sha256:9f02e069ccb0e0a7e536833e789666e85f039c9024a77ac2f219c42cb1dfcf01
Hardware: 8x H20
Workload: 65,536 samples per step, 20 steps, steady 6-20
Placement: Rollout rank i and Actor rank i on the same physical CUDA GPU
```

### 7.2 Fetch, worktree, and placement

Set `REVIEW_REMOTE` to a Git remote pointing to the public `davidmlw/RLinf` fork.
The example uses the remote name `david`; replace it with the name configured in
your checkout. Fetch the remote, verify the public commit, and create the worktree
directly from that commit:

```bash
REVIEW_REMOTE=david
git fetch "$REVIEW_REMOTE"
git cat-file -e aec9e9f8241b87a193c6fa8b0a885d6a61d13ce6^{commit}
git worktree add ../gr00t-feature-candidate \
  aec9e9f8241b87a193c6fa8b0a885d6a61d13ce6
git -C ../gr00t-feature-candidate status --short
git -C ../gr00t-feature-candidate log --oneline --reverse \
  be8d5c2d28015146355dd2b47bce97331339f9d1...aec9e9f8241b87a193c6fa8b0a885d6a61d13ce6
nvidia-smi -L
nvidia-smi topo -m
```

Equal world size is not sufficient. Verify that Rollout rank `i` and Actor rank `i`
resolve to the same physical GPU. Cross-GPU and cross-host placement must be rejected;
borrowed IPC does not fall back to NCCL or Gloo.

### 7.3 Configuration and test order

First run with `rollout_backbone_feature_transport: null` to confirm the original
image-to-Actor-backbone path. Then use the production opt-in configuration:

```yaml
actor:
  model:
    rollout_backbone_feature_transport: borrowed_ipc_pinned
rollout:
  pinned_feature_ipc_batch_blocks: 16
  pinned_feature_ipc_timeout_seconds: 300
  pinned_feature_verify_trajectory: false
```

The portable test order is:

```bash
# 1. Clean source and input hashes
git status --short
sha256sum <config> <runtime-overlay> <task-input>

# 2. Model, protocol, and default-path unit suite
pytest -q tests/unit_tests/test_backbone_feature_cache.py \
  tests/unit_tests/test_borrowed_ipc_validation.py \
  tests/unit_tests/test_gr00t_backbone_cache_interface.py \
  tests/unit_tests/test_pinned_feature_actor.py \
  tests/unit_tests/test_pinned_feature_sender.py \
  tests/unit_tests/test_pinned_feature_stream.py \
  tests/unit_tests/test_rollout_backbone_feature_reuse.py

# 3. Real same-GPU CUDA IPC alias test
pytest -q tests/unit_tests/test_intra_gpu_comm.py::TestSameDeviceCommunication::test_borrowed_tensor_list_aliases_producer_storage

# 4. Four-epoch optimizer replay
python <optimizer-parity-harness> --output-dir <artifact-dir>/optimizer-parity

# 5. Production-shape debug parity, then the production 20-step candidate
bash <candidate-launcher> 2 <artifact-dir>/smoke verify
bash <candidate-launcher> 20 <artifact-dir>/production production
```

The Docker and IsaacLab launcher details are environment-specific. Adapt the mounts
and wrapper paths rather than copying machine-local paths. The debug run must use
`verify_trajectory=true`; the production run must disable it.

### 7.4 Artifact retention checklist

For every arm and gate, retain at least:

- source commit, clean status, runtime head, and resolved config SHA;
- image inspection output, task input commit, placement, and world-size manifest;
- command line, environment summary, stdout, stderr, and exit code;
- parsed step summary, throughput, phase timing, and step time;
- per-device GPU memory samples and CPU or pinned-memory peak;
- stream/cache marker counts, fallback/error counts, and lease/ACK/NACK counts;
- optimizer loss, gradient, parameter, and AdamW hashes plus the parity summary;
- evaluation seed, episode count, `success_once`, `success_at_end`, and video metadata.

Use an immutable run ID for every artifact set and verify source, config, image, and
task hashes before comparison. Runtime instrumentation is shared repeatability input
only. The IsaacLab fix is a shared task input for both arms and must not be injected
into only one arm.

## 8. IsaacLab modification and explicit questions to owners

### 8.1 Modification and current execution semantics

The IsaacLab branch `fix/assemble-trocar-stage-update` at commit
`5db390fe6b615c10d2b57a7a12aa36159d928815` changes the Trocar `update_stage`
`RewTerm.weight` from `0.0` to `1.0`. `update_task_stage()` always returns a zero
tensor, so the term itself does not add to total reward.

The current `RewardManager.compute()` implementation explicitly continues when
`term_cfg.weight == 0.0`. Under the old configuration, the side-effecting
`update_task_stage()` function was never called and `task_stage` remained zero.
Under the current implementation, the nonzero weight enables the state-machine
update; it does not change the numeric reward contribution of that term.

The new train and eval regression test checks only that
`env_cfg.rewards.update_stage.weight == 1.0`. It does not yet verify actual stage
transition, reset, sparse reward accounting, or total reward behavior.

Therefore, the change is mechanically necessary under the current RewardManager
behavior, but the architectural and task semantics remain open for owner review.

### 8.2 Questions for IsaacLab and task owners

1. Should state progression be implemented as a reward term at all?
2. Is a zero-return function with `weight=1` the right minimal fix, or should the
   side effect move to an event, step hook, or dedicated manager?
3. Should RewardManager change its zero-weight skip semantics, or should IsaacLab
   introduce an `always_run` or `state-update` term type?
4. Is the ordering dependency, with `update_stage` before sparse reward terms, an
   accepted contract that should be explicitly documented and tested?
5. Does this change affect task definition or training comparability? Which upstream
   regression tests are required for train/eval, stage transition, reset, reward
   accounting, and checkpoint replay?

## 9. Limitations and next steps

- The candidate has only `3 MiB` of remaining margin under the prototype memory gate;
  validate larger shapes, long runs, and combined configurations with memory guards.
- Complete H20 failure-injection E2E remains to be tested for ACK timeout, consumer
  exception, malformed metadata, and partial-cache cleanup.
- The crossed 64-episode evaluation cannot establish convergence; use longer training
  and a larger evaluation set.
- The design covers GR00T N1.5, the synchronous runner with
  `pipeline_stage_num=1`, and same-GPU identity placement. It does not generalize
  to cross-GPU or cross-host placement.
- `aec9e9f8` is an upstream-oriented implementation and should not be described as
  already merged into the RLinf upstream product branch.

## 10. Public links

- [Current RLinf PR diff](https://github.com/davidmlw/RLinf/compare/be8d5c2d28015146355dd2b47bce97331339f9d1...aec9e9f8241b87a193c6fa8b0a885d6a61d13ce6)
- [Current RLinf base commit](https://github.com/davidmlw/RLinf/commit/be8d5c2d28015146355dd2b47bce97331339f9d1)
- [Current RLinf candidate commit](https://github.com/davidmlw/RLinf/commit/aec9e9f8241b87a193c6fa8b0a885d6a61d13ce6)
- [Historically validated RLinf diff](https://github.com/davidmlw/RLinf/compare/c90951a0c799a750cb5294ed10587c61cc2af8bf...c6696188d14db68fdbebd584eff49110cb53a387)
- [RLinf repeatability runtime commit](https://github.com/davidmlw/RLinf/commit/3e648e8b245f12ffeebda1f1c40284bcbb46d2be)
- [IsaacLab stage-update commit](https://github.com/davidmlw/IsaacLab/commit/5db390fe6b615c10d2b57a7a12aa36159d928815)
