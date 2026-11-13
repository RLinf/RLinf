# LIBERO-Mem evaluation initial states

These ten files contain 20 initial states per task from the public
[LIBERO-Mem-Raw dataset](https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw).
The dataset declares the MIT license; the upstream license is included here.

For each task, select `demo_101` through `demo_120` in numeric order from its
`*_demo.hdf5` file. Stack each demonstration's `init_state` array (or its first
`states` row when `init_state` is absent) with NumPy, then save the array with
`torch.save` under the task's BDDL stem and the `.pruned_init` extension.

The included files are the original serialized files used for the PR #1780
paired evaluation, with 200 trials per policy. All 200 state arrays match the
source demonstrations, and all ten SHA256 digests match the retained evaluation
manifests for both policies. Use these files to preserve the exact state order.

Verify the files from this directory:

```bash
sha256sum -c SHA256SUMS
```
