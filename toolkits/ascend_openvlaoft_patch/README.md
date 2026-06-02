# Temporary OpenVLA-OFT NPU attention patch

**Standalone — not part of RLinf.** This directory is a *temporary* workaround
for an Ascend NPU operator bug: the fused attention kernel (`aclnn`
FlashAttention) accumulates bf16 matmuls in a different order than the GPU
kernels the OpenVLA-OFT policy was trained with, which is the dominant source of
GPU↔NPU output divergence.

It replaces `transformers`' fused LLaMA SDPA attention with an explicit
`matmul → softmax → matmul`, so the NPU runs the same primitive ops as the GPU.
**Nothing in RLinf is modified.** When the NPU operator is fixed, uninstall and
delete this directory.

## Usage

On the NPU host, before launching the OpenVLA-OFT eval/train:

```bash
# install (writes a .pth into the active venv -> patch loads in the head
# process AND every Ray worker automatically)
bash toolkits/ascend_openvlaoft_patch/patch.sh
# ... run your OpenVLA-OFT eval/train as usual ...
bash toolkits/ascend_openvlaoft_patch/patch.sh --uninstall   # when done / operator fixed
```

Target a specific venv with `PYTHON=/path/to/venv/bin/python bash patch.sh`,
or call the Python directly: `python toolkits/ascend_openvlaoft_patch/patch.py --install`.

## Knobs (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `OFT_NPU_ATTN_DTYPE` | `bf16` | Compute dtype: `bf16` / `fp32` / `fp16`. `fp32` gives the maximum GPU↔NPU agreement. |
| `OFT_NPU_ATTN_DISABLE` | – | `1` skips the patch (A/B baseline). |
| `OFT_NPU_ATTN_FORCE` | – | `1` patches even if the NPU stack isn't detected (e.g. to test on GPU). |
| `OFT_NPU_ATTN_VERBOSE` | – | `1` logs the patch state in every process. |

On a host without the NPU stack (`torch_npu`), the patch is a **no-op** — a
shared GPU venv is unaffected.

## Notes

- Supports `transformers` ≤4.45 (`LlamaSdpaAttention` subclass) and ≥4.46
  (module-level `sdpa_attention_forward`). Idempotent.
- The Wan world-model attention parity is handled **inside RLinf** via the
  shared `Patcher` (auto on NPU), not by this directory — see
  `rlinf/envs/world_model/`.