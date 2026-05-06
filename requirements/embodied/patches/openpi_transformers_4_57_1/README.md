# OpenPI transformers 4.57.1 patch

These replacement files are based on the vanilla `transformers==4.57.1` wheel.
They port OpenPI's original `transformers_replace` changes, which targeted
`transformers==4.53.2`, onto the Qwen3-VL-compatible 4.57.1 runtime.

The patch preserves OpenPI-specific behavior for:

- Gemma ADARMS config/model support.
- PaliGemma image feature scaling.
- SigLIP bfloat16 activation casting.
