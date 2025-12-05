## 📊 Model Performance Comparison on LIBERO Benchmark

| Model | Spatial | Object | Goal | Long | Avg | Paper | HF Model | Code | Setup |
|-------|---------|--------|------|------|-----|-------|----------|------|-------|
| OCTO | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 | [PDF](https://arxiv.org/pdf/2405.12213) | - | [GitHub](https://github.com/octo-models/octo) | 3rd, lang |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | [PDF](https://arxiv.org/pdf/2406.09246) | [Checkpoint](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10) | [GitHub](https://github.com/openvla/openvla?tab=readme-ov-file) | 3rd, lang |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | [PDF](https://arxiv.org/pdf/2502.19645) | [Checkpoint](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) | [GitHub](https://github.com/moojink/openvla-oft) | 3rd + wrist, prop, lang |
| OpenVLA-OFT* | 96.2 | 98.3 | 96.2 | 90.7 | 95.3 | [PDF](https://arxiv.org/pdf/2502.19645) | - | [GitHub](https://github.com/moojink/openvla-oft) | 3rd, lang |
| MiniVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | [PDF](https://arxiv.org/pdf/2406.09246) | [Checkpoint](https://huggingface.co/InspireVLA/minivla-inspire-libero-union4/tree/main) | [GitHub](https://github.com/Stanford-ILIAD/openvla-mini) | 3rd, lang |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69 | 81.13 | - | - | - | 3rd, lang |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 | [PDF](https://arxiv.org/pdf/2501.15830) | - | [GitHub](https://github.com/SpatialVLA/SpatialVLA) | 3rd, lang |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 | [PDF](https://arxiv.org/pdf/2506.19850) | [Checkpoint](https://huggingface.co/Yuqi1997/UniVLA/tree/main) | [GitHub](https://github.com/baaivision/UniVLA) | 3rd, lang |
| PI0_LIBERO | 96.8 | 98.8 | 95.8 | 85.2 | 94.15 | [PDF](https://www.physicalintelligence.company/download/pi0.pdf) | [Checkpoint](https://storage.googleapis.com/openpi-assets/checkpoints/pi0_libero) | [GitHub](https://github.com/Physical-Intelligence/openpi) | 3rd + wrist, prop, lang |
| PI0_FAST_LIBERO | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 | [PDF](https://arxiv.org/pdf/2501.09747) | [Checkpoint](https://storage.googleapis.com/openpi-assets/checkpoints/pi0_fast_libero) | [GitHub](https://github.com/Physical-Intelligence/openpi) | 3rd + wrist, prop, lang |
| SmolVLA (0.24B) | 87 | 93 | 88 | 63 | 82.75 | [PDF](https://arxiv.org/pdf/2506.01844) | - | [GitHub](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla) | 3rd + wrist, prop, lang |
| SmolVLA (0.45B) | 90 | 96 | 92 | 71 | 87.3 | [PDF](https://arxiv.org/pdf/2506.01844) | - | [GitHub](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla) | 3rd + wrist, prop, lang |
| SmolVLA (2.25B) | 93 | 94 | 91 | 77 | 88.75 | [PDF](https://arxiv.org/pdf/2506.01844) | - | [GitHub](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla) | 3rd + wrist, prop, lang |
| FLOWER | 97.1 | 96.7 | 95.6 | 93.5 | 93.2 | [PDF](https://openreview.net/pdf?id=ifo8oWSLSq) | [Checkpoint](https://huggingface.co/mbreuss/flower_libero_object) | [GitHub](https://github.com/intuitive-robots/flower_vla_calvin) | 3rd, lang |
| NORA-Fine-tuned-AC | 85.6 | 89.4 | 80 | 63 | 79.5 | [PDF](https://www.arxiv.org/pdf/2504.19854) | [Checkpoint](https://huggingface.co/declare-lab/nora) | [GitHub](https://github.com/declare-lab/nora) | 3rd, lang |
| NORA-Long-Fine-tuned | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 | [PDF](https://www.arxiv.org/pdf/2504.19854) | [Checkpoint](https://huggingface.co/declare-lab/nora-long) | [GitHub](https://github.com/declare-lab/nora) | 3rd, lang |
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 | [PDF](https://arxiv.org/pdf/2412.10345) |  | - | 3rd, lang, visual trace |
| WorldVLA-512×512 | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 | [PDF](https://arxiv.org/pdf/2506.21539) | [Checkpoint](https://huggingface.co/Alibaba-DAMO-Academy/WorldVLA) | [GitHub](https://github.com/alibaba-damo-academy/WorldVLA) | 3rd, lang |
| WorldVLA-256×256 | 85.6 | 89.0 | 82.6 | 59.0 | 79.1 | [PDF](https://arxiv.org/pdf/2506.21539) | [Checkpoint](https://huggingface.co/Alibaba-DAMO-Academy/WorldVLA) | [GitHub](https://github.com/alibaba-damo-academy/WorldVLA) | 3rd, lang |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 | [PDF](https://arxiv.org/pdf/2507.16815) | - | - | 3rd, lang, prop |
| Otter | 84 | 89 | 82 | 59 | 85 | [PDF](https://arxiv.org/pdf/2503.03734) | - | [GitHub](https://github.com/Max-Fu/otter) | 3rd, lang, prop |
| RIPT-VLA | 99.0 | 98.6 | 98.6 | 93.8 | 97.5 | [PDF](https://www.arxiv.org/pdf/2505.17016) | [Checkpoint](https://huggingface.co/tanshh97/RIPT_VLA/tree/main) | [GitHub](https://github.com/Ariostgx/ript-vla) | 3rd, lang |

### 📝 Notes:
- **SFT**: Supervised Fine-tuning
- **Mix**: Mixed training data
- **3rd**: Third-person view
- **wrist**: Wrist camera
- **prop**: Proprioception
- **lang**: Language instructions