<h1 align="center">
LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
</h1>

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2510.13626"><strong>Paper</strong></a> |   
  🏗️ <a href="https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main"><strong>Assets</strong></a> | 
  🌐 <a href="https://sylvestf.github.io/LIBERO-plus"><strong>Website</strong></a> | 
  🤗 <a href="https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata"><strong>Model</strong></a> | 
  📁 <a href="https://huggingface.co/datasets/Sylvest/libero_plus_rlds"><strong>RldsDataset</strong></a>
  📁 <a href="https://huggingface.co/datasets/Sylvest/libero_plus_lerobot"><strong>LerobotDataset</strong></a>
</p>

![libero-plus](./static/images/libero-plus.jpg)

## 🔥 Overview
This repository contains the official implementation and benchmark for our paper "In-depth Robustness Analysis for Vision-Language-Action Models". We systematically expose the hidden vulnerabilities of contemporary VLA models through comprehensive robustness evaluation across seven perturbation dimensions. You can simply replace the original `libero` with a `pip install -e .` without modifying your code.

## 🚀 Key Findings
- **Significant Fragility**: VLA models exhibit extreme sensitivity to camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations
- **Language Ignorance**: Models largely ignore language instructions, functioning more like Vision-Action models
- **Negative Compositional Generalization**: Combined perturbations reveal complex interaction effects beyond independent factors

## 📊 LIBERO-plus Benchmark

### 7 Perturbation Dimensions
We introduce **LIBERO-plus**, a comprehensive benchmark with 10,030 tasks spanning:

1. **Objects Layout** - Confounding objects and target object displacement
2. **Camera Viewpoints** - Position, orientation, and field-of-view changes
3. **Robot Initial States** - Manipulator initial pose variations
4. **Language Instructions** - LLM-based instruction rewriting
5. **Light Conditions** - Intensity, direction, color, and shadow variations
6. **Background Textures** - Scene and surface appearance changes
7. **Sensor Noise** - Photometric distortions and image degradation

### Evaluated Models
- OpenVLA and variants (OFT, OFT_w, OFT_m)
- π₀ and π₀-fast
- Nora, WorldVLA, UniVLA, RIPT-VLA

## 🛠️ Installation
The usage of this project is identical to [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). Simply replace the originally installed LIBERO repository with our repository without modifying your code.

```bash
# Clone our repository
git clone https://github.com/sylvestf/LIBERO-plus.git
cd LIBERO-plus
```

If you have LIBERO installed, please uninstall or remove it first. Please verify if the repo path in the following configuration file needs to be updated to path_to_liberoplus_repo.
Here are the default paths for the configuration files: `/root/.libero/config.yaml`. You can check your `libero_config_path` at `path_to_your_LIBERO_repo/libero/libero/__init__.py`.

Then install our new LIBERO repository
```bash
# Install the new LIBERO package
pip install -e .

# New dependencies installed on top of LIBERO
apt install libexpat1
apt install libfontconfig1-dev
apt install libpython3-stdlib
apt-get install libmagickwand-dev
pip install -r extra_requirements.txt
```

Please download our assets from [LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main), including hundreds of new objects, textures, and other required assets. Please unzip the `assets.zip` file to `/LIBERO-plus/libero/libero` path. You can also find the [RLDS training dataset](https://huggingface.co/datasets/Sylvest/libero_plus_rlds/tree/main) mentioned in our paper and the [OpenVLA-OFT weights after mix-SFT](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/main) on this dataset. We also provide [LEROBOT training dataset](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) and [training dataset for each suite](https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite/tree/main).

The extracted directory structure should look like:

```text
LIBERO-plus/
└── libero/
    └── libero/
        └── assets/
            ├── articulated_objects/
            ├── new_objects/
            ├── scenes/
            ├── stable_hope_objects/
            ├── stable_scanned_objects/
            ├── textures/
            ├── turbosquid_objects/
            ├── serving_region.xml
            ├── wall_frames.stl
            └── wall.xml
```

## 🔧 Evaluation
The evaluation method is almost identical to `LIBERO`. The only required modification is adjusting `num_trials_per_task` from 50 to 1 in your configuration.

## 📊 LIBERO-Plus Benchmark Leaderboard
| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|-------|--------|-------|----------|-------|------------|-------|--------|-------|
| [OpenVLA](https://github.com/openvla/openvla) | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 17.3 |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 70.0 |
| [OpenVLA-OFT_w](https://github.com/moojink/openvla-oft) | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | 56.4 |
| [NORA](https://github.com/declare-lab/nora) | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.8 |
| [WorldVLA](https://github.com/alibaba-damo-academy/WorldVLA) | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.3 |
| [UniVLA](https://github.com/OpenDriveLab/UniVLA) | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 43.9 |
| [π₀](https://github.com/Physical-Intelligence/openpi) | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 54.6 |
| [π₀-Fast](https://github.com/Physical-Intelligence/openpi) | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 64.2 |
| [RIPT-VLA](https://github.com/Ariostgx/ript-vla) | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 69.3 |
| [OpenVLA-OFT_m](https://github.com/moojink/openvla-oft) | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 68.1 |
| **[OpenVLA-OFT+ (Ours)](https://github.com/moojink/openvla-oft)** | **92.8** | **30.3** | **85.8** | **94.9** | **93.9** | **89.3** | **77.6** | **79.6** |

- **OpenVLA-OFT+** shows the performance of [OpenVLA-OFT with a mix-sft on LIBERO-plus dataset](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/main).
- **OpenVLA-OFT_w** shows the performance of [OpenVLA-OFT without wrist observation input](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-without-wrist).
- **OpenVLA-OFT_m** shows the performance of [OpenVLA-OFT with a mix-sft](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial).

### Origin LIBERO Benchmark Leaderboard

To make it easier to get all the results in one place, we've compiled the evaluation results of current VLA models on the original LIBERO benchmark in this [table](./libero_res.md).


## Citation
If you find this work useful for your research, please cite our paper:
```bibtex
@article{fei25libero-plus,
    title={LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models},
    author={Senyu Fei and Siyin Wang and Junhao Shi and Zihao Dai and Jikun Cai and Pengfang Qian and Li Ji and Xinzhe He and Shiduo Zhang and Zhaoye Fei and Jinlan Fu and Jingjing Gong and Xipeng Qiu},
    journal = {arXiv preprint arXiv:2510.13626},
    year={2025},
}
```
