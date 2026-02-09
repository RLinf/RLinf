<h1 align="center">RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training</h1>

<div align="center">

[**📄 Paper**](https://arxiv.org/abs/2510.06710) | [**💻 Code**](https://github.com/RLinf/RLinf/tree/main/examples/rlinf-vla) | [**🤗 Models(RLinf-OpenVLA)**](https://huggingface.co/collections/RLinf/openvla) | [**🤗 Models(RLinf-OpenVLAOFT)**](https://huggingface.co/collections/RLinf/openvla-oft)

</div>


## 📝 Overview

![overview](https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/rlinf_vla_overview.png)


We introduce RLinf-VLA, a unified and efficient framework for scalable RL training of VLA models.
RLinf-VLA achieves unification by providing a unified interface that standardizes the integration of diverse VLA architectures, multiple RL algorithms, and heterogeneous simulators, enabling extensibility. 
To ensure efficiency, the system adopts a flexible resource allocation architecture for rendering, inference, and training workloads in RL pipelines. 
In particular, for GPU-parallelized simulators, RLinf-VLA introduces a hybrid fine-grained pipeline allocation strategy, yielding a 1.61x–1.88x training speedup.
Using this unified system, models trained with RLinf-VLA demonstrate consistent performance improvements of approximately 20–85\% across multiple simulation benchmarks, including LIBERO, ManiSkill, and RoboTwin. 
Furthermore, we distill a set of training practices for effective RL-based VLA training. 
We position RLinf-VLA as a foundational system to enable efficient, unified, and reproducible research in embodied intelligence.


## 🏆 Results

<div align="center">
<table border="0">
  <tr>
    <td align="center">
      <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/mani_openvla.png" alt="mani_openvla" width="350"/>
      <br/>
      <strong>OpenVLA</strong>
    </td>
    <td align="center">
      <img src="https://github.com/RLinf/misc/raw/main/pic/rlinf-vla/mani_openvlaoft.png" alt="mani_openvlaoft" width="350"/>
      <br/>
      <strong>OpenVLA-OFT</strong>
    </td>
  </tr>
</table>
</div>

- Training curves on ManiSkill “PutOnPlateInScene25Mani-v3” with OpenVLA and
OpenVLA-OFT models, using PPO and GRPO algorithms. PPO consistently outperforms GRPO
and exhibits greater stability.

<div align="center">
<table style="text-align:center;">
  <tr>
    <th colspan="6" style="text-align:center;"> <strong>Evaluation results on ManiSkill. Values denote success rates</strong></th>
  </tr>
  <tr>
    <td style="text-align:center;"></td>
    <th rowspan="2" colspan="1" style="text-align:center;">In-Distribution</th>
    <td colspan="4" style="text-align:center;"><strong>Out-Of-Distribution</strong></td>
  
  </tr>
  <tr>
    <th style="text-align:center;"></th>
    <th style="text-align:center;">Vision</th>
    <th style="text-align:center;">Semantic</th>
    <th style="text-align:center;">Execution</th>
    <th style="text-align:center;">Avg.</th>
  </tr>
  <tr>
    <td style="text-align:center;">OpenVLA (Base)</td>
    <td style="text-align:center;">53.91%</td>
    <td style="text-align:center;">38.75%</td>
    <td style="text-align:center;">35.94%</td>
    <td style="text-align:center;">42.11%</td>
    <td style="text-align:center;">39.10%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/gen-robot/openvla-7b-rlvla-rl"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">RL4VLA (PPO)</a></td>
    <td style="text-align:center;">93.75%</td>
    <td style="text-align:center;">80.47%</td>
    <td style="text-align:center;">75.00%</td>
    <td style="text-align:center;">81.77%</td>
    <td style="text-align:center;">79.15%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-GRPO-ManiSkill3-25ood"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA (RLinf-GRPO)</a></td>
    <td style="text-align:center;">84.38%</td>
    <td style="text-align:center;">74.69%</td>
    <td style="text-align:center;">72.99%</td>
    <td style="text-align:center;">77.86%</td>
    <td style="text-align:center;">75.15%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLA-PPO-ManiSkill3-25ood"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA (RLinf-PPO)</a></td>
    <td style="text-align:center;"><strong>96.09%</strong></td>
    <td style="text-align:center;">82.03%</td>
    <td style="text-align:center;"><strong>78.35%</strong></td>
    <td style="text-align:center;"><strong>85.42%</strong></td>
    <td style="text-align:center;"><strong>81.93%</strong></td>
  </tr>
  <tr>
    <th colspan="6" style="text-align:center;"></th>
  </tr>
  <tr>
    <td style="text-align:center;">OpenVLA-OFT (Base)</td>
    <td style="text-align:center;">28.13%</td>
    <td style="text-align:center;">27.73%</td>
    <td style="text-align:center;">12.95%</td>
    <td style="text-align:center;">11.72%</td>
    <td style="text-align:center;">18.29%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</a></td>
    <td style="text-align:center;">94.14%</td>
    <td style="text-align:center;">84.69%</td>
    <td style="text-align:center;">45.54%</td>
    <td style="text-align:center;">44.66%</td>
    <td style="text-align:center;">60.64%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-PPO-ManiSkill3-25ood"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-PPO)</a></td>
    <td style="text-align:center;"><strong>97.66%</strong></td>
    <td style="text-align:center;"><strong>92.11%</strong></td>
    <td style="text-align:center;">64.84%</td>
    <td style="text-align:center;">73.57%</td>
    <td style="text-align:center;">77.05%</td>
  </tr>
</table>
</div>


<div align="center">
<table style="text-align:center;">
  <tr>
    <th colspan="7" style="text-align:center;"><strong>Evaluation results of the unified model on the five LIBERO task groups</strong></th>
  </tr>
  <tr>
    <th style="text-align:center;">Model</th>
    <th style="text-align:center;">Spatial</th>
    <th style="text-align:center;">Object</th>
    <th style="text-align:center;">Goal</th>
    <th style="text-align:center;">Long</th>
    <th style="text-align:center;">90</th>
    <th style="text-align:center;">Avg.</th>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (Base)</a></td>
    <td style="text-align:center;">72.18%</td>
    <td style="text-align:center;">71.48%</td>
    <td style="text-align:center;">64.06%</td>
    <td style="text-align:center;">48.44%</td>
    <td style="text-align:center;">70.97%</td>
    <td style="text-align:center;">65.43%</td>
  </tr>
  <tr>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</a></td>
    <td style="text-align:center;"><strong>99.40%</strong></td>
    <td style="text-align:center;"><strong>99.80%</strong></td>
    <td style="text-align:center;"><strong>98.79%</strong></td>
    <td style="text-align:center;"><strong>93.95%</strong></td>
    <td style="text-align:center;"><strong>98.59%</strong></td>
    <td style="text-align:center;"><strong>98.11%</strong></td>
  </tr>
  <tr>
    <td style="text-align:center;">Δ Improvement</td>
    <td style="text-align:center;">+27.22</td>
    <td style="text-align:center;">+28.32</td>
    <td style="text-align:center;">+34.73</td>
    <td style="text-align:center;">+45.51</td>
    <td style="text-align:center;">+27.62</td>
    <td style="text-align:center;">+32.68</td>
  </tr>
</table>
</div>

<div align="center">
<table style="text-align:center;">
  <tr>
    <th colspan="9" style="text-align:center;"><strong>Evaluation results of OpenVLA-OFT models on six RoboTwin tasks</strong></th>
  </tr>
  <tr>
    <th style="text-align:center;">Model</th>
    <th style="text-align:center;">beat_block_hammer</th>
    <th style="text-align:center;">pick_dual_bottles</th>
    <th style="text-align:center;">place_empty_cup</th>
    <th style="text-align:center;">move_can_pot</th>
    <th style="text-align:center;">lift_pot</th>
    <th style="text-align:center;">handover_block</th>
    <th style="text-align:center;">Average</th>
    <th style="text-align:center;">Δ Avg.</th>
  </tr>
  <tr>
    <td style="text-align:center;"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (SFT)</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer">10.15%</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles">20.31%</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup">75.78%</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot">9.37%</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot">3.13%</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block">28.13%</a></td>
    <td style="text-align:center;">24.48%</td>
    <td style="text-align:center;">---</td>
  </tr>
  <tr>
    <td style="text-align:center;"><img src="https://github.com/RLinf/misc/raw/main/pic/hf-logo.svg" alt="HF" width="16" height="16" style="vertical-align: middle;">OpenVLA-OFT (RLinf-GRPO)</a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-beat_block_hammer"><strong>96.09%</strong></a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-pick_dual_bottles"><strong>92.96%</strong></a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup"><strong>94.53%</strong></a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-move_can_pot"><strong>83.59%</strong></a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-lift_pot"><strong>70.31%</strong></a></td>
    <td style="text-align:center;"><a href="https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-handover_block"><strong>70.31%</strong></a></td>
    <td style="text-align:center;"><strong>84.63%</strong></td>
    <td style="text-align:center;"><strong>+60.15%</strong></td>
  </tr>
</table>
</div>

> **Note:** "Base" and "SFT" both refer to supervised fine-tuned models before RL training.

## 📦 QuickStart

[Example for ManiSkill](https://rlinf.readthedocs.io/en/latest/rst_source/examples/maniskill.html).

[Example for Libero](https://rlinf.readthedocs.io/en/latest/rst_source/examples/libero.html).

[Example for RoboTwin](https://rlinf.readthedocs.io/en/latest/rst_source/examples/robotwin.html).

More examples can be found in [example gallery](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html).

## 📚 Citation
```
@article{zang2025rlinf,
  title   = {Rlinf-vla: A unified and efficient framework for vla+ rl training},
  author  = {Zang, Hongzhi and Wei, Mingjie and Xu, Si and Wu, Yongji and Guo, Zhen and Wang, Yuanqing and Lin, Hao and Shi, Liangzhi and Xie, Yuqing and Xu, Zhexuan and others},
  journal = {arXiv preprint arXiv:2510.06710},
  year    = {2025}
}
```