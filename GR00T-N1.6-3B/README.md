---
datasets:
- nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
tags:
- robotics
---

<div align="center">
  <a href="https://github.com/NVIDIA/Isaac-GR00T">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/8bFQa2ZIGCsOQQ2ho2N_U.png">
  </a>
  <div align="center">
  <a href="https://github.com/NVIDIA/Isaac-GR00T">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub Badge">
  </a>
  <a href="https://developer.nvidia.com/isaac/gr00t">
    <img src="https://img.shields.io/badge/Website-green" alt="Website Badge">
  </a>
  <!-- <a href=""">
    <img src="https://img.shields.io/badge/Project%20Page-blue?style=plastic" alt="Project Page Badge">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Research_Blog-black?style=flat" alt="Research Blog Badge">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Dataset-Overview-brightgreen?logo=googleforms" alt="Research Blog Badge">
  </a>
  -->
</div>
</div>

# GR00T-N1.6-3B

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/67b8da81d01134f89899b4a7/ZCLLXZk2LQBG0YH_BmiIN.gif"
       style="width:100%; max-width:1000px; height:auto;">
</p>

## Description:

NVIDIA Isaac GR00T N1.6 is an open vision-language-action (VLA) model for generalized humanoid robot skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.
GR00T N1.6 is trained on a diverse mixture of robot data including bimanual, semi-humanoid and an expansive humanoid dataset, consisting of real captured data, synthetic data generated using the components of NVIDIA Isaac GR00T Blueprint. It is adaptable through post-training for specific embodiments, tasks and environments.

The neural network architecture of GR00T N1.6 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. 

## License/Terms of Use
[Nvidia License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyIsIm5jaWQiOiJzby15b3V0LTg3MTcwMS12dDQ4In0=)<br>
You are responsible for ensuring that your use of NVIDIA AI Foundation Models complies with all applicable laws. <br>

### Deployment Geography:
Global

### Use Case:
Researchers, Academics, Open-Source Community: AI-driven robotics research and algorithm development.
Developers: Integrate and customize AI for various robotic applications.
Startups & Companies: Accelerate robotics development and reduce training costs.

## Reference(s):
Eagle VLM: Chen, Guo, et al. "Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models." arXiv:2504.15271 (2025).<br>
Liu, Xingchao, and Chengyue Gong. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." The Eleventh International Conference on Learning Representations”.<br>
Flow Matching Policy:
Black, Kevin, et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv preprint arXiv:2410.24164 (2024).<br>

## Model Architecture:
**Architecture Type:** Vision Transformer, Multilayer Perceptron, Flow matching Transformer

GR00T N1.6 uses vision and text transformers to encode the robot's image observations and text instructions. The architecture handles a varying number of views per embodiment by concatenating image token embeddings from all frames into a sequence, followed by language token embeddings. 

To model proprioception and a sequence of actions conditioned on observations, GR00T N1.6 uses a flow matching transformer. The flow matching transformer interleaves self-attention over proprioception and actions with cross-attention to the vision and language embeddings. During training, the input actions are corrupted by randomly interpolating between the clean action vector and a gaussian noise vector. At inference time, the policy first samples a gaussian noise vector and iteratively reconstructs a continuous-value action using its velocity prediction.

In GR00T N1.6, the MLP connector between the vision-language features and the diffusion-transformer (DiT) has been modified for improved performance on our sim benchmarks. Also, it was trained jointly with flow matching and world-modeling objectives.

**Network Architecture:** 
![image/png](https://github.com/NVIDIA/Isaac-GR00T/blob/main/media/model-architecture.png?raw=true)
The schematic diagram is shown in the illustration above.
Red, Green, Blue (RGB) camera frames are processed through a pre-trained vision transformer (SigLip2).
Text is encoded by a pre-trained transformer (T5)
Robot proprioception is encoded using a multi-layer perceptron (MLP) indexed by the embodiment ID. To handle variable-dimension proprio, inputs are padded to a configurable max length before feeding into the MLP.
Actions are encoded and velocity predictions decoded by an MLP, one per unique embodiment.
The flow matching transformer is implemented as a diffusion transformer (DiT), in which the diffusion step conditioning is implemented using adaptive layernorm (AdaLN).

## Input:
**Input Type:**
* Vision: Image Frames<br>
* State: Robot Proprioception<br>
* Language Instruction: Text<br>

**Input Format:**
* Vision: Variable number of image frames from robot cameras<br>
* State: Floating Point<br>
* Language Instruction: String<br>

**Input Parameters:**
* Vision: 2D - RGB image, any resolution<br>
* State: 1D - Floating number vector<br>
* Language Instruction: 1D - String<br>

## Output:
**Output Type(s):** Actions<br>
**Output Format** Continuous-value vectors<br>
**Output Parameters:** [Two-Dimensional (2D)] <br>
**Other Properties Related to Output:** Continuous-value vectors correspond to different motor controls on a robot, which depends on Degrees of Freedom of the robot embodiment.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):** PyTorch

**Supported Hardware Microarchitecture Compatibility:** 
All of the below:
* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Jetson
* NVIDIA Hopper
* NVIDIA Lovelace

**[Preferred/Supported] Operating System(s):**
* Linux

## Model Version(s):
Version 1.6

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

For more detailed information on ethical considerations for this model, please see the Model Card++ [Explainability](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/EXPLAINABILITY.md), [Bias](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/BIAS.md), [Safety & Security](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/SAFETY_and_SECURITY.md)), and [Privacy](https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/PRIVACY.md) Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Resources

* Previous Version: https://huggingface.co/nvidia/GR00T-N1.5-3B<br>
* Research Blog: https://research.nvidia.com/labs/gear/gr00t-n1_6/
* GR00T Website: https://developer.nvidia.com/isaac/gr00t
