# RLinf-USER

## 🚀 Overview

<div align="center">
  <img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-HEAD.png" alt="Project Header" width="800"/>
</div>


Welcome to **RLinf-USER**, a Unified and extensible SystEm for Real-world online policy learning. In this codebase, we provide extensible abstractions for rewards, algorithms, and policies, supporting online imitation or reinforcement learning of CNN/MLP, generative policies, and large vision–language–action (VLA) models within a unified pipeline.

## ⚙️Tasks

- **Peg-Insertion**: Aligning and inserting a peg into a hole.
- **Charger Task**: Plugging a charger into a socket.
- **Pick-and-Place**: Grasping and transporting a randomly initialized object (a rubber duck) to a target container.
- **Cap Tightening**: Rotating and tightening a bottle cap to a specified pose.
- **Table Clean-up**: Cleaning cluttered objects from the tabletop into a designated box, then close the lid.

## 🧠 Algorithms

We utilize a suite of RL components optimized for real-world efficiency:

* **SAC (Soft Actor-Critic)**: Classical algorithm in realworld RL.
* **RLPD (RL with Prior Data)**: Enhances learning efficiency by incorporating prior demonstration data with high update-to-data ratios.
* **SAC Flow**: Designed for sample-efficient flow-based policy RL.
* **HG-DAgger**: An interactive imitation learning algorithm.

---

## 🛠 Hardware Setup

| Component | Specification |
| --- | --- |
| **Robotic Arm** | Franka Emika Panda |
| **Cameras** | Intel RealSense (RGB support) |
| **Computing Unit** | RTX 4090 GPU (for CNN/Flow model), A100 * 4 (for $\pi_0$ model) |
| **Robot Controller** | NUC (w/o GPU) |
| **Space Mouse** | 3D Connection SpaceMouse Compact |

---

## 📥 Installation

The setup is split between the **Controller Node** and **Training/Rollout Nodes**.

### 1. Robot Controller Node

* **Firmware**: Version `< 5.9.0` required (5.7.2 recommended).
* **OS**: Ubuntu 20.04 (required for ROS Noetic).
* **Real-time Kernel**: Recommended for performance. [Follow Franka Docs](https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html).

> ⚠️ **Warning**:
> If you have already installed ROS Noetic, libfranka, franka_ros and serl_franka_controllers manually, you can skip the installation of these packages by setting the environment variable export `SKIP_ROS=1` before running the installation script.  
> If you have skipped these installations, please make sure that you have sourced the ROS setup script (usually at `/opt/ros/noetic/setup.bash`), as well as the `franka_ros` and `serl_franka_controllers` setup scripts (usually at `<your_catkin_ws>/devel/setup.bash`) in your `~/.bashrc`. Also, make sure the libfranka shared library is in your `LD_LIBRARY_PATH` or installed in the system library path `/usr/lib`.  
> This is important every time before you start ray on the controller node to ensure that the Franka control packages can be correctly found.

```bash
# Into Repository
cd RLinf
bash requirements/install.sh embodied --env franka
source .venv/bin/activate
```

### 2. Training/Rollout Nodes

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate
```

---

## 🤖 Running the Experiment

### Step 1: Data Collection

Before training, collect initial demonstrations using a Space Mouse.

1. Configure `robot_ip` and `target_ee_pose` in `examples/embodiment/config/realworld_collect_data.yaml`.
2. Run `bash examples/embodiment/collect_data.sh`.

### Step 2: Cluster Setup

RLinf-USER uses **Ray** for distributed management. You must set environment variables on every node:

```bash
# Set unique rank for each node (0 to N-1)
export RLINF_NODE_RANK=<0 to N-1>
export PYTHONPATH=$PYTHONPATH:<path_to_RLinf>
# On Head Node
ray start --head --port=6379
# On Worker Nodes
ray start --address='<head_node_ip>:6379'
```

### Step 3: Launch Training

Execute the training script from the head node:

Example: **RLPD with single franka arm**:
```bash
bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async
```

Example: **SAC with multi franka arm**:
```bash
bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async_2arms
```

Example: **HG-DAgger with single franka arm**:
```bash
bash examples/embodiment/run_realworld_dagger_async.sh realworld_dagger_openpi_async
```

---

## 📊 Main Results

We evaluate USER on a suite of 5 real-world manipulation tasks, demonstrating its capability in unified hardware abstraction, high-throughput asynchronous learning, and support for heterogeneous policies (from CNNs to VLA models).

### 1. Robust Real-world Performance

USER supports diverse learning paradigms. Below shows the training curves for RL algorithms (RLPD, SAC, SAC-Flow) on diverse tasks, as well as the performance gain for VLA models ($\pi_0$) after online fine-tuning.

<div align="center">
<img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-main_rl.jpg" alt="RL Training Curves" width="800"/>
<figcaption><strong>RL Training Curves of Diverse Tasks & Algorithms</strong></figcaption>
</div>

**Performance on VLA Models:**
Using HG-DAgger, USER significantly improves the success rate of foundation VLA models in real-world settings with minimal interventions.

<div align="center">
<table style="text-align:center;">
<tr>
<th colspan="3" style="text-align:center;"><strong>Online Training Improvement for &pi;<sub>0</sub> </strong></th>
</tr>
<tr>
<th style="text-align:center;">Task</th>
<th style="text-align:center;">Before Online Training</th>
<th style="text-align:center;">After Online Training</th>
</tr>
<tr>
<td style="text-align:center;">Pick-and-Place</td>
<td style="text-align:center;">39/60 (65%)</td>
<td style="text-align:center;"><strong>58/60 (96.7%)</strong></td>
</tr>
<tr>
<td style="text-align:center;">Table Clean-up</td>
<td style="text-align:center;">9/20 (45%)</td>
<td style="text-align:center;"><strong>16/20 (80%)</strong></td>
</tr>
</table>
</div>

### 2. System Efficiency: Asynchronous vs. Synchronous

USER adopts a fully asynchronous pipeline that decouples data generation, training, and weight synchronization. This design significantly outperforms traditional synchronous pipelines, especially for large models.

<div align="center">
<table style="text-align:center;">
<tr>
<th colspan="4" style="text-align:center;"><strong>Profiling Results: Generation & Training Throughput</strong></th>
</tr>
<tr>
<th style="text-align:center;">Model + Algorithm</th>
<th style="text-align:center;">Pipeline Mode</th>
<th style="text-align:center;">Generation Period (s/episode) ↓</th>
<th style="text-align:center;">Training Period (s/update) ↓</th>
</tr>
<tr>
  <td rowspan="3" style="text-align:center;">
    <strong>&pi;<sub>0</sub> + HG-DAgger</strong>
  </td>
  <td style="text-align:center;">Synchronous</td>
  <td style="text-align:center;">45.07</td>
  <td style="text-align:center;">45.01</td>
</tr>
<tr>
<td style="text-align:center;"><strong>Asynchronous (USER)</strong></td>
<td style="text-align:center;"><strong>37.54</strong></td>
<td style="text-align:center;"><strong>7.90</strong></td>
</tr>
<tr>
<td style="text-align:center;"><em>Speed Up</em></td>
<td style="text-align:center;"><span style="color:green"><strong>1.20x</strong></span></td>
<td style="text-align:center;"><span style="color:green"><strong>5.70x</strong></span></td>
</tr>
<tr>
<td rowspan="3" style="text-align:center;"><strong>CNN + SAC</strong></td>
<td style="text-align:center;">Synchronous</td>
<td style="text-align:center;">20.29</td>
<td style="text-align:center;">0.64</td>
</tr>
<tr>
<td style="text-align:center;"><strong>Asynchronous (USER)</strong></td>
<td style="text-align:center;"><strong>13.11</strong></td>
<td style="text-align:center;"><strong>0.14</strong></td>
</tr>
<tr>
<td style="text-align:center;"><em>Speed Up</em></td>
<td style="text-align:center;"><span style="color:green"><strong>1.55x</strong></span></td>
<td style="text-align:center;"><span style="color:green"><strong>4.61x</strong></span></td>
</tr>
</table>
</div>

### 3. Multi-Robot & Heterogeneous Support

With the Unified Hardware Abstraction Layer, USER treats robots as first-class resources, enabling:

* **Parallel Training**: Training policies on multiple robots simultaneously (e.g., 2x Franka arms) to scale data collection.
* **Heterogeneous Training**: Training a unified policy across different robot embodiments (e.g., Franka 7-DoF + ARX 6-DoF).

<div align="center">
<table border="0">
<tr>
<td align="center">
<img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-multi.jpg" alt="Multi-Robot Training" width="350"/>
<strong>Parallel Training (2x Franka)</strong>
</td>
<td align="center">
<img src="https://github.com/RLinf/misc/raw/main/pic/USER/USER-hetero.jpg" alt="Heterogeneous Training" width="350"/>
<strong>Heterogeneous (Franka + ARX)</strong>
</td>
</tr>
</table>
</div>

Under all multi/heterogeneous settings, USER achieves full convergence of the policy within comparable time, indicating USER's ability to effectively scale real-world policy learning.

## 📈 Visualization & Monitoring

Ready to witness your robots getting smarter? Simply launch **Tensorboard** and watch the curves go up 🚀:

```bash
tensorboard --logdir ./logs
```

Dive in and happy training! ✨