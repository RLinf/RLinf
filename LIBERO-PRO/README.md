<div align="center">

<img src="https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images/liberopro_logo.png" width="360">

## LIBERO-Pro: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization

**Xueyang Zhou<sup>1</sup>, Yangming Xu<sup>1</sup>, Guiyao Tie<sup>1</sup>, Yongchao Chen<sup>2,3</sup>, Guowen Zhang<sup>1</sup>, Duanfeng Chu<sup>4</sup>, Pan Zhou<sup>1</sup>, Lichao Sun<sup>5</sup>**

<sup>1</sup> **Huazhong University of Science and Technology**  
<sup>2</sup> **Harvard University** <sup>3</sup> **Massachusetts Institute of Technology**  
<sup>4</sup> **Wuhan University of Technology** <sup>5</sup> **Lehigh University**

[![Tests Passing](https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg)](https://github.com/Zxy-MLlab/LIBERO-PRO/actions)
[![Contributors](https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO)](https://github.com/Zxy-MLlab/LIBERO-PRO/graphs/contributors)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2510.03827-b31b1b.svg)](https://arxiv.org/pdf/2510.03827)
[![Website](https://img.shields.io/badge/Project-Webpage-4b7bec.svg)](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/)
[![Code](https://img.shields.io/badge/Code-LIBERO--PRO-2b9348.svg)](https://github.com/Zxy-MLlab/LIBERO-PRO/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


📄 [Paper](https://arxiv.org/pdf/2510.03827) |
💻 [Code](https://github.com/Zxy-MLlab/LIBERO-PRO) |
🌐 [Webpage](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/) |
🤗 [Dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) |
📱 [XHS](http://xhslink.com/o/5vmaip7wQCE/) |
💬 [WeChat](https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images/wechat.png)

---

![pull_figure](https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images/overall.png)
</div>

## ✨ News ✨​
- [2025/11/05] 📊 All bddl and init files have been uploaded to Huggingface (supports fast parallel evaluation): [Dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)
- [2025/10/29] 🌐 We launched the official project website for LIBERO-Pro (with more demos & details): [Webpage](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/)
- [2025/10/22] 📱 We have shared a project promotion post on Xhs: [Xhs](http://xhslink.com/o/5vmaip7wQCE/)
- [2025/10/20] 💬 We have created an official WeChat account (join discussions, get quick Q&A) [WeChat](https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images/wechat.png)​
- [2025/10/05] 🤖 We have released the full LIBERO-Pro code on GitHub: [Code​](https://github.com/Zxy-MLlab/LIBERO-PRO)
- [2025/10/04] 🎉 Our paper, LIBERO-Pro: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization is now available on arXiv: [Paper](https://arxiv.org/pdf/2510.03827)

## 🌟 Follow Us
We are committed to continuously improving **LIBERO-Pro** based on your feedback. Our goal is to establish a fair and simple evaluation environment for Vision-Language-Action (VLA) models. Your input is invaluable in helping us achieve this goal!

---

## 🔍 Motivation
Recent VLA models have demonstrated impressive performance on known tasks; however, our observations suggest that such success largely stems from mechanical memorization of training scenarios rather than genuine acquisition of transferable task-solving strategies.


| **Model** | **Goal** | **P1** | **P2** | **Spatial** | **P1** | **P2** | **10** | **P1** | **P2** | **Object** | **P1** | **P2** |
|:----------|:--------------------:|:----------------:|:--------------------:|:-----------------------:|:------------------:|:-----------------------:|:------------------:|:---------------:|:-------------------:|:----------------------:|:------------------:|:----------------------:|
| **OpenVLA** | ![](https://img.shields.io/badge/0.98-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.95-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.93-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.99-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) |
| **Pi0** | ![](https://img.shields.io/badge/0.92-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.90-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.82-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.98-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) |
| **Pi0.5** | ![](https://img.shields.io/badge/0.97-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.4-green) | ![](https://img.shields.io/badge/0.96-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.2-green) | ![](https://img.shields.io/badge/0.93-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.1-green) | ![](https://img.shields.io/badge/0.98-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.2-green) |
| **UniVLA** | ![](https://img.shields.io/badge/0.89-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.1-green) | ![](https://img.shields.io/badge/0.85-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.1-green) | ![](https://img.shields.io/badge/0.61-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) | ![](https://img.shields.io/badge/0.98-blue) | ![](https://img.shields.io/badge/0.0-orange) | ![](https://img.shields.io/badge/0.0-green) |

> 🟦 **Original** 🟧 **+ P1: Task Perturbation** 🟩 **+ P2: Position Perturbation**  
> 📉 *All models achieve >0.9 on original LIBERO tasks but collapse under LIBERO-PRO perturbations, showing poor true generalization.*

---
## 🌍 Fairer Environment
**LIBERO-Pro** calls for a more *rigorous, standardized, and transparent* approach to measuring generalization, helping the community move **beyond memorization and toward true understanding**.


## ⚙️ Five Core Generalization Dimensions
<div align="center">

<table>
<thead>
<tr>
<th align="center">Dimension</th>
<th align="center">Description</th>
<th align="center">Example Evaluation</th>
</tr>
</thead>
<tbody>

<tr>
<td align="center"><b>Object</b></td>
<td align="center">Modifies object appearance, color, and scale to test adaptability to visual shifts.</td>
<td align="center">"red cup" → "yellow cup"</td>
</tr>

<tr>
<td align="center"><b>Position</b></td>
<td align="center">Relocates objects within feasible spatial bounds to evaluate the model’s adaptability to spatial position changes.</td>
<td align="center">Change the position of "cup" and "bowl"</td>
</tr>

<tr>
<td align="center"><b>Semantic</b></td>
<td align="center">Paraphrases natural language commands to probe linguistic robustness.</td>
<td align="center">"Grasp the mug" → "Pick up the cup"</td>
</tr>

<tr>
<td align="center"><b>Task</b></td>
<td align="center">Redefines task logic and target states to test procedural generalization.</td>
<td align="center">"Pick up the mug" → "Pick up the butter"</td>
</tr>

<tr>
<td align="center"><b>Environment</b></td>
<td align="center">Replaces working environments to evaluate cross-environment robustness.</td>
<td align="center">"Main table" → "Kitchen table"</td>
</tr>

</tbody>
</table>

</div>


> 🧩 These perturbations are **combinable and configurable** via YAML for scalable and controlled generalization studies.

---

**Welcome to join our wechat discussion group, we will answer any questions in real time, and also welcome more in-depth academic discussion.**

<img src="https://github.com/Zxy-MLlab/LIBERO-OOD/blob/master/images/wechat.png" width="300">

---

# Contents
- [Installation](#Installation)
- [Evaluation](#libero-pro-evaluation)
  - [OpenVLA](#evaluation-on-openvla)
  - [LeaderBoard](#-libero-pro-model-leaderboard)
- [Position Evaluation](#initial-position-perturbation-experiment)
- [Citation](#Citation)
- [License](#License)

---

# Installation

Clone the official LIBERO-PRO repository by run:
```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO/
```

LIBERO-PRO is developed based on the original LIBERO benchmark, so it uses the same runtime environment as LIBERO—no separate environment configuration for LIBERO-PRO is needed. You only need to install the environment in accordance with LIBERO’s official requirements, as shown below:

```bash
conda create -n libero_pro python=3.8.13
conda activate libero_pro
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git
cd LIBERO-PRO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```


---

# LIBERO-Pro Evaluation

## ⚡️ Quick Start
Follow the steps below to quickly set up and run **LIBERO-Pro** for your own evaluations.

💡 **Note:**  
To enable stable and fast parallel evaluation, we updated `libero/libero/benchmark/__init__.py` and `libero/libero/benchmark/libero_suite_task_map.py`. If you cloned the repo **before 2025/11/05**, please re-download and replace these two files.

### 1️⃣ Download Required Files
First, download all `bddl_files` and `init_files` from our official Huggingface dataset: 👉 [LIBERO-Pro Dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)

### 2️⃣ Move Files into LIBERO-Pro Structure
Move the downloaded files into the correct LIBERO-Pro directory structure:
```
mv libero_data/bddl_files/* libero/libero/bddl_files/
mv libero_data/init_files/* libero/libero/init_files/
```

## 3️⃣ Configure Evaluation Settings
All evaluation parameters can be set in the file:
```
evaluation_config.yaml
```

In the this evaluation mode, only one perturbation type can be active at a time. To specify the desired perturbation, modify the corresponding field in the config file:
```
use_swap: false
use_object: false
use_language: false
use_task: true
```

## Custom Evaluation (Optional)
To specify combined-type generalization evaluation, modify `evaluation_config.yaml` in your project directory.  

| Parameter | Function |
| ----------------- | -------------------------------------------------------------------------------------- |
| use_environment | Enable/disable environment generalization evaluation |
| use_swap | Enable/disable position generalization evaluation |
| use_object | Enable/disable object generalization evaluation |
| use_language | Enable/disable semantic (language) generalization evaluation |
| use_task | Enable/disable task generalization evaluation |

*Note: task generalization (`use_task: true`) cannot be combined with others.*

## Evaluation on OpenVLA
Below is a reference code snippet for conducting LIBERO-PRO generalization evaluation on OpenVLA. Please place LIBERO-PRO in the following directory:
```
# 📁 openvla-oft-main
.
├── .idea/
├── experiments/
│   └── robot/
│       ├── aloha/
│       └── libero/
│           ├── experiments/
│           ├── LIBERO-PRO/ # our project
│           ├── libero_utils.py
│           ├── regenerate_libero_dataset.py
│           ├── run_libero_eval.py
│           ├── sample_libero_spatial_observation.pkl
│           ├── openvla_utils.py
│           └── robot_utils.py
```
Before evaluating, modify the `run_libero_eval.py` to adapt to LIBERO-RPO:
```
from LIBERO-PRO import perturbation

# Register for temporary evaluation tasks
class TaskSuite(str, Enum):
  ...
  LIBERO_GOAL_TEMP = "libero_goal_temp"
  LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
  LIBERO_10_TEMP = "libero_10_temp"
  LIBERO_OBJECT_TEMP = "libero_object_temp"
  LIBERO_GOAL_LAN = "libero_goal_lan"
  LIBERO_SPATIAL_LAN = "libero_spatial_lan"
  LIBERO_10_LAN = "libero_10_lan"
  LIBERO_OBJECT_LAN = "libero_object_lan"
  LIBERO_GOAL_OBJECT = "libero_goal_object"
  LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
  LIBERO_10_OBJECT = "libero_10_object"
  LIBERO_OBJECT_OBJECT = "libero_object_object"
  LIBERO_GOAL_SWAP = "libero_goal_swap"
  LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
  LIBERO_10_SWAP = "libero_10_swap"
  LIBERO_OBJECT_SWAP = "libero_object_swap"
  LIBERO_GOAL_TASK = "libero_goal_task"
  LIBERO_SPATIAL_TASK = "libero_spatial_task"
  LIBERO_10_TASK = "libero_10_task"
  LIBERO_OBJECT_TASK = "libero_object_task"
  LIBERO_GOAL_ENV = "libero_goal_env"
  LIBERO_SPATIAL_ENV = "libero_spatial_env"
  LIBERO_10_ENV = "libero_10_env"
  LIBERO_OBJECT_ENV = "libero_object_env"

TASK_MAX_STEPS = {
  ...
  TaskSuite.LIBERO_GOAL_TEMP: 300,
  TaskSuite.LIBERO_SPATIAL_TEMP: 220,
  TaskSuite.LIBERO_10_TEMP: 520,
  TaskSuite.LIBERO_OBJECT_TEMP: 280,
  TaskSuite.LIBERO_GOAL_LAN: 300,
  TaskSuite.LIBERO_SPATIAL_LAN: 220,
  TaskSuite.LIBERO_10_LAN: 520,
  TaskSuite.LIBERO_OBJECT_LAN: 280,
  TaskSuite.LIBERO_GOAL_OBJECT: 300,
  TaskSuite.LIBERO_SPATIAL_OBJECT: 220,
  TaskSuite.LIBERO_10_OBJECT: 520,
  TaskSuite.LIBERO_OBJECT_OBJECT: 280,
  TaskSuite.LIBERO_GOAL_SWAP: 300,
  TaskSuite.LIBERO_SPATIAL_SWAP: 220,
  TaskSuite.LIBERO_10_SWAP: 520,
  TaskSuite.LIBERO_OBJECT_SWAP: 280,
  TaskSuite.LIBERO_GOAL_TASK: 300,
  TaskSuite.LIBERO_SPATIAL_TASK: 220,
  TaskSuite.LIBERO_10_TASK: 520,
  TaskSuite.LIBERO_OBJECT_TASK: 280,
  TaskSuite.LIBERO_GOAL_ENV: 300,
  TaskSuite.LIBERO_SPATIAL_ENV: 220,
  TaskSuite.LIBERO_10_ENV: 520,
  TaskSuite.LIBERO_OBJECT_ENV: 280,
}

# Modify this line
def check_unnorm_key(cfg: GenerateConfig, model) -> None:
  ...
  unnorm_key = cfg.unnorm_key
  ...

# Modify this line
def eval_libero(cfg: GenerateConfig) -> float:
  ...
      with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f)

    evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
    evaluation_cfg["task_suite_name"] = cfg.task_suite_name

    use_swap = evaluation_cfg.get("use_swap", False)
    use_object = evaluation_cfg.get("use_object", False)
    use_language = evaluation_cfg.get("use_language", False)
    use_task = evaluation_cfg.get("use_task", False)
    use_environment = evaluation_cfg.get("use_environment", False)

    # Step 1: Check if only one of the use_xxx flags is True
    if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
        # If more than one flag is True, use the temp environment
        bddl_file_path = evaluation_cfg.get("bddl_files_path", "") + cfg.task_suite_name + "_temp/"

        init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_temp/"

        # Check if the directories exist and the log.txt file contents match
        if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
            # If directories don't exist, create them and the log.txt file
            os.makedirs(init_file_path, exist_ok=True)
            os.makedirs(bddl_file_path, exist_ok=True)

            # Create the log.txt dynamically based on current flag values
            log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
            with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                log_file.write(log_content)  # Write the dynamic state to the log file

            perturbation.create_env(configs=evaluation_cfg)
        else:
            # If directories exist, check the contents of the log.txt file
            with open(os.path.join(bddl_file_path, "log.txt"), "r") as log_file:
                log_contents = log_file.read().strip()

            # Define the expected log content based on the current flags
            expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

            # If the log contents don't match, clean up and recreate the environment
            if log_contents != expected_log:
                # Remove existing files in both directories
                for folder in [bddl_file_path, init_file_path]:
                    for root, dirs, files in os.walk(folder, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                # Create the environment again
                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                # Write the updated log content based on current flags
                with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                    log_file.write(expected_log)  # Write the updated log

                perturbation.create_env(configs=evaluation_cfg)

        # Update task_suite_name with "_temp" suffix
        cfg.task_suite_name = cfg.task_suite_name + "_temp"

    # Step 2: Handle the case when only one use_xxx flag is True
    else:
        if use_swap:
            perturb_key = "use_swap"
        elif use_object:
            perturb_key = "use_object"
        elif use_language:
            perturb_key = "use_language"
        elif use_task:
            perturb_key = "use_task"
        elif use_environment:
            perturb_key = "use_environment"

        init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_" + evaluation_cfg.get(
            "perturbation_mapping", {}).get(perturb_key, "")

        if not os.path.exists(init_file_path):
            perturbation.create_env(configs=evaluation_cfg)

        cfg.task_suite_name = cfg.task_suite_name + "_" + evaluation_cfg.get("perturbation_mapping", {}).get(perturb_key, "")
  ...
```


Note!!! For unknown reasons, in some cases replacing the environment will cause the objects on the table to move randomly. After many tests, replacing the environment with 'main_table' works and we are actively in contact with the authors of LIBERO to fix this issue.


## 🏆 LIBERO-Pro Model Leaderboard

The following table summarizes model performance under **five generalization perturbations** in LIBERO-Pro. Each cell represents the normalized success rate (**0.00–1.00**).

<table align="center">
<thead>
<tr>
<th rowspan="2" align="left">Model</th>
<th colspan="5" align="center">LIBERO-Goal</th>
<th colspan="5" align="center">LIBERO-Spatial</th>
<th colspan="5" align="center">LIBERO-10</th>
<th colspan="5" align="center">LIBERO-Object</th>
<th rowspan="2" align="center">Total</th>
</tr>
<tr>
<th align="center">Obj</th>
<th align="center">Pos</th>
<th align="center">Sem</th>
<th align="center">Task</th>
<th align="center">Env</th>
<th align="center">Obj</th>
<th align="center">Pos</th>
<th align="center">Sem</th>
<th align="center">Task</th>
<th align="center">Env</th>
<th align="center">Obj</th>
<th align="center">Pos</th>
<th align="center">Sem</th>
<th align="center">Task</th>
<th align="center">Env</th>
<th align="center">Obj</th>
<th align="center">Pos</th>
<th align="center">Sem</th>
<th align="center">Task</th>
<th align="center">Env</th>
</tr>
</thead>
<tbody>

<tr>
<td><b>OpenVLA</b></td>
<td align="center" style="background-color:#f08080;">0.96</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffd6cc;">0.98</td>
<td align="center" style="background-color:#ff9999;">0.00</td>
<td align="center" style="background-color:#ffc2b3;">0.98</td>
<td align="center" style="background-color:#f08080;">0.97</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffd6cc;">0.97</td>
<td align="center" style="background-color:#ff9999;">0.00</td>
<td align="center" style="background-color:#ffc2b3;">0.89</td>
<td align="center" style="background-color:#f08080;">0.81</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffd6cc;">0.96</td>
<td align="center" style="background-color:#ff9999;">0.00</td>
<td align="center" style="background-color:#ffc2b3;">0.85</td>
<td align="center" style="background-color:#f08080;">0.98</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffd6cc;">0.98</td>
<td align="center" style="background-color:#ff9999;">0.00</td>
<td align="center" style="background-color:#ffc2b3;">0.00</td>
<td align="center" style="background-color:#e8f0fa;"><b>0.52</b></td>
</tr>

<tr>
<td><b>Pi0</b></td>
<td align="center" style="background-color:#f28c8c;">0.94</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffddcc;">0.93</td>
<td align="center" style="background-color:#ffb3b3;">0.00</td>
<td align="center" style="background-color:#ffc6b3;">0.39</td>
<td align="center" style="background-color:#f28c8c;">0.95</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffddcc;">0.97</td>
<td align="center" style="background-color:#ffb3b3;">0.00</td>
<td align="center" style="background-color:#ffc6b3;">0.60</td>
<td align="center" style="background-color:#f28c8c;">0.79</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffddcc;">0.82</td>
<td align="center" style="background-color:#ffb3b3;">0.00</td>
<td align="center" style="background-color:#ffc6b3;">0.27</td>
<td align="center" style="background-color:#f28c8c;">0.94</td>
<td align="center" style="background-color:#ffcccc;">0.00</td>
<td align="center" style="background-color:#ffddcc;">0.90</td>
<td align="center" style="background-color:#ffb3b3;">0.00</td>
<td align="center" style="background-color:#ffc6b3;">0.29</td>
<td align="center" style="background-color:#e8f0fa;"><b>0.44</b></td>
</tr>

<tr>
<td><b>Pi0.5</b></td>
<td align="center" style="background-color:#f29c9c;">0.97</td>
<td align="center" style="background-color:#ffdddd;">0.38</td>
<td align="center" style="background-color:#ffe6cc;">0.97</td>
<td align="center" style="background-color:#ffc6b3;">0.00</td>
<td align="center" style="background-color:#ffd9b3;">0.46</td>
<td align="center" style="background-color:#f29c9c;">0.97</td>
<td align="center" style="background-color:#ffdddd;">0.20</td>
<td align="center" style="background-color:#ffe6cc;">0.97</td>
<td align="center" style="background-color:#ffc6b3;">0.01</td>
<td align="center" style="background-color:#ffd9b3;">0.46</td>
<td align="center" style="background-color:#f29c9c;">0.92</td>
<td align="center" style="background-color:#ffdddd;">0.08</td>
<td align="center" style="background-color:#ffe6cc;">0.93</td>
<td align="center" style="background-color:#ffc6b3;">0.01</td>
<td align="center" style="background-color:#ffd9b3;">0.46</td>
<td align="center" style="background-color:#f29c9c;">0.98</td>
<td align="center" style="background-color:#ffdddd;">0.17</td>
<td align="center" style="background-color:#ffe6cc;">0.96</td>
<td align="center" style="background-color:#ffc6b3;">0.01</td>
<td align="center" style="background-color:#ffd9b3;">0.73</td>
<td align="center" style="background-color:#e0f5e0;"><b>0.53</b></td>
</tr>

</tbody>
</table>

✅ *We will continue to expand the LIBERO-PRO leaderboard with new model evaluations. Researchers are warmly invited to use LIBERO-PRO to assess their Vision-Language-Action (VLA) models and share the results with us for inclusion in the official online leaderboard.*

# Initial Position Perturbation Experiment

This guide provides a step-by-step procedure for reproducing the **Object Position Perturbation Evaluation** and replicating the results shown in **Figure 6** of the paper.

> 💡 We have pre-packaged all necessary `.init` and `.bddl` files required for evaluation. You can easily reproduce the experiment by following the steps below.

---

## 🚀 **Quick Start**

### **1️⃣ Prepare the BDDL Files**

Execute the following commands to set up the perturbed BDDL configuration:

```bash
# Navigate to the BDDL directory
cd libero/libero/bddl_files/

# Create a new folder for the perturbation experiment
mkdir -p libero_object_temp

# Copy the target perturbation configuration (e.g., x0.1)
cp -r libero_object_temp_x0.1/* libero_object_temp/
```

> 🧩 This creates the `libero_object_temp` directory containing all `.bddl` files required  for the **object position perturbation** experiment.

---

### **2️⃣ Prepare the Initialization Files**

Similarly, set up the initialization configuration directory:

```bash
# Navigate to the initialization directory
cd libero/libero/init_files/

# Create a matching subdirectory
mkdir -p libero_object_temp

# Copy the initialization configuration (e.g., x0.1)
cp -r libero_object_temp_x0.1/* libero_object_temp/
```

> 💡 Ensure that both `bddl_files` and `init_files` share consistent naming conventions (e.g., `libero_object_temp_x0.1` → `libero_object_temp`).

---

### **3️⃣ Configure Perturbation Intensity (Optional)**

You can adjust the perturbation intensity based on your experimental requirements.  
The following levels are supported:

| **Perturbation Axis** | **Available Levels** | **Description** |
|------------------------|----------------------|-----------------|
| **X-axis Perturbation** | `x0.1`, `x0.2`, `x0.3`, `x0.4`, `x0.5` | Object translation along the X-axis |
| **Y-axis Perturbation** | `y0.1`, `y0.2`, `y0.3`, `y0.4`, `y0.5` | Object translation along the Y-axis |

Example: to test a specific perturbation level, simply copy the corresponding configuration:

```bash
# Example: apply perturbation magnitude x0.3
cp -r libero_object_temp_x0.3/* libero_object_temp/

# Example: apply perturbation magnitude y0.5
cp -r libero_object_temp_y0.5/* libero_object_temp/
```

> ⚙️ Modify the perturbation axis and magnitude to simulate different spatial displacement conditions.

---

### **4️⃣ Run the Evaluation**

Using **OpenVLA** as an example, execute the following command to perform the evaluation:

```bash
# Navigate to the project root
cd libero/

# Run the perturbation evaluation
python run_libero_eval.py
```

> The script automatically detects and loads perturbation data from `libero/libero/bddl_files/libero_object_temp/` and `libero/libero/init_files/libero_object_temp/`.

---

# Citation

If you use LIBERO-PRO in your research, please cite both **LIBERO** and **LIBERO-PRO**:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}

@article{zhou2025liberopro,
  title={LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization},
  author={Xueyang Zhou and Yangming Xu and Guiyao Tie and Yongchao Chen and Guowen Zhang and Duanfeng Chu and Pan Zhou and Lichao Sun},
  journal={[arXiv preprint arXiv:2510.03827]},
  year={2025},
  publisher={[Publisher]} / eprint={[arXiv ID]}
}
```

---

# License

| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                              |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |

---

<div align="center">

💡 *LIBERO-Pro — advancing the frontier of robust and fair generalization evaluation for Vision-Language-Action Models.*

</div>
