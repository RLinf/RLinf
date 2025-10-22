**Summary**
feat(embodied): pi0 data generation and postprocessing. 


**Detailed Functionalities**
Tonghe also provides the script for

1. Visualizing the trajectories
2. Helper function to append the last 15 frames to a motion trajectory, to reinforce task completion awareness. 
3. A whole set of conversion script that compresses the raw generated data to highly compact lerobot dataeset format (compression rate: 10x or more)
4. Data filtering (remove idle actions, which is inherited from RL4VLA) and normalization script, for VLA with state input. 
5. Comprehensive bash source scripts to walk you through this data creation and post-processing pipeline. (`data_generation/generate_mp_mainv3.sh`) and (`data_generation/generate_mp_single_task.sh`)
6. How to merge generated single tasks to a multitask dataset. (`data_generation/merge_single_tasks.sh`)
7. Assets for the ManiSkill environment. This is zipped into a .rar file and uploaded to Google Drive at [this repo](https://drive.google.com/file/d/1lQygQpeg4kn3s8XT9MRcHHuyuPuo8CnR/view?usp=sharing)
   Please place this assets under ./maniskill/assets or wherever your script reads the robot, environment, obj STLs. 

**Application**
Tonghe tested this multiple times when fine-tuning PI0. It can successfuly make data for SFT. 
The same data format could also be used for tuning GR00T family. 

**Fix Env Bug**
Tonghe also spotted an environment issue, that the zqlenv_0729 version does not have a proper widowx.py file. I have also put this to the repo `widowx.py`, and maintainers should replace the widowx.py file in the maniskill environment with this version, consistent with 
RL4VLA paper. 

**To Maintainers**

Tonghe developed this software package with hard-coded env path and script path links in the 3 .sh files on an old branch, with env=`zqlenv_wmj_0729`. 

Please replace them with the newest env names that RLinf currently have. 

**Authorship**
Since Tonghe was not au author of the RLinf papaer, he would act as an outside collaborator. Tonghe would like the RLinf maintainence team to keep the MIT license attached to the beginning of Tonghe's custom files. 