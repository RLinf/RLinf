# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# "StackGreenCubeOnYellowCubeBakedTexInScene-v1": SolveStackCube,
# "PutSpoonOnTableClothInScene-v1": SolvePutSpoon,
# "PutEggplantInBasketScene-v1": SolvePutEggplant,
# "PutCarrotOnPlateInScene-v1": SolvePutCarr

## Create motion planning data for a single task
export PYTHONPATH="/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe:$PYTHONPATH"

# On I machine:
```
export PYTHONPATH="/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe:$PYTHONPATH"
conda activate /mnt/mnt/public/mjwei/conda_envs/zqlenv_wmj_0729
cd /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/
cuda=0
CUDA_VISIBLE_DEVICES=$cuda
python maniskill_custom_package/motionplanning/widowx/collect_simpler.py \
  -e "PutSpoonOnTableClothInScene-v1" \
  --save_data \
  --record_dir /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill \
  --num_procs 16 --num_traj 150 --seed=0
```

# on J2 machine:
```
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe:$PYTHONPATH"
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/
cuda=0
CUDA_VISIBLE_DEVICES=$cuda
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python maniskill_custom_package/motionplanning/widowx/collect_simpler.py \
  -e "PutCarrotOnPlateInScene-v1" \
  --save_data \
  --record_dir /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill \
  --num_procs 16 --num_traj 150 --seed=0
```





# Append last 15 frames to the trajectory, discard 'info', further compress the file. 
#!/bin/bash#!/bin/bash
# on I machine:
n_last_frames=15
N_PROC=64
conda activate rlinf_env
cd /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing/
input_dir=/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data/
output_dir=/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data_append/
mkdir -p "$output_dir"
process_file() {
  local raw_mp_data_path=$1
  local output_dir=$2
  local n_last_frames=$3
  filename=$(basename "$raw_mp_data_path")
  appended_mp_data_path="${output_dir}${filename}"
  echo "Processing: $filename"
  python append_last_frames.py \
    --npz_path "$raw_mp_data_path" \
    --output "$appended_mp_data_path" \
    --n_last_frames "$n_last_frames"
}
export -f process_file
export n_last_frames
export output_dir
find "$input_dir" -maxdepth 1 -name "*.npz" | \
  xargs -I {} -P ${N_PROC} bash -c 'process_file "$@"' _ {} "$output_dir" "$n_last_frames"
echo "All files processed!"



# on J2 machine:
# "StackGreenCubeOnYellowCubeBakedTexInScene-v1": SolveStackCube,
# "PutSpoonOnTableClothInScene-v1": SolvePutSpoon,
# "PutEggplantInBasketScene-v1": SolvePutEggplant,
# "PutCarrotOnPlateInScene-v1": SolvePutCarr
n_last_frames=15
N_PROC=64
TASK_NAME="PutCarrotOnPlateInScene-v1"
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing/
input_dir=/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/${TASK_NAME}/150/data/
output_dir=/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/${TASK_NAME}/150/data_append/
mkdir -p "$output_dir"
process_file() {
  local raw_mp_data_path=$1
  local output_dir=$2
  local n_last_frames=$3
  filename=$(basename "$raw_mp_data_path")
  appended_mp_data_path="${output_dir}${filename}"
  echo "Processing: $filename"
  /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python append_last_frames.py \
    --npz_path "$raw_mp_data_path" \
    --output "$appended_mp_data_path" \
    --n_last_frames "$n_last_frames"
}
export -f process_file
export n_last_frames
export output_dir
find "$input_dir" -maxdepth 1 -name "*.npz" | \
  xargs -I {} -P ${N_PROC} bash -c 'process_file "$@"' _ {} "$output_dir" "$n_last_frames"
echo "All files processed!"



# visualize the trajectory
# On I machine:
conda activate rlinf_env
cd /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing/
python visualize_trajectory.py \
  --npz_path /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data_last_frames/success_proc_0_numid_0_epsid_3.npz \
  --output /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data_last_frames/success_proc_0_numid_0_epsid_3.mp4 \
  --fps 24

# on J2 machine:
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate  
TASK_NAME="PutEggplantInBasketScene-v1"
EPISODE_ID=0
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python visualize_trajectory.py \
  --npz_path /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/${TASK_NAME}/150/data_append/success_proc_0_numid_0_epsid_${EPISODE_ID}.npz \
  --output /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/${TASK_NAME}/150/data_append/success_proc_0_numid_0_epsid_${EPISODE_ID}.mp4 \
  --fps 24