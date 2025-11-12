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


## Create motion planning data for multiple tasks (train set)
# it takes 4.5 hours and generates a file of size 213 GB
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe:$PYTHONPATH"
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python maniskill_custom_package/motionplanning/widowx/collect_simpler.py \
  -e "PutOnPlateInScene25Main-v3" \
  --save_data \
  --record_dir /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill \
  --num_procs 16 --num_traj 16384 --seed=0

### Append last 15 frames to the raw motion planning data to make the robot learn to stop after the task is done.
### starts at 10/19/2025 3:58 PM
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing/
n_last_frames=15
N_PROC=64
input_dir=/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutOnPlateInScene25Main-v3/16384/data/
output_dir=/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutOnPlateInScene25Main-v3/16384/data_append/
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
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing/
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python visualize_trajectory.py \
  --npz_path /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutOnPlateInScene25Main-v3/16384/data_append/success_proc_1_numid_9_epsid_14.npz \
  --output /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutOnPlateInScene25Main-v3/16384/videos/success_proc_1_numid_9_epsid_14.mp4 \
  --fps 24

# export to lerobot dataset format   (7 hours)
# Main V3:
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe:$PYTHONPATH"
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export"
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python cli.py \
--input_dir /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutOnPlateInScene25Main-v3/16384/data_append \
--output_root /mnt/mnt/public/zhangtonghe/openpi-main/data/maniskill/PutOnPlateInScene25Main-v3 \
--n_eps_per_shard 128 \
--fps 24 \
--keep_last_n 15 \
--num_workers 40 \
--export_videos \
--use_temp \
--val_split_percent 0.0 \
--verbose
