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

# merge raw appended trajectories
#!/bin/bash

# Create destination directory
mkdir -p /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/432/data_append

# Copy and rename eggplant files
for file in /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutEggplantInBasketScene-v1/150/data_append/*.npz; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/432/data_append/eggplant_"$filename"
    fi
done

# Copy and rename carrot files
for file in /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/PutCarrotOnPlateInScene-v1/150/data_append/*.npz; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/432/data_append/carrot_"$filename"
    fi
done

# Copy and rename stack files
for file in /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/StackGreenCubeOnYellowCubeBakedTexInScene-v1/150/data_append/*.npz; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/432/data_append/stack_"$filename"
    fi
done

echo "Done! Files copied and renamed to /mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/432/data_append"

# export to lerobot dataset format (30mins)
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe:$PYTHONPATH"
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export"
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python cli.py \
--input_dir /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/432/data_append \
--output_root /mnt/mnt/public/zhangtonghe/openpi-main/data/maniskill/simpler3tasks/432 \
--n_eps_per_shard 36 \
--fps 24 \
--keep_last_n 15 \
--num_workers 16 \
--export_videos \
--use_temp \
--val_split_percent 0.0 \
--verbose


# export to lerobot dataset: (4 tasks)
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe:$PYTHONPATH"
export PYTHONPATH="/mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export"
source /mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/activate
cd /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export
/mnt/mnt/public/chenkang/zqlenv_wmj_0729/bin/python cli.py \
--input_dir /mnt/mnt/public/zhangtonghe/RLinf_openpi_tonghe/data/maniskill/simpler3tasks/576/data_append \
--output_root /mnt/mnt/public/zhangtonghe/openpi-main/data/maniskill/simpler4tasks \
--n_eps_per_shard 48 \
--fps 24 \
--keep_last_n 15 \
--num_workers 16 \
--export_videos \
--use_temp \
--val_split_percent 0.0 \
--verbose
