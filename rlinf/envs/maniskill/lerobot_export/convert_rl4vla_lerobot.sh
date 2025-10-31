


# Zone I:
export PYTHONPATH="/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe:$PYTHONPATH"
export PYTHONPATH="/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export"
conda activate /mnt/mnt/public/mjwei/conda_envs/zqlenv_wmj_0729
cd /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/rlinf/envs/maniskill/lerobot_export
python cli.py \
--input_dir /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data_append \
--output_root /mnt/mnt/public/liuzhihao/openpi-main/data/maniskill/PutSpoonOnTableClothInScene-v1 \
--n_eps_per_shard 30 \
--fps 24 \
--keep_last_n 15 \
--num_workers 40 \
--export_videos \
--use_temp \
--val_split_percent 0.0 \
--verbose

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



