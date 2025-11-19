To benchmark, we must install Habitat. Please ensure your current working directory is in the root of the repo.

Please perform the steps in [../README.md](../README.md) first.

```
# NOTE: ensure your current working directory is the root of this repo.
conda create -n mshab-habitat python=3.9 cmake=3.14.0
conda activate mshab-habitat

conda install habitat-sim=0.3.1 withbullet -c conda-forge -c aihabitat
pip install habitat-lab==0.3.20231024 habitat-baselines==0.3.20231024 gymnasium shortuuid
pip install -e ManiSkill

python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
echo '{"physics_simulator": "bullet", "timestep": 0.008, "gravity": [0, -9.8, 0], "friction_coefficient": 0.3, "restitution_coefficient": 0, "rigid object paths": ["objects"]}' > data/default.physics_config.json

python -m habitat_sim.utils.datasets_download --uids hab2_bench_assets
```