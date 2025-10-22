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


from pathlib import Path
import yaml
from datetime import datetime

def main():
    stats = {}

    # old stats
    sold_path = Path(__file__).parent.parent / "scripts" / "stats"
    solds = sold_path.glob("stats-*.yaml")
    solds = sorted(list(solds), key=lambda x: x.name)
    for s in solds:
        print(f"{s.name}")
    for sold in solds:
        cfg = yaml.safe_load(sold.read_text())
        for load_path, envs in cfg.items():
            if load_path not in stats:
                stats[load_path] = {}
            for env_name, seeds in envs.items():
                if env_name not in stats[load_path]:
                    stats[load_path][env_name] = {}
                for seed, stat in seeds.items():
                    if seed not in stats[load_path][env_name]:
                        stats[load_path][env_name][seed] = {}
                    stats[load_path][env_name][seed].update(stat)

    # wandb
    wandb_path = Path(__file__).parent.parent / "wandb"
    runs = wandb_path.glob("offline-run-*")

    for run in runs:
        cfg = yaml.safe_load((run / "glob" / "config.yaml").read_text())

        load_path = "/".join(cfg["vla_load_path"].split("/")[-3:])
        env_name = cfg["env_id"]
        seed = cfg["seed"]

        if load_path not in stats:
            stats[load_path] = {}
        if env_name not in stats[load_path]:
            stats[load_path][env_name] = {}

        train_vis_dir = run / "glob" / "vis_0_train" / "stats.yaml"
        if train_vis_dir.exists():
            train_stats = yaml.safe_load(train_vis_dir.read_text())
            if "stats" in train_stats:
                if "train" not in stats[load_path][env_name]:
                    stats[load_path][env_name]["train"] = {}
                stats[load_path][env_name]["train"][seed] = train_stats["stats"]
                stats[load_path][env_name]["train"][seed]["path"] = str(run)

        test_vis_dir = run / "glob" / "vis_0_test" / "stats.yaml"
        if test_vis_dir.exists():
            test_stats = yaml.safe_load(test_vis_dir.read_text())
            if "stats" in test_stats:
                if "test" not in stats[load_path][env_name]:
                    stats[load_path][env_name]["test"] = {}
                stats[load_path][env_name]["test"][seed] = test_stats["stats"]
                stats[load_path][env_name]["test"][seed]["path"] = str(run)

    # save stats
    tt = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).parent.parent / "scripts" / "stats" / f"stats-{tt}.yaml"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)

if __name__ == "__main__":
    main()
