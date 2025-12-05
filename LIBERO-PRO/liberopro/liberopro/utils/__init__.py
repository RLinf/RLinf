import os
import yaml

# This is a default path for localizing all the benchmark related files
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
)
config_file = os.path.join(libero_config_path, "config.yaml")


def get_path_dict(root_location=os.path.dirname(os.path.abspath(__file__))):
    benchmark_root_path = root_location

    # This is a default path for localizing all the default bddl files
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # This is a default path for localizing all the default bddl files
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # This is a default path for localizing all the default datasets
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
    }


def get_libero_path(key):
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))
    assert key in config, f"Key {key} not found in config file {config_file}"
    return config[key]


def set_libero_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    new_config = get_path_dict(custom_location)
    with open(config_file, "w") as f:
        yaml.dump(new_config, f)


if not os.path.exists(libero_config_path):
    os.makedirs(libero_config_path)

if not os.path.exists(config_file):
    # Create a default config file

    # write all the paths into a yaml file
    with open(config_file, "w") as f:
        yaml.dump(get_path_dict(), f)
