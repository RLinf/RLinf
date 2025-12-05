import os
import yaml

# This is a default path for localizing all the benchmark related files
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("/data/zxy/.libero")
)
config_file = os.path.join(libero_config_path, "config.yaml")

def get_default_path_dict(custom_location=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    benchmark_root_path = current_dir

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": os.path.join(benchmark_root_path, "bddl_files"),
        "init_states": os.path.join(benchmark_root_path, "init_files"),
        "assets": os.path.join(benchmark_root_path, "assets"),
        "datasets": os.path.join(benchmark_root_path, "../../datasets"), 
    }

def get_libero_path(query_key):
    paths = get_default_path_dict()
    
    if query_key not in paths:
        print(f"[Error] Key {query_key} not found.")
        return None
    
    return paths[query_key]

def set_libero_default_path(custom_location=None):
    pass