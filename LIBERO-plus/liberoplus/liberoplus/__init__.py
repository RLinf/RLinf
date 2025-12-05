import os
from pathlib import Path

def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = str(Path(__file__).resolve().parent)
    else:
        benchmark_root_path = custom_location

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": os.path.join(benchmark_root_path, "bddl_files"),
        "init_states": os.path.join(benchmark_root_path, "init_files"),
        "assets": os.path.join(benchmark_root_path, "assets"),
        "datasets": os.path.join(benchmark_root_path, "../datasets"),
    }

def get_libero_path(query_key):
    paths = get_default_path_dict()
    
    if query_key not in paths:
        return os.path.join(paths["benchmark_root"], query_key)
    
    return paths[query_key]

def set_libero_default_path(custom_location=None):
    pass