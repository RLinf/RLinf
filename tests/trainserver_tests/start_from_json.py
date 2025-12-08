import os
import sys
import json
import requests


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Prefer API key, fall back to cookie
API_KEY = os.getenv("INFINI_API_KEY")
INF_COOKIES = os.getenv("INFINI_COOKIES")

if API_KEY:
    AUTH_MODE = "api_key"
elif INF_COOKIES:
    AUTH_MODE = "cookie"
else:
    raise RuntimeError("Missing required environment variable: need at least one of INFINI_API_KEY or INF_COOKIES")

# Full URLs for train_plan generate
TRAIN_PLAN_OPEN_URL = require_env("INFINI_OPEN_TRAIN_PLAN_URL")
TRAIN_PLAN_USER_URL = require_env("INFINI_USER_TRAIN_PLAN_URL")

# Full URLs for train_service create
TRAIN_SERVICE_OPEN_URL = require_env("INFINI_OPEN_TRAIN_SERVICE_URL")
TRAIN_SERVICE_USER_URL = require_env("INFINI_USER_TRAIN_SERVICE_URL")

# Common headers without auth info
_BASE_COMMON_HEADERS = {
    "content-type": "application/json",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
}

# Build final headers based on auth mode
if AUTH_MODE == "api_key":
    COMMON_HEADERS = {
        **_BASE_COMMON_HEADERS,
        "Authorization": f"Bearer {API_KEY}",
    }
else:  # cookie
    COMMON_HEADERS = {
        **_BASE_COMMON_HEADERS,
        "Cookie": INF_COOKIES,
    }


def load_jobs(config_path: str):
    """
    Parse all enabled models from JSON into a list of job configs.
    Each job is a dict containing all fields needed to call the APIs.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    models_cfg = cfg.get("models", {})
    train_server_cfg = cfg.get("train_server", {})
    framework_cfg = cfg.get("frame_work", {})

    jobs = []

    for model_name, m in models_cfg.items():
        if not m.get("enabled"):
            continue

        print(f"Found enabled model: {model_name}")

        mirror = m.get("mirror")
        if not mirror:
            raise KeyError(f"Model {model_name} is missing 'mirror' field")

        image_id = m.get("image_id")
        if not image_id:
            raise KeyError(f"Model {model_name} is missing 'image_id' field (image id mapped from mirror)")

        train_cfg = m.get("train") or {}

        train_server_name = train_cfg.get("train_server")
        if not train_server_name:
            raise KeyError(f"Model {model_name} 'train' config is missing 'train_server' field")

        run_gpus = str(train_cfg.get("run_gpus"))
        if not run_gpus or run_gpus == "None":
            raise KeyError(f"Model {model_name} 'train' config is missing 'run_gpus' field")

        worker_num = int(train_cfg.get("worker_num", 1))

        # RDMA rule: disable when worker_num == 1 or single worker gpu count < 8
        rdma_enable = not (worker_num == 1 or run_gpus < "8")
        framework_id = framework_cfg.get("Distributed" if worker_num > 1 else "Single")

        expect_train_complete_time = int(train_cfg.get("train_time", 3600))

        job_name = train_cfg.get("job_name", f"{model_name}_train_job")
        entry_point = train_cfg.get("entry_point", "sleep 999")
        tb_log_dir = train_cfg.get("tb_log_dir", "")

        # Pick cluster + GPU profile from train_server
        cluster_cfg = train_server_cfg.get(train_server_name)
        if cluster_cfg is None:
            raise KeyError(f"No configuration named '{train_server_name}' found in 'train_server'")

        gpu_profile = cluster_cfg.get(run_gpus)
        if gpu_profile is None:
            raise KeyError(
                f"'train_server.{train_server_name}' has no configuration for run_gpus={run_gpus}"
            )

        region_id = gpu_profile.get("region_id")
        resource_spec_id = gpu_profile.get("resource_spec_id")
        shared_mem = int(gpu_profile.get("shared_mem", 0))
        mount_volume_id = gpu_profile.get("mount_volume_id")

        if not region_id or not resource_spec_id:
            raise KeyError(
                f"'train_server.{train_server_name}[{run_gpus}]' is missing 'region_id' or 'resource_spec_id'"
            )

        job = {
            "model_name": model_name,
            "mirror": mirror,
            "image_id": image_id,
            "train_server": train_server_name,
            "run_gpus": run_gpus,
            "region_id": region_id,
            "resource_spec_id": resource_spec_id,
            "shared_mem": shared_mem,
            "mount_volume_id": mount_volume_id,
            "worker_num": worker_num,
            "expect_train_complete_time": expect_train_complete_time,
            "job_name": job_name,
            "entry_point": entry_point,
            "tb_log_dir": tb_log_dir,
            "framework_id": framework_id,
            "rdma_enable": rdma_enable,
        }
        jobs.append(job)

    if not jobs:
        raise RuntimeError("No models with enabled=true found in config.")

    print(f"Total jobs to start: {len(jobs)}")
    return jobs


def generate_train_plan(job: dict) -> str:
    # Choose API path based on auth mode (URLs are fully from env)
    if AUTH_MODE == "api_key":
        url = TRAIN_PLAN_OPEN_URL
    else:
        url = TRAIN_PLAN_USER_URL

    payload = {
        "expect_train_complete_time": job["expect_train_complete_time"],
        "region_id": job["region_id"],
        "resource_spec_id": job["resource_spec_id"],
        "worker_num": job["worker_num"],
    }

    print(f"\n[Model {job['model_name']}] requesting train_plan generation... (auth={AUTH_MODE})")
    resp = requests.post(url, headers=COMMON_HEADERS, data=json.dumps(payload))
    print("generate status:", resp.status_code)
    print("generate resp:", resp.text)

    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"[{job['model_name']}] generate failed: {body.get('msg')}")

    plans = body["data"]["train_plan_list"]
    if not plans:
        raise RuntimeError(f"[{job['model_name']}] generate succeeded but 'train_plan_list' is empty")

    # For now always pick the first plan
    chosen_idx = 0
    chosen = plans[chosen_idx]
    train_plan_id = chosen["train_plan_id"]

    print(f"[{job['model_name']}] chosen plan index: {chosen_idx}")
    print("  train_plan_id:", train_plan_id)
    print("  train_expire_time:", chosen.get("train_expire_time"))
    print("  price:", chosen.get("price"), "discount:", chosen.get("discount"))

    return train_plan_id


def create_job(job: dict, train_plan_id: str):
    # Choose API path based on auth mode (URLs are fully from env)
    if AUTH_MODE == "api_key":
        url = TRAIN_SERVICE_OPEN_URL
    else:
        url = TRAIN_SERVICE_USER_URL

    print(f"\n[Model {job['model_name']}] creating training job... (auth={AUTH_MODE})")
    print(f"  mirror={job['mirror']}, image_id={job['image_id']}")

    payload = {
        "job_name": job["job_name"],
        "expect_train_complete_time": job["expect_train_complete_time"],
        "region_id": job["region_id"],
        "resource_spec_id": job["resource_spec_id"],
        "worker_num": job["worker_num"],
        "framework_id": job["framework_id"],
        "rdma_enable": job["rdma_enable"],
        "image_id": job["image_id"],
        "entry_point": job["entry_point"],
        "tb_log_dir": job["tb_log_dir"],
        "train_plan_id": train_plan_id,
        "shared_mem": job["shared_mem"],
        "mount": [
            {
                "path": "/mnt/mnt/public",
                "volume_id": job["mount_volume_id"],
                "rw_setting": "can_write",
            }
        ],
    }

    try:
        resp = requests.post(
            url,
            headers=COMMON_HEADERS,
            data=json.dumps(payload),
            timeout=15,  # e.g. 15 seconds
        )
    except requests.exceptions.RequestException as e:
        print(f"[{job['model_name']}] create request failed with exception:", repr(e))
        raise

    print("create status:", resp.status_code)
    print("create resp:", resp.text)


if __name__ == "__main__":
    # CLI args:
    #   --config/-c PATH    : Path to JSON config file (default: Task_train.json in current directory)
    #   --expect-time/-t N  : Override expect_train_complete_time in seconds for all enabled models

    expect_time_arg = None    # config file path (assumed to be in the same directory by default)
    config_path = "Task_train.json"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--config", "-c"):
            if i + 1 >= len(args):
                raise SystemExit("Error: --config/-c must be followed by a config file path")
            config_path = args[i + 1]
            i += 2
        elif arg in ("--expect-time", "-t"):
            if i + 1 >= len(args):
                raise SystemExit("Error: --expect-time/-t must be followed by an integer (seconds)")
            try:
                expect_time_arg = int(args[i + 1])
            except ValueError:
                raise SystemExit("Error: --expect-time/-t argument must be an integer (seconds)")
            i += 2
        else:
            print(f"Warning: ignore unknown argument {arg}")
            i += 1

    # Load job configs for all enabled models
    jobs = load_jobs(config_path)

    # Override expect_train_complete_time from CLI if provided
    if expect_time_arg is not None:
        for job in jobs:
            job["expect_train_complete_time"] = expect_time_arg
        print(
            f"\nUsing command line expect_train_complete_time={expect_time_arg} "
            "(applied to all enabled models)"
        )
    else:
        print("\nUsing expect_train_complete_time from JSON config")

    print(f"Current auth mode: {AUTH_MODE}")
    if AUTH_MODE == "api_key":
        print("Using Open API with INFINI_API_KEY to create jobs")
    else:
        print("Using Cookie and user APIs to create jobs")

    # Launch all enabled models
    for job in jobs:
        print("\n" + "=" * 60)
        print(f"Start processing model: {job['model_name']}")
        tp_id = generate_train_plan(job)
        print(f"[{job['model_name']}] got train_plan_id:", tp_id)
        create_job(job, tp_id)
        print(f"[{job['model_name']}] job creation request sent")
