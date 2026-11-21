"""Select one physical GPU from nvidia-smi CSV supplied on stdin."""
import csv
import sys


def select_gpu(inventory, requested="auto"):
    devices = [(index.strip(), int(free.strip())) for index, free in csv.reader(inventory)]
    candidates = [
        (index, free)
        for index, free in devices
        if free >= 10240 and (requested == "auto" or index == requested)
    ]
    if not candidates:
        raise ValueError(
            f"No GPU matches {requested!r} with at least 10240 MiB free; "
            "retry after other work releases memory"
        )
    return max(candidates, key=lambda item: item[1])[0]


if __name__ == "__main__":
    try:
        print(select_gpu(sys.stdin, sys.argv[1] if len(sys.argv) > 1 else "auto"))
    except (ValueError, TypeError) as exc:
        sys.exit(f"Stage1 GPU selection failed: {exc}")
