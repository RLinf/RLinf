"""Build RLT's synchronized observation configs without editing SDK defaults."""
import argparse
import copy
import os
from pathlib import Path

import yaml


def prepare(
    infra: Path,
    output: Path,
    recordings: Path,
    sync_hz: int = 10,
    reset_pose: Path | None = None,
    hil_raw_root: Path | None = None,
) -> dict:
    """Build client.yaml and collection.yaml under ``output`` for LoopInfra."""
    infra, output, recordings = infra.resolve(), output.resolve(), recordings.resolve()
    client = yaml.safe_load((infra / "config/tianji_collection_client.yaml").read_text())
    collection = yaml.safe_load((infra / "config/tianji_collection.yaml").read_text())
    client = copy.deepcopy(client)
    client.setdefault("raw_recording", {})["enabled"] = False
    client["sync_frequency_hz"] = client["upload_hz"] = int(sync_hz)
    client["camera_selection_file"] = str(
        infra / client.get("camera_selection_file", ".loopinfra-cameras.json")
    )
    client["robot"]["sdk_root"] = str(infra / client["robot"]["sdk_root"])
    if hil_raw_root is not None:
        root = str(Path(hil_raw_root).resolve())
        client["raw_recording"].update(enabled=False, hil=True, storage_roots=[root])
        collection["hil_storage_roots"] = [root]
    collection.pop("mode", None)
    if reset_pose is not None:
        if not Path(reset_pose).is_file():
            raise FileNotFoundError(f"reset pose {reset_pose} not found")
        collection["reset_pose"] = str(Path(reset_pose).resolve())
    else:
        collection["reset_pose"] = str(infra / collection["reset_pose"])
    collection["sdk_root"] = str(infra / collection["sdk_root"])
    if collection.get("camera_calibration"):
        collection["camera_calibration"] = str(infra / collection["camera_calibration"])
    collection["directory"] = str(recordings)
    collection["storage_roots"] = [str(recordings)]
    collection["camera_client_config"] = str(output / "client.yaml")
    output.mkdir(parents=True, exist_ok=True)
    result = {"client": output / "client.yaml", "collection": output / "collection.yaml"}
    for name, value in [("client", client), ("collection", collection)]:
        content = yaml.safe_dump(value, sort_keys=False)
        path = result[name]
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infra", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("recordings", type=Path)
    parser.add_argument("--sync-hz", type=int, choices=(10, 15, 30), default=10)
    parser.add_argument("--reset-pose", type=Path, default=None)
    args = parser.parse_args()
    hil_root = (
        args.recordings.parent
        if os.environ.get("TIANJI_HIL_RAW_RECORDING", "0") == "1"
        else None
    )
    prepare(args.infra, args.output, args.recordings, args.sync_hz, args.reset_pose, hil_root)
