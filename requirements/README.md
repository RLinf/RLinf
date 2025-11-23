## Dependency Installation Guide

We recommend using [`uv`](https://docs.astral.sh/uv/) to install the necessary Python dependencies.
You can install `uv` via `pip`.
```shell
pip install --upgrade uv
```

After installing `uv`, you can install the dependencies for the target experiments using the `install.sh` script under the `requirements/` folder.
The script accepts one argument which specifies the target experiment, including `openvla`, `openvla-oft`, `openpi`, and `reason`.
For example, to install the dependencies for the openvla experiment, you would run:
```shell
bash requirements/install.sh openvla
```

This will create a virtual environment under the current path named `.venv`.
To activate the virtual environment, you can use the following command:
```shell
source .venv/bin/activate
```

To deactivate the virtual environment, simply run:
```shell
deactivate
```