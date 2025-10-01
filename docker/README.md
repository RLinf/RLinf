## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning and embodied environments, and can switch between them using the `BUILD_TARGET` build argument, which can be `reason` or `embodied`.
To build the Docker image, run the following command in the `docker/torch-x.x` directory, replacing `x.x` with the desired PyTorch version (e.g., `2.6` or `2.7`):

```shell
export BUILD_TARGET=reason # or embodied for the embodied environment
docker build --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:reason .
```

# Using the Docker Image

The built Docker image contains one or multiple Python venv in the `/opt/venv` directory, depending on the `BUILD_TARGET`.

Currently, the reasoning environment contains one venv named `reason` in `/opt/venv/reason`, while the embodied environment contains three venv named `openvla`, `openvla-oft` and `pi0` in `/opt/venv/`.

To switch to the desired venv, we have a built-in script `switch_env` that can switch among venvs in a single command.

```shell
switch_env <env_name> # e.g., switch_env reason, switch_env openvla, etc.
```