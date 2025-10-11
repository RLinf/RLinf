## Building Docker Images

RLinf provides a unified Dockerfile for both the math reasoning and embodied images, and can switch between the two images using the `BUILD_TARGET` build argument, which can be `reason` or `embodied`.
To build the Docker image, run the following command in the `docker/torch-x.x` directory, replacing `x.x` with the desired PyTorch version (e.g., `2.6` or `2.7`):

```shell
export BUILD_TARGET=reason # or embodied for the embodied image
docker build --build-arg BUILD_TARGET=$BUILD_TARGET -t rlinf:$BUILD_TARGET .
```

If you are building the `reason` image and run into OOM during build, it might be because the `APEX` package's number of compile threads is set too high (default 24 and may require over 200 GB memory).
You can reduce the number of compile threads by adding `--build-arg APEX_BUILD_THREADS=<num_threads>` to the `docker build` command, where `<num_threads>` is the number of threads you want to use (e.g., 8 or 12).

# Using the Docker Image

The built Docker image contains one or multiple Python virtual environments (venv) in the `/opt/venv` directory, depending on the `BUILD_TARGET`.

Currently, the reasoning image contains one venv named `reason` in `/opt/venv/reason`, while the embodied image contains three venvs named `openvla`, `openvla-oft` and `pi0` in `/opt/venv/`.

To switch to the desired venv, we have a built-in script `switch_env` that can switch among venvs in a single command.

```shell
source switch_env <env_name> # e.g., source switch_env openvla-oft, source switch_env pi0, etc.
```