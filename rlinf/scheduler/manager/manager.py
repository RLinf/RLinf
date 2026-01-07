# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import TypeVar

import ray
from ray._private import worker

from ..cluster import Cluster

ManagerClsType = TypeVar("ManagerClsType")


class ManagerProxy:
    """Singleton proxy for the Manager class to handle remote calls."""

    def __init__(self, manager_cls: "type[Manager]", no_wait: bool):
        """Launch the Manager class as a remote actor if not already running."""
        from ..worker import Worker

        if not ray.is_initialized():
            ray.init(
                address="auto",
                namespace=Cluster.NAMESPACE,
                logging_level=Cluster.LOGGING_LEVEL,
            )

        count = 0
        while True:
            try:
                self._manager = ray.get_actor(
                    name=manager_cls.MANAGER_NAME, namespace=Cluster.NAMESPACE
                )
                break
            except ValueError as e:
                if no_wait:
                    raise e
                count += 1
                time.sleep(0.001)
                if count % Cluster.TIMEOUT_WARN_TIME == 0:
                    Worker.logger.warning(
                        f"Waiting for manager named {manager_cls.MANAGER_NAME} to be ready for {count // 1000} seconds..."
                    )

        # Suppress warning for blocking get inside asyncio
        if (
            hasattr(worker.global_worker, "core_worker")
            and worker.global_worker.core_worker.current_actor_is_asyncio()
        ):
            worker.blocking_get_inside_async_warned = True

        # Attach Manager methods to the Proxy instance
        sched_fun_list = [
            func
            for func in dir(manager_cls)
            if callable(getattr(manager_cls, func)) and not func.startswith("_")
        ]

        class ProxyMethod:
            def __init__(self, func_name, manager):
                self._func_name = func_name
                self._manager = manager

            def __call__(self, *args, **kwargs):
                return ray.get(
                    getattr(self._manager, self._func_name).remote(*args, **kwargs)
                )

        for func in sched_fun_list:
            setattr(self, func, ProxyMethod(func, self._manager))


class Manager:
    """Ray-style global manager, which is launched from the driver process and offers a proxy singleton per worker process to access the global manager."""

    MANAGER_NAME = ""
    proxy: ManagerProxy = None
    PID = None
    ENV_LIST = ["RAY_ADDRESS"]

    @classmethod
    def get_proxy(cls: type[ManagerClsType], no_wait: bool = False) -> ManagerClsType:
        """Get the singleton proxy for the Manager class.

        Args:
            no_wait (bool): If True, do not wait for the manager to be ready.

        Returns:
            ManagerProxy: The singleton proxy instance for the Manager class.

        """
        if (
            cls.proxy is None or os.getpid() != cls.PID
        ):  # Reinitialize if PID has changed
            cls.PID = os.getpid()
            cls.proxy = ManagerProxy(cls, no_wait)
        return cls.proxy

    @classmethod
    def get_runtime_env_vars(cls) -> dict:
        """Get the runtime environment variables required for the manager.

        Returns:
            dict: A dictionary of environment variables.
        """
        runtime_env = {}
        for var in cls.ENV_LIST:
            if var in os.environ:
                runtime_env[var] = os.environ[var]
        return runtime_env
