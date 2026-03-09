import pickle
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import portal

from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.robots.utils import Timeout


def remote(serialization_needed: bool = False) -> Callable:
    """
    Decorator to mark a method as remotely accessible.

    Args:
        serialization_needed (bool): Indicates if the return value needs serialization. Default is False.
    """

    def decorator(method: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            # Handle case where args might be empty
            remote_mode = getattr(args[0], "_remote_mode", False) if args else False
            if remote_mode and serialization_needed:
                return pickle.dumps(result)
            return result

        wrapper._is_remote = True  # type: ignore
        wrapper._serialization_needed = serialization_needed  # type: ignore
        return wrapper

    return decorator


class RemoteServer:
    def __init__(
        self,
        obj: Any,
        port: int,
        host: str = "127.0.0.1",
        custom_remote_methods: Optional[Dict[str, bool]] = None,  # Dict of method name to serialization_needed
    ) -> None:
        self.obj = obj
        self.obj._remote_mode = True
        # keys are method names, values are whether serialization is needed
        self._custom_remote_methods = {} if custom_remote_methods is None else custom_remote_methods
        self._server = portal.Server(port)
        self._bind_remote_methods()
        self._port = port
        self._host = host

    def __del__(self):
        """Ensure server is closed on destruction"""
        try:
            if hasattr(self, "_server") and self._server:
                self._server.close()
        except Exception:
            pass

    def _bind_remote_methods(self):
        """Bind methods marked as `@remote` to the server."""
        for attr_name in dir(self.obj):
            method = getattr(self.obj, attr_name)
            if callable(method) and (
                getattr(method, "_is_remote", False) or attr_name in self._custom_remote_methods.keys()
            ):
                if attr_name in self._custom_remote_methods.keys() and self._custom_remote_methods[attr_name]:
                    self._server.bind(attr_name, lambda _m=method, *args, **kwargs: pickle.dumps(_m(*args, **kwargs)))
                else:
                    self._server.bind(attr_name, method)
        self._server.bind("get_supported_remote_methods", self.get_supported_remote_methods)

    @remote()
    def get_supported_remote_methods(self) -> List[str]:
        """Get a list of methods that are marked as `@remote`."""
        _result = [
            attr_name
            for attr_name in dir(self.obj)
            if callable(getattr(self.obj, attr_name)) and getattr(getattr(self.obj, attr_name), "_is_remote", False)
        ]

        result = []
        for name in _result:
            serialization_needed = getattr(getattr(self.obj, name), "_serialization_needed", False)
            result.append((name, serialization_needed))
        result.append(("get_supported_remote_methods", False))

        for k, v in self._custom_remote_methods.items():
            result.append((k, v))
        return result

    def serve(self) -> None:
        self._server.start()

    def close(self):
        """Close the server"""
        try:
            if hasattr(self, "_server") and self._server:
                self._server.close()
        except Exception:
            pass


class WrappedFuture:
    """A wrapper for futures to handle pickle deserialization on calling .result()."""

    def __init__(self, future: Any):
        self._future = future

    def result(self):
        """Return result, deserializing if needed."""
        result = self._future.result()
        return pickle.loads(result)


class Client:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")
        self._supported_remote_methods = self._get_supported_remote_methods()
        self._supported_remote_methods = {  # convert from list of tuples to dict
            name: serialization_needed
            for name, serialization_needed in self._supported_remote_methods  # type: ignore
        }
        self._use_future = False

    def __del__(self):
        """Ensure client is closed on destruction"""
        try:
            if hasattr(self, "_client") and self._client:
                self._client.close()
        except Exception:
            pass

    @property
    def use_future(self):
        return self._use_future

    def set_use_future(self, use_future: bool) -> None:
        self._use_future = use_future

    def __getattr__(self, name):
        # This method is used for attributes that are not found on the class
        if name in self._supported_remote_methods:
            # Return a callable proxy for the remote method
            if self._use_future:

                def remote_method_proxy(*args: Any, **kwargs: Any) -> Any:
                    if self._supported_remote_methods[name]:  # type: ignore
                        return WrappedFuture(self._client.__getattr__(name)(*args, **kwargs))
                    else:
                        return self._client.__getattr__(name)(*args, **kwargs)
            else:

                def remote_method_proxy(*args: Any, **kwargs: Any) -> Any:
                    if self._supported_remote_methods[name]:  # type: ignore
                        return pickle.loads(self._client.__getattr__(name)(*args, **kwargs).result())  # type: ignore
                    else:
                        return self._client.__getattr__(name)(*args, **kwargs).result()  # type: ignore

            return remote_method_proxy
        else:
            raise AttributeError(f"Method '{name}' is not supported by the remote object.")

    def _get_supported_remote_methods(self):
        return self._client.get_supported_remote_methods().result()

    @property
    def supported_remote_methods(self):
        return self._supported_remote_methods

    def close(self):
        """Close the client"""
        try:
            if hasattr(self, "_client") and self._client:
                self._client.close()
        except Exception:
            pass


@contextmanager
def return_futures(*clients: Client):
    """Context manager to set use_future=True for all provided Client instances."""
    assert all(isinstance(client, Client) for client in clients)
    previous_states = [client.use_future for client in clients]  # Save old states
    try:
        # Set use_future=True for all clients
        for client in clients:
            client.set_use_future(True)
        yield  # Yield control to the block
    finally:
        # Restore previous states
        for client, prev_state in zip(clients, previous_states, strict=False):
            client.set_use_future(prev_state)


def launch_remote_get_local_handler(
    cfg: Any,
    port: Optional[int] = None,
    host: str = "127.0.0.1",
    launch_remote: bool = True,
    process_pool: Any = None,
    custom_remote_methods: Optional[Dict[str, bool]] = None,
    logging_config_path: str | None = None,
) -> tuple[Any, Client]:
    if port is None:
        port = portal.free_port()
    p = launch_remote_server(cfg, port, host, launch_remote, process_pool, custom_remote_methods, logging_config_path)

    with Timeout(20, f"launching client: {cfg} at port {port}"):
        assert port is not None
        client = Client(port, host)
    return p, client


def launch_remote_server(
    cfg: Any,
    port: Optional[int] = None,
    host: str = "127.0.0.1",
    launch_remote: bool = True,
    process_pool: Any = None,
    custom_remote_methods: Optional[Dict[str, bool]] = None,
    logging_config_path: str | None = None,
) -> Any:
    if port is None:
        port = portal.free_port()

    def _launch() -> None:
        # setup_logging(logging_config_path)
        obj = instantiate(cfg)
        assert port is not None
        remote_server = RemoteServer(obj, port, host, custom_remote_methods=custom_remote_methods)
        remote_server.serve()

    # wrapper that makes the remote server ignores ctrl+c, so then we can reset the robot position and close server.
    def _launch_wrapper():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _launch()

    p = None
    if launch_remote:
        # p = portal.Process(_launch, start=True)
        p = portal.Process(_launch_wrapper, start=True)
    if process_pool is not None: 
        process_pool.append(p)
    return p


if __name__ == "__main__":
    from yam_realtime.configs.loader import DictLoader

    configs_dict = DictLoader.load("~/yam_realtime/configs/yam_viser.yaml")
    agent_cfg = configs_dict.pop("agent")

    p, client = launch_remote_get_local_handler(agent_cfg, 3339)
    # client = instantiate(agent_cfg)
    from tqdm import tqdm

    for _ in tqdm(range(1000)):
        client.act({})
        action_spec = client.action_spec()
        time.sleep(0.01)
    p.kill()
