# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import os
import threading
from typing import Any, Optional

import requests

_THREAD_LOCAL = threading.local()
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_LOCAL_NO_PROXY_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def direct_request_env(extra_no_proxy: Optional[list[str]] = None) -> dict[str, str]:
    """Return an environment for child processes making internal HTTP calls."""
    env = dict(os.environ)
    for name in _PROXY_ENV_VARS:
        env.pop(name, None)

    no_proxy_hosts = list(_LOCAL_NO_PROXY_HOSTS)
    if extra_no_proxy:
        no_proxy_hosts.extend(extra_no_proxy)
    no_proxy = ",".join(dict.fromkeys(no_proxy_hosts))
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    return env


def get_session() -> requests.Session:
    """Return a per-thread session that ignores environment proxy settings."""
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _THREAD_LOCAL.session = session
    return session


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    return get_session().request(method, url, **kwargs)


def get(url: str, **kwargs: Any) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return request("POST", url, **kwargs)


def delete(url: str, **kwargs: Any) -> requests.Response:
    return request("DELETE", url, **kwargs)
