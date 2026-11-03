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

"""RoboLab simulator transport and owned process cleanup."""

import signal
import socket
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rlinf.envs.sim.robolab.protocol import receive, send
from rlinf.envs.sim.robolab.robolab_env import RoboLabEnv


def test_action_array_roundtrip():
    actions = np.arange(2 * 15 * 8, dtype=np.float32).reshape(2, 15, 8)
    sender, receiver = socket.socketpair()
    with sender, receiver:
        send(sender, {"command": "chunk_step", "actions": actions})
        received = receive(receiver)
    assert received["command"] == "chunk_step"
    np.testing.assert_array_equal(received["actions"], actions)
    assert received["actions"].flags.writeable


@pytest.mark.parametrize("partial", [b"", b"short", struct.pack("!Q", 10) + b"short"])
def test_receive_reports_disconnected_peer(partial):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(partial)
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError, match="closed the connection"):
            receive(receiver)


def test_large_message_roundtrip():
    sender, receiver = socket.socketpair()
    values = np.arange(300_000, dtype=np.float32)
    with sender, receiver, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send, sender, {"values": values})
        np.testing.assert_array_equal(receive(receiver)["values"], values)
        future.result(timeout=5)


@pytest.mark.parametrize("responsive", [True, False])
def test_close_reaps_simulator_process(responsive):
    parent, child = socket.socketpair()
    code = (
        "import socket,sys,time; "
        "from rlinf.envs.sim.robolab.protocol import receive; "
        "sock=socket.socket(fileno=int(sys.argv[1])); "
        "message=receive(sock); "
        "assert message['command']=='close'; "
        + ("sock.close()" if responsive else "time.sleep(120)")
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(child.fileno())],
        pass_fds=(child.fileno(),),
        start_new_session=True,
    )
    child.close()
    env = object.__new__(RoboLabEnv)
    env._socket = parent
    env._process = process
    env._closed = False
    try:
        env.close()
        assert process.returncode == (0 if responsive else -signal.SIGKILL)
        assert parent.fileno() == -1
        env.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        parent.close()
