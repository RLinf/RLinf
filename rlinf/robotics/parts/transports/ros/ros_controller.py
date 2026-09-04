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
import pathlib
import sys
import threading
import time
from typing import Any, Callable, ClassVar, Optional

import psutil
import rospy
from filelock import FileLock

from rlinf.utils.logging import get_logger


class ROSController:
    """Manage ROS 1 publishers and subscribers for one robot.

    Prefer :meth:`shared` over constructing this directly. ROS 1 allows one node
    per process, so parts that each build their own controller end up on one
    node anyway, and only the first call to :func:`rospy.init_node` decides what
    that node is called.
    """

    #: The session this process is already using, and the lock that keeps two
    #: threads from racing to open it.
    _shared: "ClassVar[Optional[ROSController]]" = None
    _shared_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def shared(cls, ros_version: int = 1) -> "ROSController":
        """Return the ROS session for this process, opening it on first use.

        ROS 1 gives a process one node, so an arm, the end effector mounted on
        it, and anything else ROS-backed have to share a session rather than
        each opening their own. Asking for it here is what lets a part reach ROS
        without being handed a session by whichever part happened to start
        first.

        The session is never closed. ``rospy`` has no supported way to bring a
        node back up once it is shut down, so a part that has finished with ROS
        leaves the node standing for whatever is still running.
        """
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls(ros_version)
            return cls._shared

    def __init__(self, ros_version: int = 1) -> None:
        """Initialize the ROS controller."""
        self._logger = get_logger()
        self._ros_version = ros_version
        assert self._ros_version == 1, "Currently only ROS 1 is supported."

        # Serialize access to the node-wide ROS core.
        ros_lock_file = "/tmp/.ros.lock"
        # Fall back to the user directory if the temporary path is unavailable.
        if not os.path.exists(os.path.dirname(ros_lock_file)):
            ros_lock_file = os.path.join(pathlib.Path.home(), ".ros.lock")
        self._ros_lock = FileLock(ros_lock_file)

        if self._ros_version == 1:
            with self._ros_lock:
                self._ros_core = None
                # Reuse an existing core or launch one for this node.
                for proc in psutil.process_iter():
                    if proc.name() == "roscore":
                        self._ros_core = proc

                if self._ros_core is None:
                    self._ros_core = psutil.Popen(
                        ["roscore"], stdout=sys.stdout, stderr=sys.stdout
                    )
                    time.sleep(1)  # Allow roscore to accept connections.

        # Initialize the ROS node.
        rospy.init_node("franka_controller", anonymous=True)

        # ROS channels.
        self._output_channels: dict[str, rospy.Publisher] = {}
        self._input_channels: dict[str, rospy.Subscriber] = {}
        self._input_channel_status: dict[str, bool] = {}

    def get_input_channel_status(self, name: str) -> bool:
        """Return whether a subscriber has received a message.

        Args:
            name: The name of the ROS input channel.

        """
        if name not in self._input_channel_status:
            return False
        return self._input_channel_status.get(name, False)

    def create_ros_channel(
        self, name: str, data_class: rospy.Message, queue_size: Optional[int] = None
    ) -> None:
        """Create a ROS publisher.

        Args:
            name: ROS topic name.
            data_class: Message type published on the topic.
            queue_size: Publisher queue size. Zero selects an unbounded queue;
                ``None`` enables synchronous publishing.
        """
        self._output_channels[name] = rospy.Publisher(
            name, data_class, queue_size=queue_size
        )

    def connect_ros_channel(
        self, name: str, data_class: rospy.Message, callback: Callable
    ) -> None:
        """Create a ROS subscriber.

        Parts share this session, so several of them may subscribe to one
        topic. A later subscriber does not reset what the topic has already
        delivered: readiness describes the topic, and a part that has been
        reading it does not stop because something else started listening too.

        Args:
            name: ROS topic name.
            data_class: Message type received from the topic.
            callback: Function invoked for each message.
        """

        def callback_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Mark the subscriber active after its first message.
            self._input_channel_status[name] = True
            return callback(*args, **kwargs)

        self._input_channel_status.setdefault(name, False)
        self._input_channels[name] = rospy.Subscriber(
            name, data_class, callback_wrapper
        )

    def put_channel(self, name: str, data: rospy.Message) -> None:
        """Publish a message on a configured channel.

        Args:
            name: ROS topic name.
            data: Message to publish.
        """
        if name in self._output_channels:
            assert isinstance(data, self._output_channels[name].data_class), (
                f"Invalid data type for ROS channel '{name}'. Expected {self._output_channels[name].data_class}, got {type(data)}."
            )
            self._output_channels[name].publish(data)
        else:
            self._logger.warning(f"ROS channel '{name}' is not created.")
