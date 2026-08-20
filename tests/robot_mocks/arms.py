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

"""Fake arm SDKs: franky, ROS, the GimArm CAN stack, and the DOSW1 SDK."""

from __future__ import annotations

import threading
import time
import types
from typing import Any

import numpy as np

from ._fakes import module

#: A pose that is reachable and obviously not zero, so a test can tell the
#: difference between "read the arm" and "read nothing".
HOME_JOINTS = (0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8)
HOME_TCP = (0.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0)


def franky() -> types.ModuleType:
    """A ``franky`` whose robot answers with a fixed pose."""

    class Affine:
        def __init__(self, matrix=None):
            self.matrix = matrix
            self.translation = np.asarray(HOME_TCP[:3], dtype=np.float64)
            self.quaternion = np.asarray(HOME_TCP[3:], dtype=np.float64)

    class Gripper:
        def __init__(self, *_args, **_kwargs):
            self.position = 0.04
            self.is_open = True

        def is_ready(self):
            return True

        def open(self, speed=1.0):
            self.is_open = True

        def close(self, speed=1.0):
            self.is_open = False

        def cleanup(self):
            pass

    class Model:
        def zero_jacobian(self, _frame, _state):
            return np.zeros((6, 7))

    class Robot:
        def __init__(self, ip):
            self.ip = ip
            self.relative_dynamics_factor = 1.0
            self.model = Model()
            self.moved = []

        @property
        def state(self):
            return types.SimpleNamespace(
                O_T_EE=Affine(),
                q=np.asarray(HOME_JOINTS, dtype=np.float64),
                dq=np.zeros(7),
                # Wrench in the stiffness frame, which is what the arm reads.
                K_F_ext_hat_K=np.zeros(6),
                O_F_ext_hat_K=np.zeros(6),
                elbow=np.zeros(2),
            )

        def recover_from_errors(self):
            return True

        def set_collision_behavior(self, *_args):
            return True

        def move(self, motion):
            self.moved.append(motion)

        def join_motion(self):
            return True

    class _Tracker:
        """Keeps the targets it was given, so a test can read them back."""

        def __init__(self, *_args, **_kwargs):
            self.targets: list[Any] = []

        def set_target(self, target, **kwargs):
            self.targets.append((target, kwargs))

        def stop(self):
            return None

    return module(
        "franky",
        Robot=Robot,
        Gripper=Gripper,
        Affine=Affine,
        Frame=types.SimpleNamespace(EndEffector="EndEffector"),
        JointImpedanceTracker=_Tracker,
        CartesianImpedanceTracker=_Tracker,
        JointMotion=lambda *a, **k: ("joint", a, k),
        CartesianMotion=lambda *a, **k: ("cartesian", a, k),
        # A joint waypoint, which a reset motion is built from.
        JointState=lambda position=None, **_k: types.SimpleNamespace(
            position=np.asarray(
                HOME_JOINTS if position is None else position, dtype=np.float64
            )
        ),
        ReferenceType=types.SimpleNamespace(Absolute="Absolute"),
    )


def ros() -> dict[str, types.ModuleType]:
    """A ``rospy`` and the message packages the Franka arm imports."""
    published: list[tuple[str, Any]] = []

    class Publisher:
        """Keeps its message type: the transport checks what is put on it."""

        def __init__(self, name, data_class, **_kwargs):
            self.name = name
            self.data_class = data_class

        def publish(self, message):
            published.append((self.name, message))

    class Subscriber:
        """Delivers one message as soon as it subscribes.

        A ROS channel counts as up once its callback has fired, so a
        subscriber that never delivers leaves the arm waiting for a robot that
        looks switched off. One message is what a live robot publishing state
        looks like from here.
        """

        def __init__(self, name, data_class, callback, **_kwargs):
            self.name = name
            self.callback = callback
            self.data_class = data_class
            self.publish()

        def publish(self, message=None):
            """Deliver a message, defaulting to a fresh one of this type."""
            self.callback(message if message is not None else self.data_class())

    class _Timer:
        """A ROS timer that actually fires.

        Parts drive themselves from timers: Turtle2 interpolates toward its
        commanded pose on one, and an arm that never moves leaves the env
        waiting for a reset that cannot converge. The thread is a daemon and
        stops on ``shutdown``, as rospy's does.
        """

        def __init__(self, period, callback, oneshot=False, **_kwargs):
            self.period = float(period)
            self.callback = callback
            self._running = True
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

        def _tick(self):
            while self._running:
                time.sleep(max(self.period, 0.001))
                if not self._running:
                    return
                try:
                    self.callback(types.SimpleNamespace(current_real=0.0))
                except Exception:  # pragma: no cover - a timer must not die
                    return

        def shutdown(self):
            self._running = False

    def _message(**fields):
        """A message *class*: the transport isinstance-checks what it publishes."""

        names = list(fields)

        def __init__(self, *positional, **overrides):
            # ROS message types take their fields positionally too, which is
            # how geometry_msgs.Point(x, y, z) is spelled at the call site.
            values = dict(fields)
            values.update(dict(zip(names, positional)))
            values.update(overrides)
            for name, value in values.items():
                setattr(self, name, value)

        return type("Message", (), {"__init__": __init__, "_fields": fields})

    rospy = module(
        "rospy",
        init_node=lambda *a, **k: None,
        Publisher=Publisher,
        Subscriber=Subscriber,
        Message=object,
        set_param=lambda *a, **k: None,
        get_param=lambda *a, **k: None,
        Time=types.SimpleNamespace(now=lambda: 0.0),
        Duration=lambda seconds: seconds,
        Timer=_Timer,
        Rate=lambda _hz: types.SimpleNamespace(sleep=lambda: None),
        sleep=lambda _seconds: None,
        loginfo=lambda *_a, **_k: None,
        logwarn=lambda *_a, **_k: None,
        is_shutdown=lambda: False,
        signal_shutdown=lambda *_a: None,
    )
    rospy.published = published

    pose = _message(
        header=types.SimpleNamespace(stamp=0.0, frame_id=""),
        pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    geometry = module(
        "geometry_msgs.msg",
        PoseStamped=pose,
        Pose=_message(),
        Point=_message(x=0.0, y=0.0, z=0.0),
        Quaternion=_message(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    franka_msgs = module(
        "franka_msgs.msg",
        FrankaState=_message(
            # Column-major, as libfranka publishes it.
            O_T_EE=list(np.eye(4).flatten(order="F")),
            q=list(HOME_JOINTS),
            dq=[0.0] * 7,
            # Wrench in the stiffness frame; the arm reads force and torque
            # out of this one message.
            K_F_ext_hat_K=[0.0] * 6,
            O_F_ext_hat_K=[0.0] * 6,
            robot_mode=2,
        ),
        ErrorRecoveryActionGoal=_message(),
    )
    serl = module(
        "serl_franka_controllers.msg", ZeroJacobian=_message(zero_jacobian=[0.0] * 42)
    )
    reconfigure = module(
        "dynamic_reconfigure.client",
        Client=lambda *a, **k: types.SimpleNamespace(
            update_configuration=lambda *_: None
        ),
    )

    def _gripper_goal():
        """The goal a grasp or move message carries, with room for its fields."""
        return types.SimpleNamespace(
            width=0.0,
            speed=0.0,
            force=0.0,
            epsilon=types.SimpleNamespace(inner=0.0, outer=0.0),
        )

    gripper_msgs = module(
        "franka_gripper.msg",
        GraspActionGoal=_message(goal=_gripper_goal()),
        MoveActionGoal=_message(goal=_gripper_goal()),
    )
    sensor_msgs = module("sensor_msgs.msg", JointState=_message(position=[0.0, 0.0]))
    bridge = module(
        "cv_bridge",
        CvBridge=lambda: types.SimpleNamespace(
            imgmsg_to_cv2=lambda *_a, **_k: np.zeros((48, 64, 3), dtype=np.uint8)
        ),
    )

    made = {
        "rospy": rospy,
        "geometry_msgs": module("geometry_msgs"),
        "geometry_msgs.msg": geometry,
        "franka_msgs": module("franka_msgs"),
        "franka_msgs.msg": franka_msgs,
        "serl_franka_controllers": module("serl_franka_controllers"),
        "serl_franka_controllers.msg": serl,
        "dynamic_reconfigure": module("dynamic_reconfigure"),
        "dynamic_reconfigure.client": reconfigure,
        "franka_gripper": module("franka_gripper"),
        "franka_gripper.msg": gripper_msgs,
        "sensor_msgs": module("sensor_msgs"),
        "sensor_msgs.msg": sensor_msgs,
        "cv_bridge": bridge,
    }
    for dotted, leaf in (
        ("geometry_msgs", geometry),
        ("franka_msgs", franka_msgs),
        ("serl_franka_controllers", serl),
        ("dynamic_reconfigure", reconfigure),
        ("franka_gripper", gripper_msgs),
        ("sensor_msgs", sensor_msgs),
    ):
        setattr(made[dotted], dotted.split(".")[-1] if False else "msg", leaf)
    made["dynamic_reconfigure"].client = reconfigure
    return made


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Every arm SDK, by the name a part imports it as."""
    made = {"franky": franky()}
    made.update(ros())
    return made
