"""
Abstract base class for bimanual robot headless visualization.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import viser
import viser.extras
import viser.transforms as vtf

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description as load_urdf_robot_description
except ImportError as e:
    print(f"Error importing robot_descriptions: {e}")
    print("Please run: pip install git+https://github.com/uynitsuj/robot_descriptions.py.git")
    exit()


@dataclass
class TransformHandle:
    """Data class to store transform handles."""

    tcp_offset_frame: viser.FrameHandle
    control: Optional[viser.TransformControlsHandle] = None


class ViserAbstractBase(ABC):
    """
    Abstract base class for bimanual robot visualization.
    - This class provides common functionality for different IK solvers
    - Subclasses must implement the solve_ik method with their specific solver
    - robot_description: the name of the robot description to load (default: yam_description)
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server: Optional[viser.ViserServer] = None,
        robot_description: str = "yam_description",
        bimanual: bool = False,
        coordinate_frame: Literal["base", "world"] = "base",
    ):
        self.rate = rate
        self.bimanual = bimanual
        self.coordinate_frame = coordinate_frame

        # Load robot description with error handling
        try:
            self.urdf = load_urdf_robot_description(robot_description)
        except Exception as e:
            print(f"Error loading robot description '{robot_description}': {e}")
            print("Please ensure the robot description is properly installed.")
            exit()

        self.viser_server = viser_server if viser_server is not None else viser.ViserServer()

        self.joints = {"left": np.zeros(6)}
        if bimanual:
            self.joints["right"] = np.zeros(6)

        self._setup_solver_specific()

        self._setup_visualization()
        self._setup_gui()
        self._setup_transform_handles()

    def _setup_visualization(self):
        """Setup basic visualization elements."""
        self.base_frame = self.viser_server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis_left = viser.extras.ViserUrdf(self.viser_server, self.urdf, root_node_name="/base")

        # Add ground grid with configurable size
        self.ground_grid = self.viser_server.scene.add_grid("ground", width=2, height=2, cell_size=0.1)

    def _setup_gui(self):
        """Setup enhanced GUI elements."""
        # Timing display
        self.timing_handle = self.viser_server.gui.add_number("Time (ms)", 0.01, disabled=True)

        # Transform control size
        self.tf_size_handle = self.viser_server.gui.add_slider(
            "Gizmo size", min=0.05, max=0.4, step=0.01, initial_value=0.2
        )

        # Coordinate frame selector
        self.coordinate_frame_handle = self.viser_server.gui.add_dropdown(
            "Coordinate Frame", options=["base", "world"], initial_value=self.coordinate_frame
        )

        # Reset button
        self.reset_button = self.viser_server.gui.add_button("Reset to Rest Pose")

        # Reset to rest pose
        @self.reset_button.on_click
        def _(_evt):
            self.home()

    def _setup_transform_handles(self):
        """Setup transform handles for end effectors with enhanced features."""
        self.transform_handles = {
            "left": TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "target_left/tcp_offset", show_axes=False, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
                ),
                control=self.viser_server.scene.add_transform_controls("target_left", scale=self.tf_size_handle.value),
            ),
        }
        if self.bimanual:
            self.transform_handles["right"] = TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "target_right/tcp_offset", show_axes=False, position=(0.0, 0.0, 0.0), wxyz=(1, 0, 0, 0)
                ),
                control=self.viser_server.scene.add_transform_controls(
                    "target_right", scale=self.tf_size_handle.value
                ),
            )

        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for handle in self.transform_handles.values():
                if handle.control:
                    handle.control.scale = self.tf_size_handle.value
            self._update_optional_handle_sizes()

        self._initialize_transform_handles()

    @property
    def urdf_joint_names(self):
        """Get URDF joint names."""
        return self.urdf.joint_names

    @abstractmethod
    def _update_optional_handle_sizes(self):
        """Override in subclasses to update optional handle sizes."""
        pass

    def update_visualization(self):
        """Update visualization with current state."""
        self.urdf_vis_left.update_cfg(self.joints["left"])

    def get_target_poses(self):
        """Get target poses from transform controls with optional TCP offset."""
        target_poses = {}

        for side, handle in self.transform_handles.items():
            if handle.control is None:
                continue

            # Combine control handle with TCP offset
            control_tf = vtf.SE3(np.array([*handle.control.wxyz, *handle.control.position]))
            tcp_offset_tf = vtf.SE3(np.array([*handle.tcp_offset_frame.wxyz, *handle.tcp_offset_frame.position]))
            target_poses[side] = control_tf @ tcp_offset_tf

        return target_poses

    def set_ee_targets(self, left_wxyz_xyz: np.ndarray, right_wxyz_xyz: np.ndarray):
        """
        Set end effector targets.
        left_wxyz_xyz: [wxyz, xyz]
        right_wxyz_xyz: [wxyz, xyz]
        """
        if "left" in self.transform_handles and self.transform_handles["left"].control is not None:
            self.transform_handles["left"].control.wxyz = left_wxyz_xyz[:4]  # type: ignore
            self.transform_handles["left"].control.position = left_wxyz_xyz[4:]  # type: ignore
        if self.bimanual and "right" in self.transform_handles and self.transform_handles["right"].control is not None:
            self.transform_handles["right"].control.wxyz = right_wxyz_xyz[:4]  # type: ignore
            self.transform_handles["right"].control.position = right_wxyz_xyz[4:]  # type: ignore

    def set_tcp_offsets(self, left_offset: np.ndarray, right_offset: np.ndarray):
        """
        Set TCP offset frames.
        left_offset: [wxyz, xyz]
        right_offset: [wxyz, xyz]
        """
        if "left" in self.transform_handles:
            self.transform_handles["left"].tcp_offset_frame.wxyz = left_offset[:4]
            self.transform_handles["left"].tcp_offset_frame.position = left_offset[4:]
        if self.bimanual and "right" in self.transform_handles:
            self.transform_handles["right"].tcp_offset_frame.wxyz = right_offset[:4]
            self.transform_handles["right"].tcp_offset_frame.position = right_offset[4:]

    @abstractmethod
    def home(self):
        """Reset robot to rest pose. Must be implemented by subclasses."""
        raise NotImplementedError

    def run(self):
        """Main run loop with enhanced timing."""
        while True:
            start_time = time.time()

            self.solve_ik()
            self.update_visualization()

            # Update timing with exponential moving average
            elapsed_time = time.time() - start_time
            if hasattr(self, "timing_handle"):
                self.timing_handle.value = 0.99 * self.timing_handle.value + 0.01 * (elapsed_time * 1000)

            # Sleep to maintain desired rate
            sleep_time = max(0, (1.0 / self.rate) - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _setup_solver_specific(self):
        """Setup solver-specific components. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _initialize_transform_handles(self):
        """Initialize transform handle positions. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def solve_ik(self):
        """Solve inverse kinematics. Must be implemented by subclasses (only necessary if is an IK solver class)."""
        raise NotImplementedError
