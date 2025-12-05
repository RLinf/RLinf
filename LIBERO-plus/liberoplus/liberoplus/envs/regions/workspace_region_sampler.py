import numpy as np

from .base_region_sampler import MultiRegionRandomSampler
from robosuite.utils.transform_utils import quat_multiply


class TableRegionSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class Libero100TableRegionSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class ObjectBasedSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )
