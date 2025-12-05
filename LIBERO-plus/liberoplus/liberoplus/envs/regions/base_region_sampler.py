import collections
import numpy as np
import os
import robosuite
import xml.etree.ElementTree as ET

from copy import copy
from robosuite.utils.errors import RandomizationError
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.utils.transform_utils import quat_multiply
import robosuite.utils.transform_utils as T


class MultiRegionRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.
    Args:
        name (str): Name of this sampler.
        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models
        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects
        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects
        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation
        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation
        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): If True, will check for correct (valid) object placements
        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        assert len(self.x_ranges) == len(self.y_ranges)
        self.num_ranges = len(self.x_ranges)
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.idx = 0

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class SiteRegionRandomSampler(ObjectPositionSampler):
    """
    Places all objects on a site
    Args:
        name (str): Name of this sampler.
        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models
        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects
        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects
        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation
        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation
        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): If True, will check for correct (valid) object placements
        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
        sim=None,
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        assert len(self.x_ranges) == len(self.y_ranges)
        self.num_ranges = len(self.x_ranges)
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.idx = 0
        self.sim = sim
        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object
        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for
        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

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

    def sample(self, sim, fixtures=None, reference=None, site_name="", on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            site_x, site_y, site_z = T.quat2mat(
                T.convert_quat(ref_quat, to="xyzw")
            ) @ sim.data.get_site_xpos(site_name)
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(horizontal_radius) + base_offset[0] + site_x
                object_y = self._sample_y(horizontal_radius) + base_offset[1] + site_y
                object_z = self.z_offset + base_offset[2] + site_z
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()
                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class InSiteRegionRandomSampler(SiteRegionRandomSampler):
    """
    Places an object inside a site
    Args:
        name (str): Name of this sampler.
        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models
        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects
        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects
        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation
        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation
        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): If True, will check for correct (valid) object placements
        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            x_ranges=x_ranges,
            y_ranges=y_ranges,
            rotation=rotation,
            rotation_axis=rotation_axis,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
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

    def sample(self, sim, fixtures=None, reference=None, site_name="", on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            # if on_top:
            #     base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            site_x, site_y, site_z = T.quat2mat(
                T.convert_quat(ref_quat, to="xyzw")
            ) @ sim.data.get_site_xpos(site_name)
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(0) + base_offset[0] + site_x
                object_y = self._sample_y(0) + base_offset[1] + site_y
                object_z = self.z_offset + base_offset[2] + site_z
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                import pdb

                pdb.set_trace()
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class SiteSequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially. Allows chaining
    multiple placement initializers together - so that object locations can
    be sampled on top of other objects or relative to other object placements.
    Args:
        name (str): Name of this sampler.
    """

    def __init__(self, name):
        # Samplers / args will be filled in later
        self.samplers = collections.OrderedDict()
        self.sample_args = collections.OrderedDict()

        super().__init__(name=name)

    def append_sampler(self, sampler, sample_args=None):
        """
        Adds a new placement initializer with corresponding @sampler and arguments
        Args:
            sampler (ObjectPositionSampler): sampler to add
            sample_args (None or dict): If specified, should be additional arguments to pass to @sampler's sample()
                call. Should map corresponding sampler's arguments to values (excluding @fixtures argument)
        Raises:
            AssertionError: [Object name in samplers]
        """
        # Verify that all added mujoco objects haven't already been added, and add to this sampler's objects dict
        for obj in sampler.mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        self.samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args

    def hide(self, mujoco_objects):
        """
        Helper method to remove an object from the workspace.
        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to hide
        """
        sampler = UniformRandomSampler(
            name="HideSampler",
            mujoco_objects=mujoco_objects,
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis="z",
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
        )
        self.append_sampler(sampler=sampler)

    def add_objects(self, mujoco_objects):
        """
        Override super method to make sure user doesn't call this (all objects should implicitly belong to sub-samplers)
        """
        raise AttributeError(
            "add_objects() should not be called for SequentialCompsiteSamplers!"
        )

    def add_objects_to_sampler(self, sampler_name, mujoco_objects):
        """
        Adds specified @mujoco_objects to sub-sampler with specified @sampler_name.
        Args:
            sampler_name (str): Existing sub-sampler name
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to add
        """
        # First verify that all mujoco objects haven't already been added, and add to this sampler's objects dict
        mujoco_objects = (
            [mujoco_objects]
            if isinstance(mujoco_objects, MujocoObject)
            else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        # Make sure sampler_name exists
        assert sampler_name in self.samplers.keys(), (
            "Invalid sub-sampler specified, valid options are: {}, "
            "requested: {}".format(self.samplers.keys(), sampler_name)
        )
        # Add the mujoco objects to the requested sub-sampler
        self.samplers[sampler_name].add_objects(mujoco_objects)

    def reset(self):
        """
        Resets this sampler. In addition to base method, iterates over all sub-samplers and resets them
        """
        super().reset()
        for sampler in self.samplers.values():
            sampler.reset()

    def sample(self, sim, fixtures=None, reference=None, on_top=True):
        """
        Sample from each placement initializer sequentially, in the order
        that they were appended.
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. This will override each
                sampler's @reference argument if not already specified. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object. This will override each
                sampler's @on_top argument if not already specified. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        Raises:
            RandomizationError: [Cannot place all objects]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)

        # Iterate through all samplers to sample
        for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):
            # Pre-process sampler args
            if s_args is None:
                s_args = {}
            for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg
            # Run sampler
            new_placements = sampler.sample(sim=sim, fixtures=placed_objects, **s_args)
            # Update placements
            placed_objects.update(new_placements)

        return placed_objects
