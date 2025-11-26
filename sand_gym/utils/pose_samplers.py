import collections
import numpy as np
from robosuite.utils.transform_utils import quat_multiply

class PoseSampler:
    """
    Base class of object placement sampler.

    Args:
        name (str): Name of this sampler.

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
    """

    def __init__(
        self,
        name,
        reference_pos=(0, 0, 0),
        reference_quat=(0.0, 0.0, 0.0, 1.0),
    ):
        # Setup attributes
        self.name = name
        self.reference_pos = reference_pos
        self.reference_quat = reference_quat

    def sample(self, reference=None):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            reference (str or 3-tuple or None): if provided, sample relative placement.

        Return:
            tuple: ((x,y,z),(qx,qy,qz,qw))
        """
        raise NotImplementedError


class UniformRandomPoseSampler(PoseSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly sample poses

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly sample poses

        z_range (2-array of float): Specify the (min, max) relative z_range used to uniformly sample poses

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_boundary (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
    """

    def __init__(
        self,
        name,
        x_range=(0, 0),
        y_range=(0, 0),
        z_range=(0, 0),
        rotation=0.0,
        rotation_axis="z",
        reference_pos=(0.0, 0.0, 0.0),
        reference_quat=(0.0, 0.0, 0.0, 1.0),
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis

        super().__init__(
            name=name,
            reference_pos=reference_pos,
            reference_quat = reference_quat,
        )

    def _sample_x(self):
        """
        Samples the x location

        Args:
            None

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self):
        """
        Samples the y location for a given object

        Args:
            None

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_z(self):
        """
        Samples the z location for a given object

        Args:
            None

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.z_range
        return np.random.uniform(high=maximum, low=minimum)
    
    def _sample_quat(self):
        """
        Samples the orientation for a given pose

        Returns:
            np.array: sampled pose quaternion in (x,y,z,w) form

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(high=max(self.rotation), low=min(self.rotation))
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            quat = (np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2))
        elif self.rotation_axis == "y":
            quat =  (0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2))
        elif self.rotation_axis == "z":
            quat =  (0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2))
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis)
            )
        
        return quat

    def sample(self, reference=None):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

        Return:
            tuple: (x,y,z,qx,qy,qz,qw)

        Raises:
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        if reference is None:
            base_offset = self.reference_pos
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(base_offset)

        # Sample pos and quat
        pos_x = self._sample_x() + base_offset[0]
        pos_y = self._sample_y() + base_offset[1]
        pos_z = self._sample_z() + base_offset[2]

        # random rotation
        quat = self._sample_quat()
        quat = tuple(quat_multiply(quat, self.reference_quat))

        # location is valid, put the pos down
        pos = (pos_x, pos_y, pos_z)

        sampled_pose = (pos, quat)

        return sampled_pose