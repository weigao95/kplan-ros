from sensor_msgs.msg import JointState
import numpy as np
import kplan.utils.transformations as transformation

# The ros
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Transform

# The drake staff
import pydrake
try:
    import pydrake.attic.multibody.shapes as shapes
    from pydrake.attic.multibody.collision import CollisionElement
    from pydrake.all import (
        RigidBodyTree,
        AddFlatTerrainToWorld,
        AddModelInstanceFromUrdfFile,
        AddModelInstancesFromSdfString,
        RigidBodyFrame,
        FloatingBaseType
    )
except ImportError:
    import pydrake.multibody.shapes as shapes
    from pydrake.multibody.collision import CollisionElement
    from pydrake.multibody.rigid_body_tree import (
        RigidBodyTree,
        AddFlatTerrainToWorld,
        AddModelInstanceFromUrdfFile,
        AddModelInstancesFromSdfString,
        RigidBodyFrame,
        FloatingBaseType
    )


def get_kuka_flatten_q(rbt, joint_state):
    # type: (RigidBodyTree, JointState) -> np.ndarray
    num_q = rbt.get_num_positions()
    q = np.zeros(shape=(num_q,))

    # Iterate over all positions
    for i in range(num_q):
        position_i_name = rbt.get_position_name(i)
        for j in range(len(joint_state.name)):
            if position_i_name == joint_state.name[j]:
                q[i] = joint_state.position[j]

    # OK
    return q


def transform_to_np(msg):
    # type: (Transform) -> np.ndarray
    w = msg.rotation.w
    x = msg.rotation.x
    y = msg.rotation.y
    z = msg.rotation.z
    quat = [w, x, y, z]
    matrix = transformation.quaternion_matrix(np.asanyarray(quat))
    matrix[0, 3] = msg.translation.x
    matrix[1, 3] = msg.translation.y
    matrix[2, 3] = msg.translation.z
    return matrix


class TFWrapper(object):

    def __init__(self):
        self._tf_buffer = None
        self._tf_listener = None

    def setup(self):
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

    def get_transform(self, from_frame_name, to_frame_name, ros_time=None):
        # type: (str, str, rospy.Time) -> TransformStamped
        """
        Get the transform from the 'from_frame' to the 'to_frame'. In other words,
        if x_from is the coordinate of a vector expressed in 'from_frame', we have
        x_to = get_transform() x_from
        Note that this definition is different from the default lookup_transform() in ros
        :param from_frame_name:
        :param to_frame_name:
        :param ros_time:
        :return:
        """
        if ros_time is None:
            ros_time = rospy.Time(0)

        # Invoke the transform finder
        transform = self._tf_buffer.lookup_transform(
            to_frame_name,
            from_frame_name,
            ros_time
        )
        return transform
