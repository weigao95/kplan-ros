#! /usr/bin/env python
import os
import yaml
import numpy as np
import argparse
from threading import Lock
from kplan.kinematic_planning.rbt_env_builder import KukaIIwaEnvironmentBuilder
from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
import kplan.utils.ros_utils as ros_utils

# The ros staff
import rospy
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs.point_cloud2 as point_cloud2


# The argument for the planner
parser = argparse.ArgumentParser()
parser.add_argument('--env_config_path',
                    type=str,
                    default='/home/wei/catkin_ws/src/kplan_ros/kplan/kinematic_planning/config/rack_env.yaml',
                    help='The path to environment yaml description.')
parser.add_argument('--camera_name',
                    type=str,
                    default='carmine_1',
                    help='The name of the camera')


class MeshcatPointCloudStreamer(object):

    def __init__(self, rbt_env_config_path, camera_name='carmine_1'):  # type: (str, str) -> None
        # Load the map
        assert os.path.exists(rbt_env_config_path)
        with open(rbt_env_config_path, 'r') as config_file:
            load_map = yaml.load(config_file, Loader=yaml.CLoader)

        # Get the element list
        self._element_list = []
        if 'element_list' in load_map:
            self._element_list = KukaIIwaEnvironmentBuilder.parse_element(load_map['element_list'])

        # The name of camera
        self.camera_name = camera_name

        # The robot env
        self._rbt_env = KukaIIwaEnvironmentBuilder.build_rbt_env(self._element_list)

        # The tf
        self._tf = None  # type: ros_utils.TFWrapper

        # The visualizer
        self._meshcat_vis = MeshCatVisualizer()
        rbt = self._rbt_env.rbt
        self._meshcat_vis.add_rigidbody_tree(rbt, rbt.getZeroConfiguration(), draw_collision=False)

        # the lock
        self._vis_lock = Lock()

    def initialize(self):
        self._tf = ros_utils.TFWrapper()
        self._tf.setup()

    def get_camera2world(self):  # type: () -> np.ndarray
        rgb_optical_frame = 'camera_' + self.camera_name + "_rgb_optical_frame"
        base_frame = 'base'
        camera2world = self._tf.get_transform(rgb_optical_frame, base_frame)
        return ros_utils.transform_to_np(camera2world.transform)

    def _on_robot_joint_state(self, msg):  # type: (JointState) -> None
        # The gripper msg, ignored
        if len(msg.name) < 7:
            return

        # The robot msg
        q = ros_utils.get_kuka_flatten_q(self._rbt_env.rbt, joint_state=msg)

        # Send to meshcat
        self._vis_lock.acquire()
        self._meshcat_vis.update_rigidbody_tree_configuration(q)
        self._vis_lock.release()

    def _on_point_cloud(self, msg):  # type: (PointCloud2) -> None
        point_list = point_cloud2.read_points_list(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        n_point = len(point_list)

        # The point cloud in camera frame
        np_cloud_camera = np.zeros(shape=(n_point, 3))
        for i in range(n_point):
            point_i = point_list[i]
            np_cloud_camera[i, 0] = point_i.x
            np_cloud_camera[i, 1] = point_i.y
            np_cloud_camera[i, 2] = point_i.z

        # Transform into world frame
        camera2world = self.get_camera2world()
        np_cloud_world = camera2world[0:3, 0:3].dot(np_cloud_camera.T)
        np_cloud_world = np_cloud_world.T
        for i in range(3):
            np_cloud_world[:, i] += camera2world[i, 3]

        # Set the color
        color = self._meshcat_vis._make_meshcat_color(n_point, 254, 254, 254)

        # Update the frame
        self._vis_lock.acquire()
        self._meshcat_vis.add_pointcloud(np_cloud_world, color)
        self._vis_lock.release()

    def run(self):
        # Build the node
        rospy.init_node('meshcat_streamer')

        # Must be called after init_node
        self.initialize()

        # The point cloud topic
        cloud_topic = '/camera_' + str(self.camera_name) + '/depth_registered/points'
        rospy.Subscriber(cloud_topic, PointCloud2, self._on_point_cloud)

        # The joint state topic
        joint_topic = '/joint_states'
        rospy.Subscriber(joint_topic, JointState, self._on_robot_joint_state)

        # Run it
        print('The meshcat streaming initialization OK!')
        rospy.spin()


def main():
    # Parse the argument
    args, unknown = parser.parse_known_args()
    env_config_path = args.env_config_path
    camera_name = args.camera_name

    # Run it
    streamer = MeshcatPointCloudStreamer(env_config_path, camera_name)
    streamer.run()


if __name__ == '__main__':
    main()
