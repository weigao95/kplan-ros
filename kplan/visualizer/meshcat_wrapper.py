import numpy as np
import math
from typing import List, Dict, Optional
import random
import time
import meshcat
import meshcat.transformations as tf
import meshcat.geometry as meshcat_geometry

# The rigid body tree
try:
    from pydrake.multibody.rigid_body_tree import RigidBodyTree
    from pydrake.multibody.shapes import Shape
except ImportError:
    from pydrake.all import RigidBodyTree, Shape

# The geometry type line
try:
    from meshcat.geometry import Line
except ImportError:
    from meshcat.geometry import Object
    class Line(Object):
        _type = u"Line"


class MeshCatVisualizer(object):
    """
    The meshcat based visualizer for both robot and other
    things like point-cloud.
    The visualizer is stateful, but some commonly used drawing
    function would be provided such as draw_rigidbody_tree or
    animate_rigid_body_tree
    """
    def __init__(
            self,
            prefix='RBViz',  # type: str
            zmq_url='tcp://127.0.0.1:6000'  # type: str
    ):
        # The help info
        print('Please make sure meshcat-server is running in another terminal')
        # Setup the visualizer
        self._prefix = prefix
        self._meshcat_vis = meshcat.Visualizer(zmq_url=zmq_url)
        self._meshcat_vis[self._prefix].delete()

        # The map from key to rigid body tree
        # Note that rbt would be copied
        self._rbt_map = dict()  # type: Dict[str, RigidBodyTree]
        self._rbt_geometry_element_set = set()

        # The set of point cloud keys
        self._pointcloud_set = set()

    def add_rigidbody_tree(self, rbt, q, rbt_key='rbt_0', draw_collision=False):
        # type: (RigidBodyTree, np.ndarray, str, bool) -> None
        # Check the key
        assert len(rbt_key) > 0
        if rbt_key in self._rbt_map:
            print('The rigidbody tree %s is already in the visualizer.' % rbt_key)
            print('You should use non-default key if more than on RigidBodyTree are presented')
            return

        # The key is OK, insert the geometry. Note that this should not be the copied one
        self._add_rbt_geometry(rbt, rbt_key, draw_collision)

        # Copy the tree and store in the map for kinematic computation
        rbt_clone = rbt.Clone()
        self._rbt_map[rbt_key] = rbt_clone

        # Update the pose
        self.update_rigidbody_tree_configuration(q, rbt_key)

    def _add_rbt_geometry(self, rbt, rbt_key, draw_collision=False):
        # type: (RigidBodyTree, str, bool) -> None
        """
        Add the geometry of a RigidBodyTree to the meshcat visualizer.
        Note that later update to rbt WOULD NOT be reflected
        :param rbt:
        :param rbt_key: The key to index the rigid body tree
        :param draw_collision:
        :return:
        """
        n_bodies = rbt.get_num_bodies() - 1
        # all_meshcat_geometry = {}
        for body_i in range(n_bodies):
            # Get the body index
            body = rbt.get_body(body_i + 1)
            body_name = rbt_key + body.get_name() + ("(%d)" % body_i)

            if draw_collision:
                draw_elements = [rbt.FindCollisionElement(k) for k in body.get_collision_element_ids()]
            else:
                draw_elements = body.get_visual_elements()

            for element_i, element in enumerate(draw_elements):
                element_local_tf = element.getLocalTransform()
                if element.hasGeometry():
                    geom = element.getGeometry()

                    geom_type = geom.getShape()
                    if geom_type == Shape.SPHERE:
                        meshcat_geom = meshcat.geometry.Sphere(geom.radius)
                    elif geom_type == Shape.BOX:
                        meshcat_geom = meshcat.geometry.Box(geom.size)
                    elif geom_type == Shape.CYLINDER:
                        meshcat_geom = meshcat.geometry.Cylinder(
                            geom.length, geom.radius)
                        # In Drake, cylinders are along +z
                        # In meshcat, cylinders are along +y
                        # Rotate to fix this misalignment
                        extra_rotation = tf.rotation_matrix(
                            math.pi / 2., [1, 0, 0])
                        element_local_tf[0:3, 0:3] = \
                            element_local_tf[0:3, 0:3].dot(
                                extra_rotation[0:3, 0:3])
                    elif geom_type == Shape.MESH:
                        meshcat_geom = \
                            meshcat.geometry.ObjMeshGeometry.from_file(
                                geom.resolved_filename[0:-3] + "obj")
                        # respect mesh scale
                        element_local_tf[0:3, 0:3] *= geom.scale
                    elif geom_type == Shape.MESH_POINTS:
                        # The shape of points is (3, N)
                        points = geom.getPoints()  # type: np.ndarray
                        n_points = points.shape[1]
                        color = MeshCatVisualizer._make_meshcat_color(n_points, r=255, g=0, b=0)
                        meshcat_geom = meshcat_geometry.PointCloud(points, color.T, size=0.01)
                        self._meshcat_vis[self._prefix][body_name][str(element_i)] \
                            .set_object(meshcat_geom)
                        self._meshcat_vis[self._prefix][body_name][str(element_i)]. \
                            set_transform(element_local_tf)
                        self._rbt_geometry_element_set.add(body_name)
                        continue
                    else:
                        print "UNSUPPORTED GEOMETRY TYPE ", \
                            geom.getShape(), " IGNORED"
                        continue

                    # The color of the body
                    rgba = [1., 0.7, 0., 1.]
                    if not draw_collision:
                        rgba = element.getMaterial()
                    self._meshcat_vis[self._prefix][body_name][str(element_i)] \
                        .set_object(meshcat_geom,
                                    meshcat.geometry.MeshLambertMaterial(
                                        color=MeshCatVisualizer.rgba2hex(rgba)))
                    self._meshcat_vis[self._prefix][body_name][str(element_i)]. \
                        set_transform(element_local_tf)
                    self._rbt_geometry_element_set.add(body_name)

    def update_rigidbody_tree_configuration(self, q, rbt_key='rbt_0'):
        # type: (np.ndarray, str) -> None
        """
        Update the configuration of the RigidBodyTree corresponds to rbt_key
        :param q: The new configuration of the robot
        :param rbt_key: The key to index the robot
        :return:
        """
        assert rbt_key in self._rbt_map
        rbt = self._rbt_map[rbt_key]

        # The kinematic of the rbt
        kinsol = rbt.doKinematics(q)

        # Update the pose of each body
        for body_i in range(rbt.get_num_bodies()-1):
            tf = rbt.relativeTransform(kinsol, 0, body_i+1)
            body = rbt.get_body(body_i+1)
            body_name = rbt_key + body.get_name() + ("(%d)" % body_i)
            if body_name in self._rbt_geometry_element_set:
                self._meshcat_vis[self._prefix][body_name].set_transform(tf)

    def add_pointcloud(self, cloud, color=None, cloud_key='cloud_0'):
        # type: (np.ndarray, np.ndarray, str) -> None
        """
        Add a point cloud to the visualizer
        :param cloud: (N, 3) np.ndarray
        :param color: (N, 3) np.ndarray of (r, g, b) color or None
        :param cloud_key: The key used to index the point cloud
        :return:
        """
        if cloud_key in self._pointcloud_set:
            print('Duplicate pointcloud key detected')
            print('You should use update_pointcloud instread')
            return

        # The color
        if color is None:
            color = MeshCatVisualizer._random_color_meshcat(cloud.shape[0])

        # Check shape
        assert cloud.shape[0] == color.shape[0]
        assert cloud.shape[1] == 3
        assert color.shape[1] == 3

        # Add to meshcat
        self._meshcat_vis[self._prefix][cloud_key].set_object(meshcat_geometry.PointCloud(cloud.T, color.T))

    def remove_cloud(self, cloud_key):
        # type: (str) -> bool
        if cloud_key not in self._pointcloud_set:
            return False

        # Remove it
        self._meshcat_vis[self._prefix][cloud_key].delete()
        self._pointcloud_set.remove(cloud_key)
        return True

    def update_pointcloud(self, cloud, color=None, cloud_key='cloud_0'):
        # type: (np.ndarray, np.ndarray, str) -> None
        self.remove_cloud(cloud_key)
        self.add_pointcloud(cloud, color, cloud_key)

    def add_frame(self, T_frame, frame_name='frame_0', frame_scale=1.0):
        # type: (np.ndarray, str, float) -> None
        """
        :param T_frame: 4x4 np.ndarray that is the frame expressed in world frame
        :param frame_name:
        :param frame_scale: The length of the frame line
        :return:
        """
        # Build the vertices
        vertices_x = np.zeros(shape=(3, 2))
        vertices_x[:, 0] = T_frame[0:3, 3]
        vertices_x[:, 1] = vertices_x[:, 0] + frame_scale * T_frame[0:3, 0]

        vertices_y = np.zeros(shape=(3, 2))
        vertices_y[:, 0] = T_frame[0:3, 3]
        vertices_y[:, 1] = vertices_y[:, 0] + frame_scale * T_frame[0:3, 1]

        vertices_z = np.zeros(shape=(3, 2))
        vertices_z[:, 0] = T_frame[0:3, 3]
        vertices_z[:, 1] = vertices_z[:, 0] + frame_scale * T_frame[0:3, 2]

        # Send to meshcat
        line_x_name = frame_name + 'x_axis'
        line_y_name = frame_name + 'y_axis'
        line_z_name = frame_name + 'z_axis'
        self._meshcat_vis[self._prefix][line_x_name].set_object(geometry=Line(
            meshcat_geometry.PointsGeometry(vertices_x), meshcat_geometry.MeshBasicMaterial(color=0xff0000)
        ))
        self._meshcat_vis[self._prefix][line_y_name].set_object(geometry=Line(
            meshcat_geometry.PointsGeometry(vertices_y), meshcat_geometry.MeshBasicMaterial(color=0x00ff00)
        ))
        self._meshcat_vis[self._prefix][line_z_name].set_object(geometry=Line(
            meshcat_geometry.PointsGeometry(vertices_z), meshcat_geometry.MeshBasicMaterial(color=0x0000ff)
        ))

    def update_frame(self, T_frame, frame_name='frame_0', frame_scale=1.0):
        # type: (np.ndarray, str, float) -> None
        self.add_frame(T_frame, frame_name, frame_scale)

    def add_point(self, point_in_world, point_name='point_0', radius=0.02):
        # type: (np.ndarray, str, float) -> None
        """
        Add a point to the world, represent by sphere
        :param point_in_world: (3,) np.ndarray represent the point
        :param point_name: the key used to index the point
        :param radius: the radius in METER (by default 5 cm)
        :return:
        """
        meshcat_sphere_geom = meshcat.geometry.Sphere(radius)
        self._meshcat_vis[self._prefix][point_name].set_object(
            meshcat_sphere_geom,
            meshcat_geometry.MeshBasicMaterial(color=0x0000ff)
        )

        point_in_world_tf = np.eye(4)
        point_in_world_tf[0:3, 3] = point_in_world
        self._meshcat_vis[self._prefix][point_name].set_transform(point_in_world_tf)

    @staticmethod
    def _random_color_meshcat(N):
        # type: (int) -> np.ndarray
        rand_r = random.randint(0, 255)
        rand_g = random.randint(0, 255)
        rand_b = random.randint(0, 255)
        return MeshCatVisualizer._make_meshcat_color(N, rand_r, rand_g, rand_b)

    @staticmethod
    def rgba2hex(rgb):
        # type: (List[int]) -> int
        """
        Turn a list of R,G,B elements (any indexable
        list of >= 3 elements will work), where each element
        is specified on range [0., 1.], into the equivalent
        24-bit value 0xRRGGBB.
        :param rgb:
        :return:
        """
        val = 0
        for i in range(3):
            val += (256 ** (2 - i)) * int(255 * rgb[i])
        return val

    @staticmethod
    def _make_meshcat_color(N, r, g, b):
        # type: (int, int, int, int) -> np.ndarray
        """
        Turn int8 space rgb into meshcat color
        :param N: the nominal size of cloud
        :param r: [0, 255]
        :param g: [0, 255]
        :param b: [0, 255]
        :return: np.ndarray of shape (N, 3)
        """
        color = np.zeros(shape=(N, 3))
        color[:, 0] = (float(r)/255.0)
        color[:, 1] = (float(g) / 255.0)
        color[:, 2] = (float(b) / 255.0)
        return color

    @staticmethod
    def draw_rigidbody_tree(rbt, q, draw_collision=False):
        # type: (RigidBodyTree, np.ndarray, bool) -> None
        visualizer = MeshCatVisualizer()
        visualizer.add_rigidbody_tree(rbt, q, draw_collision=draw_collision)

    @staticmethod
    def animate_rigidbody_tree(rbt, q_array, traj_interval=None, draw_collision=False, repeat_n_times=1):
        # type: (RigidBodyTree, np.ndarray, Optional[float], bool, int) -> None
        """
        Draw a robot with a sequence of pose
        :param rbt:
        :param q_array: (N, nq) np.ndarray where N is the number of steps
        :param traj_interval:
        :param draw_collision: original geometry or collision geometry
        :param repeat_n_times: Draw the trajectory with repetition
        :return:
        """
        if traj_interval is None:
            traj_interval = q_array.shape[0] * 0.2
        delay_interval = traj_interval / float(q_array.shape[0])
        visualizer = MeshCatVisualizer()
        visualizer.add_rigidbody_tree(rbt, q_array[0, :], draw_collision=draw_collision)

        # The drawing loop
        for j in range(repeat_n_times):
            for i in range(1, q_array.shape[0]):
                visualizer.update_rigidbody_tree_configuration(q_array[i, :])
                time.sleep(delay_interval)

    @staticmethod
    def draw_pointcloud(point, color=None):
        # type: (np.ndarray, np.ndarray) -> None
        """
        Handy helper for cloud drawing
        :param point: (N, 3) np.ndarray
        :param color: None or (N, 3) np.ndarray
        :return:
        """
        visualizer = MeshCatVisualizer()
        visualizer.add_pointcloud(point, color)


# The below are debug/test code
def run_pendulum_static_visualization():
    import os
    from pydrake.all import FloatingBaseType
    vis_path = os.path.dirname(__file__)
    pendulum_path = os.path.join(vis_path, 'data/pendulum.urdf')
    rbt = RigidBodyTree(pendulum_path, floating_base_type=FloatingBaseType.kFixed)

    # The zero config
    nq = rbt.get_num_positions()
    q = np.zeros(shape=nq)

    # Draw it
    MeshCatVisualizer.draw_rigidbody_tree(rbt, q)


def construct_iiwa_simple():
    import os
    import pydrake
    from pydrake.all import AddFlatTerrainToWorld, RigidBodyFrame, AddModelInstanceFromUrdfFile, FloatingBaseType

    # The rigid body tree
    rbt = RigidBodyTree()
    AddFlatTerrainToWorld(rbt)

    # The iiwa
    iiwa_urdf_path = os.path.join(
        pydrake.getDrakePath(),
        "manipulation", "models", "iiwa_description", "urdf",
        "iiwa14_polytope_collision.urdf")
    robot_base_frame = RigidBodyFrame(
        "robot_base_frame", rbt.world(),
        [0.0, 0.0, 0.0], [0, 0, 0])
    AddModelInstanceFromUrdfFile(iiwa_urdf_path, FloatingBaseType.kFixed,
                                 robot_base_frame, rbt)
    return rbt


def run_iiwa_static_visualization():
    rbt = construct_iiwa_simple()
    q = np.zeros(rbt.get_num_positions())

    # Draw it
    MeshCatVisualizer.draw_rigidbody_tree(rbt, q)


def run_static_pointcloud_visualization():
    import open3d, os
    vis_path = os.path.dirname(__file__)
    cloud_path = os.path.join(vis_path, 'data/bunny.pcd')
    pcd = open3d.read_point_cloud(cloud_path)
    pcd_np = np.asarray(pcd.points)
    MeshCatVisualizer.draw_pointcloud(pcd_np)


def run_point_visualization():
    visualizer = MeshCatVisualizer()
    point_in_world = np.array([0.5, 0.5, 0.0])
    visualizer.add_point(point_in_world)


if __name__ == '__main__':
    # run_iiwa_static_visualization()
    # run_static_pointcloud_visualization()
    run_point_visualization()
