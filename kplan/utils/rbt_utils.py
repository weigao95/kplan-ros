import attr
import os
from typing import List, Dict
import numpy as np

# The drake staff
try:
    import pydrake
    import pydrake.attic.multibody.shapes as shapes
    from pydrake.attic.multibody.collision import CollisionElement
    from pydrake.attic.multibody.rigid_body import RigidBody
    from pydrake.attic.multibody.joints import FixedJoint
    from pydrake.all import (
        RigidBodyTree,
        AddFlatTerrainToWorld,
        AddModelInstanceFromUrdfFile,
        AddModelInstancesFromSdfString,
        RigidBodyFrame,
        FloatingBaseType
    )
except ImportError:
    import pydrake
    import pydrake.multibody.shapes as shapes
    from pydrake.multibody.collision import CollisionElement
    from pydrake.multibody.rigid_body import RigidBody
    from pydrake.multibody.joints import FixedJoint
    from pydrake.multibody.rigid_body_tree import (
        RigidBodyTree,
        RigidBodyFrame,
        FloatingBaseType
    )


# Ideally, everything should be written in a urdf file.
# However, it might be easier to write some of commonly used them
# into config file that is easily modifiable
class ElementBase(object):

    def from_data_map(self, data_map):  # type: (Dict) -> None
        raise NotImplementedError

    def to_data_map(self):  # type: () -> Dict
        raise NotImplementedError

    @staticmethod
    def type_key():
        raise NotImplementedError


@attr.s
class BoxConfig(ElementBase):
    dim_xyz = list()  # type: List[float]
    box_in_world = np.ndarray(shape=(4, 4))

    def from_data_map(self, data_map):  # type: (Dict) -> None
        self.dim_xyz = data_map['dim_xyz']
        mat_list = data_map['box_in_world']  # type: List[List[float]]
        self.box_in_world = np.eye(4)
        for i in range(4):
            for j in range(4):
                self.box_in_world[i, j] = mat_list[i][j]

    def to_data_map(self):  # type: () -> Dict
        data_map = dict()
        data_map['dim_xyz'] = self.dim_xyz
        mat_list = []
        for i in range(4):
            row_i = []
            for j in range(4):
                row_i.append(float(self.box_in_world[i, j]))
            mat_list.append(row_i)
        data_map['box_in_world'] = mat_list
        return data_map

    @staticmethod
    def type_key():
        return 'Box'


@attr.s
class BoxContainerConfig(ElementBase):
    dim_xyz = list()
    center_xyz = list()  # Only support axis-aligned now
    thickness = 0.01  # 1cm

    def to_data_map(self):  # type: () -> Dict
        data_map = dict()
        data_map['dim_xyz'] = self.dim_xyz
        data_map['center_xyz'] = self.center_xyz
        data_map['thickness'] = self.thickness
        return data_map

    def from_data_map(self, data_map):  # type: (Dict) -> None
        self.dim_xyz = data_map['dim_xyz']
        self.center_xyz = data_map['center_xyz']
        self.thickness = data_map['thickness']

    @staticmethod
    def type_key():
        return 'BoxContainer'


@attr.s
class CylinderConfig(ElementBase):
    start_point = list()
    end_point = list()
    radius = 0.01

    def to_data_map(self):  # type: () -> Dict
        data_map = dict()
        data_map['start_point'] = self.start_point
        data_map['end_point'] = self.end_point
        data_map['radius'] = self.radius
        return data_map

    def from_data_map(self, data_map):  # type: (Dict) -> None
        self.start_point = data_map['start_point']
        self.end_point = data_map['end_point']
        self.radius = data_map['radius']

    @staticmethod
    def type_key():
        return 'Cylinder'


def add_box_to_rbt(rbt, box_config, name='box_0'):
    # type: (RigidBodyTree, BoxConfig, str) -> List[str]
    rigid_body = RigidBody()
    rigid_body.set_name(name)

    # The fixed joint of rigid body
    fixed_joint = FixedJoint(name+'joint', box_config.box_in_world)
    rigid_body.add_joint(rbt.world(), fixed_joint)

    # The visual element
    box_element = shapes.Box(box_config.dim_xyz)
    box_visual_element = shapes.VisualElement(
        box_element, np.eye(4), [1., 0., 0., 1.])
    rigid_body.AddVisualElement(box_visual_element)
    rbt.add_rigid_body(rigid_body)

    # The collision element
    box_collision_element = CollisionElement(box_element, np.eye(4))
    box_collision_element.set_body(rigid_body)
    rbt.addCollisionElement(box_collision_element, rigid_body, 'default')
    return [name]


def add_cylinder_to_rbt(rbt, cylinder_config, name='cylinder_0'):
    # type: (RigidBodyTree, CylinderConfig, str) -> List[str]
    # Construct the shape
    start_point = np.asarray(cylinder_config.start_point)
    end_point = np.asarray(cylinder_config.end_point)
    length = np.linalg.norm(start_point - end_point)  # type: float
    radius = cylinder_config.radius  # type: float
    cylinder_element = shapes.Cylinder(radius=radius, length=length)

    # Compute the transform in world
    cylinder_in_world = np.eye(4)
    z_axis = (end_point - start_point) / (length)
    x_axis = np.array([z_axis[1], -z_axis[0], 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    center = 0.5 * (start_point + end_point)
    cylinder_in_world[0:3, 0] = x_axis
    cylinder_in_world[0:3, 1] = y_axis
    cylinder_in_world[0:3, 2] = z_axis
    cylinder_in_world[0:3, 3] = center

    # The fixed joint of rigid body
    rigid_body = RigidBody()
    rigid_body.set_name(name)
    fixed_joint = FixedJoint(name + 'joint', cylinder_in_world)
    rigid_body.add_joint(rbt.world(), fixed_joint)

    # The visual element
    cylinder_visual_element = shapes.VisualElement(
        cylinder_element, np.eye(4), [1., 0., 0., 1.])
    rigid_body.AddVisualElement(cylinder_visual_element)
    rbt.add_rigid_body(rigid_body)

    # The collision element
    cylinder_collision_element = CollisionElement(cylinder_element, np.eye(4))
    cylinder_collision_element.set_body(rigid_body)
    rbt.addCollisionElement(cylinder_collision_element, rigid_body, 'default')
    return [name]


def add_box_container_to_rbt(rbt, container_config, name='container_0'):
    # type: (RigidBodyTree, BoxContainerConfig, str) -> List[str]
    bottom_box = BoxConfig()
    bottom_box.dim_xyz = [container_config.dim_xyz[0], container_config.dim_xyz[1], container_config.thickness]
    bottom_box.box_in_world = np.eye(4)
    bottom_box.box_in_world[0:3, 3] = np.asarray([
        container_config.center_xyz[0],
        container_config.center_xyz[1],
        container_config.center_xyz[2] - 0.5 * container_config.dim_xyz[2]])
    add_box_to_rbt(rbt, bottom_box, name=name+'bottom')

    front_box = BoxConfig()
    front_box.dim_xyz = [container_config.thickness, container_config.dim_xyz[1], container_config.dim_xyz[2]]
    front_box.box_in_world = np.eye(4)
    front_box.box_in_world[0:3, 3] = np.asarray([
        container_config.center_xyz[0] - 0.5 * container_config.dim_xyz[0],
        container_config.center_xyz[1],
        container_config.center_xyz[2]])
    add_box_to_rbt(rbt, front_box, name=name+'front')

    back_box = BoxConfig()
    back_box.dim_xyz = [container_config.thickness, container_config.dim_xyz[1], container_config.dim_xyz[2]]
    back_box.box_in_world = np.eye(4)
    back_box.box_in_world[0:3, 3] = np.asarray([
        container_config.center_xyz[0] + 0.5 * container_config.dim_xyz[0],
        container_config.center_xyz[1],
        container_config.center_xyz[2]])
    add_box_to_rbt(rbt, back_box, name=name + 'back')

    left_box = BoxConfig()
    left_box.dim_xyz = [container_config.dim_xyz[0], container_config.thickness, container_config.dim_xyz[2]]
    left_box.box_in_world = np.eye(4)
    left_box.box_in_world[0:3, 3] = np.asarray([
        container_config.center_xyz[0],
        container_config.center_xyz[1] - 0.5 * container_config.dim_xyz[1],
        container_config.center_xyz[2]])
    add_box_to_rbt(rbt, left_box, name=name + 'left')

    right_box = BoxConfig()
    right_box.dim_xyz = [container_config.dim_xyz[0], container_config.thickness, container_config.dim_xyz[2]]
    right_box.box_in_world = np.eye(4)
    right_box.box_in_world[0:3, 3] = np.asarray([
        container_config.center_xyz[0],
        container_config.center_xyz[1] + 0.5 * container_config.dim_xyz[1],
        container_config.center_xyz[2]])
    add_box_to_rbt(rbt, right_box, name=name + 'right')

    # The name of rigid body
    return [
        name+'bottom',
        name+'front',
        name+'back',
        name+'left',
        name+'right'
    ]


# The code below are for visualization
def add_robot(rbt):  # type: (RigidBodyTree) -> None
    # The iiwa link path
    iiwa_urdf_path = os.path.join(
        pydrake.getDrakePath(),
        "manipulation", "models", "iiwa_description", "urdf",
        "iiwa14_primitive_collision.urdf")
    robot_base_frame = RigidBodyFrame(
        "robot_base_frame", rbt.world(),
        [0.0, 0.0, 0.0], [0, 0, 0])
    AddModelInstanceFromUrdfFile(iiwa_urdf_path, FloatingBaseType.kFixed,
                                 robot_base_frame, rbt)


def test_box_shape():
    # Empty rigidbody tree
    rbt = RigidBodyTree()

    # The iiwa link path
    add_robot(rbt)

    # The box
    box_config = BoxConfig()
    box_config.dim_xyz = [1, 1, 1]
    box_config.box_in_world = np.eye(4)
    add_box_to_rbt(rbt, box_config)
    rbt.compile()

    # Visualize the box
    from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
    visualizer = MeshCatVisualizer()
    visualizer.add_rigidbody_tree(rbt, rbt.getZeroConfiguration(), draw_collision=True)


def test_box_container():
    # Empty rigidbody tree
    rbt = RigidBodyTree()

    # The iiwa link path
    add_robot(rbt)

    # The box
    container_config = BoxContainerConfig()
    container_config.dim_xyz = [2, 1, 1]
    container_config.center_xyz = [0, 0, 0]
    add_box_container_to_rbt(rbt, container_config)
    rbt.compile()

    # Visualize the box
    from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
    visualizer = MeshCatVisualizer()
    visualizer.add_rigidbody_tree(rbt, rbt.getZeroConfiguration(), draw_collision=False)


def test_cylinder():
    # Empty rigidbody tree
    rbt = RigidBodyTree()

    # The iiwa link path
    add_robot(rbt)

    # Compute the transform
    import kplan.utils.transformations as transformations
    quat = [0.6784430069620968, 0, 0, 0.7346530380419237]
    rack2world = transformations.quaternion_matrix(quat)
    rack2world[0, 3] = 0.3285152706168804
    rack2world[1, 3] = 0.007816573364084667
    rack2world[2, 3] = -0.045688056593718634

    start_point = np.array([0.64248, -0.00062, 0.2854])
    end_point = [0.56776, 0.00287, 0.31459]
    transformed_start_point = rack2world[0:3, 0:3].dot(start_point) + rack2world[0:3, 3]
    transformed_end_point = rack2world[0:3, 0:3].dot(end_point) + rack2world[0:3, 3]

    # The cylinder
    cylinder = CylinderConfig()
    cylinder.start_point = []
    cylinder.end_point = []
    for i in range(3):
        cylinder.start_point.append(float(transformed_start_point[i]))
        cylinder.end_point.append(float(transformed_end_point[i]))
    cylinder.radius = 0.02
    print(cylinder.start_point)
    print(cylinder.end_point)
    add_cylinder_to_rbt(rbt, cylinder)
    rbt.compile()

    # Visualize the cylinder
    from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
    visualizer = MeshCatVisualizer()
    visualizer.add_rigidbody_tree(rbt, rbt.getZeroConfiguration(), draw_collision=False)


if __name__ == '__main__':
    # test_box_shape()
    # test_box_container()
    test_cylinder()
