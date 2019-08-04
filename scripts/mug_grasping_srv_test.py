import numpy as np
import open3d, os
from kplan.utils.transformations import quaternion_matrix

import rospy
from geometry_msgs.msg import Point, Point32
from kplan_ros.srv import KeypointGraspingService, KeypointGraspingServiceRequest, KeypointGraspingServiceResponse


def run_grasp_planning():
    # The transformation matrix
    camera2world = quaternion_matrix(
        [0.13363039718756375, -0.6601503565555793, 0.7226078483153184, -0.1555066597935349])
    camera2world[0, 3] = 0.29972335333107686
    camera2world[1, 3] = -0.00016958877060524544
    camera2world[2, 3] = 0.8278206244574912

    # The keypoint in camera frame
    keypoint_camera = np.zeros((3, 3))
    keypoint_camera[0, :] = np.array([-0.0710507483878, -0.0075068516829, 0.909739379883])
    keypoint_camera[1, :] = np.array([-0.0641933829504, 0.0298324972555, 0.838331695557])
    keypoint_camera[2, :] = np.array([-0.0684536122998, -0.0403231180828, 0.821209274292])

    # Transformed into world
    keypoint_world = np.zeros_like(keypoint_camera)
    for i in range(3):
        keypoint_world[i, :] = camera2world[0:3, 0:3].dot(keypoint_camera[i, :]) + camera2world[0:3, 3]

    # Read the point cloud
    project_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)
    project_path = os.path.abspath(project_path)
    pcdpath = os.path.join(project_path, 'kplan/keypoint_grasp/data/completed_shape.pcd')
    pcd = open3d.read_point_cloud(pcdpath)
    pc_camera_frame = np.asarray(pcd.points)
    pc_world_frame = camera2world[0:3, 0:3].dot(pc_camera_frame.T)
    pc_world_frame[0, :] = pc_world_frame[0, :] + camera2world[0, 3]
    pc_world_frame[1, :] = pc_world_frame[1, :] + camera2world[1, 3]
    pc_world_frame[2, :] = pc_world_frame[2, :] + camera2world[2, 3]
    pc_world_frame = pc_world_frame.T

    # Construct the request
    request = KeypointGraspingServiceRequest()

    # The camera pose
    request.camera2world.orientation.w = 0.13363039718756375
    request.camera2world.orientation.x = -0.6601503565555793
    request.camera2world.orientation.y = 0.7226078483153184
    request.camera2world.orientation.z = -0.1555066597935349
    request.camera2world.position.x = 0.29972335333107686
    request.camera2world.position.y = -0.00016958877060524544
    request.camera2world.position.z = 0.8278206244574912

    # The keypoint
    point_0, point_1, point_2 = Point(), Point(), Point()
    point_0.x, point_0.y, point_0.z = keypoint_world[0, 0], keypoint_world[0, 1], keypoint_world[0, 2]
    point_1.x, point_1.y, point_1.z = keypoint_world[1, 0], keypoint_world[1, 1], keypoint_world[1, 2]
    point_2.x, point_2.y, point_2.z = keypoint_world[2, 0], keypoint_world[2, 1], keypoint_world[2, 2]
    request.keypoint_world_frame.append(point_0)
    request.keypoint_world_frame.append(point_1)
    request.keypoint_world_frame.append(point_2)

    # The point cloud
    request.pointcloud_world_frame = []
    for i in range(pc_world_frame.shape[0]):
        point_i = Point32()
        point_i.x = pc_world_frame[i, 0]
        point_i.x = pc_world_frame[i, 1]
        point_i.x = pc_world_frame[i, 2]
        request.pointcloud_world_frame.append(point_i)

    # Construct the service
    rospy.wait_for_service('plan_grasp')
    plan_grasp = rospy.ServiceProxy('plan_grasp', KeypointGraspingService)

    # Call the service
    response = plan_grasp(request)  # type: KeypointGraspingServiceResponse

    # To 4x4 matrix
    quat_fingertip_in_world = [
        response.gripper_fingertip_in_world.orientation.w,
        response.gripper_fingertip_in_world.orientation.x,
        response.gripper_fingertip_in_world.orientation.y,
        response.gripper_fingertip_in_world.orientation.z]
    T_fingertip_in_world = quaternion_matrix(quat_fingertip_in_world)
    T_fingertip_in_world[0, 3] = response.gripper_fingertip_in_world.position.x
    T_fingertip_in_world[1, 3] = response.gripper_fingertip_in_world.position.y
    T_fingertip_in_world[2, 3] = response.gripper_fingertip_in_world.position.z

    # visualize it
    from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
    visualizer = MeshCatVisualizer()
    visualizer.add_frame(T_fingertip_in_world, frame_scale=0.1)
    visualizer.add_pointcloud(pc_world_frame)


if __name__ == '__main__':
    run_grasp_planning()
