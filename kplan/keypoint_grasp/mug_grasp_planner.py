import numpy as np
import kplan.utils.transformations as transformations
import kplan.utils.SE3_utils as SE3_utils


class MugUprightGraspPlanner(object):

    def __init__(self):
        pass

    @staticmethod
    def plan(mug_keypoint_world, pc_in_world):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        A simple dispatcher
        :param mug_keypoint_world:
        :param pc_in_world:
        :return:
        """
        success, grasp_frame = MugUprightGraspPlanner.plan_with_geometry(mug_keypoint_world, pc_in_world)
        if success:
            # print('Use geometry planning')
            return grasp_frame
        else:
            return MugUprightGraspPlanner.plan_heuristic(mug_keypoint_world, pc_in_world)

    @staticmethod
    def plan_heuristic(mug_keypoint_world, pc_in_world):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        Plan the grasp frame of mugs that are placed upright on the table.
        The output is the grasp finger tip frame
        :param mug_keypoint_world: (N, 3) mug keypoints in the order of (bottom_center, handle_center, top_center)
        :param pc_in_world: (N, 3) mug point cloud expressed in world frame.
        :return: 4x4 np.ndarray of gripper fingertip frame
        """
        # Read the keypoint
        assert mug_keypoint_world.shape[0] == 3
        assert mug_keypoint_world.shape[1] == 3
        bottom_center = mug_keypoint_world[0, :]
        handle_center = mug_keypoint_world[1, :]
        top_center = mug_keypoint_world[2, :]

        # The top center averaged in xy plane
        center_avg = (bottom_center + top_center) / 2.0  # average of top and bottom keypoint detections
        top_center_avg = np.copy(top_center)
        top_center_avg[2] = top_center[2]

        # The orientation of the finger-tip frame
        grasp_x_axis = np.array([0, 0, -1])
        grasp_y_axis = center_avg - handle_center
        grasp_y_axis[2] = 0
        grasp_y_axis_orig = grasp_y_axis
        grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
        grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)

        # The center of grasping
        center_to_handle_dist = np.linalg.norm(grasp_y_axis_orig)
        grasp_center = top_center_avg + grasp_y_axis * (center_to_handle_dist - 0.01)
        grasp_center[2] = (top_center[2] + bottom_center[2]) / 2.0

        # Fix the center of grasping for tall mugs
        if top_center[2] - grasp_center[2] > 0.04:
            grasp_center[2] = top_center[2] - 0.04

        # OK
        grasp_fingertip_in_world = np.eye(4)
        grasp_fingertip_in_world[0:3, 0] = grasp_x_axis
        grasp_fingertip_in_world[0:3, 1] = grasp_y_axis
        grasp_fingertip_in_world[0:3, 2] = grasp_z_axis
        grasp_fingertip_in_world[0:3, 3] = grasp_center
        return grasp_fingertip_in_world

    @staticmethod
    def plan_with_geometry(mug_keypoint_world, pc_in_world):
        # type: (np.ndarray, np.ndarray) -> (bool, np.ndarray)
        """
        Plan the grasp frame of mugs that are placed upright on the table.
        The output is the grasp finger tip frame
        :param mug_keypoint_world: (N, 3) mug keypoints in the order of (bottom_center, handle_center, top_center)
        :param pc_in_world: (N, 3) mug point cloud expressed in world frame.
        :return: 4x4 np.ndarray of gripper fingertip frame
        """
        # Read the keypoint
        assert mug_keypoint_world.shape[0] == 3
        assert mug_keypoint_world.shape[1] == 3
        assert pc_in_world.shape[1] == 3
        bottom_center = mug_keypoint_world[0, :]
        handle_center = mug_keypoint_world[1, :]
        top_center = mug_keypoint_world[2, :]

        # The top center averaged in xy plane
        center_avg = (bottom_center + top_center) / 2.0  # average of top and bottom keypoint detections
        top_center_avg = np.copy(top_center)
        top_center_avg[2] = top_center[2]

        # The orientation of the finger-tip frame
        grasp_x_axis = np.array([0, 0, -1])
        grasp_y_axis = center_avg - handle_center
        grasp_y_axis[2] = 0
        grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
        grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)

        # The cropping box
        crop_frame_origin = top_center_avg
        T_crop_box_frame_in_world = np.eye(4)
        T_crop_box_frame_in_world[0:3, 0] = grasp_x_axis
        T_crop_box_frame_in_world[0:3, 1] = grasp_y_axis
        T_crop_box_frame_in_world[0:3, 2] = grasp_z_axis
        T_crop_box_frame_in_world[0:3, 3] = crop_frame_origin
        T_world_to_crop_frame = transformations.inverse_matrix(T_crop_box_frame_in_world)

        # Transform the pc to crop frame and crop it
        pc_in_crop_frame = SE3_utils.transform_point_cloud(T_world_to_crop_frame, pc_in_world)

        # The cropping bbox
        bbox_min = np.array([-0.02, 0.0, -0.01])
        bbox_max = np.array([0.02, 0.1, 0.01])
        mask_x = np.logical_and(pc_in_crop_frame[:, 0] > bbox_min[0], pc_in_crop_frame[:, 0] < bbox_max[0])
        mask_y = np.logical_and(pc_in_crop_frame[:, 1] > bbox_min[1], pc_in_crop_frame[:, 1] < bbox_max[1])
        mask_z = np.logical_and(pc_in_crop_frame[:, 2] > bbox_min[2], pc_in_crop_frame[:, 2] < bbox_max[2])
        mask_xy = np.logical_and(mask_x, mask_y)
        mask_xyz = np.logical_and(mask_xy, mask_z)
        cropped_pc_in_crop_frame = pc_in_crop_frame[mask_xyz]  # type: np.ndarray

        # Not enough points, just return
        if cropped_pc_in_crop_frame.size < 20:
            return False, np.ndarray(shape=(4, 4))

        # The grasp center in cropped frame
        grasp_center_crop_frame = [0, 0, 0]
        grasp_center_crop_frame[0] = np.min(cropped_pc_in_crop_frame[:, 0])
        grasp_center_crop_frame[1] = np.average(cropped_pc_in_crop_frame[:, 1])
        grasp_center_world_frame = SE3_utils.transform_point(
            T_crop_box_frame_in_world,
            np.asarray(grasp_center_crop_frame))

        # OK
        grasp_fingertip_in_world = np.eye(4)
        grasp_fingertip_in_world[0:3, 0] = grasp_x_axis
        grasp_fingertip_in_world[0:3, 1] = grasp_y_axis
        grasp_fingertip_in_world[0:3, 2] = grasp_z_axis
        grasp_fingertip_in_world[0:3, 3] = grasp_center_world_frame
        return True, grasp_fingertip_in_world


def test_planner_simple():
    test_keypoint = np.zeros((3, 3))
    test_keypoint[0, :] = np.array([0, 0, 0])
    test_keypoint[1, :] = np.array([0, 0.03, 0.04])
    test_keypoint[2, :] = np.array([0, 0, 0.07])  # 7cm tall
    gripper_fingertip_frame = MugUprightGraspPlanner.plan_heuristic(test_keypoint, None)

    # Visualize
    from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
    visualizer = MeshCatVisualizer()
    visualizer.add_frame(gripper_fingertip_frame, frame_scale=0.1)


def test_planner_real_data():
    from kplan.utils.transformations import quaternion_matrix
    # The transformation matrix
    camera2world = quaternion_matrix([0.13363039718756375, -0.6601503565555793, 0.7226078483153184, -0.1555066597935349])
    camera2world[0, 3] = 0.29972335333107686
    camera2world[1, 3] = -0.00016958877060524544
    camera2world[2, 3] = 0.8278206244574912

    # The keypoint in camera frame
    keypoint_camera = np.zeros((3, 3))
    keypoint_camera[0, :] = np.array([-0.0710507483878, -0.0075068516829, 0.909739379883])
    keypoint_camera[1, :] = np.array([-0.0641933829504,  0.0298324972555, 0.838331695557])
    keypoint_camera[2, :] = np.array([-0.0684536122998, -0.0403231180828, 0.821209274292])

    # Transformed into world
    keypoint_world = np.zeros_like(keypoint_camera)
    for i in range(3):
        keypoint_world[i, :] = camera2world[0:3, 0:3].dot(keypoint_camera[i, :]) + camera2world[0:3, 3]

    # Read the point cloud
    import open3d, os
    filepath = os.path.dirname(os.path.realpath(__file__))
    pcdpath = os.path.join(filepath, 'data/completed_shape.pcd')

    # Read the point cloud
    pcd = open3d.read_point_cloud(pcdpath)
    pc_camera_frame = np.asarray(pcd.points)
    pc_world_frame = camera2world[0:3, 0:3].dot(pc_camera_frame.T)
    pc_world_frame[0, :] = pc_world_frame[0, :] + camera2world[0, 3]
    pc_world_frame[1, :] = pc_world_frame[1, :] + camera2world[1, 3]
    pc_world_frame[2, :] = pc_world_frame[2, :] + camera2world[2, 3]
    pc_world_frame = pc_world_frame.T

    # Test it
    gripper_fingertip_frame = MugUprightGraspPlanner.plan(keypoint_world, pc_world_frame)

    # visualize it
    from kplan.visualizer.meshcat_wrapper import MeshCatVisualizer
    visualizer = MeshCatVisualizer()
    visualizer.add_frame(gripper_fingertip_frame, frame_scale=0.1)
    visualizer.add_pointcloud(pc_world_frame)


if __name__ == '__main__':
    test_planner_real_data()
