import numpy as np
import kplan.utils.transformations as transformations
import kplan.utils.SE3_utils as SE3_utils


class ShoeUprightGraspPlanner(object):

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
        success, grasp_frame = ShoeUprightGraspPlanner.plan_with_geometry(mug_keypoint_world, pc_in_world)
        if success:
            return grasp_frame
        else:
            return ShoeUprightGraspPlanner.plan_heuristic(mug_keypoint_world, pc_in_world)

    @staticmethod
    def plan_heuristic(shoe_keypoint_world, pc_in_world):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        Plan the grasp frame of mugs that are placed upright on the table.
        The output is the grasp finger tip frame
        :param shoe_keypoint_world: (N, 3) mug keypoints in the order of (bottom_center, handle_center, top_center)
        :param pc_in_world: (N, 3) mug point cloud expressed in world frame.
        :return: 4x4 np.ndarray of gripper fingertip frame
        """
        # Read the keypoint
        assert shoe_keypoint_world.shape[0] == 6
        assert shoe_keypoint_world.shape[1] == 3
        heel_bottom = shoe_keypoint_world[2, :]
        heel_top = shoe_keypoint_world[4, :]
        tongue_kp = shoe_keypoint_world[5, :]

        # Construct the grasping
        grasp_x_axis = np.array([0, 0, -1])
        heel_center = 1 / 2.0 * (heel_bottom + heel_top)
        grasp_y_axis = heel_center - tongue_kp
        grasp_y_axis[2] = 0
        grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
        grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)
        grasp_center = heel_top

        # OK
        grasp_fingertip_in_world = np.eye(4)
        grasp_fingertip_in_world[0:3, 0] = grasp_x_axis
        grasp_fingertip_in_world[0:3, 1] = grasp_y_axis
        grasp_fingertip_in_world[0:3, 2] = grasp_z_axis
        grasp_fingertip_in_world[0:3, 3] = grasp_center
        return grasp_fingertip_in_world

    @staticmethod
    def plan_with_geometry(shoe_keypoint_world, pc_in_world):
        # type: (np.ndarray, np.ndarray) -> (bool, np.ndarray)
        # Read the keypoint
        assert shoe_keypoint_world.shape[0] == 6
        assert shoe_keypoint_world.shape[1] == 3
        heel_bottom = shoe_keypoint_world[2, :]
        heel_top = shoe_keypoint_world[4, :]
        tongue_kp = shoe_keypoint_world[5, :]

        # Construct the grasping
        grasp_x_axis = np.array([0, 0, -1])
        heel_center = 1 / 2.0 * (heel_bottom + heel_top)
        grasp_y_axis = heel_center - tongue_kp
        grasp_y_axis[2] = 0
        grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
        grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)

        # The center of crop frame
        crop_frame_origin = heel_center
        crop_frame_origin[2] = 1 / 2.0 * (heel_top[2] + tongue_kp[2])
        T_crop_box_frame_in_world = np.eye(4)
        T_crop_box_frame_in_world[0:3, 0] = grasp_x_axis
        T_crop_box_frame_in_world[0:3, 1] = grasp_y_axis
        T_crop_box_frame_in_world[0:3, 2] = grasp_z_axis
        T_crop_box_frame_in_world[0:3, 3] = crop_frame_origin
        T_world_to_crop_frame = transformations.inverse_matrix(T_crop_box_frame_in_world)

        # Transform the pc to crop frame and crop it
        pc_in_crop_frame = SE3_utils.transform_point_cloud(T_world_to_crop_frame, pc_in_world)

        # The cropping bbox
        bbox_min = np.array([-0.05, -0.02, -0.01])
        bbox_max = np.array([ 0.10,  0.1,   0.01])
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
        grasp_center_crop_frame[1] -= 0.01  # to help us not hit the heel
        # grasp_center_crop_frame[1] = np.max(cropped_pc_in_crop_frame[:, 1])
        # grasp_center_crop_frame[1] -= 0.01  # to help us not hit the heel
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
