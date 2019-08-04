#! /usr/bin/env python
import rospy
import numpy as np
from kplan.utils.transformations import quaternion_from_matrix

from kplan.keypoint_grasp.mug_grasp_planner import MugUprightGraspPlanner
from kplan_ros.srv import KeypointGraspingService, KeypointGraspingServiceRequest, KeypointGraspingServiceResponse


class MugKeypointGraspPlanningServer(object):

    def __init__(self):
        self._planner = MugUprightGraspPlanner()

    def handle_grasp_planning_request(self, request):
        # type: (KeypointGraspingServiceRequest) -> KeypointGraspingServiceResponse
        # Decode the keypoint
        n_keypoint = len(request.keypoint_world_frame)
        mug_keypoint_in_world = np.zeros(shape=(n_keypoint, 3))
        for i in range(n_keypoint):
            mug_keypoint_in_world[i, 0] = request.keypoint_world_frame[i].x
            mug_keypoint_in_world[i, 1] = request.keypoint_world_frame[i].y
            mug_keypoint_in_world[i, 2] = request.keypoint_world_frame[i].z

        # Decode the cloud point
        n_cloud_point = len(request.pointcloud_world_frame)
        pc_in_world = np.zeros(shape=(n_cloud_point, 3))
        for i in range(n_cloud_point):
            pc_in_world[i, 0] = request.pointcloud_world_frame[i].x
            pc_in_world[i, 1] = request.pointcloud_world_frame[i].y
            pc_in_world[i, 2] = request.pointcloud_world_frame[i].z

        # Invoke the method
        T_fingertip_in_world = self._planner.plan(mug_keypoint_in_world, pc_in_world)
        quat_fingertip_in_world = quaternion_from_matrix(T_fingertip_in_world)

        # Constrcut the response
        response = KeypointGraspingServiceResponse()
        response.gripper_fingertip_in_world.orientation.w = quat_fingertip_in_world[0]
        response.gripper_fingertip_in_world.orientation.x = quat_fingertip_in_world[1]
        response.gripper_fingertip_in_world.orientation.y = quat_fingertip_in_world[2]
        response.gripper_fingertip_in_world.orientation.z = quat_fingertip_in_world[3]
        response.gripper_fingertip_in_world.position.x = T_fingertip_in_world[0, 3]
        response.gripper_fingertip_in_world.position.y = T_fingertip_in_world[1, 3]
        response.gripper_fingertip_in_world.position.z = T_fingertip_in_world[2, 3]

        # This one always success
        response.success = 1
        return response

    def run(self):
        rospy.init_node('keypoint_grasp_planning_server')
        rospy.Service('plan_grasp', KeypointGraspingService, self.handle_grasp_planning_request)
        print('The server for grasp planning initialization OK!')
        rospy.spin()


def main():
    server = MugKeypointGraspPlanningServer()
    server.run()


if __name__ == '__main__':
    main()
