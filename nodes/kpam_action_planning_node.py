#! /usr/bin/env python
import os
import numpy as np
import argparse
from kplan.utils.transformations import quaternion_from_matrix

import rospy
from geometry_msgs.msg import Pose
from kplan_ros.srv import ActionPlanningkPAM, ActionPlanningkPAMRequest, ActionPlanningkPAMResponse
from kplan.kpam_opt.optimization_spec import OptimizationProblemSpecification
from kplan.kpam_opt.mp_builder import OptimizationBuilderkPAM
from kplan.kpam_opt.optimization_problem import solve_kpam


# The argument for the planner
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    default='',
                    help='The path to the specification of the kpam planning problem')


# The server
class ActionPlannerkPAMServer(object):

    def __init__(self, config_path):  # type: (str) -> None
        # Load the specification
        assert os.path.exists(config_path)
        self._optimization_sepcification = OptimizationProblemSpecification()
        self._optimization_sepcification.load_from_yaml(config_path)

        # The problem builder
        self._mp_builder = OptimizationBuilderkPAM(self._optimization_sepcification)

    def handle_planning_request(self, request):
        # type: (ActionPlanningkPAMRequest) -> ActionPlanningkPAMResponse
        n_keypoints = len(request.keypoint_world_frame)
        assert n_keypoints == len(self._optimization_sepcification.keypoint_name2idx)
        keypoints = np.zeros(shape=(n_keypoints, 3))
        for i in range(n_keypoints):
            keypoints[i, 0] = request.keypoint_world_frame[i].x
            keypoints[i, 1] = request.keypoint_world_frame[i].y
            keypoints[i, 2] = request.keypoint_world_frame[i].z

        # Build and solve it
        problem = self._mp_builder.build_optimization(keypoints)
        solve_kpam(problem)

        # Store the result
        response = ActionPlanningkPAMResponse()
        if not problem.has_solution:
            response.success = 0
            return response

        # The resolution is OK
        response.success = 1
        response.T_action = Pose()

        # The rotation part
        T_action_rot = np.eye(4)
        T_action_rot[0:3, 0:3] = problem.T_action[0:3, 0:3]
        T_action_quat = quaternion_from_matrix(T_action_rot)
        response.T_action.orientation.w = T_action_quat[0]
        response.T_action.orientation.x = T_action_quat[1]
        response.T_action.orientation.y = T_action_quat[2]
        response.T_action.orientation.z = T_action_quat[3]

        # The translation part
        response.T_action.position.x = problem.T_action[0, 3]
        response.T_action.position.y = problem.T_action[1, 3]
        response.T_action.position.z = problem.T_action[2, 3]
        return response

    def run(self):
        rospy.init_node('kpam_optimization_server')
        rospy.Service('plan_kpam_action', ActionPlanningkPAM, self.handle_planning_request)
        print('The server for kpam action planning initialization OK!')
        rospy.spin()


def main():
    # Parse the argument
    args, unknown = parser.parse_known_args()
    config_path = args.config_path  # type: str
    if len(config_path) <= 0:
        # The default
        node_path = os.path.dirname(__file__)
        kplan_ros_path = os.path.join(node_path, os.path.pardir)
        kplan_ros_path = os.path.abspath(kplan_ros_path)
        config_path = os.path.join(kplan_ros_path, 'kplan/kpam_opt/config/mug_on_shelf.yaml')

    # OK
    server = ActionPlannerkPAMServer(config_path)
    server.run()


if __name__ == '__main__':
    main()
