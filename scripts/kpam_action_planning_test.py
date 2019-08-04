import numpy as np
from kplan.utils.transformations import quaternion_matrix
import rospy
from geometry_msgs.msg import Point
from kplan_ros.srv import ActionPlanningkPAM, ActionPlanningkPAMRequest, ActionPlanningkPAMResponse


def run_kpam_action_planning():
    # The ros service
    rospy.wait_for_service('plan_kpam_action')
    plan_kpam_action = rospy.ServiceProxy('plan_kpam_action', ActionPlanningkPAM)

    # Some keypoint location
    point_0 = Point()
    point_0.x = 0.7765539536119968
    point_0.y = 0.2988056134902533
    point_0.z = -0.011688694634577002

    point_1 = Point()
    point_1.x = 0.6836153987929412
    point_1.y = 0.24241428970220136
    point_1.z = 0.13068491379520353

    point_2 = Point()
    point_2.x = 0.7278681586707589
    point_2.y = 0.2742936891124349
    point_2.z = 0.025218065353234786

    # Construct the request and plan it
    request = ActionPlanningkPAMRequest()
    request.keypoint_world_frame = []
    request.keypoint_world_frame.extend([point_0, point_1, point_2])
    result = plan_kpam_action(request)  # type: ActionPlanningkPAMResponse

    # The transformed keypoint
    quat = [
        result.T_action.orientation.w,
        result.T_action.orientation.x,
        result.T_action.orientation.y,
        result.T_action.orientation.z]
    T_action = quaternion_matrix(quat)
    T_action[0, 3] = result.T_action.position.x
    T_action[1, 3] = result.T_action.position.y
    T_action[2, 3] = result.T_action.position.z
    print 'The result on client'
    print T_action

    # Print the result
    keypoint_loc = np.zeros(shape=(3, 3))
    keypoint_loc[0, :] = np.asarray([0.7765539536119968, 0.2988056134902533, -0.011688694634577002])
    keypoint_loc[1, :] = np.asarray([0.6836153987929412, 0.24241428970220136, 0.13068491379520353])
    keypoint_loc[2, :] = np.asarray([0.7278681586707589, 0.2742936891124349, 0.025218065353234786])

    # Transform the keypoint
    from kplan.utils import SE3_utils
    print(
        'The transformed bottom center is: ',
        SE3_utils.transform_point(T_action, keypoint_loc[0, :]))
    print(
        'The transformed handle center is: ',
        SE3_utils.transform_point(T_action, keypoint_loc[1, :]))
    print(
        'The transformed top center is: ',
        SE3_utils.transform_point(T_action, keypoint_loc[2, :]))


if __name__ == '__main__':
    run_kpam_action_planning()
