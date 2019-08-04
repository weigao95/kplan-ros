#### KeyPoint Grasping

This python package contains the "hacky" grasp planner used for generalizable manipulation. More general grasp planner should be used instead, but we don't want to spend too much time on the integration of a learning based grasp planner as well as its training/data processing code.

An interesting direction is to jointly optimize the grasp quality with task dependent trajectory, similar to the work of "[Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision](<https://sites.google.com/view/task-oriented-grasp/>)". Based on dense geometry and keypoint, we might be able to do something better than directly RL.