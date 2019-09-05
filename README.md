# kplan_ros

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides ros service for kPAM optimization and grasp planning.

### Install Instruction

- Clone this repo into your `catkin workspace` by `git clone https://github.com/weigao95/kplan-ros kplan_ros` (Note the underscore)
- Build the message types by `catkin_make`
- To run the code in `nodes/` and `scripts/`, you need to add `${project_path}` to `PYTHONPATH` [1]. You might run `export PYTHONPATH="${project_path}:${PYTHONPATH}"`

### Run kPAM Action Planning

The costs and constraints to specifiy the manipulation task are serialized into `yaml` files. Examples for config files are provided in `${project_root}/kplan/kpam_opt/config/`. To test kPAM optimization

- Start the server by `python nodes/kpam_action_planning_node.py --config_path path/to/config`
- Run the test by `python scripts/kpam_action_planning_test.py` 

### Run Grasp Planning

Since the [kPAM](https://github.com/weigao95/kPAM)  project is mainly about object *placing*, the grasp planner we use is little bit hacky. To try it, please run `python nodes/mug_keypoint_grasping_node.py`

### MISC

[1] The official way for python package installation is the `setup.py` file in `ros`. However, I always forgot to re-run `catkin_make` after modifying the python code, which causes confusion. Thus, I prefer to directly add `${project_path}` to `PYTHONPATH`.
