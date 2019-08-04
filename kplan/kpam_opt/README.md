# The interface for kpam optimization

The kpam optimization module is initialized by a configuration file that describes the task name, object name and cost/constraint used in this task. In later query, the kpam optimization acts as a server that takes in sequence of keypoints and return the transformation that maps the current pose to target pose. 

The format of the yaml file is

```yaml
task_name: mug_on_shelf
category_name: mug
keypoint_name_list: [bottom_center, handle_center, top_center]
keypoint_nominal_target_position: {}
constraint_list:
- keypoint_idx: 0
  keypoint_name: bottom_center
  target_position: [0.68501, -0.012, 0.01673]
  tolerance: 0.001
  type: point2point_constraint
- axis_from_keypoint_idx: 0
  axis_from_keypoint_name: bottom_center
  axis_to_keypoint_idx: 2
  axis_to_keypoint_name: top_center
  target_axis: [0, 0, 1.0]
  tolerance: 0.1
  type: keypoint_axis_alignment
cost_list:
- keypoint_idx: 2
  keypoint_name: top_center
  penalty_weight: 1.0
  target_position: [0.59347, 0.01889, 0.01464]
  type: point2point_cost
- keypoint_idx: -1
  keypoint_name: handle_center
  penalty_weight: 1.0
  target_position: [0.61868, -0.04039, -0.00272]
  type: point2point_cost
```

Note that the `keypoint_name_list` must be the same order as the detected keypoint. 

The `keypoint_idx` in cost/constraint are just for checking and can be left as -1. The `keypoint_name` is the main index.

The yaml file can be constructed directly or the optimization can be defined in python and exported to yaml. Please refer to `optimization_spec.py` for details.