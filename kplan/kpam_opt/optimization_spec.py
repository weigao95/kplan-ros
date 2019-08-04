from kplan.kpam_opt.term_spec import OptimizationTermSpec, Point2PointConstraintSpec, Point2PointCostL2Spec
from typing import List, Dict
import copy
import yaml
import os


class OptimizationProblemSpecification(object):
    """
    The class serves as the interface between the
    config file and solver
    """
    def __init__(
        self,
        task_name='',  # type: str
        category_name='',  # type: str
        keypoint_name_list=None  # type: List[str]
    ):
        """
        The optimization spec can be constructed in python code
        or load from yaml config path. In latter case, these
        parameters can left default and use load_from_yaml method
        :param task_name:
        :param category_name:
        :param keypoint_name_list:
        """
        self._task_name = task_name  # type: str
        self._category_name = category_name  # type: str

        # The default construction of list
        if keypoint_name_list is not None:
            self._keypoint_name_list = keypoint_name_list  # type: List[str]
        else:
            self._keypoint_name_list = []

        # By default, nothing here
        self._cost_list = []  # type: List[OptimizationTermSpec]
        self._constraint_list = []  # type: List[OptimizationTermSpec]

        # Build from keypoint name list
        self._keypoint_name2idx = {}  # type: Dict[str, int]
        self.setup_keypoint_mapping()

        # The container for explicit nominal position
        # The nominal position can either explicit declared
        # Or implicit added using Point2PointCost/Constraint
        self._keypoint_nominal_target_position = {}  # type: Dict[str, List[float]]

    def setup_keypoint_mapping(self):
        self._keypoint_name2idx.clear()
        for i in range(len(self._keypoint_name_list)):
            name = self._keypoint_name_list[i]
            self._keypoint_name2idx[name] = i

    # The access interface
    @property
    def task_name(self):
        return self._task_name

    @property
    def category_name(self):
        return self._category_name

    @property
    def keypoint_name2idx(self):
        return self._keypoint_name2idx

    @property
    def cost_list(self):
        return self._cost_list

    @property
    def constraint_list(self):
        return self._constraint_list

    # The method to manipulate nominal target
    def add_nominal_target_position(self, keypoint_name, nominal_target):  # type: (str, List[float]) -> bool
        # Check the existence of keypoint
        if keypoint_name not in self._keypoint_name2idx:
            return False

        # Check the shape of target
        if len(nominal_target) != 3:
            return False

        # OK
        self._keypoint_nominal_target_position[keypoint_name] = nominal_target
        return True

    def get_nominal_target_position(self, keypoint_name):  # type: (str) -> (bool, List[float])
        # If explicitly defined
        if keypoint_name in self._keypoint_nominal_target_position:
            return True, self._keypoint_nominal_target_position[keypoint_name]

        # Else search for constraints
        for cost_term in self.cost_list:
            if isinstance(cost_term, Point2PointCostL2Spec):
                if cost_term.keypoint_name == keypoint_name:
                    return True, cost_term.target_position
        for constraint_term in self.constraint_list:
            if isinstance(constraint_term, Point2PointConstraintSpec):
                if constraint_term.keypoint_name == keypoint_name:
                    return True, constraint_term.target_position

        # Not available
        return False, []

    # The method to modify the specification from python
    def add_cost(self, cost_term):  # type: (OptimizationTermSpec) -> bool
        if not cost_term.is_cost():
            return False
        copied = copy.deepcopy(cost_term)
        self._cost_list.append(copied)
        return True

    def add_constraint(self, constraint_term):  # type: (OptimizationTermSpec) -> bool
        if constraint_term.is_cost():
            return False
        copied = copy.deepcopy(constraint_term)
        self._constraint_list.append(copied)
        return True

    def add_optimization_term(self, optimization_term):  # type: (OptimizationTermSpec) -> bool
        if optimization_term.is_cost():
            return self.add_cost(optimization_term)
        else:
            return self.add_constraint(optimization_term)

    # The interface from/to yaml
    def write_to_yaml(self, yaml_save_path):  # type: (str) -> None
        data_map = dict()
        data_map['task_name'] = self._task_name
        data_map['category_name'] = self._category_name
        data_map['keypoint_name_list'] = self._keypoint_name_list

        # For cost terms
        cost_map_list = []
        for cost in self._cost_list:
            cost_i_map = cost.to_dict()
            cost_i_map['type'] = cost.type_name()
            cost_map_list.append(cost_i_map)
        data_map['cost_list'] = cost_map_list

        # For constraint terms
        constraint_map_list = []
        for constraint in self._constraint_list:
            constraint_i_map = constraint.to_dict()
            constraint_i_map['type'] = constraint.type_name()
            constraint_map_list.append(constraint_i_map)
        data_map['constraint_list'] = constraint_map_list

        # For nominal location
        data_map['keypoint_nominal_target_position'] = self._keypoint_nominal_target_position

        # Save to yaml
        with open(yaml_save_path, mode='w') as save_file:
            yaml.dump(data_map, save_file)
            save_file.close()

    def load_from_yaml(self, yaml_load_path):  # type: (str) -> bool
        if not os.path.exists(yaml_load_path):
            return False

        # Read the file
        load_file = open(yaml_load_path, mode='r')
        data_map = yaml.load(load_file, Loader=yaml.SafeLoader)
        load_file.close()

        # Basic meta
        self._task_name = data_map['task_name']
        self._category_name = data_map['category_name']
        self._keypoint_name_list = data_map['keypoint_name_list']
        self._keypoint_name2idx.clear()
        self.setup_keypoint_mapping()

        # The nominal location
        self._keypoint_nominal_target_position = data_map['keypoint_nominal_target_position']

        # For cost terms
        import kplan.kpam_opt.term_spec as term_spec
        cost_map_list = data_map['cost_list']
        self._cost_list = []
        for cost in cost_map_list:
            cost_type = cost['type']  # type: str
            if cost_type == term_spec.Point2PointCostL2Spec.type_name():
                cost_spec = term_spec.Point2PointCostL2Spec()
                cost_spec.from_dict(cost)
                self._cost_list.append(cost_spec)
            elif cost_type == term_spec.Point2PlaneCostSpec.type_name():
                cost_spec = term_spec.Point2PlaneCostSpec()
                cost_spec.from_dict(cost)
                self._cost_list.append(cost_spec)
            else:
                raise RuntimeError('Unknown cost type %s' % cost_type)

        # For constraint terms
        constraint_map_list = data_map['constraint_list']
        self._constraint_list = []
        for constraint in constraint_map_list:
            constraint_type = constraint['type']
            if constraint_type == term_spec.Point2PointConstraintSpec.type_name():
                constraint_spec = term_spec.Point2PointConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            elif constraint_type == term_spec.AxisAlignmentConstraintSpec.type_name():
                constraint_spec = term_spec.AxisAlignmentConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            elif constraint_type == term_spec.KeypointAxisAlignmentConstraintSpec.type_name():
                constraint_spec = term_spec.KeypointAxisAlignmentConstraintSpec()
                constraint_spec.from_dict(constraint)
                self._constraint_list.append(constraint_spec)
            else:
                raise RuntimeError('Unknown constraint type %s' % constraint_type)

        # OK
        return True


# The debugging code
def build_mug2rack():  # type: () -> OptimizationProblemSpecification
    # The init
    optimization = OptimizationProblemSpecification(
        'mug_on_rack',
        'mug',
        ['bottom_center', 'handle_center', 'top_center'])
    # Add different cost/constraint terms
    assert len(optimization.cost_list) == 0

    # Add cost term
    from kplan.kpam_opt.term_spec import Point2PointCostL2Spec, Point2PointConstraintSpec
    # The bottom center
    cost_bottom_center = Point2PointCostL2Spec()
    cost_bottom_center.keypoint_name = 'bottom_center'
    cost_bottom_center.keypoint_idx = 0
    cost_bottom_center.target_position = [0.68501, -0.0120, 0.01673]
    assert optimization.add_optimization_term(cost_bottom_center)

    # The top center
    cost_top_center = Point2PointCostL2Spec()
    cost_top_center.keypoint_name = 'top_center'
    cost_top_center.keypoint_idx = 2
    cost_top_center.target_position = [0.59347, 0.01889, 0.01464]
    assert optimization.add_optimization_term(cost_top_center)

    # The constraint on handle center
    constraint_handle_center = Point2PointConstraintSpec()
    constraint_handle_center.keypoint_name = 'handle_center'
    constraint_handle_center.keypoint_id = 1
    constraint_handle_center.target_position = [0.61868, -0.04039, -0.00272]
    constraint_handle_center.tolerance = 1e-3
    assert optimization.add_optimization_term(constraint_handle_center)

    # Finsh check
    assert len(optimization.cost_list) == 2
    assert len(optimization.constraint_list) == 1
    return optimization


def build_mug2shelf():
    # The init
    optimization = OptimizationProblemSpecification(
        'mug_on_shelf',
        'mug',
        ['bottom_center', 'handle_center', 'top_center'])
    # Add different cost/constraint terms
    assert len(optimization.cost_list) == 0

    # Add cost term
    from kplan.kpam_opt.term_spec import (
        Point2PointCostL2Spec,
        Point2PointConstraintSpec,
        KeypointAxisAlignmentConstraintSpec,
        Point2PlaneCostSpec
    )

    # The bottom center
    constraint_bottom_center = Point2PointConstraintSpec()
    constraint_bottom_center.keypoint_name = 'bottom_center'
    constraint_bottom_center.keypoint_idx = 0
    constraint_bottom_center.target_position = [0.68501, -0.0120, 0.01673]
    constraint_bottom_center.tolerance = 0.001  # 1 [mm]
    assert optimization.add_optimization_term(constraint_bottom_center)

    # The axis alignment
    axis_alignment = KeypointAxisAlignmentConstraintSpec()
    axis_alignment.axis_from_keypoint_name = 'bottom_center'
    axis_alignment.axis_to_keypoint_name = 'top_center'
    axis_alignment.axis_from_keypoint_idx = 0
    axis_alignment.axis_to_keypoint_idx = 2
    axis_alignment.target_axis = [0, 0, 1.0]
    axis_alignment.tolerance = 0.1  # 1 - eps <= dot(from, to) <= 1
    assert optimization.add_optimization_term(axis_alignment)

    # The top center
    cost_top_center = Point2PointCostL2Spec()
    cost_top_center.keypoint_name = 'top_center'
    cost_top_center.keypoint_idx = 2
    cost_top_center.target_position = [0.59347, 0.01889, 0.01464]
    assert optimization.add_optimization_term(cost_top_center)

    # The constraint on handle center
    cost_handle_center = Point2PointCostL2Spec()
    cost_handle_center.keypoint_name = 'handle_center'
    cost_handle_center.keypoint_id = 1
    cost_handle_center.target_position = [0.61868, -0.04039, -0.00272]
    assert optimization.add_optimization_term(cost_handle_center)

    # The point to plane cost
    cost_bottom_center = Point2PlaneCostSpec()
    cost_bottom_center.keypoint_name = 'bottom_center'
    cost_bottom_center.keypoint_idx = 0
    cost_bottom_center.target_position = [0.68501, -0.0120, 0.01673]
    cost_bottom_center.plane_normal = [0.0, 0.0, 1.0]
    assert optimization.add_optimization_term(cost_bottom_center)

    # Finish check
    optimization.write_to_yaml('test.yaml')
    optimization_load = OptimizationProblemSpecification()
    assert optimization_load.load_from_yaml('test.yaml')
    assert len(optimization_load.cost_list) == 3
    assert len(optimization_load.constraint_list) == 2
    return optimization_load


if __name__ == '__main__':
    opt = build_mug2shelf()
    print(opt.get_nominal_target_position('top_center'))
