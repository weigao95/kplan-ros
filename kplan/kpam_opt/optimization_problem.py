import attr
import numpy as np
import kplan.utils.SE3_utils as SE3_utils
try:
    from pydrake.solvers.mathematicalprogram import MathematicalProgram, SolutionResult
except ImportError:
    from pydrake.all import MathematicalProgram, Solve

@attr.s
class OptimizationProblemkPAM(object):
    # The property about initial transformation
    # There is always an initial guess, either identity or from keypoint matching
    T_init = np.ndarray(shape=(4, 4))  # type: np.ndarray

    # The final mathematical program
    mp = None  # type: MathematicalProgram
    xyzrpy = None  # type: np.ndarray

    # The solution to the program
    has_solution = False
    xyzrpy_sol = np.ndarray(shape=(6,))
    # T_action.dot(observed_keypoint) = target_keypoint
    T_action = np.ndarray(shape=(4, 4))

    # Build empty problem
    def build_empty_mp(self):
        # Construct the problem
        mp = MathematicalProgram()
        xyz = mp.NewContinuousVariables(3, 'xyz')
        rpy = mp.NewContinuousVariables(3, 'rpy')
        xyz_rpy = np.concatenate((xyz, rpy))
        mp.SetInitialGuessForAllVariables(np.zeros(6))

        # Store the result to problem
        self.mp = mp
        self.xyzrpy = xyz_rpy


def solve_kpam(problem):  # type: (OptimizationProblemkPAM) -> bool
    result = problem.mp.Solve()
    if not result is not SolutionResult.kSolutionFound:
        problem.has_solution = False
        return False

    # Save the result to problem
    problem.xyzrpy_sol = problem.mp.GetSolution(problem.xyzrpy)
    T_eps = SE3_utils.xyzrpy_to_matrix(xyzrpy=problem.xyzrpy_sol)
    problem.T_action = np.dot(problem.T_init, T_eps)
    problem.has_solution = True

    # OK
    return True
