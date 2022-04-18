from solution import VRPTWSolution
from neighborhood import VRPTWNeighborhood
from solution_space import VRPTWSolutionSpace
from flexible_vrptw import FlexVRPTWSolution, FlexVRPTWNeighborhood, FlexVRPTWSolutionSpace
from metaheuristics.base_problem import Problem


class VRPTW(Problem):
    solution = VRPTWSolution
    neighborhood = VRPTWNeighborhood
    solution_space = VRPTWSolutionSpace


class FlexVRPTW(Problem):
    solution = FlexVRPTWSolution
    neighborhood = FlexVRPTWNeighborhood
    solution_space = FlexVRPTWSolutionSpace
