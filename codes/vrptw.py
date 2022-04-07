from solution import VRPTWSolution
from neighborhood import VRPTWNeighborhood
from solution_space import VRPTWSolutionSpace
from metaheuristics.base_problem import Problem


class VRPTW(Problem):
    solution = VRPTWSolution
    neighborhood = VRPTWNeighborhood
    solution_space = VRPTWSolutionSpace

