from solution import VRPTWSolution
from neighborhood import VRPTWNeighborhood
from metaheuristics.base_problem import Problem


class VRPTW(Problem):
    solution = VRPTWSolution
    neighborhood = VRPTWNeighborhood

