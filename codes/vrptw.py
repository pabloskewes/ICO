from context import VRPTWContext
from solution import VRPTWSolution
from neighborhood import VRPTWNeighborhood
from metaheuristics.base_problem import Problem


class VRPTW(Problem):
    solution = VRPTWSolution
    neighborhood = VRPTWNeighborhood

    def __init__(self, context):
        self.context = context
        VRPTW.solution.set_class_context(context)




