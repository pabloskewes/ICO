from typing import List, Set, Union, Optional, Dict
from pprint import pformat
from metaheuristics.base_problem import Solution
from context import VRPTWContext


class VRPTWSolutionSpace(SolutionSpace):
    context: VRPTWContext = None

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def distance(self, s1: Solution, s2: Solution):
        A, B = set(s1.graph), set(s2.graph)
        return len(A.symmetric_difference(B))
    