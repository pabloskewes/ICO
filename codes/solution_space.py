from typing import List, Set, Union, Optional, Dict
from pprint import pformat

from metaheuristics.base_problem import SolutionSpace, Solution
from context import VRPTWContext
from solution import VRPTWSolution


class VRPTWSolutionSpace(SolutionSpace):
    context: VRPTWContext = None

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def distance(self, s1: VRPTWSolution, s2: VRPTWSolution):
        A, B = set(s1.graph), set(s2.graph)
        return len(A.symmetric_difference(B))


