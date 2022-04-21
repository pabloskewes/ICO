from metaheuristics.base_problem import Context, Solution, Neighborhood, SolutionSpace, Problem
from dataclasses import dataclass
import numpy as np
from typing import List
import random


# TODO: MUST IMPLEMENT FUNCTION TO IMPORT DATA OF MACHINES IN ANOTHER MODULE
# def load_flowshop(path, ...) -> FlowShopContext


@dataclass
class FlowShopContext(Context):
    machines: np.array
    num_machines: int


class FlowShopSolution(Solution):
    context: FlowShopContext = None

    def __init__(self, flowshop: List[int], params=None):
        super().__init__()

        self.flowshop = flowshop

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def cost(self) -> float:
        """  Returns the cost of a solution """
        pass


class FlowShopNeighborhood(Neighborhood):
    context: FlowShopContext = None

    def __init__(self, params=None):
        super().__init__()

        self.init_sol = 'random'

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def initial_solution(self) -> Solution:
        if self.init_sol == 'random':
            flowshop = list(range(self.context.num_machines))
            random.shuffle(flowshop)
            init_sol = FlowShopSolution(flowshop)
        elif isinstance(self.init_sol, FlowShopSolution):
            init_sol = self.init_sol
        else:
            raise Exception('Not a valid init sol')
        return init_sol

    def get_neighbor(self, solution: Solution) -> Solution:
        """ TODO: MUST IMPLEMENT """

    # TODO: MUST IMPLEMENT NEIGHBORHOOD FUNCTIONS


class FlowShopSolutionSpace(SolutionSpace):
    context: FlowShopContext = None

    def __init__(self, params=None):
        super().__init__()

        self.valid_params = []
        if params is not None:
            self.set_params(params)

    def distance(self, s1: Solution, s2: Solution) -> float:
        """ TODO: MUST IMPLEMENT """


class FlowShop(Problem):
    """  Flow Shop Scheduling Problem """
    solution = FlowShopSolution
    neighborhood = FlowShopNeighborhood
    solution_space = FlowShopSolutionSpace
