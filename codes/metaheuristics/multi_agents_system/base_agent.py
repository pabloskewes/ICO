from .sequencial_models import SequentialModel
from ..base_problem import Solution
from .pools import BasePool

from mesa import Agent as MesaAgent


class BaseAgent(MesaAgent):
    def __init__(self, unique_id: int, model: SequentialModel,
                 push_to: List[BasePool] = None, pull_from: List[BasePool] = None):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.problem = model.problem
        self.pools = model.pools
        self.N = self.problem.NEIGHBORHOOD()
        self.push_to = push_to
        self.pull_from = pull_from
        self.in_solution = None
        self.out_solution = None

<<<<<<< HEAD
    def step(self):
        pass

    def push(self) -> None:
        if self.push_to is None:
            return
        for pool in self.push_to:
            pool.push(self.out_solution)

    def explore(self) -> Solution:
        """ Basic explore that simply looks for a neighborhood with the default configuration """
        N = self.N
        self.out_solution = N(self.in_solution)
        return self.out_solution
=======
    def explore(self, solution: Solution) -> Solution:
        """ Basic explore that simply looks for a neighborhood with the default configuration. It's normally overridden"""
        N = self.N
        new_sol = N(solution)
        return new_sol

>>>>>>> 8c43f0ae18ac863777f9d05864673cb4a51a334e
